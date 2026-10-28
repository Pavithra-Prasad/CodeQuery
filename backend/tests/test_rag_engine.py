"""Tests for the RAG engine: prompt context, source formatting and the Groq -> Gemini failover."""

import asyncio
import sys
import types

import pytest

import rag_engine
import vector_store as vs
from fakes import fake_embed


def results(docs, metas=None, distances=None):
    return {
        "documents": docs,
        "metadatas": metas or [{} for _ in docs],
        "distances": distances or [0.1 for _ in docs],
    }


# ── build_context / format_sources ──────────────────────────────

def test_build_context_with_no_results():
    assert rag_engine.build_context(results([])) == "No relevant code found in the indexed codebase."


def test_build_context_labels_each_source_with_file_and_lines():
    ctx = rag_engine.build_context(results(
        ["def login(): ..."],
        [{"file_path": "auth.py", "language": "python", "start_line": 4, "end_line": 9}],
    ))
    assert "### Source 1: `auth.py` (lines 4-9)" in ctx
    assert "```python\ndef login(): ...\n```" in ctx


def test_format_sources_removes_duplicates_and_shortens_snippets():
    meta = {"file_path": "auth.py", "language": "python", "start_line": 1, "end_line": 40}
    sources = rag_engine.format_sources(results(["x" * 300, "duplicate"], [meta, meta], [0.25, 0.3]))
    assert len(sources) == 1
    assert sources[0]["snippet"] == "x" * 200 + "..."
    assert sources[0]["relevance"] == 0.875              # cosine similarity: 1 - 0.25 / 2


def test_relevance_drops_as_distance_grows():
    metas = [{"file_path": f"{n}.py", "start_line": 1} for n in ("close", "far", "opposite")]
    close, far, opposite = rag_engine.format_sources(results(["a", "b", "c"], metas, [0.2, 1.4, 4.0]))
    assert close["relevance"] > far["relevance"] > opposite["relevance"] == 0


# ── LLM failover ────────────────────────────────────────────────

@pytest.fixture
def providers(monkeypatch):
    """Fake Groq and Gemini calls that record which provider was used."""
    calls = []
    state = {"groq_fails": False, "gemini_fails": False}

    def groq(prompt):
        calls.append("groq")
        if state["groq_fails"]:
            raise RuntimeError("rate limited")
        return "answer from groq"

    def gemini(prompt):
        calls.append("gemini")
        if state["gemini_fails"]:
            raise RuntimeError("quota exceeded")
        return "answer from gemini"

    monkeypatch.setattr(rag_engine, "_call_groq", groq)
    monkeypatch.setattr(rag_engine, "_call_gemini", gemini)
    return calls, state


def test_groq_is_used_first(providers, monkeypatch):
    calls, _ = providers
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    assert rag_engine._call_llm("q") == "answer from groq"
    assert calls == ["groq"]


def test_falls_back_to_gemini_when_groq_fails(providers, monkeypatch):
    calls, state = providers
    state["groq_fails"] = True
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    assert rag_engine._call_llm("q") == "answer from gemini"
    assert calls == ["groq", "gemini"]


def test_uses_gemini_directly_when_only_its_key_is_set(providers, monkeypatch):
    calls, _ = providers
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    assert rag_engine._call_llm("q") == "answer from gemini"
    assert calls == ["gemini"]


def test_error_names_both_providers_when_both_fail(providers, monkeypatch):
    _, state = providers
    state["groq_fails"] = state["gemini_fails"] = True
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    with pytest.raises(Exception, match="All LLM providers failed") as err:
        rag_engine._call_llm("q")
    assert "Groq: rate limited" in str(err.value)
    assert "Gemini: quota exceeded" in str(err.value)


def test_error_when_groq_fails_and_there_is_no_gemini_key(providers, monkeypatch):
    calls, state = providers
    state["groq_fails"] = True
    monkeypatch.setenv("GROQ_API_KEY", "test")
    with pytest.raises(Exception, match="Groq: rate limited"):
        rag_engine._call_llm("q")
    assert calls == ["groq"]


def test_clear_message_when_no_api_key_is_set(providers):
    calls, _ = providers
    with pytest.raises(ValueError, match="No API key set"):
        rag_engine._call_llm("q")
    assert calls == []


# ── Provider calls (SDKs replaced by fakes, no network) ─────────

def test_call_groq_sends_the_system_prompt_and_model(monkeypatch):
    sent = {}

    class FakeGroq:
        def __init__(self, api_key):
            sent["api_key"] = api_key
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            sent.update(kwargs)
            message = types.SimpleNamespace(content="groq says hi")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    monkeypatch.setitem(sys.modules, "groq", types.SimpleNamespace(Groq=FakeGroq))
    monkeypatch.setenv("GROQ_API_KEY", "key-123")

    assert rag_engine._call_groq("What does login do?") == "groq says hi"
    assert sent["api_key"] == "key-123"
    assert sent["model"] == "llama-3.3-70b-versatile"
    assert sent["messages"][0] == {"role": "system", "content": rag_engine.SYSTEM_PROMPT}
    assert sent["messages"][1] == {"role": "user", "content": "What does login do?"}


def test_call_gemini_sends_the_system_prompt(monkeypatch):
    sent = {}

    class FakeModel:
        def __init__(self, model_name, system_instruction):
            sent.update(model=model_name, system=system_instruction)

        def generate_content(self, prompt):
            sent["prompt"] = prompt
            return types.SimpleNamespace(text="gemini says hi")

    fake = types.SimpleNamespace(configure=lambda api_key: sent.update(api_key=api_key), GenerativeModel=FakeModel)
    import google
    monkeypatch.setitem(sys.modules, "google.generativeai", fake)
    monkeypatch.setattr(google, "generativeai", fake, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "key-456")

    assert rag_engine._call_gemini("Explain orders") == "gemini says hi"
    assert sent == {"api_key": "key-456", "model": "gemini-2.0-flash-lite",
                    "system": rag_engine.SYSTEM_PROMPT, "prompt": "Explain orders"}


@pytest.mark.parametrize("call, key", [(rag_engine._call_groq, "GROQ_API_KEY"), (rag_engine._call_gemini, "GEMINI_API_KEY")])
def test_provider_calls_need_their_key(call, key):
    with pytest.raises(ValueError, match=key):
        call("q")


# ── ask_question (full pipeline, LLM mocked) ────────────────────

def index(repo, files: dict[str, str]):
    names = list(files)
    vs.add_documents(
        repo,
        ids=names,
        documents=[files[n] for n in names],
        embeddings=[fake_embed(files[n]) for n in names],
        metadatas=[{"file_path": n, "language": "python", "start_line": 1, "end_line": 10} for n in names],
    )


def test_ask_question_sends_retrieved_code_to_the_llm(monkeypatch):
    index("shop", {"auth.py": "def login(password): check password hash",
                   "cart.py": "def total(items): sum item prices"})
    prompts = []
    monkeypatch.setattr(rag_engine, "_call_llm", lambda p: prompts.append(p) or "Login hashes the password.")

    out = asyncio.run(rag_engine.ask_question("How does login check the password?", "shop", top_k=1))

    assert out["answer"] == "Login hashes the password."
    assert [s["file_path"] for s in out["sources"]] == ["auth.py"]
    assert "`auth.py`" in prompts[0] and "How does login check the password?" in prompts[0]


def test_ask_question_on_an_empty_index_gives_a_friendly_message(monkeypatch):
    monkeypatch.setattr(rag_engine, "query_documents", lambda *a, **k: results([]))
    out = asyncio.run(rag_engine.ask_question("anything?", "empty-repo"))
    assert "couldn't find any relevant code" in out["answer"]
    assert out["sources"] == []
