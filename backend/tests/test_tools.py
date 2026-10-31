"""Tests for the agent's tools and the function -> tool-schema helpers. No model is used."""

import pytest

import vector_store as vs
from fakes import fake_embed
from tools import RepoTools, ToolError, bind_arguments, function_to_tool
from tools.schema import parse_docstring

AUTH = "def hash_password(password):\n    return sha256(password)\n\ndef check_password(password, stored):\n    return hash_password(password) == stored"
ORDERS = "function totalPrice(items) {\n  return items.reduce((s, i) => s + i.price, 0);\n}"


def index(repo: str, chunks: list[tuple[str, int, str]]) -> None:
    """chunks: (file_path, start_line, content)"""
    vs.add_documents(
        repo,
        ids=[f"{path}:{start}-{start + text.count(chr(10))}" for path, start, text in chunks],
        documents=[text for _, _, text in chunks],
        embeddings=[fake_embed(text) for _, _, text in chunks],
        metadatas=[{"file_path": path, "language": "python" if path.endswith(".py") else "javascript",
                    "start_line": start, "end_line": start + text.count("\n")} for path, start, text in chunks],
    )


@pytest.fixture
def tools():
    index("shop", [
        ("app/auth.py", 1, AUTH),
        ("app/orders.js", 1, ORDERS),
        ("legacy/auth.py", 1, "def old_login(): pass"),
        ("README.md", 1, "# Shop\nA shop with login."),
    ])
    return RepoTools("shop", summarize_fn=lambda prompt: f"SUMMARY of {len(prompt)} chars")


# ── search_docs ─────────────────────────────────────────────────

def test_search_returns_passages_with_file_and_chunk_id(tools):
    out = tools.search_docs("check password hash")
    first = out["results"][0]
    assert first["file"] == "app/auth.py"
    assert first["chunk_id"] == "app/auth.py:1-5"
    assert (first["start_line"], first["end_line"]) == (1, 5)
    assert "def check_password" in first["text"]
    assert 0 <= first["relevance"] <= 1
    assert len(out["results"]) == 4  # only 4 chunks exist


def test_search_keeps_top_k_between_5_and_10():
    index("big", [(f"f{i}.py", 1, f"def handler_{i}(): login") for i in range(12)])
    big = RepoTools("big")
    assert len(big.search_docs("login", top_k=1)["results"]) == 5
    assert len(big.search_docs("login", top_k=7)["results"]) == 7
    assert len(big.search_docs("login", top_k=99)["results"]) == 10


def test_search_rejects_an_empty_query(tools):
    with pytest.raises(ToolError, match="empty"):
        tools.search_docs("   ")


def test_search_on_an_unindexed_repo_returns_nothing():
    assert RepoTools("nope").search_docs("anything")["results"] == []


# ── list_documents ──────────────────────────────────────────────

def test_list_documents_groups_chunks_by_file(tools):
    index("shop", [("app/auth.py", 6, "def logout():\n    pass")])
    out = tools.list_documents()
    assert out["total_documents"] == 4
    auth = next(d for d in out["documents"] if d["file"] == "app/auth.py")
    assert auth == {"file": "app/auth.py", "language": "python", "chunks": 2, "lines": 7}
    assert [d["file"] for d in out["documents"]] == sorted(d["file"] for d in out["documents"])


# ── read_document_section ───────────────────────────────────────

def test_read_section_returns_numbered_lines(tools):
    out = tools.read_document_section("app/auth.py", 4, 5)
    assert out["text"] == "4: def check_password(password, stored):\n5:     return hash_password(password) == stored"
    assert (out["start_line"], out["end_line"], out["total_lines"]) == (4, 5, 5)


def test_read_section_clamps_to_the_end_of_the_file(tools):
    out = tools.read_document_section("app/orders.js", 2, 500)
    assert out["end_line"] == 3
    assert out["text"].startswith("2:")


def test_read_section_accepts_a_unique_file_name(tools):
    assert tools.read_document_section("orders.js", 1, 1)["file"] == "app/orders.js"


def test_read_section_asks_for_the_full_path_when_a_name_is_ambiguous(tools):
    with pytest.raises(ToolError, match="several files: app/auth.py, legacy/auth.py"):
        tools.read_document_section("auth.py", 1, 2)


def test_read_section_explains_unknown_documents(tools):
    with pytest.raises(ToolError, match="list_documents"):
        tools.read_document_section("missing.py", 1, 2)


@pytest.mark.parametrize("start, end, message", [(5, 2, "before start"), (40, 50, "only has 5 lines")])
def test_read_section_rejects_bad_ranges(tools, start, end, message):
    with pytest.raises(ToolError, match=message):
        tools.read_document_section("app/auth.py", start, end)


def test_read_section_caps_long_ranges(monkeypatch, tools):
    monkeypatch.setattr("tools.repo_tools.MAX_SECTION_LINES", 2)
    out = tools.read_document_section("app/auth.py", 1, 5)
    assert out["end_line"] == 2
    assert "Ask for 3-5 next" in out["note"]


# ── summarize_document ──────────────────────────────────────────

def test_summarize_sends_the_whole_file_to_the_summarizer(tools):
    prompts = []
    tools._summarize = lambda prompt: prompts.append(prompt) or "  Hashes and checks passwords.  "
    out = tools.summarize_document("app/auth.py")
    assert out == {"file": "app/auth.py", "total_lines": 5, "summary": "Hashes and checks passwords.",
                   "truncated_input": False}
    assert "def check_password" in prompts[0] and "File: app/auth.py" in prompts[0]


def test_summarize_truncates_very_long_files(monkeypatch, tools):
    monkeypatch.setattr("tools.repo_tools.MAX_SUMMARY_INPUT_CHARS", 20)
    assert tools.summarize_document("app/auth.py")["truncated_input"] is True


# ── schema helpers ──────────────────────────────────────────────

def test_tool_schemas_mark_only_arguments_without_defaults_as_required(tools):
    schema = function_to_tool(tools.search_docs)["function"]
    assert schema["name"] == "search_docs"
    assert schema["description"].startswith("Semantic search over the indexed code.")
    assert schema["parameters"]["required"] == ["query"]
    assert schema["parameters"]["properties"]["top_k"] == {
        "type": "integer", "description": "How many passages to return, 5 to 10.", "default": 5}


def test_every_tool_has_a_description_for_each_argument(tools):
    for func in tools.functions():
        params = function_to_tool(func)["function"]["parameters"]["properties"]
        assert all(p.get("description") for p in params.values()), func.__name__


def test_parse_docstring_joins_wrapped_argument_descriptions():
    description, args = parse_docstring("""Do a thing.

    Args:
        name: The first line
            continues here.
        count (int): How many.

    Returns:
        Nothing.
    """)
    assert description == "Do a thing."
    assert args == {"name": "The first line continues here.", "count": "How many."}


def test_bind_arguments_converts_types_and_drops_unknown_arguments(tools):
    kwargs = bind_arguments(tools.read_document_section, {"doc_name": "a.py", "start": "3", "end": 7.0, "extra": 1})
    assert kwargs == {"doc_name": "a.py", "start": 3, "end": 7}


@pytest.mark.parametrize("args, error, message", [
    ({"start": 1, "end": 2}, TypeError, "missing required argument 'doc_name'"),
    ({"doc_name": "a.py", "start": "two", "end": 3}, ValueError, "'start' must be an integer"),
    ({"doc_name": "a.py", "start": 1.5, "end": 3}, ValueError, "whole number"),
    ({"doc_name": ["a.py"], "start": 1, "end": 3}, ValueError, "'doc_name' must be a string"),
])
def test_bind_arguments_rejects_bad_input(tools, args, error, message):
    with pytest.raises(error, match=message):
        bind_arguments(tools.read_document_section, args)
