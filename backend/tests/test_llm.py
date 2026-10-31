"""Tests for the settings and the Ollama client wrapper (the Ollama server is faked)."""

import types

import ollama
import pytest

import llm
from config import get_settings
from llm import check_ollama as real_check_ollama  # conftest stubs llm.check_ollama for other tests


def fake_client(models=(), error=None, chat=None):
    def list_models():
        if error:
            raise error
        return types.SimpleNamespace(models=[types.SimpleNamespace(model=m) for m in models])

    return types.SimpleNamespace(list=list_models, chat=chat)


def test_defaults_are_local_and_free():
    s = get_settings()
    assert (s.llm_provider, s.llm_model, s.judge_model) == ("ollama", "qwen2.5:3b", "qwen2.5:3b")
    assert s.ollama_host == "http://localhost:11434"
    assert s.agent_max_iterations == 6


def test_settings_come_from_env_vars(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", " Cloud ")
    monkeypatch.setenv("LLM_MODEL", "llama3.1:8b")
    monkeypatch.setenv("AGENT_MAX_ITERATIONS", "3")
    s = get_settings()
    assert (s.llm_provider, s.llm_model, s.judge_model, s.agent_max_iterations) == ("cloud", "llama3.1:8b", "llama3.1:8b", 3)


def test_check_passes_when_the_model_is_pulled(monkeypatch):
    monkeypatch.setattr(llm, "get_client", lambda: fake_client(["qwen2.5:3b", "nomic-embed-text:latest"]))
    real_check_ollama()
    real_check_ollama(["qwen2.5:3b", "nomic-embed-text"])  # a bare name means :latest


def test_check_says_how_to_start_ollama_when_it_isnt_running(monkeypatch):
    monkeypatch.setattr(llm, "get_client", lambda: fake_client(error=ConnectionError("refused")))
    with pytest.raises(llm.OllamaNotReady, match="ollama serve") as err:
        real_check_ollama()
    assert "https://ollama.com" in str(err.value)


def test_check_gives_the_exact_pull_command_for_each_missing_model(monkeypatch):
    monkeypatch.setattr(llm, "get_client", lambda: fake_client(["qwen2.5:3b"]))
    with pytest.raises(llm.OllamaNotReady) as err:
        real_check_ollama(["qwen2.5:3b", "llama3.1:8b", "llama3.1:8b"])
    message = str(err.value)
    assert "  ollama pull llama3.1:8b" in message
    assert message.count("ollama pull") == 1


def test_chat_sends_the_model_and_options(monkeypatch):
    sent = {}
    monkeypatch.setenv("OLLAMA_NUM_CTX", "4096")
    monkeypatch.setattr(llm, "get_client", lambda: fake_client(chat=lambda **kw: sent.update(kw) or "reply"))
    assert llm.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}], temperature=0) == "reply"
    assert sent["model"] == "qwen2.5:3b"
    assert sent["tools"] == [{"type": "function"}]
    assert sent["options"] == {"temperature": 0, "num_ctx": 4096}


def test_chat_turns_a_missing_model_into_a_pull_command(monkeypatch):
    def missing(**kw):
        raise ollama.ResponseError("model not found", status_code=404)

    monkeypatch.setattr(llm, "get_client", lambda: fake_client(chat=missing))
    with pytest.raises(llm.OllamaNotReady, match="ollama pull qwen2.5:3b"):
        llm.chat([])


def test_chat_explains_a_stopped_server(monkeypatch):
    def down(**kw):
        raise ConnectionError("Failed to connect to Ollama")

    monkeypatch.setattr(llm, "get_client", lambda: fake_client(chat=down))
    with pytest.raises(llm.OllamaNotReady, match="ollama serve"):
        llm.chat([])
