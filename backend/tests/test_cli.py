"""Tests for the command-line interface (indexing is real, the LLM is faked)."""

import pytest

import cli
import llm
import rag_engine
from agent import AgentResult, TraceStep


def test_parser_accepts_mode_and_verbose():
    args = cli.build_parser().parse_args(["ask", "--repo", "shop", "--mode", "agent", "--verbose", "How?"])
    assert (args.repo, args.mode, args.verbose, args.question) == ("shop", "agent", True, "How?")
    assert cli.build_parser().parse_args(["ask", "--repo", "shop", "How?"]).mode == "rag"
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["ask", "--repo", "shop", "--mode", "magic", "How?"])


def test_index_then_ask_in_rag_mode(sample_repo, monkeypatch, capsys):
    monkeypatch.setattr(rag_engine, "_call_llm", lambda prompt: "auth.py hashes passwords.")
    assert cli.main(["index", str(sample_repo), "--name", "shop"]) == 0
    assert cli.main(["ask", "--repo", "shop", "How are passwords hashed?"]) == 0
    out = capsys.readouterr().out
    assert "auth.py hashes passwords." in out
    assert "Sources: auth.py" in out


def test_verbose_agent_mode_prints_the_trace(sample_repo, monkeypatch, capsys):
    cli.main(["index", str(sample_repo), "--name", "shop"])
    trace = [TraceStep(1, "tool_call", tool="search_docs", arguments={"query": "password"},
                       result_chars=1234, latency_ms=12, model_ms=2500),
             TraceStep(2, "answer", result_chars=40, model_ms=3100)]
    monkeypatch.setattr(cli, "run_agent", lambda q, repo: AgentResult("auth.py (lines 1-3)", [], trace, 2, 1, "answered"))

    assert cli.main(["ask", "--repo", "shop", "--mode", "agent", "--verbose", "How?"]) == 0
    out = capsys.readouterr().out
    assert 'round 1: search_docs({"query": "password"})' in out
    assert "1,234 chars · tool 12 ms · model 2.5 s" in out
    assert "1 tool calls · 2 rounds" in out


def test_ask_about_an_unindexed_repo_explains_how_to_index(capsys):
    assert cli.main(["ask", "--repo", "nope", "How?"]) == 1
    assert "python cli.py index" in capsys.readouterr().err


def test_check_prints_the_fix_when_ollama_is_missing(monkeypatch, capsys):
    def not_ready():
        raise llm.OllamaNotReady("Can't reach Ollama.\n  ollama serve")

    monkeypatch.setattr(llm, "check_ollama", not_ready)
    assert cli.main(["check"]) == 1
    assert "ollama serve" in capsys.readouterr().err
