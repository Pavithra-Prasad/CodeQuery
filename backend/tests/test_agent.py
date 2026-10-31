"""Tests for the agent loop. The model is a scripted fake, so these run offline in milliseconds."""

import types

import pytest

import llm
import main
import vector_store as vs
from agent import Agent, run_agent
from agent.loop import EMPTY_REPO_ANSWER, FORCE_ANSWER_PROMPT, NO_ANSWER
from fakes import fake_embed
from tools import RepoTools

AUTH = "def check_password(password, stored):\n    return hash_password(password) == stored"


@pytest.fixture(autouse=True)
def shop():
    vs.add_documents("shop", ids=["auth.py:10-11"], documents=[AUTH], embeddings=[fake_embed(AUTH)],
                     metadatas=[{"file_path": "auth.py", "language": "python", "start_line": 10, "end_line": 11}])


def call(name, **arguments):
    return {"function": {"name": name, "arguments": arguments}}


class ScriptedModel:
    """Replays a list of replies and records what the agent sent each time."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, messages, tools):
        self.requests.append({"messages": [dict(m) for m in messages], "tools": tools})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, str):
            return {"message": {"role": "assistant", "content": reply}}
        return {"message": {"role": "assistant", "content": "", "tool_calls": reply}}


def make_agent(model, **kwargs):
    return Agent(RepoTools("shop", summarize_fn=lambda p: "summary"), chat_fn=model, **kwargs)


def test_searches_then_answers_with_citations():
    model = ScriptedModel([call("search_docs", query="password check")],
                          "`check_password` in `auth.py` (lines 10-11) compares hashes.")
    result = make_agent(model).run("How are passwords checked?")

    assert result.answer.startswith("`check_password`")
    assert result.stopped_reason == "answered"
    assert (result.iterations, result.tool_calls) == (2, 1)
    assert [s["file_path"] for s in result.sources] == ["auth.py"]
    assert "def check_password" in result.context

    tool_step, answer_step = result.trace
    assert (tool_step.kind, tool_step.tool, tool_step.arguments) == ("tool_call", "search_docs", {"query": "password check"})
    assert tool_step.result_chars > 0 and tool_step.latency_ms >= 0
    assert answer_step.kind == "answer"

    # The tool result went back to the model as a tool message, with the tool schemas offered
    second = model.requests[1]
    assert second["messages"][-1]["role"] == "tool"
    assert second["messages"][-1]["tool_name"] == "search_docs"
    assert "auth.py:10-11" in second["messages"][-1]["content"]
    assert {t["function"]["name"] for t in second["tools"]} == {
        "search_docs", "list_documents", "read_document_section", "summarize_document"}


def test_several_tool_calls_in_one_round_all_run():
    model = ScriptedModel([call("list_documents"), call("read_document_section", doc_name="auth.py", start=10, end=11)],
                          "See auth.py.")
    result = make_agent(model).run("q")
    assert [s.tool for s in result.trace[:2]] == ["list_documents", "read_document_section"]
    assert result.trace[1].model_ms == 0  # model time is only counted once per round
    assert result.tool_calls == 2


def test_unknown_tool_is_reported_back_and_the_loop_continues():
    model = ScriptedModel([call("delete_everything")], "I couldn't find this in the indexed codebase.")
    result = make_agent(model).run("q")
    assert result.trace[0].kind == "tool_error"
    assert "Unknown tool 'delete_everything'" in result.trace[0].error
    assert "search_docs" in model.requests[1]["messages"][-1]["content"]
    assert result.stopped_reason == "answered"


@pytest.mark.parametrize("arguments, message", [
    ("{not json", "aren't valid JSON"),
    ("[1, 2]", "must be a JSON object"),
    ({"start": 1}, "missing required argument 'doc_name'"),
    ({"doc_name": "auth.py", "start": "ten", "end": 11}, "'start' must be an integer"),
])
def test_malformed_arguments_are_reported_back(arguments, message):
    model = ScriptedModel([{"function": {"name": "read_document_section", "arguments": arguments}}], "done")
    result = make_agent(model).run("q")
    assert result.trace[0].kind == "tool_error"
    assert message in result.trace[0].error


def test_arguments_sent_as_a_json_string_still_work():
    model = ScriptedModel([{"function": {"name": "search_docs", "arguments": '{"query": "password"}'}}], "auth.py")
    result = make_agent(model).run("q")
    assert result.trace[0].kind == "tool_call"
    assert result.trace[0].arguments == {"query": "password"}


def test_a_crashing_tool_doesnt_crash_the_agent():
    def broken(prompt):
        raise RuntimeError("model unavailable")

    model = ScriptedModel([call("summarize_document", doc_name="auth.py")], "Couldn't summarize.")
    agent = Agent(RepoTools("shop", summarize_fn=broken), chat_fn=model)
    result = agent.run("q")
    assert result.trace[0].error == "summarize_document failed: RuntimeError: model unavailable"
    assert result.answer == "Couldn't summarize."


def test_tool_errors_with_advice_are_passed_on_verbatim():
    model = ScriptedModel([call("read_document_section", doc_name="nope.py", start=1, end=2)], "done")
    result = make_agent(model).run("q")
    assert result.trace[0].error == "No document named 'nope.py'. Call list_documents to see the available files."


def test_stops_after_max_iterations_and_forces_an_answer_without_tools():
    model = ScriptedModel(*([[call("search_docs", query="x")]] * 3), "Best effort: see auth.py.")
    result = make_agent(model, max_iterations=3).run("q")

    assert result.stopped_reason == "max_iterations"
    assert (result.tool_calls, result.iterations) == (3, 4)
    assert result.answer == "Best effort: see auth.py."
    final = model.requests[-1]
    assert final["tools"] is None
    assert final["messages"][-1] == {"role": "user", "content": FORCE_ANSWER_PROMPT}


def test_forced_answer_that_is_still_a_tool_call_becomes_a_clear_message():
    model = ScriptedModel([call("search_docs", query="x")], '{"name": "search_docs", "arguments": {"query": "y"}}')
    result = make_agent(model, max_iterations=1).run("q")
    assert result.answer == NO_ANSWER


def test_tool_calls_written_as_text_are_recognised():
    model = ScriptedModel('<tool_call>\n{"name": "search_docs", "arguments": {"query": "password"}}\n</tool_call>',
                          "auth.py checks it.")
    result = make_agent(model).run("q")
    assert result.trace[0].kind == "tool_call"
    assert result.answer == "auth.py checks it."


def test_json_in_an_answer_is_not_mistaken_for_a_tool_call():
    model = ScriptedModel([call("search_docs", query="x")], '{"name": "not_a_tool", "value": 1}')
    result = make_agent(model).run("q")
    assert result.trace[-1].kind == "answer"
    assert result.answer == '{"name": "not_a_tool", "value": 1}'


def test_an_empty_reply_gets_a_nudge():
    model = ScriptedModel("", [call("search_docs", query="x")], "Answer after nudge.")
    result = make_agent(model).run("q")
    assert [s.kind for s in result.trace] == ["model_error", "tool_call", "answer"]
    assert result.answer == "Answer after nudge."


def test_answering_without_searching_gets_sent_back_to_search_once():
    model = ScriptedModel("I couldn't find this in the indexed codebase.",
                          [call("search_docs", query="password")], "auth.py (lines 10-11) checks it.")
    result = make_agent(model).run("q")
    assert [s.kind for s in result.trace] == ["model_error", "tool_call", "answer"]
    assert "without searching" in result.trace[0].error
    assert model.requests[1]["messages"][-1]["content"].startswith("You answered without looking anything up")
    assert result.answer == "auth.py (lines 10-11) checks it."


def test_a_second_answer_without_searching_is_accepted():
    model = ScriptedModel("I couldn't find this.", "Still couldn't find it.")
    result = make_agent(model).run("q")
    assert result.answer == "Still couldn't find it."
    assert result.tool_calls == 0


def test_a_model_error_is_retried_then_raised():
    model = ScriptedModel(RuntimeError("error parsing tool call"), [call("search_docs", query="x")], "Recovered.")
    assert make_agent(model).run("q").answer == "Recovered."

    model = ScriptedModel(*[RuntimeError("error parsing tool call")] * 3)
    with pytest.raises(RuntimeError):
        make_agent(model).run("q")


def test_ollama_not_ready_is_raised_immediately():
    model = ScriptedModel(llm.OllamaNotReady("Run:\n  ollama pull qwen2.5:3b"))
    with pytest.raises(llm.OllamaNotReady):
        make_agent(model).run("q")


def test_sources_prefer_files_the_answer_cites():
    vs.add_documents("shop", ids=["cart.py:1-1"], documents=["def total(): password"],
                     embeddings=[fake_embed("def total(): password")],
                     metadatas=[{"file_path": "cart.py", "language": "python", "start_line": 1, "end_line": 1}])
    model = ScriptedModel([call("search_docs", query="password")], "It's in auth.py.")
    result = make_agent(model).run("q")
    assert [s["file_path"] for s in result.sources] == ["auth.py"]


def test_run_agent_skips_the_model_for_an_empty_repo():
    result = run_agent("q", "never-indexed", chat_fn=lambda *a: pytest.fail("model should not be called"))
    assert result.answer == EMPTY_REPO_ANSWER
    assert result.stopped_reason == "empty_repo"


def test_default_max_iterations_comes_from_config(monkeypatch):
    monkeypatch.setenv("AGENT_MAX_ITERATIONS", "4")
    assert make_agent(ScriptedModel()).max_iterations == 4
    monkeypatch.delenv("AGENT_MAX_ITERATIONS")
    assert make_agent(ScriptedModel()).max_iterations == 6


def test_real_ollama_response_objects_are_understood():
    from ollama import ChatResponse, Message

    responses = iter([
        ChatResponse(message=Message(role="assistant", content="", tool_calls=[
            Message.ToolCall(function=Message.ToolCall.Function(name="search_docs", arguments={"query": "password"}))])),
        ChatResponse(message=Message(role="assistant", content="auth.py does it.")),
    ])
    result = make_agent(lambda messages, tools: next(responses)).run("q")
    assert result.trace[0].tool == "search_docs"
    assert result.answer == "auth.py does it."


# ── API ─────────────────────────────────────────────────────────

def test_api_agent_mode_returns_the_trace(client, monkeypatch):
    model = ScriptedModel([call("search_docs", query="password")], "`auth.py` (lines 10-11).")
    monkeypatch.setattr(main, "run_agent", lambda q, repo: run_agent(q, repo, chat_fn=model))
    res = client.post("/api/query", json={"question": "How?", "repo_name": "shop", "mode": "agent"})
    assert res.status_code == 200
    body = res.json()
    assert body["mode"] == "agent"
    assert body["answer"] == "`auth.py` (lines 10-11)."
    assert [s["kind"] for s in body["trace"]] == ["tool_call", "answer"]
    assert (body["tool_calls"], body["iterations"]) == (1, 2)
    assert body["sources"][0]["file_path"] == "auth.py"


def test_api_rag_mode_is_still_the_default(client, monkeypatch):
    async def fake(question, repo_name, top_k):
        return {"answer": "rag answer", "sources": []}

    monkeypatch.setattr(main, "ask_question", fake)
    body = client.post("/api/query", json={"question": "How?", "repo_name": "shop"}).json()
    assert (body["answer"], body["mode"], body["trace"]) == ("rag answer", "rag", [])


def test_api_rejects_unknown_modes(client):
    res = client.post("/api/query", json={"question": "How?", "repo_name": "shop", "mode": "magic"})
    assert res.status_code == 422


def test_api_explains_when_ollama_isnt_ready(client, monkeypatch):
    def not_ready(q, repo):
        raise llm.OllamaNotReady("Run:\n  ollama pull qwen2.5:3b")

    monkeypatch.setattr(main, "run_agent", not_ready)
    res = client.post("/api/query", json={"question": "How?", "repo_name": "shop", "mode": "agent"})
    assert res.status_code == 503
    assert "ollama pull qwen2.5:3b" in res.json()["detail"]
