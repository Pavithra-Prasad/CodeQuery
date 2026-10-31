"""
Agent loop with native Ollama tool calling.

Each round the model sees the conversation so far and either calls tools (we run them and add
the results to the conversation) or writes its final answer. After `max_iterations` rounds the
tools are taken away and the model has to answer with what it has gathered.

Nothing the model sends can crash the loop: unknown tools, malformed or missing arguments and
exceptions inside a tool all come back to the model as an error result it can recover from.
"""

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

import llm
from config import get_settings
from tools.repo_tools import RepoTools, ToolError
from tools.schema import bind_arguments, function_to_tool
from vector_store import get_collection_count

logger = logging.getLogger("codequery.agent")

SYSTEM_PROMPT = """You are CodeQuery, an assistant that answers questions about one indexed codebase.
You can't see the code directly; use the tools to look things up:
- search_docs: semantic search over code chunks. Start here.
- list_documents: list the files in the codebase.
- read_document_section: read exact lines of a file (e.g. a whole function).
- summarize_document: get an overview of one file.

RULES:
1. Always call search_docs before answering. Never invent code, file names or line numbers.
2. Cite every claim with the file path and line range from the tool results, like `app/auth.py` (lines 10-24).
3. If a question has several parts (a calculation, or a flow through several functions), search for each part.
4. If a result shows only part of a function or mentions a constant you haven't seen, call
   read_document_section or search for it before answering.
5. Only after searching, if the results don't contain the answer, say "I couldn't find this in the indexed codebase."
6. Stop calling tools once you have enough evidence, and answer concisely in markdown."""

SEARCH_FIRST_PROMPT = ("You answered without looking anything up. Call search_docs first "
                       "(with a short query about the question), then answer from what it returns.")

FORCE_ANSWER_PROMPT = (
    "You've reached the tool-call limit. Answer the question now using only the tool results above, "
    "with file and line citations. If they don't contain the answer, say you couldn't find it in the indexed codebase."
)
NUDGE_PROMPT = "Your last reply was empty. Call a tool, or answer the question with citations."
NO_ANSWER = "I couldn't find enough in the indexed codebase to answer this within the tool-call limit."
EMPTY_REPO_ANSWER = ("I couldn't find any relevant code in the indexed codebase. "
                     "Please make sure the repository has been indexed first.")

MAX_TOOL_RESULT_CHARS = 6000
MAX_CONSECUTIVE_MODEL_ERRORS = 2

# chat_fn(messages, tool_schemas_or_None) -> an Ollama ChatResponse/Message, or a dict shaped like one
ChatFn = Callable[[list[dict], Optional[list[dict]]], Any]

_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_CODE_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)


@dataclass
class TraceStep:
    """One step of the agent's reasoning trace."""
    iteration: int
    kind: str                       # "tool_call", "tool_error", "model_error" or "answer"
    tool: Optional[str] = None
    arguments: dict = field(default_factory=dict)
    result_chars: int = 0           # size of the tool result (or answer) the model received
    latency_ms: float = 0.0         # time spent running the tool
    model_ms: float = 0.0           # time the model took to decide this step (first step of a round only)
    error: Optional[str] = None


@dataclass
class AgentResult:
    """The agent's answer plus everything needed to inspect or grade it."""
    answer: str
    sources: list[dict]
    trace: list[TraceStep]
    iterations: int                 # model rounds used, including the final answer
    tool_calls: int
    stopped_reason: str             # "answered", "max_iterations" or "empty_repo"
    context: str = ""               # every successful tool result the model saw (used by the eval judge)

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("context")
        return data


@dataclass
class _ToolCall:
    name: str
    arguments: dict
    parse_error: Optional[str] = None


def _get(obj: Any, key: str) -> Any:
    """Read a field from an Ollama response object or a plain dict."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _parse_arguments(raw: Any) -> tuple[dict, Optional[str]]:
    """Tool arguments should be a JSON object; some models send a JSON string instead."""
    if raw is None:
        return {}, None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError as e:
            return {}, f"arguments aren't valid JSON ({e.msg})"
    if not isinstance(raw, dict):
        return {}, f"arguments must be a JSON object, got {type(raw).__name__}"
    return dict(raw), None


def _parse_text_tool_calls(content: str, tool_names: set[str]) -> list[_ToolCall]:
    """
    Qwen sometimes writes a tool call into the reply text instead of the tool_calls field,
    either inside <tool_call> tags or as a bare JSON object. Only known tool names count,
    so an answer that happens to contain JSON isn't mistaken for a call.
    """
    blobs = _TOOL_CALL_TAG.findall(content)
    if not blobs:
        stripped = content.strip()
        fenced = _CODE_FENCE.match(stripped)
        stripped = fenced.group(1) if fenced else stripped
        if stripped.startswith("{") and stripped.endswith("}"):
            blobs = [stripped]

    calls = []
    for blob in blobs:
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("name") in tool_names:
            arguments, error = _parse_arguments(data.get("arguments", data.get("parameters")))
            calls.append(_ToolCall(data["name"], arguments, error))
    return calls


def _default_chat(messages: list[dict], tools: Optional[list[dict]]) -> Any:
    return llm.chat(messages, tools=tools)


class Agent:
    """Runs the tool-calling loop for one repository."""

    def __init__(self, tools: RepoTools, chat_fn: Optional[ChatFn] = None, max_iterations: Optional[int] = None):
        """
        Args:
            tools: The repository tools the model may call.
            chat_fn: Sends messages (and tool schemas) to the model. Defaults to the local Ollama model.
            max_iterations: Most tool-calling rounds before a final answer is forced (default AGENT_MAX_ITERATIONS).
        """
        self.tools = {func.__name__: func for func in tools.functions()}
        self.schemas = [function_to_tool(func) for func in self.tools.values()]
        self.chat_fn = chat_fn or _default_chat
        self.max_iterations = max_iterations or get_settings().agent_max_iterations

    def run(self, question: str) -> AgentResult:
        """Answer one question, calling tools as the model decides."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
        trace: list[TraceStep] = []
        sources: dict[tuple, dict] = {}
        contexts: list[str] = []
        tool_calls = 0
        model_errors = 0
        nudged_to_search = False

        for iteration in range(1, self.max_iterations + 1):
            try:
                content, calls, model_ms = self._ask(messages, self.schemas)
            except llm.OllamaNotReady:
                raise
            except Exception as e:
                # e.g. Ollama rejecting a tool call the model wrote with broken JSON
                model_errors += 1
                trace.append(self._log(TraceStep(iteration, "model_error", error=f"{type(e).__name__}: {e}")))
                if model_errors > MAX_CONSECUTIVE_MODEL_ERRORS:
                    raise
                messages.append({"role": "user", "content": "Your last tool call couldn't be parsed. "
                                 "Call a tool with valid JSON arguments, or answer the question."})
                continue
            model_errors = 0

            if not calls:
                if not content.strip():
                    trace.append(self._log(TraceStep(iteration, "model_error", model_ms=model_ms, error="empty reply")))
                    messages.append({"role": "user", "content": NUDGE_PROMPT})
                    continue
                if tool_calls == 0 and not nudged_to_search:
                    # Small models sometimes skip the tools and reply "couldn't find it"; make them look once
                    nudged_to_search = True
                    trace.append(self._log(TraceStep(iteration, "model_error", model_ms=model_ms,
                                                     error="answered without searching; asked it to search first")))
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": SEARCH_FIRST_PROMPT})
                    continue
                trace.append(self._log(TraceStep(iteration, "answer", result_chars=len(content), model_ms=model_ms)))
                return self._result(content, sources, trace, iteration, tool_calls, "answered", contexts)

            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": [{"function": {"name": c.name, "arguments": c.arguments}} for c in calls],
            })
            for i, call in enumerate(calls):
                tool_calls += 1
                payload, error, tool_ms = self._execute(call)
                text = json.dumps(payload, ensure_ascii=False, default=str)
                if len(text) > MAX_TOOL_RESULT_CHARS:
                    text = text[:MAX_TOOL_RESULT_CHARS] + " ... [truncated]"
                messages.append({"role": "tool", "content": text, "tool_name": call.name})
                trace.append(self._log(TraceStep(
                    iteration, "tool_error" if error else "tool_call", tool=call.name, arguments=call.arguments,
                    result_chars=len(text), latency_ms=tool_ms, model_ms=model_ms if i == 0 else 0.0, error=error,
                )))
                if not error:
                    contexts.append(f"[{call.name}({json.dumps(call.arguments)})]\n{text}")
                    self._collect_sources(call.name, payload, sources)

        # Out of rounds: take the tools away and make the model answer
        messages.append({"role": "user", "content": FORCE_ANSWER_PROMPT})
        content, calls, model_ms = self._ask(messages, None)
        answer = content.strip()
        if not answer or calls or _parse_text_tool_calls(answer, set(self.tools)):
            answer = NO_ANSWER
        trace.append(self._log(TraceStep(self.max_iterations + 1, "answer", result_chars=len(answer), model_ms=model_ms)))
        return self._result(answer, sources, trace, self.max_iterations + 1, tool_calls, "max_iterations", contexts)

    # ── Steps ─────────────────────────────────────────────────

    def _ask(self, messages: list[dict], schemas: Optional[list[dict]]) -> tuple[str, list[_ToolCall], float]:
        """Call the model once and pull out its text and tool calls."""
        start = time.perf_counter()
        response = self.chat_fn(messages, schemas)
        model_ms = (time.perf_counter() - start) * 1000

        message = _get(response, "message") or response
        content = _get(message, "content") or ""
        calls = []
        for raw in _get(message, "tool_calls") or []:
            function = _get(raw, "function") or {}
            arguments, error = _parse_arguments(_get(function, "arguments"))
            calls.append(_ToolCall(str(_get(function, "name") or ""), arguments, error))
        if not calls and schemas:
            calls = _parse_text_tool_calls(content, set(self.tools))
            if calls:
                content = ""
        return content, calls, model_ms

    def _execute(self, call: _ToolCall) -> tuple[dict, Optional[str], float]:
        """Run one tool call. Returns (payload for the model, error message or None, latency in ms)."""
        start = time.perf_counter()
        error = None
        func = self.tools.get(call.name)

        if call.parse_error:
            error = f"Couldn't read the arguments for {call.name or 'the tool'}: {call.parse_error}."
        elif func is None:
            error = f"Unknown tool '{call.name}'. Available tools: {', '.join(self.tools)}."
        else:
            try:
                payload = func(**bind_arguments(func, call.arguments))
            except (TypeError, ValueError) as e:
                # ToolError is a ValueError: a clear message about what to change
                error = str(e) if isinstance(e, ToolError) else f"Bad arguments for {call.name}: {e}"
            except Exception as e:
                logger.exception("Tool %s failed", call.name)
                error = f"{call.name} failed: {type(e).__name__}: {e}"

        latency = (time.perf_counter() - start) * 1000
        if error:
            return {"error": error}, error, latency
        return payload, None, latency

    @staticmethod
    def _collect_sources(tool: str, payload: dict, sources: dict[tuple, dict]) -> None:
        """Remember which code the model has seen, in the same shape as the RAG mode's sources."""
        if tool == "search_docs":
            for r in payload.get("results", []):
                key = (r["file"], r["start_line"], r["end_line"])
                sources.setdefault(key, {
                    "file_path": r["file"], "language": r.get("language", ""),
                    "start_line": r["start_line"], "end_line": r["end_line"],
                    "snippet": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"],
                    "relevance": r.get("relevance", 0),
                })
        elif tool in ("read_document_section", "summarize_document"):
            start, end = payload.get("start_line", 1), payload.get("end_line", payload.get("total_lines", 0))
            snippet = payload.get("text", payload.get("summary", ""))
            sources.setdefault((payload["file"], start, end), {
                "file_path": payload["file"], "language": "", "start_line": start, "end_line": end,
                "snippet": snippet[:200] + "..." if len(snippet) > 200 else snippet, "relevance": 0,
            })

    @staticmethod
    def _result(answer, sources, trace, iterations, tool_calls, reason, contexts) -> AgentResult:
        # Prefer the sources the answer actually cites; fall back to everything the model saw
        seen = list(sources.values())
        cited = [s for s in seen if s["file_path"] in answer or os.path.basename(s["file_path"]) in answer]
        return AgentResult(answer.strip(), cited or seen, trace, iterations, tool_calls, reason, "\n\n".join(contexts))

    @staticmethod
    def _log(step: TraceStep) -> TraceStep:
        if step.kind in ("tool_call", "tool_error"):
            logger.info("[round %d] %s(%s) -> %d chars in %.0f ms%s (model %.0f ms)", step.iteration, step.tool,
                        json.dumps(step.arguments), step.result_chars, step.latency_ms,
                        f" ERROR: {step.error}" if step.error else "", step.model_ms)
        elif step.kind == "answer":
            logger.info("[round %d] answer, %d chars (model %.0f ms)", step.iteration, step.result_chars, step.model_ms)
        else:
            logger.info("[round %d] model error: %s", step.iteration, step.error)
        return step


def run_agent(question: str, repo_name: str, chat_fn: Optional[ChatFn] = None,
              max_iterations: Optional[int] = None) -> AgentResult:
    """Answer a question about an indexed repo with the tool-calling agent."""
    if get_collection_count(repo_name) == 0:
        return AgentResult(EMPTY_REPO_ANSWER, [], [], 0, 0, "empty_repo")
    return Agent(RepoTools(repo_name), chat_fn=chat_fn, max_iterations=max_iterations).run(question)
