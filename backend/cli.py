"""
Command-line interface for CodeQuery AI.

Examples:
    python cli.py check
    python cli.py index https://github.com/user/repo --name repo
    python cli.py ask --repo repo --mode agent --verbose "How does login work?"
    python cli.py ask --repo repo --mode rag "How does login work?"
"""

import argparse
import asyncio
import json
import logging
import sys
import time

import llm
from agent import run_agent
from config import get_settings
from indexer import extract_repo_name, index_codebase
from rag_engine import ask_question
from vector_store import get_collection_count


def _check(args) -> int:
    settings = get_settings()
    if settings.llm_provider == "cloud":
        print("LLM_PROVIDER=cloud: RAG answers use Groq/Gemini. Agent mode still needs Ollama.")
    try:
        llm.check_ollama()
    except llm.OllamaNotReady as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1
    print(f"✅ Ollama is running at {settings.ollama_host} and {settings.llm_model} is pulled.")
    return 0


def _index(args) -> int:
    name = args.name or extract_repo_name(args.source)
    result = index_codebase(args.source, name)
    if result["status"] != "success":
        print(f"❌ {result['message']}", file=sys.stderr)
        return 1
    print(f"✅ {result['message']} (repo name: {name})")
    return 0


def print_trace(trace) -> None:
    """Print the agent's steps: tool, arguments, result size and timings."""
    print("\n── Agent trace " + "─" * 45)
    for step in trace:
        if step.kind in ("tool_call", "tool_error"):
            status = f"ERROR: {step.error}" if step.error else f"{step.result_chars:,} chars"
            print(f"  round {step.iteration}: {step.tool}({json.dumps(step.arguments)})")
            print(f"           -> {status} · tool {step.latency_ms:.0f} ms · model {step.model_ms / 1000:.1f} s")
        elif step.kind == "answer":
            print(f"  round {step.iteration}: answer ({step.result_chars:,} chars) · model {step.model_ms / 1000:.1f} s")
        else:
            print(f"  round {step.iteration}: model error: {step.error}")


def _ask(args) -> int:
    if get_collection_count(args.repo) == 0:
        print(f"❌ Repo '{args.repo}' isn't indexed. Run: python cli.py index <path-or-url> --name {args.repo}",
              file=sys.stderr)
        return 1
    if args.mode == "agent" or get_settings().llm_provider != "cloud":
        try:
            llm.check_ollama()
        except llm.OllamaNotReady as e:
            print(f"❌ {e}", file=sys.stderr)
            return 1

    start = time.perf_counter()
    if args.mode == "agent":
        result = run_agent(args.question, args.repo)
        answer, sources = result.answer, result.sources
    else:
        out = asyncio.run(ask_question(args.question, args.repo, top_k=args.top_k))
        answer, sources = out["answer"], out["sources"]
    elapsed = time.perf_counter() - start

    if args.verbose and args.mode == "agent":
        print_trace(result.trace)
    print("\n" + answer + "\n")
    print("Sources: " + (", ".join(f"{s['file_path']} (L{s['start_line']}-{s['end_line']})" for s in sources) or "none"))
    if args.verbose:
        extra = f" · {result.tool_calls} tool calls · {result.iterations} rounds" if args.mode == "agent" else ""
        print(f"\n[{args.mode} mode · {elapsed:.1f} s{extra}]")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cli.py", description="Ask questions about an indexed codebase.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="check that Ollama is running and the model is pulled")

    p_index = sub.add_parser("index", help="index a local folder or GitHub URL")
    p_index.add_argument("source", help="local path or GitHub URL")
    p_index.add_argument("--name", help="repo name to store it under")

    p_ask = sub.add_parser("ask", help="ask a question about an indexed repo")
    p_ask.add_argument("question")
    p_ask.add_argument("--repo", required=True, help="name of the indexed repo")
    p_ask.add_argument("--mode", choices=["rag", "agent"], default="rag",
                       help="rag: one retrieval + one LLM call; agent: the model calls tools in a loop")
    p_ask.add_argument("--top-k", type=int, default=5, help="chunks to retrieve in rag mode")
    p_ask.add_argument("--verbose", action="store_true", help="show the agent's step-by-step trace and timings")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "verbose", False):
        logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    return {"check": _check, "index": _index, "ask": _ask}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
