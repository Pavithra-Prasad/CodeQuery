"""
Evaluate RAG mode vs agent mode on eval/dataset.jsonl, graded by a local Ollama judge.

Usage (from the repo root, with Ollama running):
    backend/venv/bin/python eval/run_eval.py                  # all 20 questions, both modes
    backend/venv/bin/python eval/run_eval.py --limit 4        # quick partial run
    backend/venv/bin/python eval/run_eval.py --resume         # continue a stopped run

Outputs:
    eval/results.jsonl  every answer, score and timing (written as it goes, so a run can be resumed)
    eval/results.md     summary tables comparing the two modes
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent / "backend"))
sys.path.insert(0, str(EVAL_DIR))

# The eval keeps its own index and always runs locally; set before importing the backend
os.environ.setdefault("CHROMA_DIR", str(EVAL_DIR / ".chroma"))
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("LLM_TEMPERATURE", "0")
os.environ["LLM_PROVIDER"] = "ollama"

import llm  # noqa: E402
from agent import run_agent  # noqa: E402
from config import get_settings  # noqa: E402
from indexer import index_codebase  # noqa: E402
from judge import CRITERIA, VERDICT_SCHEMA, judge_answer  # noqa: E402
from rag_engine import ask_question  # noqa: E402
from vector_store import get_collection_count  # noqa: E402

REPO_NAME = "bookshelf-eval"
SAMPLE_REPO = EVAL_DIR / "sample_repo"
RESULTS_JSONL = EVAL_DIR / "results.jsonl"
RESULTS_MD = EVAL_DIR / "results.md"
TYPES = ("single_hop", "multi_hop", "unanswerable")


def load_dataset(path: Path = EVAL_DIR / "dataset.jsonl") -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_rag(question: str) -> dict:
    start = time.perf_counter()
    out = asyncio.run(ask_question(question, REPO_NAME, top_k=5, include_context=True))
    return {"answer": out["answer"], "context": out.get("context", ""), "tool_calls": 0, "iterations": 1,
            "latency_s": time.perf_counter() - start, "tool_errors": 0, "stopped_reason": "answered"}


def run_agent_mode(question: str) -> dict:
    start = time.perf_counter()
    result = run_agent(question, REPO_NAME)
    errors = sum(1 for s in result.trace if s.kind in ("tool_error", "model_error"))
    return {"answer": result.answer, "context": result.context, "tool_calls": result.tool_calls,
            "iterations": result.iterations, "latency_s": time.perf_counter() - start,
            "tool_errors": errors, "stopped_reason": result.stopped_reason}


RUNNERS = {"rag": run_rag, "agent": run_agent_mode}


def judge_chat(prompt: str) -> str:
    settings = get_settings()
    response = llm.chat([{"role": "user", "content": prompt}], model=settings.judge_model,
                        format=VERDICT_SCHEMA, temperature=0)
    return response.message.content


def evaluate(item: dict, mode: str) -> dict:
    """Answer one question in one mode and grade it. A crash counts as a failed answer (all 1s)."""
    row = {"id": item["id"], "type": item["type"], "mode": mode, "question": item["question"]}
    try:
        row.update(RUNNERS[mode](item["question"]))
    except llm.OllamaNotReady:
        raise
    except Exception as e:
        row.update({"answer": "", "context": "", "tool_calls": 0, "iterations": 0, "latency_s": 0.0,
                    "tool_errors": 0, "stopped_reason": "crashed", "error": f"{type(e).__name__}: {e}"})
        row.update({k: 1 for k in CRITERIA}, justification="The run crashed.", judge_error=None)
        return row

    verdict = judge_answer(judge_chat, item["question"], item["reference_answer"], row["context"], row["answer"])
    row.update({k: getattr(verdict, k) for k in CRITERIA},
               justification=verdict.justification, judge_error=verdict.parse_error)
    return row


# ── Reporting ─────────────────────────────────────────────────

def _mean(values) -> float:
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else float("nan")


def summarize(rows: list[dict]) -> dict:
    by_mode = defaultdict(list)
    for r in rows:
        by_mode[r["mode"]].append(r)
    summary = {}
    for mode, rs in by_mode.items():
        summary[mode] = {
            "n": len(rs),
            **{k: _mean(r[k] for r in rs) for k in CRITERIA},
            "overall": _mean(_mean(r[k] for k in CRITERIA) for r in rs),
            "pass_rate": sum(1 for r in rs if (r["correctness"] or 0) >= 4) / len(rs),
            "latency_s": _mean(r["latency_s"] for r in rs),
            "tool_calls": _mean(r["tool_calls"] for r in rs),
            "iterations": _mean(r["iterations"] for r in rs),
            "by_type": {t: _mean(r["correctness"] for r in rs if r["type"] == t) for t in TYPES},
            "judge_errors": sum(1 for r in rs if r.get("judge_error")),
            "crashes": sum(1 for r in rs if r.get("stopped_reason") == "crashed"),
            "hit_max_iterations": sum(1 for r in rs if r.get("stopped_reason") == "max_iterations"),
        }
    return summary


def console_table(summary: dict) -> str:
    modes = [m for m in ("rag", "agent") if m in summary]
    lines = [f"{'metric':<26}" + "".join(f"{m:>12}" for m in modes)]
    rows = [("correctness (1-5)", "correctness", "{:.2f}"), ("faithfulness (1-5)", "faithfulness", "{:.2f}"),
            ("citation quality (1-5)", "citation_quality", "{:.2f}"), ("overall (1-5)", "overall", "{:.2f}"),
            ("pass rate (correct >= 4)", "pass_rate", "{:.0%}"), ("avg latency (s)", "latency_s", "{:.1f}"),
            ("avg tool calls", "tool_calls", "{:.1f}"), ("avg model rounds", "iterations", "{:.1f}")]
    for label, key, fmt in rows:
        lines.append(f"{label:<26}" + "".join(f"{fmt.format(summary[m][key]):>12}" for m in modes))
    for t in TYPES:
        lines.append(f"{'correctness: ' + t:<26}" + "".join(f"{summary[m]['by_type'][t]:>12.2f}" for m in modes))
    return "\n".join(lines)


def write_markdown(rows: list[dict], summary: dict, settings) -> None:
    modes = [m for m in ("rag", "agent") if m in summary]
    md = [
        "# Eval results: RAG vs agent",
        "",
        f"- Run: {datetime.now():%Y-%m-%d %H:%M} on the Bookshelf sample repo (`eval/sample_repo/`)",
        f"- Answer model: `{settings.llm_model}` · Judge model: `{settings.judge_model}` · temperature {settings.temperature}",
        f"- Questions: {len({r['id'] for r in rows})} "
        f"({', '.join(f'{sum(1 for r in rows if r['type'] == t and r['mode'] == modes[0])} {t}' for t in TYPES)})",
        "",
        "## Summary",
        "",
        "| Metric | " + " | ".join(m.upper() for m in modes) + " |",
        "|---|" + "---|" * len(modes),
    ]
    for label, key, fmt in [("Correctness (1-5)", "correctness", "{:.2f}"), ("Faithfulness (1-5)", "faithfulness", "{:.2f}"),
                            ("Citation quality (1-5)", "citation_quality", "{:.2f}"), ("Overall (1-5)", "overall", "{:.2f}"),
                            ("Pass rate (correctness ≥ 4)", "pass_rate", "{:.0%}"), ("Avg latency (s)", "latency_s", "{:.1f}"),
                            ("Avg tool calls", "tool_calls", "{:.1f}"), ("Avg model rounds", "iterations", "{:.1f}"),
                            ("Runs that hit the iteration limit", "hit_max_iterations", "{}"),
                            ("Crashed runs", "crashes", "{}"), ("Judge replies that couldn't be parsed", "judge_errors", "{}")]:
        md.append(f"| {label} | " + " | ".join(fmt.format(summary[m][key]) for m in modes) + " |")

    md += ["", "## Correctness by question type", "", "| Type | " + " | ".join(m.upper() for m in modes) + " |",
           "|---|" + "---|" * len(modes)]
    for t in TYPES:
        md.append(f"| {t} | " + " | ".join(f"{summary[m]['by_type'][t]:.2f}" for m in modes) + " |")

    md += ["", "## Per question", "", "| ID | Type | Mode | Correct | Faithful | Citations | Tools | Rounds | Latency (s) | Judge's reason |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["id"], r["mode"])):
        reason = (r.get("error") or r.get("justification") or "").replace("|", "/").replace("\n", " ")
        md.append(f"| {r['id']} | {r['type']} | {r['mode']} | {r['correctness']} | {r['faithfulness']} | "
                  f"{r['citation_quality']} | {r['tool_calls']} | {r['iterations']} | {r['latency_s']:.1f} | {reason[:160]} |")

    md += ["", "## Caveats", "",
           "- The judge is a small local model. When it is the same model that wrote the answers, scores tend to be "
           "generous; set `JUDGE_MODEL` to a different model for a stricter grade.",
           "- 20 questions is a small sample: treat differences under ~0.3 points as noise.",
           "- Latency was measured on the machine that ran the eval and includes the first model load."]
    RESULTS_MD.write_text("\n".join(md) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modes", default="rag,agent", help="comma-separated: rag, agent")
    parser.add_argument("--limit", type=int, help="only run the first N questions")
    parser.add_argument("--ids", help="only run these question ids, comma-separated (e.g. s01,m01,u01)")
    parser.add_argument("--resume", action="store_true", help="skip (question, mode) pairs already in results.jsonl")
    parser.add_argument("--reindex", action="store_true", help="re-index the sample repo first")
    args = parser.parse_args(argv)

    settings = get_settings()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    try:
        llm.check_ollama([settings.llm_model, settings.judge_model])
    except llm.OllamaNotReady as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1

    if args.reindex or get_collection_count(REPO_NAME) == 0:
        result = index_codebase(str(SAMPLE_REPO), REPO_NAME)
        if result["status"] != "success":
            print(f"❌ Indexing failed: {result['message']}", file=sys.stderr)
            return 1

    dataset = load_dataset()
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",")}
        dataset = [item for item in dataset if item["id"] in wanted]
    dataset = dataset[: args.limit]
    rows = []
    if args.resume and RESULTS_JSONL.exists():
        rows = [json.loads(line) for line in RESULTS_JSONL.read_text().splitlines() if line.strip()]
    else:
        RESULTS_JSONL.write_text("")
    done = {(r["id"], r["mode"]) for r in rows}

    total = len(dataset) * len(modes)
    print(f"Running {total} answers with {settings.llm_model} (judge: {settings.judge_model})...")
    for item in dataset:
        for mode in modes:
            if (item["id"], mode) in done:
                continue
            row = evaluate(item, mode)
            rows.append(row)
            with RESULTS_JSONL.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"  [{len(rows)}/{total}] {item['id']} {mode:<5} correct={row['correctness']} "
                  f"faithful={row['faithfulness']} cite={row['citation_quality']} "
                  f"tools={row['tool_calls']} {row['latency_s']:.1f}s")

    ids = {item["id"] for item in dataset}
    rows = [r for r in rows if r["id"] in ids and r["mode"] in modes]
    summary = summarize(rows)
    print("\n" + console_table(summary))
    write_markdown(rows, summary, settings)
    print(f"\nSaved {RESULTS_MD.relative_to(EVAL_DIR.parent)} and {RESULTS_JSONL.relative_to(EVAL_DIR.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
