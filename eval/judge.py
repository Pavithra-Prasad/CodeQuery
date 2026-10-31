"""
LLM-as-judge: a local Ollama model scores an answer 1-5 on correctness, faithfulness to the
retrieved sources, and citation quality, with a short justification.

The judge is asked for a strict JSON object (Ollama's structured output). parse_verdict still
copes with fenced, wrapped or slightly broken replies, because small models don't always comply.
"""

import json
import re
from dataclasses import dataclass
from typing import Callable, Optional

CRITERIA = ("correctness", "faithfulness", "citation_quality")

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "correctness": {"type": "integer", "minimum": 1, "maximum": 5},
        "faithfulness": {"type": "integer", "minimum": 1, "maximum": 5},
        "citation_quality": {"type": "integer", "minimum": 1, "maximum": 5},
        "justification": {"type": "string"},
    },
    "required": [*CRITERIA, "justification"],
}

JUDGE_PROMPT = """You are grading an AI assistant's answer to a question about a codebase.

QUESTION:
{question}

REFERENCE ANSWER (written by a human who knows the code):
{reference}

SOURCES THE ASSISTANT RETRIEVED:
{context}

ASSISTANT'S ANSWER:
{answer}

Score each criterion from 1 (bad) to 5 (excellent):
- correctness: does the answer agree with the reference answer? If the reference says the feature
  isn't in the codebase, give 5 when the assistant says it couldn't find it and 1 when it invents one.
- faithfulness: is every claim supported by the retrieved sources, with nothing made up?
- citation_quality: does it cite real file paths (and line ranges) from the sources for its claims?
  For a correct "couldn't find it" answer with nothing to cite, give 5 when it doesn't invent citations.

Reply with only this JSON object and nothing else:
{{"correctness": <1-5>, "faithfulness": <1-5>, "citation_quality": <1-5>, "justification": "<one or two sentences>"}}"""

MAX_CONTEXT_CHARS = 8000


@dataclass
class Verdict:
    correctness: Optional[int]
    faithfulness: Optional[int]
    citation_quality: Optional[int]
    justification: str
    parse_error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.parse_error is None


def _clamp(value) -> Optional[int]:
    """Turn a judge score into an int from 1 to 5, or None if it isn't a number."""
    try:
        number = round(float(str(value).strip()))
    except (TypeError, ValueError):
        return None
    return max(1, min(5, number))


def parse_verdict(text: str) -> Verdict:
    """
    Parse the judge's reply. Accepts plain JSON, JSON inside ```json fences or surrounding prose,
    and scores written as strings or floats (clamped to 1-5).
    """
    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        candidates.append(fenced.group(1))
    braces = re.search(r"\{.*\}", text, re.S)
    if braces:
        candidates.append(braces.group(0))

    data = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            data = parsed
            break

    if data is None:
        # Last resort: pull "name": number pairs out of broken JSON
        data = {k: m.group(1) for k in CRITERIA if (m := re.search(rf'"?{k}"?\s*[:=]\s*"?(\d(?:\.\d+)?)', text))}
        if not data:
            return Verdict(None, None, None, "", parse_error=f"no JSON in judge reply: {text[:120]!r}")

    scores = {k: _clamp(data.get(k)) for k in CRITERIA}
    missing = [k for k, v in scores.items() if v is None]
    justification = str(data.get("justification", "")).strip()
    return Verdict(**scores, justification=justification,
                   parse_error=f"missing scores: {', '.join(missing)}" if missing else None)


def judge_answer(
    chat_fn: Callable[[str], str],
    question: str,
    reference: str,
    context: str,
    answer: str,
    retries: int = 1,
) -> Verdict:
    """
    Score one answer.

    Args:
        chat_fn: Sends a prompt to the judge model and returns its reply text.
        question: The eval question.
        reference: The human-written reference answer.
        context: What the system retrieved (truncated to keep the prompt small).
        answer: The answer being graded.
        retries: Extra attempts when the reply can't be parsed.
    """
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n... [truncated]"
    prompt = JUDGE_PROMPT.format(question=question, reference=reference,
                                 context=context or "(nothing retrieved)", answer=answer)
    verdict = parse_verdict(chat_fn(prompt))
    for _ in range(retries):
        if verdict.ok:
            break
        verdict = parse_verdict(chat_fn(prompt + "\n\nReturn ONLY the JSON object."))
    return verdict
