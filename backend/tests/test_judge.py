"""Tests for the eval judge's prompt and its JSON parsing (no model needed)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))

from judge import judge_answer, parse_verdict  # noqa: E402

DATASET = Path(__file__).resolve().parents[2] / "eval" / "dataset.jsonl"


def test_parses_strict_json():
    v = parse_verdict('{"correctness": 5, "faithfulness": 4, "citation_quality": 3, "justification": "Good."}')
    assert (v.correctness, v.faithfulness, v.citation_quality, v.justification, v.ok) == (5, 4, 3, "Good.", True)


def test_parses_json_inside_fences_and_prose():
    v = parse_verdict('Here you go:\n```json\n{"correctness": "4", "faithfulness": 4.6, "citation_quality": 2, '
                      '"justification": "ok"}\n```\nThanks')
    assert (v.correctness, v.faithfulness, v.citation_quality) == (4, 5, 2)


def test_clamps_scores_to_one_through_five():
    v = parse_verdict('{"correctness": 9, "faithfulness": 0, "citation_quality": -3, "justification": ""}')
    assert (v.correctness, v.faithfulness, v.citation_quality) == (5, 1, 1)


def test_recovers_scores_from_broken_json():
    v = parse_verdict('{"correctness": 3, "faithfulness": 4, "citation_quality": 2, "justification": "cut off')
    assert (v.correctness, v.faithfulness, v.citation_quality, v.ok) == (3, 4, 2, True)


def test_reports_missing_scores():
    v = parse_verdict('{"correctness": 4, "justification": "partial"}')
    assert v.correctness == 4
    assert v.parse_error == "missing scores: faithfulness, citation_quality"


def test_reports_replies_with_no_json():
    v = parse_verdict("I think the answer is good.")
    assert not v.ok and v.correctness is None


def test_judge_retries_once_when_the_reply_cant_be_parsed():
    replies = iter(["no json here", '{"correctness": 5, "faithfulness": 5, "citation_quality": 5, "justification": "ok"}'])
    prompts = []
    v = judge_answer(lambda p: prompts.append(p) or next(replies), "Q?", "Ref", "ctx", "Ans")
    assert v.ok and len(prompts) == 2
    assert prompts[1].endswith("Return ONLY the JSON object.")


def test_judge_prompt_includes_everything_and_truncates_long_context():
    prompts = []
    judge_answer(lambda p: prompts.append(p) or '{"correctness":1,"faithfulness":1,"citation_quality":1,"justification":""}',
                 "Q?", "The reference", "x" * 20000, "The answer")
    prompt = prompts[0]
    assert "Q?" in prompt and "The reference" in prompt and "The answer" in prompt
    assert "[truncated]" in prompt and len(prompt) < 12000


def test_dataset_has_20_questions_of_each_kind_with_references():
    items = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    assert len(items) == 20
    assert len({i["id"] for i in items}) == 20
    counts = {t: sum(1 for i in items if i["type"] == t) for t in ("single_hop", "multi_hop", "unanswerable")}
    assert counts == {"single_hop": 8, "multi_hop": 8, "unanswerable": 4}
    assert all(i["question"].strip() and i["reference_answer"].strip() for i in items)


@pytest.mark.parametrize("name", ["README.md", "config.py", "auth.py", "loans.py", "fees.py", "api.py"])
def test_dataset_reference_files_exist(name):
    assert (DATASET.parent / "sample_repo" / name).exists()
