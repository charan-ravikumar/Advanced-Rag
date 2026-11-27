"""Unit tests for the eval metric helpers and the Qasper golden set."""

import json
from pathlib import Path

import pytest

from eval.metrics import (
    answer_f1,
    best_over_references,
    doc_hit_rank,
    exact_match,
    first_relevant_rank,
    is_refusal,
)

GOLDEN_SET = Path(__file__).resolve().parents[2] / "eval" / "datasets" / "qasper_dev" / "golden_set.json"


class TestRetrievalMetrics:
    def test_first_relevant_rank(self):
        gold = ["the quick brown fox jumps over the lazy dog"]
        chunks = ["unrelated text about databases", "a quick brown fox jumps over a lazy dog"]
        assert first_relevant_rank(gold, chunks) == 2

    def test_no_relevant_chunk(self):
        assert first_relevant_rank(["alpha beta gamma"], ["delta epsilon zeta"]) is None

    def test_doc_hit_rank(self):
        assert doc_hit_rank("Paper B", ["Paper A", "Paper B", "Paper B"]) == 2
        assert doc_hit_rank("Paper C", ["Paper A"]) is None


class TestAnswerMetrics:
    def test_exact_match_normalises(self):
        assert exact_match("The Transformer.", "transformer") == 1.0

    def test_f1_partial_overlap(self):
        assert answer_f1("a large language model", "language model") == pytest.approx(0.8)

    def test_f1_no_overlap(self):
        assert answer_f1("cats", "dogs") == 0.0

    def test_best_over_references(self):
        assert best_over_references(exact_match, "yes", ["no", "yes"]) == 1.0

    def test_refusal_detection(self):
        assert is_refusal("I could not find that information in the provided documents.")
        assert not is_refusal("The answer is 42.")


@pytest.mark.skipif(not GOLDEN_SET.exists(), reason="golden set not built")
class TestGoldenSet:
    def test_schema_and_counts(self):
        data = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))
        questions = data["questions"]
        assert data["n_questions"] == len(questions)
        for q in questions:
            assert q["gold_answers"] and q["gold_answer"] == q["gold_answers"][0]
            assert bool(q["gold_evidence"]) == q["answerable"]
