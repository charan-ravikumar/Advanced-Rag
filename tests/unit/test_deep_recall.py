"""Recall@50 (chunk level) and EvidenceRecall@50: opt-in depth for eval/PREREGISTRATION_qwen3_qdrant.md. Existing metric keys must not change when it is off."""

import pytest

from eval import run_retrieval_eval as rre
from rag.eval_metrics import DEEP_KS, question_metrics, recall_at_k
from tests.unit.test_eval_runner import ARM, GOLD_CHUNK, INDEX, QUESTIONS, FakeRetriever
from tests.unit.test_relevance import chunk, span


def test_recall_is_relevant_in_the_top_k_over_relevant_that_exist():
    rel = [False, True, False, True, True]
    assert recall_at_k(rel, 2, 3) == pytest.approx(1 / 3)
    assert recall_at_k(rel, 5, 3) == 1.0
    assert recall_at_k(rel, 50, 3) == 1.0  # a list shorter than k is fine
    assert recall_at_k([False] * 5, 5, 2) == 0.0


def test_a_question_with_no_relevant_chunk_in_the_index_has_no_recall_not_zero():
    assert recall_at_k([False, False], 50, 0) is None


def test_recall_never_exceeds_one_even_if_the_index_count_is_smaller_than_the_hits():
    assert recall_at_k([True, True, True], 3, 2) == 1.0


def test_default_metric_keys_are_unchanged_and_deep_keys_appear_only_when_asked():
    gold = [span(500, 700)]
    retrieved = [chunk(500, 1000), chunk(0, 500)]
    plain = question_metrics(retrieved, gold, "p1", 1)
    deep = question_metrics(retrieved, gold, "p1", 1, deep_ks=DEEP_KS)
    assert "Recall@50" not in plain and "EvidenceRecall@50" not in plain
    assert set(deep) - set(plain) == {"Recall@50", "EvidenceRecall@50"}
    assert {k: v for k, v in deep.items() if k in plain} == plain  # nothing else moved
    assert deep["Recall@50"] == 1.0 and deep["EvidenceRecall@50"] == 1.0


def test_the_runner_passes_the_depths_through_and_summarises_them_only_when_present():
    chunks = lambda q, d: list(INDEX[d[0] if d else q.split()[-1]])  # the whole paper, in index order  # noqa: E731
    fake = FakeRetriever(chunks)
    rows = rre.evaluate_arm(ARM, fake, INDEX, QUESTIONS, eval_mode="paper", warmup=0, seed=0, top_k=50, candidate_k=100, deep_ks=(50,))
    assert all(c["top_k"] == 50 and c["candidate_k"] == 100 for c in fake.calls)
    assert all(r["metrics"]["Recall@50"] == 1.0 and r["metrics"]["EvidenceRecall@50"] == 1.0 for r in rows)
    means = rre.summarize(rows, [ARM], warmup=0, eval_mode="paper")["arms"]["A"]["means"]
    assert means["Recall@50"] == 1.0 and means["EvidenceRecall@50"] == 1.0
    plain = rre.evaluate_arm(ARM, FakeRetriever(chunks), INDEX, QUESTIONS, eval_mode="paper", warmup=0, seed=0)
    assert "Recall@50" not in plain[0]["metrics"] and "Recall@50" not in rre.summarize(plain, [ARM], warmup=0)["arms"]["A"]["means"]


def test_a_miss_scores_zero_recall_and_a_deeper_list_can_only_help():
    wrong = lambda q, d: [c for c in INDEX[d[0]] if c is not GOLD_CHUNK[d[0]]]  # noqa: E731
    rows = rre.evaluate_arm(ARM, FakeRetriever(wrong), INDEX, QUESTIONS, eval_mode="paper", warmup=0, seed=0, top_k=50, deep_ks=(50,))
    assert all(r["metrics"]["Recall@50"] == 0.0 for r in rows)


def test_the_arm_config_records_the_candidate_depth():
    assert rre.arm_config([ARM], top_k=50, candidate_k=100)["candidate_k"] == 100 and rre.arm_config([ARM])["candidate_k"] == rre.CANDIDATE_K
