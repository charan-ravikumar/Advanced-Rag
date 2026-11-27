"""Span-based, length-neutral relevance and the per-question retrieval metrics (pure functions)."""

import math

import pytest

from eval.gold_spans import GoldSpan
from eval.metrics import is_relevant as old_is_relevant
from rag.eval_metrics import (
    count_relevant_chunks,
    doc_hit_at_k,
    evidence_recall_at_k,
    group_by_evidence,
    hit_rate_at_k,
    is_relevant,
    matches_span,
    ndcg_at_k,
    question_metrics,
    reciprocal_rank,
)
from rag.schemas_ingest import Chunk


def chunk(start, end, *, doc="p1", section="Methods", text="x") -> Chunk:
    return Chunk.create(doc_id=doc, section=section, chunk_idx=0, char_start=start, char_end=end,
                        strategy="recursive", text=text, doc_version="v")


def span(start, end, *, doc="p1", section="Methods", ev=0) -> GoldSpan:
    return GoldSpan(doc, section, 0, start, end, ev, "exact")


# ---- the length-bias regression ---------------------------------------------------------------------

def test_long_chunk_containing_a_short_span_is_relevant_but_the_old_f1_metric_says_no():
    evidence = " ".join(f"evidence{i}" for i in range(30))  # a 30-word evidence sentence
    left = " ".join(f"x{i}" for i in range(170))
    right = " ".join(f"y{i}" for i in range(170))
    text = left + " " + evidence + " " + right
    text = (text + " " + " ".join(f"z{i}" for i in range(400)))[:2000]
    assert len(text) == 2000 and evidence in text and len(set(text.split())) > 300
    start = text.index(evidence)
    c = chunk(0, 2000, text=text)
    s = span(start, start + len(evidence))
    assert is_relevant(c, [s]) is True  # new rule: the whole span lies inside the chunk (overlap = len(span) = min)
    assert old_is_relevant([evidence], text) is False  # old rule: token F1 collapses as the chunk gets longer


def test_short_chunk_inside_a_long_span_is_relevant():
    assert is_relevant(chunk(400, 500), [span(0, 1500)]) is True  # overlap = len(chunk) = min


# ---- negatives -----------------------------------------------------------------------------------

def test_tiny_overlap_is_not_relevant():
    assert is_relevant(chunk(0, 1000), [span(990, 1500)]) is False  # 10 chars of 510


def test_different_doc_is_never_relevant_even_with_identical_offsets():
    assert is_relevant(chunk(0, 1000, doc="p2"), [span(0, 1000)]) is False


def test_different_section_is_not_relevant_even_with_identical_offsets():
    assert is_relevant(chunk(0, 1000, section="Intro"), [span(0, 1000)]) is False


def test_unlocated_chunk_is_never_relevant():
    assert is_relevant(chunk(-1, -1), [span(0, 1000)]) is False


def test_no_overlap_and_zero_length_are_not_relevant():
    assert is_relevant(chunk(0, 100), [span(100, 200)]) is False  # touching, no shared character
    assert is_relevant(chunk(50, 50), [span(0, 100)]) is False
    assert is_relevant(chunk(0, 100), [span(60, 60)]) is False
    assert is_relevant(chunk(0, 100), []) is False


# ---- boundary at exactly 0.5 ---------------------------------------------------------------------

def test_exactly_half_of_the_smaller_one_is_relevant_one_less_is_not():
    s = span(0, 100)  # min(100, 200) = 100 -> need 50
    assert is_relevant(chunk(50, 250), [s]) is True  # overlap exactly 50
    assert is_relevant(chunk(51, 251), [s]) is False  # overlap 49


def test_boundary_when_the_chunk_is_the_smaller_one():
    s = span(0, 1000)  # chunk length 100 -> need 50
    assert is_relevant(chunk(950, 1050), [s]) is True  # overlap exactly 50
    assert is_relevant(chunk(951, 1051), [s]) is False  # overlap 49


def test_any_of_several_spans_is_enough():
    assert is_relevant(chunk(0, 100), [span(500, 600), span(0, 100)]) is True


def test_threshold_is_a_parameter():
    assert matches_span(chunk(0, 100), span(80, 200), fraction=0.1) is True
    assert matches_span(chunk(0, 100), span(80, 200)) is False


# ---- metric functions ---------------------------------------------------------------------------

def test_hit_rate_and_reciprocal_rank():
    rels = [False, False, True, False]
    assert [hit_rate_at_k(rels, k) for k in (1, 2, 3, 10)] == [0.0, 0.0, 1.0, 1.0]
    assert reciprocal_rank(rels) == pytest.approx(1 / 3)
    assert reciprocal_rank([False, False]) == 0.0 and reciprocal_rank([]) == 0.0


def test_ndcg_binary_gain_with_known_values():
    # one relevant item at rank 2, one exists in the index: DCG = 1/log2(3); ideal = 1/log2(2) = 1
    assert ndcg_at_k([False, True], 10, 1) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k([True, False], 10, 1) == 1.0
    # two relevant exist, only one retrieved at rank 1: DCG = 1, ideal = 1 + 1/log2(3)
    assert ndcg_at_k([True], 10, 2) == pytest.approx(1 / (1 + 1 / math.log2(3)))
    assert ndcg_at_k([True, True], 10, 2) == pytest.approx(1.0)
    assert ndcg_at_k([True], 10, 0) == 0.0  # nothing relevant exists: undefined -> 0, flagged as unreachable elsewhere
    assert ndcg_at_k([False] * 10 + [True], 10, 1) == 0.0  # beyond the cut-off


def test_doc_hit_ignores_the_passage():
    got = [chunk(0, 10, doc="other"), chunk(900, 950, doc="p1")]
    assert [doc_hit_at_k(got, "p1", k) for k in (1, 2)] == [0.0, 1.0]


def test_evidence_recall_counts_paragraphs_and_an_ambiguous_paragraph_is_one_unit():
    spans = [span(0, 100, ev=0), span(500, 600, ev=1), span(900, 1000, ev=1)]  # paragraph 1 occurs twice
    assert len(group_by_evidence(spans)) == 2
    top = [chunk(0, 100), chunk(900, 1000)]  # covers paragraph 0 and the second occurrence of paragraph 1
    assert evidence_recall_at_k(top, spans, 10) == 1.0
    assert evidence_recall_at_k(top, spans, 1) == 0.5
    assert evidence_recall_at_k(top, [], 10) is None


def test_count_relevant_chunks_in_the_index():
    index = [chunk(0, 100), chunk(100, 200), chunk(200, 300)]
    assert count_relevant_chunks(index, [span(50, 250)]) == 3
    assert count_relevant_chunks(index, [span(0, 100)]) == 1


def test_question_metrics_bundle():
    spans = [span(0, 100)]
    got = [chunk(500, 600), chunk(0, 120)]
    m = question_metrics(got, spans, "p1", n_relevant_in_index=1)
    assert m["relevance"] == [False, True]
    assert (m["HitRate@1"], m["HitRate@5"], m["HitRate@10"]) == (0.0, 1.0, 1.0)
    assert m["MRR"] == 0.5 and m["DocHit@1"] == 1.0 and m["reachable"] is True
    assert m["nDCG@10"] == pytest.approx(1 / math.log2(3))
    assert m["EvidenceRecall@1"] == 0.0 and m["EvidenceRecall@5"] == 1.0
    assert question_metrics(got, spans, "p1", n_relevant_in_index=0)["reachable"] is False
