"""Retrieval metrics for Evaluation v2. Pure functions: no I/O, no model, no store.

Relevance is span-based and length-neutral: a retrieved chunk is relevant to a question when it comes
from the same document and section as a gold evidence span and the character overlap with that span
is at least ``0.5 * min(len(span), len(chunk))``. A long chunk that contains a short evidence span counts
(overlap = len(span) = min), and a short chunk inside a long evidence span counts (overlap = len(chunk) = min).

Objects only need the attributes ``doc_id``, ``section``, ``char_start``, ``char_end`` (and, for gold spans,
optionally ``evidence_idx``), so ``rag.retriever.Hit``, ``rag.schemas_ingest.Chunk`` and
``eval.gold_spans.GoldSpan`` all work. A chunk with ``char_start < 0`` (not located at ingest) is never relevant.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import NamedTuple

OVERLAP_FRACTION = 0.5
KS = (1, 5, 10)
# Opt-in depths for recall (eval/PREREGISTRATION_qwen3_qdrant.md): computed only when the caller passes ``deep_ks``, so existing runs keep exactly their metric keys.
DEEP_KS = (50,)


class Paragraph(NamedTuple):
    """A source paragraph in chunk coordinates (section title and section-relative offsets)."""

    section: str
    section_idx: int
    para_idx: int
    char_start: int
    char_end: int
    text: str


def overlap_chars(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def matches_span(chunk, span, fraction: float = OVERLAP_FRACTION) -> bool:
    """True when ``chunk`` overlaps ``span`` by at least ``fraction * min(len(span), len(chunk))`` characters."""
    if chunk.doc_id != span.doc_id or chunk.section != span.section or chunk.char_start < 0:
        return False
    span_len, chunk_len = span.char_end - span.char_start, chunk.char_end - chunk.char_start
    smaller = min(span_len, chunk_len)
    if smaller <= 0:
        return False
    ov = overlap_chars(chunk.char_start, chunk.char_end, span.char_start, span.char_end)
    return ov > 0 and ov >= fraction * smaller


def is_relevant(chunk, gold_spans: Iterable, fraction: float = OVERLAP_FRACTION) -> bool:
    return any(matches_span(chunk, s, fraction) for s in gold_spans)


def group_by_evidence(gold_spans: Iterable) -> list[list]:
    """Spans grouped by evidence paragraph. An ambiguous paragraph (several occurrences) is ONE group:
    covering any of its occurrences covers the paragraph."""
    groups: dict[int, list] = {}
    for s in gold_spans:
        groups.setdefault(getattr(s, "evidence_idx", 0), []).append(s)
    return [groups[k] for k in sorted(groups)]


def hit_rate_at_k(relevance: Sequence[bool], k: int) -> float:
    return 1.0 if any(relevance[:k]) else 0.0


def reciprocal_rank(relevance: Sequence[bool]) -> float:
    for i, r in enumerate(relevance, 1):
        if r:
            return 1.0 / i
    return 0.0


def dcg_at_k(relevance: Sequence[bool], k: int) -> float:
    return sum(1.0 / math.log2(i + 2) for i, r in enumerate(relevance[:k]) if r)


def ndcg_at_k(relevance: Sequence[bool], k: int, n_relevant: int) -> float:
    """Binary-gain nDCG. ``n_relevant`` is the number of relevant chunks that EXIST in the index for the
    question (so the ideal ranking is bounded by it). Zero when no relevant chunk exists."""
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(n_relevant, k)))
    return dcg_at_k(relevance, k) / ideal if ideal > 0 else 0.0


def recall_at_k(relevance: Sequence[bool], k: int, n_relevant: int) -> float | None:
    """Chunk-level recall (the BEIR definition): relevant chunks among the top k / relevant chunks that EXIST in the index for the question.
    ``None`` when none exists (an unreachable question: it has no recall, and is left out of the mean)."""
    if n_relevant <= 0:
        return None
    return min(sum(1 for r in relevance[:k] if r), n_relevant) / n_relevant


def doc_hit_at_k(retrieved: Sequence, gold_doc_id: str, k: int) -> float:
    return 1.0 if any(c.doc_id == gold_doc_id for c in retrieved[:k]) else 0.0


def evidence_recall_at_k(retrieved: Sequence, gold_spans: Sequence, k: int) -> float | None:
    """Fraction of the question's evidence paragraphs covered by some top-k chunk (None if no gold spans)."""
    groups = group_by_evidence(gold_spans)
    if not groups:
        return None
    top = retrieved[:k]
    covered = sum(1 for g in groups if any(is_relevant(c, g) for c in top))
    return covered / len(groups)


def count_relevant_chunks(index_chunks: Iterable, gold_spans: Sequence) -> int:
    """How many chunks of the index (any chunks, typically those of the gold document) are relevant."""
    return sum(1 for c in index_chunks if is_relevant(c, gold_spans))


# ---- paragraph-level Evidence-F1 (the official Qasper evidence metric, applied to retrieved chunks) ------------------

def paragraphs_of_chunk(chunk, paragraphs: Iterable[Paragraph], min_overlap_chars: int = 1) -> list[Paragraph]:
    """The source paragraphs a chunk's offsets overlap (same section, at least ``min_overlap_chars`` shared characters)."""
    if chunk.char_start < 0:
        return []
    return [p for p in paragraphs
            if p.section == chunk.section
            and overlap_chars(chunk.char_start, chunk.char_end, p.char_start, p.char_end) >= min_overlap_chars]


def retrieved_paragraph_texts(chunks: Iterable, paragraphs: Sequence[Paragraph], min_overlap_chars: int = 1) -> set[str]:
    """Set of paragraph TEXTS covered by the chunks. Text is the identity, as in the official metric (duplicate
    paragraphs collapse)."""
    out: set[str] = set()
    for c in chunks:
        out.update(p.text for p in paragraphs_of_chunk(c, paragraphs, min_overlap_chars))
    return out


def evidence_f1(predicted: set[str], gold_sets: Iterable[set[str]]) -> float | None:
    """Evidence-F1 of a predicted paragraph set against several annotators' evidence sets: F1 per annotator over sets of
    paragraphs, MAX over annotators (the official Qasper definition). Annotators with no evidence are skipped;
    ``None`` when none has any. An empty prediction scores 0."""
    best: float | None = None
    for gold in gold_sets:
        if not gold:
            continue
        inter = len(predicted & gold)
        if not predicted or inter == 0:
            f1 = 0.0
        else:
            precision, recall = inter / len(predicted), inter / len(gold)
            f1 = 2 * precision * recall / (precision + recall)
        best = f1 if best is None else max(best, f1)
    return best


def evidence_f1_at_k(retrieved: Sequence, paragraphs: Sequence[Paragraph], gold_sets: Iterable[set[str]], k: int) -> float | None:
    return evidence_f1(retrieved_paragraph_texts(retrieved[:k], paragraphs), list(gold_sets))


def question_metrics(retrieved: Sequence, gold_spans: Sequence, gold_doc_id: str, n_relevant_in_index: int,
                     ks: Sequence[int] = KS, paragraphs: Sequence[Paragraph] | None = None,
                     reference_evidence: Sequence[set[str]] | None = None, deep_ks: Sequence[int] = ()) -> dict:
    """All per-question retrieval metrics plus the per-hit relevance flags.

    ``EvidenceF1@k`` is computed only when ``paragraphs`` (the document's paragraph index) and
    ``reference_evidence`` (one set of evidence paragraph texts per annotator) are given, otherwise it is ``None``.
    ``deep_ks`` (default none) adds ``Recall@k`` (chunk level) and ``EvidenceRecall@k`` for those depths.
    """
    rels = [is_relevant(c, gold_spans) for c in retrieved]
    m: dict = {"relevance": rels}
    for k in ks:
        m[f"HitRate@{k}"] = hit_rate_at_k(rels, k)
        m[f"DocHit@{k}"] = doc_hit_at_k(retrieved, gold_doc_id, k)
        m[f"EvidenceRecall@{k}"] = evidence_recall_at_k(retrieved, gold_spans, k)
        m[f"EvidenceF1@{k}"] = (evidence_f1_at_k(retrieved, paragraphs, reference_evidence, k)
                                if paragraphs is not None and reference_evidence is not None else None)
    m["MRR"] = reciprocal_rank(rels)
    for k in deep_ks:
        m[f"Recall@{k}"] = recall_at_k(rels, k, n_relevant_in_index)
        m[f"EvidenceRecall@{k}"] = evidence_recall_at_k(retrieved, gold_spans, k)
    m["nDCG@10"] = ndcg_at_k(rels, 10, n_relevant_in_index)
    m["n_relevant_in_index"] = n_relevant_in_index
    m["reachable"] = n_relevant_in_index > 0
    return m
