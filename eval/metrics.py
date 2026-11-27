"""
eval/metrics.py — dependency-free metric helpers for the eval harness.

Retrieval is matched against gold evidence (token-F1 on passage text) and
against the gold paper (by title). Answer quality is scored against the
reference answers with SQuAD-style EM / token-F1.
"""

import re
import string
from collections import Counter

MATCH_THRESHOLD = 0.25
REFUSAL_MARKER = "could not find that information"


def token_f1_set(a: str, b: str) -> float:
    """Unigram-set F1 used to decide whether a chunk matches a gold passage."""
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta or not tb:
        return 0.0
    inter = ta & tb
    p = len(inter) / len(ta)
    r = len(inter) / len(tb)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def is_relevant(gold_evidence: list, chunk: str) -> bool:
    return any(token_f1_set(ev, chunk) >= MATCH_THRESHOLD for ev in gold_evidence)


def first_relevant_rank(gold_evidence: list, chunks: list) -> int | None:
    """1-based rank of the first chunk matching any gold passage, else None."""
    for rank, chunk in enumerate(chunks, start=1):
        if is_relevant(gold_evidence, chunk):
            return rank
    return None


def doc_hit_rank(gold_title: str, sources: list) -> int | None:
    """1-based rank of the first retrieved chunk that comes from the gold paper."""
    for rank, source in enumerate(sources, start=1):
        if source == gold_title:
            return rank
    return None


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def answer_f1(prediction: str, reference: str) -> float:
    pred, ref = normalize_answer(prediction).split(), normalize_answer(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def best_over_references(metric, prediction: str, references: list) -> float:
    return max(metric(prediction, ref) for ref in references)


def is_refusal(answer: str) -> bool:
    return REFUSAL_MARKER in answer.lower()
