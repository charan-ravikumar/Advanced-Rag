"""Official-style answer metrics for the generation evaluation. Pure (stdlib + numpy through ``eval.compare`` for the intervals): no network, no SDK.

* **Answer-F1** follows the Qasper evaluator: SQuAD-style normalisation (lowercase, remove punctuation, remove the articles a/an/the, collapse
  whitespace), token-level F1 over the multiset overlap, and the **maximum over the annotators' references**. Two documented deviations: punctuation also
  covers Unicode punctuation (typographic quotes and dashes, which model output contains and which are not ASCII punctuation), and extractive references
  are stored joined with a space where the official evaluator joins with ", " (identical after normalisation, which removes the comma; tested).
* A reference is **usable** unless it is ``unanswerable`` or table-only (``eval.gen_arms.usable_references``). A question with no usable reference has no Answer-F1.
* An abstention on an answerable question scores F1 = 0 (a miss) and is counted in the abstention rate.
* **Yes/no accuracy**: over questions with at least one usable yes/no reference, the answer's first word is "yes" or "no" and equals a reference.
* **Citation validity**: valid cited labels over all cited labels, per answer (None when nothing is cited), plus the pooled rate and the share of answers with no citation.
  A valid marker does not mean the claim is supported; that is what faithfulness is for.
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from eval.gen_arms import usable_references

_PUNCT = frozenset(string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(text: str | None) -> str:
    """Lowercase, remove punctuation (ASCII and Unicode), remove the articles a/an/the, collapse whitespace."""
    if not text:
        return ""
    lowered = str(text).lower()
    stripped = "".join(ch for ch in lowered if ch not in _PUNCT and not unicodedata.category(ch).startswith("P"))
    return " ".join(_ARTICLES.sub(" ", stripped).split())


def token_f1(prediction: str | None, reference: str | None) -> float:
    """Token-level F1 of the normalised texts (multiset overlap). 0 when there is no overlap, including when either side is empty."""
    p, r = normalize_answer(prediction).split(), normalize_answer(reference).split()
    overlap = sum((Counter(p) & Counter(r)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(r)
    return 2 * precision * recall / (precision + recall)


@dataclass(frozen=True)
class AnswerScore:
    f1: float
    best_type: str  # the type of the best-matching reference (ties: the first in reference order)
    best_index: int  # index into the USABLE references


def answer_f1(prediction: str | None, references: Sequence[dict]) -> AnswerScore | None:
    """Maximum token F1 over the USABLE references; ``None`` if there is none. ``prediction`` of an abstention must be passed as ``""``."""
    refs = usable_references(references)
    if not refs:
        return None
    best: AnswerScore | None = None
    for i, ref in enumerate(refs):
        f1 = token_f1(prediction, ref["answer"])
        if best is None or f1 > best.f1:  # strict: the first reference wins a tie
            best = AnswerScore(f1, ref["type"], i)
    return best


def yes_no_label(prediction: str | None) -> str | None:
    """"yes" or "no" if the answer starts with one of them (after normalisation), else None."""
    words = normalize_answer(prediction).split()
    return words[0] if words and words[0] in ("yes", "no") else None


def yes_no_accuracy(prediction: str | None, references: Sequence[dict]) -> float | None:
    """1.0 / 0.0 over questions with a usable yes/no reference (the label equals one of them); None when the question has none."""
    wanted = {normalize_answer(r["answer"]) for r in usable_references(references) if r["type"] == "yes_no"}
    if not wanted:
        return None
    return 1.0 if yes_no_label(prediction) in wanted else 0.0


def citation_counts(citations: Sequence[str], invalid_citations: Sequence[str]) -> tuple[int, int]:
    """``(valid, total)`` distinct cited labels; ``AnswerResult.citations`` are the valid ones, ``invalid_citations`` name sources that were never provided."""
    valid, invalid = set(citations), set(invalid_citations)
    return len(valid), len(valid | invalid)


_CITATION_MARKER = re.compile(r"[\[\u3010\uff3b]\s*S\d+(?:\s*,\s*S\d+)*\s*[\]\u3011\uff3d]", re.IGNORECASE)  # [S1], [S1, S2], 【S1】, ［S1］


def strip_citation_markers(text: str | None) -> str:
    """Remove ``[S1]``-style source markers: they are not part of the answer (the short prompt does not ask for them, but a model may add them)."""
    return _CITATION_MARKER.sub(" ", text or "")


def score_short(short_answer: str | None, abstained: bool, references: Sequence[dict]) -> tuple[dict, str | None]:
    """Metrics of one short-mode answer and the type of its best reference. An abstention is a prediction of nothing (F1 0 on an answerable question)."""
    prediction = "" if abstained or not short_answer else strip_citation_markers(short_answer)
    score = answer_f1(prediction, references)
    metrics = {"AnswerF1": None if score is None else score.f1, "YesNoAccuracy": yes_no_accuracy(prediction, references),
               "Abstained": 1.0 if abstained else 0.0}
    return metrics, None if score is None else score.best_type


def quote_counts(claims: Sequence) -> tuple[int, int]:
    """``(verified, total)`` quotes over the claims of one long answer (``Claim`` objects or their dicts): a quote is verified when it was found word
    for word in the source it cites (``rag_core.claims.locate_quote``). Verified does NOT mean the source supports the claim."""
    spans = [c for claim in claims for c in (claim["citations"] if isinstance(claim, dict) else claim.citations)]
    return sum(1 for c in spans if (c["verified"] if isinstance(c, dict) else c.verified)), len(spans)


def score_long(citations: Sequence[str], invalid_citations: Sequence[str], abstained: bool, claims: Sequence = ()) -> dict:
    """Citation metrics of one long-mode answer. An abstention has no citations to judge (the citation metrics are None). ``QuoteVerified`` is the
    share of the answer's quotes found word for word in their cited source (None when there is no quote)."""
    valid, total = citation_counts(citations, invalid_citations)
    if abstained:
        return {"CitationValidity": None, "HasCitation": None, "QuoteVerified": None, "Abstained": 1.0}
    verified, quotes = quote_counts(claims)
    return {"CitationValidity": None if total == 0 else valid / total, "HasCitation": 1.0 if total else 0.0,
            "QuoteVerified": None if quotes == 0 else verified / quotes, "Abstained": 0.0}


# ---- summaries with paper-clustered intervals -------------------------------------------------------------------------------------

def mean_with_interval(values: Sequence[float], clusters: Sequence[str], *, n_resamples: int = 5000, seed: int = 0) -> dict:
    """Mean with a 95% paper-clustered bootstrap interval (resample papers, keep a paper's questions together): ``eval.compare.clustered_bootstrap``
    applied to the values themselves (its 'difference' is then the plain mean). ``ci_*`` are None for fewer than two papers."""
    from eval.compare import clustered_bootstrap

    if not values:
        return {"n": 0, "n_papers": 0, "mean": None, "ci_low": None, "ci_high": None}
    n_papers = len(set(clusters))
    out = {"n": len(values), "n_papers": n_papers, "mean": float(sum(values) / len(values)), "ci_low": None, "ci_high": None}
    if n_papers >= 2:
        r = clustered_bootstrap(list(values), list(clusters), n_resamples, seed)
        out["ci_low"], out["ci_high"] = r["ci_low"], r["ci_high"]
    return out


def summarize_metric(rows: Sequence[dict], metric: str, **kw) -> dict:
    """``mean_with_interval`` of ``row['metrics'][metric]`` over the rows where it is defined."""
    pairs = [(r["metrics"][metric], r["paper_id"]) for r in rows if r["metrics"].get(metric) is not None]
    return mean_with_interval([v for v, _ in pairs], [p for _, p in pairs], **kw)


def summarize_short(rows: Sequence[dict], **kw) -> dict:
    """Short-mode summary: Answer-F1 overall and by the type of the best reference, yes/no accuracy, abstention rate."""
    by_type: dict[str, dict] = {}
    for typ in sorted({r["best_reference_type"] for r in rows if r.get("best_reference_type")}):
        subset = [r for r in rows if r.get("best_reference_type") == typ]
        by_type[typ] = summarize_metric(subset, "AnswerF1", **kw)
    return {"AnswerF1": summarize_metric(rows, "AnswerF1", **kw), "AnswerF1_by_best_reference_type": by_type,
            "YesNoAccuracy": summarize_metric(rows, "YesNoAccuracy", **kw), "AbstentionRate_on_answerable": summarize_metric(rows, "Abstained", **kw),
            "n_questions": len(rows)}


def summarize_long(rows: Sequence[dict], **kw) -> dict:
    """Long-mode summary: per-answer citation validity, the pooled validity rate, the share of non-abstained answers with no citation, the abstention rate."""
    valid = total = verified = quotes = 0
    for r in rows:
        if not r.get("abstained"):
            v, t = citation_counts(r.get("citations", []), r.get("invalid_citations", []))
            valid, total = valid + v, total + t
            qv, qt = quote_counts(r.get("claims", []))
            verified, quotes = verified + qv, quotes + qt
    answered = [r for r in rows if not r.get("abstained")]
    return {"CitationValidity_per_answer": summarize_metric(rows, "CitationValidity", **kw),
            "CitationValidity_pooled": None if total == 0 else {"valid": valid, "cited": total, "rate": valid / total},
            "QuoteVerified_per_answer": summarize_metric([r for r in rows if "QuoteVerified" in r["metrics"]], "QuoteVerified", **kw),
            "QuoteVerified_pooled": None if quotes == 0 else {"verified": verified, "quotes": quotes, "rate": verified / quotes},
            "answers_with_no_citation": {"n": sum(1 for r in answered if not citation_counts(r.get("citations", []), r.get("invalid_citations", []))[1]),
                                         "of": len(answered)},
            "AbstentionRate_on_answerable": summarize_metric(rows, "Abstained", **kw), "n_questions": len(rows)}
