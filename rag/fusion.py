"""Rank fusion. Pure functions: no I/O, no clients, no logging side effects.

``weighted_rrf`` keeps the signature used by the old pipeline. ``naive_interleave`` is the
correct "no score fusion" baseline. ``fuse_rrf`` is the deterministic ranked form the Retriever uses.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

from config import cfg


def weighted_rrf(vector_rankings, bm25_rankings, vector_weight=1.0, bm25_weight=0.7, k=None):
    """Weighted Reciprocal Rank Fusion. Returns ``{id: score}``.

    score(id) = sum over lists of weight / (k + rank + 1), rank starting at 0. It uses only ranks,
    so no calibration between BM25 and cosine distance is needed. ``k`` defaults to config.
    """
    if k is None:
        k = cfg.pipeline.rrf_k
    scores: dict[Hashable, float] = {}
    for rank, item in enumerate(vector_rankings):
        scores[item] = scores.get(item, 0.0) + vector_weight / (k + rank + 1)
    for rank, item in enumerate(bm25_rankings):
        scores[item] = scores.get(item, 0.0) + bm25_weight / (k + rank + 1)
    return scores


def fuse_rrf(
    vector_ids: Sequence[Hashable],
    lexical_ids: Sequence[Hashable],
    *,
    vector_weight: float = 1.0,
    lexical_weight: float = 0.7,
    k: int | None = None,
    limit: int | None = None,
) -> list[tuple[Hashable, float]]:
    """RRF as an ordered ``[(id, score)]`` list, best first.

    Ties on score are broken by id so the order is deterministic. ``limit`` truncates AFTER
    fusion (never before).
    """
    scores = weighted_rrf(vector_ids, lexical_ids, vector_weight, lexical_weight, k)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], str(kv[0])))
    return ranked if limit is None else ranked[:limit]


def naive_interleave(
    vector_ids: Sequence[Hashable],
    lexical_ids: Sequence[Hashable],
    k: int | None = None,
) -> list[Hashable]:
    """Round-robin merge: vec[0], lex[0], vec[1], lex[1], ... skipping ids already taken.

    The merge is complete BEFORE truncation to ``k``. Truncating a concatenated list
    (``(vec + lex)[:k]``) silently drops every lexical result once ``len(vec) >= k``, which is
    what the old harness did.
    """
    merged: list[Hashable] = []
    seen: set[Hashable] = set()
    for i in range(max(len(vector_ids), len(lexical_ids))):
        for source in (vector_ids, lexical_ids):
            if i < len(source) and source[i] not in seen:
                seen.add(source[i])
                merged.append(source[i])
    return merged if k is None else merged[:k]
