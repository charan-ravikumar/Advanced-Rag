"""Deterministic dev/test split by paper.

A paper belongs to ``dev`` when ``int(sha256(paper_id), 16) % 100 < DEV_PERCENT``, otherwise to ``test``.
The rule depends only on the paper id, so no paper (and therefore no near-duplicate question about it)
can appear in both splits, and the assignment never changes when questions are added or removed.
"""

from __future__ import annotations

import hashlib

DEV_PERCENT = 40


def split_of(paper_id: str) -> str:
    bucket = int(hashlib.sha256(paper_id.encode("utf-8")).hexdigest(), 16) % 100
    return "dev" if bucket < DEV_PERCENT else "test"


def split_questions(questions: list[dict]) -> dict[str, list[str]]:
    """``{"dev": [ids], "test": [ids]}``, ids sorted."""
    out: dict[str, list[str]] = {"dev": [], "test": []}
    for q in questions:
        out[split_of(q["paper_id"])].append(q["id"])
    return {k: sorted(v) for k, v in out.items()}
