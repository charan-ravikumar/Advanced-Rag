"""Context construction for the generation-evaluation arms (pre-registration ``eval/PREREGISTRATION_gen.md``). Pure: no SDK, no network, no store.

* ``R``  (RAG)    the retrieved chunks: NOT built here (the real Retriever supplies them).
* ``O``  (oracle) the gold evidence paragraphs: the union over annotators of their verbatim evidence paragraphs, first occurrence in the paper, in
                  document order, up to ``max_chunks`` (5). NEVER a retrieved chunk. Table and figure evidence is not retrievable and is not in the list.
* ``T``  (title)  title and abstract of the question's paper, as one source.

``FixedContextRetriever`` is the injected retriever for O and T: it returns the context it was given and nothing else, so the arm that is
supposed to know only the gold paragraphs (or only the abstract) cannot reach the index. Hit objects duck-type the retriever's hits for
``rag_core.context.build_context``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace

ARMS = ("R", "O", "T", "F")
MAX_ORACLE_CHUNKS = 5


@dataclass(frozen=True)
class ContextHit:
    chunk_id: str
    doc_id: str
    section: str
    text: str
    rank: int
    score: float = 1.0


@dataclass
class OracleContext:
    hits: list[ContextHit] = field(default_factory=list)
    n_gold_paragraphs: int = 0  # distinct gold paragraphs found in the paper
    n_dropped_over_cap: int = 0  # gold paragraphs beyond the first ``max_chunks`` in document order
    n_not_found: int = 0  # evidence texts that match no paragraph of the paper (should be 0: the golden set keeps verbatim paragraphs only)


def usable_references(references: Sequence[dict]) -> list[dict]:
    """The references that count for Answer-F1: not ``unanswerable`` and not table-only (no verbatim evidence listed while at least one
    evidence item was not retrievable: a table or figure). A reference with no evidence at all and none lost is kept."""
    out = []
    for ref in references or []:
        if ref.get("type") == "unanswerable" or ref.get("answer") is None:
            continue
        if not ref.get("evidence") and (ref.get("n_evidence_not_retrievable") or 0) > 0:
            continue
        out.append(ref)
    return out


def gold_paragraph_texts(references: Sequence[dict]) -> list[str]:
    """Distinct evidence paragraph texts over all non-unanswerable references, in first-seen order."""
    seen: dict[str, None] = {}
    for ref in references or []:
        if ref.get("type") == "unanswerable":
            continue
        for text in ref.get("evidence") or []:
            if text and text not in seen:
                seen[text] = None
    return list(seen)


def oracle_context(paper_id: str, references: Sequence[dict], paragraphs: Sequence, max_chunks: int = MAX_ORACLE_CHUNKS) -> OracleContext:
    """The gold paragraphs of one question, in document order (section index, then paragraph index), at most ``max_chunks``.

    ``paragraphs`` is ``eval.gold_spans.paragraph_index(paper_id, paper)``. A text that occurs in more than one place is taken at its first
    occurrence. ``hits`` is empty (and the question has no oracle arm) when the question has no gold paragraph."""
    first: dict[str, object] = {}
    for p in paragraphs:
        first.setdefault(p.text, p)
    wanted = gold_paragraph_texts(references)
    found = [first[t] for t in wanted if t in first]
    found.sort(key=lambda p: (p.section_idx, p.para_idx))
    kept = found[:max_chunks]
    hits = [ContextHit(chunk_id=f"oracle:{paper_id}:{p.section_idx}:{p.para_idx}", doc_id=paper_id, section=p.section, text=p.text, rank=i + 1)
            for i, p in enumerate(kept)]
    return OracleContext(hits=hits, n_gold_paragraphs=len(found), n_dropped_over_cap=max(0, len(found) - len(kept)),
                         n_not_found=len(wanted) - len(found))


def title_abstract_context(paper_id: str, title: str, abstract: str) -> list[ContextHit]:
    """Title and abstract as ONE source. Empty if the paper has neither (not expected for Qasper)."""
    title, abstract = (title or "").strip(), (abstract or "").strip()
    if not title and not abstract:
        return []
    text = f"Title: {title}\n\nAbstract: {abstract}".strip()
    return [ContextHit(chunk_id=f"title_abstract:{paper_id}", doc_id=paper_id, section="Title and abstract", text=text, rank=1)]


def context_hash(hits: Sequence) -> str:
    """Hash of exactly what is offered to the generator (ids and texts, in order): part of the generation cache key."""
    h = hashlib.sha256()
    for hit in hits:
        h.update(hit.chunk_id.encode("utf-8") + b"\x00" + hit.text.encode("utf-8") + b"\x01")
    return h.hexdigest()


class FixedContextRetriever:
    """The ``retriever`` given to ``RagService`` for the arms O and T. ``set`` the context of the next question; ``retrieve`` returns it.
    It holds no index, so it cannot return a retrieved chunk. ``calls`` records every query for the tests."""

    def __init__(self, label: str):
        self.label = label
        self.embedder = SimpleNamespace(revision=f"fixed-context:{label}")
        self._hits: list = []
        self.calls: list[dict] = []

    def set(self, hits: Sequence) -> None:
        self._hits = list(hits)

    def retrieve(self, query: str, **kw):
        self.calls.append({"query": query, **kw})
        top_k = kw.get("top_k")
        hits = self._hits[:top_k] if top_k else list(self._hits)
        return SimpleNamespace(hits=hits, latency_ms={})
