"""Paper-level router for two-stage retrieval: find the paper, then the passage.

``PaperRouter`` indexes ONE entry per paper built from its title and abstract, and nothing else: the build takes
``(doc_id, title, abstract)`` records, never questions, gold spans or labels. ``route(query, m)`` returns the top-``m``
paper ids by hybrid RRF over two legs:

* vector: exact brute-force cosine over the paper vectors (deterministic, no ANN), with the embedder's ``encode_documents``
  for the papers and ``encode_query`` for the question (asymmetric models use different task types);
* lexical: SQLite FTS5 (porter), OR of the query words (``rag.lexical.escape_query``), in an in-memory database.

Ties are broken by doc id. Import-safe: nothing is loaded or opened at import. Brute force is fine for hundreds to tens of thousands
of papers; beyond that replace the vector leg with an ANN index.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from rag.embedders import embed_documents, embed_query
from rag.fusion import fuse_rrf
from rag.lexical import escape_query


@dataclass(frozen=True)
class PaperRecord:
    doc_id: str
    title: str
    abstract: str

    @property
    def text(self) -> str:
        return f"{self.title}\n\n{self.abstract}".strip()


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1.0, n)


class PaperRouter:
    def __init__(self, papers: Sequence[PaperRecord], embedder, *, rrf_k: int = 60, vector_weight: float = 1.0,
                 lexical_weight: float = 0.7, candidate_k: int = 20):
        papers = sorted(papers, key=lambda p: p.doc_id)
        if not papers:
            raise ValueError("no papers to index")
        if len({p.doc_id for p in papers}) != len(papers):
            raise ValueError("duplicate doc_id in the paper records")
        self.embedder = embedder
        self.rrf_k, self.vector_weight, self.lexical_weight, self.candidate_k = rrf_k, vector_weight, lexical_weight, candidate_k
        self.doc_ids = [p.doc_id for p in papers]
        self._vectors = _unit(np.asarray(embed_documents(embedder, [p.text for p in papers])))
        self._conn = sqlite3.connect(":memory:")
        self._conn.execute("CREATE VIRTUAL TABLE papers USING fts5(doc_id UNINDEXED, text, tokenize='porter unicode61')")
        self._conn.executemany("INSERT INTO papers(doc_id, text) VALUES (?, ?)", [(p.doc_id, p.text) for p in papers])
        self._conn.commit()

    def __len__(self) -> int:
        return len(self.doc_ids)

    def _vector_ids(self, query: str, k: int) -> list[str]:
        q = _unit(np.asarray(embed_query(self.embedder, query)))
        sims = self._vectors @ q
        order = sorted(range(len(self.doc_ids)), key=lambda i: (-float(sims[i]), self.doc_ids[i]))
        return [self.doc_ids[i] for i in order[:k]]

    def _lexical_ids(self, query: str, k: int) -> list[str]:
        match = escape_query(query)
        if match is None:
            return []
        rows = self._conn.execute("SELECT doc_id, bm25(papers) AS s FROM papers WHERE papers MATCH ? ORDER BY s, doc_id LIMIT ?",
                                  (match, k)).fetchall()
        return [r[0] for r in rows]

    def route(self, query: str, m: int) -> list[str]:
        """The top-``m`` paper ids for ``query`` (best first). Uses the query text only."""
        if m < 1:
            raise ValueError("m must be >= 1")
        k = max(self.candidate_k, m)
        fused = fuse_rrf(self._vector_ids(query, k), self._lexical_ids(query, k), vector_weight=self.vector_weight,
                         lexical_weight=self.lexical_weight, k=self.rrf_k)
        return [doc_id for doc_id, _ in fused[:m]]

    def close(self) -> None:
        self._conn.close()


def papers_from_corpus(corpus: dict) -> list[PaperRecord]:
    """Title and abstract of every paper of a Qasper-style corpus ``{paper_id: {"title": ..., "abstract": ...}}``."""
    return [PaperRecord(pid, p.get("title") or "", p.get("abstract") or "") for pid, p in corpus.items()]
