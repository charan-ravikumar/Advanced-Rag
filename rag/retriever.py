"""Retriever: vector leg + lexical leg + fusion + optional rerank, with per-stage latency.

Import-safe: no module-level state, no clients, no network, no model. Everything is injected:
  store      db.store.VectorStore      (local Chroma)
  embedder   rag.embedders.Embedder     (encodes the query for the vector leg)
  lexical    rag.lexical.LexicalIndex   (SQLite FTS5)
  reranker   optional callable ``reranker(query, texts) -> list[float]`` (higher = better); never loaded here

Behaviour guaranteed by tests/unit/test_retriever.py
  * ``doc_ids`` restricts BOTH legs before ranking and limiting (filter parity).
  * mode="vector" never touches the lexical index; mode="lexical" never touches the embedder or the store query.
  * the fused list is cut to ``rerank_k`` (rerank on) or ``top_k`` (rerank off); it is never unbounded.
  * a chunk found by both legs appears once, with both ranks recorded.
  * ties are broken by chunk_id, so the order is deterministic.
  * with ``executors`` (``rag.concurrency``) the two legs run at the same time, embedding and reranking in their own bounded pools and the
    database reads in another; without them everything runs in the caller's thread, one step after the other. The result is the same.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import wait
from dataclasses import dataclass, field
from typing import Any

from rag.concurrency import RetrievalExecutors, Stage, run_chain
from rag.embedders import embed_query
from rag.fusion import fuse_rrf, naive_interleave

MODES = ("hybrid", "vector", "lexical")
FUSIONS = ("rrf", "naive")
STAGES = ("route", "embed", "vector", "lexical", "fuse", "hydrate", "rerank")


@dataclass(frozen=True)
class LegHit:
    rank: int  # 1-based rank inside that leg
    score: float  # vector: cosine similarity (1 - distance); lexical: -bm25 (higher = better)


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc_id: str
    text: str
    score: float  # final score: reranker score if reranked, else fused (or single-leg) score
    rank: int  # 1-based final rank
    legs: dict[str, LegHit]  # only the legs that found it: "vector" and/or "lexical"
    rerank_score: float | None = None
    # provenance (section-relative offsets, as stored with the chunk); -1 when the chunk was not located
    section: str = ""
    char_start: int = -1
    char_end: int = -1


@dataclass
class RetrievalResult:
    query: str
    mode: str
    fusion: str | None  # None unless mode == "hybrid"
    hits: list[Hit]
    latency_ms: dict[str, float]  # route, embed, vector, lexical, fuse, hydrate, rerank, total
    fused_candidates: int  # size of the fused list BEFORE truncation (diagnostic)
    reranked: bool = False
    params: dict = field(default_factory=dict)
    routed_doc_ids: list[str] | None = None  # set by retrieve_routed: the papers stage 2 was restricted to


class Retriever:
    def __init__(self, store, embedder, lexical, reranker: Callable[[str, list[str]], Sequence[float]] | None = None, cfg=None,
                 executors: RetrievalExecutors | None = None):
        """``executors``: run the legs concurrently. ``lexical`` and ``store`` must then be safe to call from several threads
        (``rag.lexical.PooledFtsIndex``); a plain ``SqliteFtsIndex`` is one connection and is not."""
        self.executors = executors
        self.store = store
        self.embedder = embedder
        self.lexical = lexical
        self.reranker = reranker
        self.cfg = cfg
        p = getattr(cfg, "pipeline", None)
        self._defaults = {
            "top_k": getattr(p, "top_k", 5),
            "candidate_k": getattr(p, "candidate_k", 20),
            "rerank_k": getattr(p, "rerank_k", 20),
            "rrf_k": getattr(p, "rrf_k", 60),
            "vector_weight": getattr(p, "vector_weight", 1.0),
            "lexical_weight": getattr(p, "bm25_weight", 0.7),
        }

    def retrieve(self, query: str, **options: Any) -> RetrievalResult:
        """Blocking retrieval. The options are those of ``_plan``."""
        plan = self._plan(query, **options)
        try:
            legs = next(plan)
            while True:
                legs = plan.send(self._run(legs))
        except StopIteration as stop:
            return stop.value

    async def aretrieve(self, query: str, **options: Any) -> RetrievalResult:
        """``retrieve`` for an event loop: no thread waits while the pools work. Without executors it runs ``retrieve`` in a thread
        (``store`` and ``lexical`` must then be usable from that thread)."""
        if self.executors is None:
            return await asyncio.to_thread(self.retrieve, query, **options)  # to_thread keeps the context
        plan = self._plan(query, **options)
        try:
            legs = next(plan)
            while True:
                futures = [asyncio.wrap_future(run_chain(self.executors, leg)) for leg in legs]
                results = await asyncio.gather(*futures, return_exceptions=True)  # wait for EVERY leg, also when one failed
                for result in results:
                    if isinstance(result, BaseException):
                        raise result
                legs = plan.send(results)
        except StopIteration as stop:
            return stop.value

    def _run(self, legs: list[list[Stage]]) -> list[Any]:
        """The result of each leg (a list of stages run in order): concurrently in the pools, or here one after the other."""
        if self.executors is None:
            results = []
            for leg in legs:
                value = None
                for _pool, stage in leg:
                    value = stage(value)
                results.append(value)
            return results
        futures = [run_chain(self.executors, leg) for leg in legs]
        wait(futures)  # every leg finishes before the first failure is raised: no work outlives its request
        return [f.result() for f in futures]

    def _plan(
        self,
        query: str,
        *,
        mode: str = "hybrid",
        fusion: str = "rrf",
        top_k: int | None = None,
        candidate_k: int | None = None,
        rerank_k: int | None = None,
        doc_ids: Sequence[str] | None = None,
        rerank: bool = False,
        rrf_k: int | None = None,
        vector_weight: float | None = None,
        lexical_weight: float | None = None,
    ) -> Generator[list[list[Stage]], list[Any], RetrievalResult]:
        """The pipeline as a generator, so that ``retrieve`` and ``aretrieve`` share it: it yields the legs to run, receives their
        results and returns the ``RetrievalResult``. What runs in this generator itself (fusion, assembly) is cheap and never blocks."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if fusion not in FUSIONS:
            raise ValueError(f"fusion must be one of {FUSIONS}, got {fusion!r}")
        d = self._defaults
        top_k = d["top_k"] if top_k is None else top_k
        candidate_k = d["candidate_k"] if candidate_k is None else candidate_k
        rerank_k = d["rerank_k"] if rerank_k is None else rerank_k
        rrf_k = d["rrf_k"] if rrf_k is None else rrf_k
        vector_weight = d["vector_weight"] if vector_weight is None else vector_weight
        lexical_weight = d["lexical_weight"] if lexical_weight is None else lexical_weight
        if top_k < 1 or candidate_k < 1:
            raise ValueError("top_k and candidate_k must be >= 1")
        if rerank and self.reranker is None:
            raise ValueError("rerank=True but no reranker was injected")
        if rerank and rerank_k < top_k:
            raise ValueError(f"rerank_k ({rerank_k}) must be >= top_k ({top_k}) when rerank is on")
        limit = rerank_k if rerank else top_k  # the fused list is never longer than this

        t_total = time.perf_counter()
        lat = {s: 0.0 for s in STAGES}
        params = {"top_k": top_k, "candidate_k": candidate_k, "rerank_k": rerank_k, "rerank": rerank,
                  "rrf_k": rrf_k, "vector_weight": vector_weight, "lexical_weight": lexical_weight,
                  "doc_ids": None if doc_ids is None else len(set(doc_ids))}
        if doc_ids is not None:
            doc_ids = list(dict.fromkeys(doc_ids))
            if not doc_ids:  # an empty allow-list matches nothing in either leg
                lat["total"] = (time.perf_counter() - t_total) * 1000
                return RetrievalResult(query, mode, fusion if mode == "hybrid" else None, [], lat, 0, False, params)

        def timed(stage: str, fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
            def run(value: Any) -> Any:  # measured inside the task: the waiting time for a worker is only in "total"
                t = time.perf_counter()
                out = fn(value)
                lat[stage] = (time.perf_counter() - t) * 1000
                return out

            return run

        def vector_search(query_vec: Any) -> list[tuple]:
            where = None if doc_ids is None else {"doc_id": {"$in": doc_ids}}
            raw = self.store.query(query_vec, k=candidate_k, where=where)
            raw.sort(key=lambda r: (r["distance"], r["chunk_id"]))  # deterministic on equal distance
            return [
                (r["chunk_id"], 1.0 - r["distance"], r["text"], r["metadata"]["doc_id"],
                 r["metadata"].get("section", ""), int(r["metadata"].get("char_start", -1)),
                 int(r["metadata"].get("char_end", -1)))
                for r in raw
            ]

        search_legs: dict[str, list[Stage]] = {}
        if mode in ("hybrid", "vector"):
            search_legs["vector"] = [("embed", timed("embed", lambda _: embed_query(self.embedder, query))),
                                     ("db", timed("vector", vector_search))]
        if mode in ("hybrid", "lexical"):
            search_legs["lexical"] = [("db", timed("lexical", lambda _: self.lexical.search(query, candidate_k, doc_ids=doc_ids)))]
        found = dict(zip(search_legs, (yield list(search_legs.values()))))
        vec_hits: list[tuple] = found.get("vector", [])  # (chunk_id, similarity, text, doc_id, section, char_start, char_end)
        lex_hits: list[tuple[str, float]] = found.get("lexical", [])

        # ---- fuse ------------------------------------------------------------------------
        t = time.perf_counter()
        legs: dict[str, dict[str, LegHit]] = {}
        for rank, (cid, sim, *_rest) in enumerate(vec_hits, 1):
            legs.setdefault(cid, {})["vector"] = LegHit(rank, sim)
        for rank, (cid, score) in enumerate(lex_hits, 1):
            legs.setdefault(cid, {})["lexical"] = LegHit(rank, score)
        vec_ids = [h[0] for h in vec_hits]
        lex_ids = [h[0] for h in lex_hits]

        if mode == "vector":
            fused = [(cid, legs[cid]["vector"].score) for cid in vec_ids]
        elif mode == "lexical":
            fused = [(cid, legs[cid]["lexical"].score) for cid in lex_ids]
        elif fusion == "rrf":
            fused = fuse_rrf(vec_ids, lex_ids, vector_weight=vector_weight,
                             lexical_weight=lexical_weight, k=rrf_k)
        else:
            merged = naive_interleave(vec_ids, lex_ids)
            fused = [(cid, 1.0 / rank) for rank, cid in enumerate(merged, 1)]
        fused_candidates = len(fused)
        fused = fused[:limit]  # truncate AFTER merging
        lat["fuse"] = (time.perf_counter() - t) * 1000

        # ---- hydrate text / doc_id for the survivors (lexical-only hits have neither yet) ------------
        known = {cid: (text, doc, sec, cs, ce) for cid, _s, text, doc, sec, cs, ce in vec_hits}
        need = [cid for cid, _ in fused if cid not in known]
        if need:
            (chunks,) = yield [[("db", timed("hydrate", lambda _: self.store.get_chunks(need)))]]
            for c in chunks:
                known[c.chunk_id] = (c.text, c.doc_id, c.section, c.char_start, c.char_end)

        # ---- optional rerank ----------------------------------------------------------------
        rerank_scores: dict[str, float] = {}
        if rerank and fused:
            texts = [known[cid][0] for cid, _ in fused]
            reranker = self.reranker
            assert reranker is not None  # checked above
            (raw_scores,) = yield [[("rerank", timed("rerank", lambda _: reranker(query, texts)))]]
            scores = list(raw_scores)
            if len(scores) != len(texts):
                raise ValueError(f"reranker returned {len(scores)} scores for {len(texts)} texts")
            rerank_scores = {cid: float(s) for (cid, _), s in zip(fused, scores)}
            fused = sorted(fused, key=lambda kv: (-rerank_scores[kv[0]], kv[0]))
        final = fused[:top_k]

        hits = [
            Hit(
                chunk_id=cid,
                doc_id=known[cid][1],
                text=known[cid][0],
                score=rerank_scores.get(cid, score),
                rank=rank,
                legs=dict(legs[cid]),
                rerank_score=rerank_scores.get(cid),
                section=known[cid][2],
                char_start=known[cid][3],
                char_end=known[cid][4],
            )
            for rank, (cid, score) in enumerate(final, 1)
        ]
        lat["total"] = (time.perf_counter() - t_total) * 1000
        return RetrievalResult(query, mode, fusion if mode == "hybrid" else None, hits, lat,
                               fused_candidates, bool(rerank_scores), params)

    def retrieve_routed(self, query: str, router, *, route_m: int, **kwargs) -> RetrievalResult:
        """Two-stage retrieval for "search all papers": (1) ``router.route(query, route_m)`` picks the top papers from a
        paper-level index, (2) ``retrieve`` runs restricted to those papers (``doc_ids`` filters BOTH legs, so no chunk
        from another paper can appear). The router sees only the query text. ``latency_ms["route"]`` is stage 1 alone
        (router query embedding included); the other stages are those of ``retrieve``; ``total`` covers both.
        """
        if kwargs.get("doc_ids") is not None:
            raise ValueError("retrieve_routed chooses doc_ids itself; do not pass doc_ids")
        if route_m < 1:
            raise ValueError("route_m must be >= 1")
        t_total = time.perf_counter()
        t = time.perf_counter()
        routed = list(dict.fromkeys(router.route(query, route_m)))[:route_m]
        route_ms = (time.perf_counter() - t) * 1000
        result = self.retrieve(query, doc_ids=routed, **kwargs)
        result.latency_ms["route"] = route_ms
        result.latency_ms["total"] = (time.perf_counter() - t_total) * 1000
        result.routed_doc_ids = routed
        result.params["route_m"] = route_m
        return result
