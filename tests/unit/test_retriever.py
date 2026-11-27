"""Retriever tests: HashEmbedder + tmp local Chroma + tmp SQLite FTS5. No network, no model, no reranker model."""

import time

import chromadb
import pytest
from chromadb.config import Settings

from config import cfg
from db.store import build_store
from rag.embedders import HashEmbedder
from rag.lexical import SqliteFtsIndex
from rag.retriever import Retriever, RetrievalResult
from rag.schemas_ingest import Chunk

EMB = HashEmbedder()
FILLER = "dataset model training corpus annotation label evaluation result"


class World:
    def __init__(self, tmp_path):
        client = chromadb.PersistentClient(path=str(tmp_path / "chroma"), settings=Settings(anonymized_telemetry=False))
        self.store = build_store(cfg, "recursive", client=client)
        self.lexical = SqliteFtsIndex(tmp_path / "lexical.sqlite3")
        self.chunks: dict[str, Chunk] = {}

    def add(self, doc_id: str, idx: int, text: str) -> Chunk:
        return self.add_many([(doc_id, idx, text)])[0]

    def add_many(self, rows) -> list[Chunk]:
        chunks = [
            Chunk.create(doc_id=d, section="S", chunk_idx=i, char_start=0, char_end=len(t),
                         strategy="recursive", text=t, doc_version="v1")
            for d, i, t in rows
        ]
        self.store.upsert(chunks, EMB.encode([c.text for c in chunks]))
        self.lexical.upsert(chunks)
        self.chunks.update({c.chunk_id: c for c in chunks})
        return chunks

    def retriever(self, **kw) -> Retriever:
        return Retriever(self.store, kw.pop("embedder", EMB), kw.pop("lexical", self.lexical), cfg=cfg, **kw)


@pytest.fixture
def world(tmp_path):
    w = World(tmp_path)
    yield w
    w.lexical.close()


# multi_doc and needle are read-only (tests only monkeypatch, which is undone per test), so they are
# built once per module: rebuilding Chroma per test made this file the slowest in the fast profile.
@pytest.fixture(scope="module")
def multi_doc(tmp_path_factory):
    """Four documents, six chunks each, every chunk shares vocabulary so both legs return hits from all docs."""
    w = World(tmp_path_factory.mktemp("multi_doc"))
    w.add_many(
        (f"doc{doc}", i, f"{FILLER} paper {doc} section {i} sentence about {doc}{i} details")
        for doc in "ABCD" for i in range(6)
    )
    yield w
    w.lexical.close()


@pytest.fixture(scope="module")
def needle(tmp_path_factory):
    """12 filler chunks plus ONE chunk reachable only through a rare word (lexical), not through meaning (vector)."""
    w = World(tmp_path_factory.mktemp("needle"))
    w.add_many((("fill", i, f"{FILLER} filler number {i} one two three") for i in range(12)))
    w.rare = w.add("rare_doc", 0, "zyxwv " + " ".join(f"unrelated{j} word{j} other{j}" for j in range(12)))
    w.query = "dataset model training zyxwv"
    yield w
    w.lexical.close()


# ---- filter parity -----------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["hybrid", "vector", "lexical"])
@pytest.mark.parametrize("fusion", ["rrf", "naive"])
@pytest.mark.parametrize("allowed", [["docB"], ["docB", "docD"], ["docA", "docC", "docD"]])
def test_doc_filter_restricts_every_mode(multi_doc, mode, fusion, allowed):
    r = multi_doc.retriever().retrieve("dataset model paper details", mode=mode, fusion=fusion,
                                       doc_ids=allowed, top_k=10, candidate_k=24)
    assert r.hits, "the fixture must produce hits so the assertion below is not vacuous"
    assert {h.doc_id for h in r.hits} <= set(allowed)
    assert {multi_doc.chunks[h.chunk_id].doc_id for h in r.hits} <= set(allowed)


def test_filter_is_passed_to_both_legs(multi_doc, monkeypatch):
    seen = {}
    real_q, real_s = multi_doc.store.query, multi_doc.lexical.search
    monkeypatch.setattr(multi_doc.store, "query", lambda *a, **k: (seen.__setitem__("where", k.get("where")), real_q(*a, **k))[1])
    monkeypatch.setattr(multi_doc.lexical, "search", lambda *a, **k: (seen.__setitem__("doc_ids", k.get("doc_ids")), real_s(*a, **k))[1])
    multi_doc.retriever().retrieve("dataset model", doc_ids=["docB", "docC"])
    assert set(seen["where"]["doc_id"]["$in"]) == {"docB", "docC"} == set(seen["doc_ids"])


def test_filter_beats_relevance(multi_doc):
    """The best matches live in docA ("A0" in the query) but the caller only allows docC."""
    r = multi_doc.retriever().retrieve("paper A section 0 sentence about A0", doc_ids=["docC"], top_k=10, candidate_k=24)
    assert r.hits and {h.doc_id for h in r.hits} == {"docC"}


def test_empty_allow_list_returns_nothing_and_calls_no_leg(multi_doc, monkeypatch):
    for target, name in ((multi_doc.store, "query"), (multi_doc.lexical, "search")):
        monkeypatch.setattr(target, name, lambda *a, **k: pytest.fail("a leg ran for an empty allow-list"))
    for mode in ("hybrid", "vector", "lexical"):
        r = multi_doc.retriever().retrieve("dataset", mode=mode, doc_ids=[])
        assert r.hits == [] and r.fused_candidates == 0


def test_unknown_doc_filter_returns_nothing(multi_doc):
    for mode in ("hybrid", "vector", "lexical"):
        assert multi_doc.retriever().retrieve("dataset", mode=mode, doc_ids=["nope"]).hits == []


# ---- a lexical-only top hit survives hybrid fusion ---------------------------------------------

def test_precondition_rare_chunk_is_lexical_only(needle):
    rt = needle.retriever()
    vec = rt.retrieve(needle.query, mode="vector", candidate_k=3, top_k=3)
    lex = rt.retrieve(needle.query, mode="lexical", candidate_k=3, top_k=3)
    assert needle.rare.chunk_id not in [h.chunk_id for h in vec.hits]
    assert lex.hits[0].chunk_id == needle.rare.chunk_id


def test_lexical_only_top_hit_survives_hybrid_rrf(needle):
    r = needle.retriever().retrieve(needle.query, mode="hybrid", fusion="rrf", candidate_k=3, top_k=5)
    hit = next(h for h in r.hits if h.chunk_id == needle.rare.chunk_id)
    assert set(hit.legs) == {"lexical"} and hit.legs["lexical"].rank == 1


def test_lexical_only_top_hit_survives_truncation_with_naive_interleave(needle):
    """top_k=3 with 3 vector candidates would drop every lexical result under the old
    truncate-the-concatenation bug; round-robin keeps the lexical top hit at rank 2."""
    r = needle.retriever().retrieve(needle.query, mode="hybrid", fusion="naive", candidate_k=3, top_k=3)
    assert [h.chunk_id for h in r.hits][1] == needle.rare.chunk_id
    assert len(r.hits) == 3


def test_old_truncated_concatenation_would_have_lost_it(needle):
    rt = needle.retriever()
    vec = [h.chunk_id for h in rt.retrieve(needle.query, mode="vector", candidate_k=3, top_k=3).hits]
    lex = [h.chunk_id for h in rt.retrieve(needle.query, mode="lexical", candidate_k=3, top_k=3).hits]
    assert needle.rare.chunk_id not in (vec + lex)[:3]  # the old harness's behaviour


# ---- bounded fused list / rerank --------------------------------------------------------------

def test_fused_list_is_cut_to_top_k_when_rerank_is_off(multi_doc):
    r = multi_doc.retriever().retrieve("dataset model", top_k=3, candidate_k=20)
    assert r.fused_candidates > 3 and len(r.hits) == 3


def test_reranker_sees_at_most_rerank_k_texts_and_result_is_top_k(multi_doc):
    seen = []

    def reranker(query, texts):
        seen.append(list(texts))
        return [float(i) for i in range(len(texts))]  # last text wins

    r = multi_doc.retriever(reranker=reranker).retrieve("dataset model", top_k=2, rerank_k=4, candidate_k=20, rerank=True)
    assert len(seen) == 1 and len(seen[0]) == 4  # not the whole fused list
    assert r.fused_candidates > 4
    assert len(r.hits) == 2 and r.reranked
    assert r.hits[0].rerank_score == 3.0 and r.hits[0].text == seen[0][3]
    assert [h.rank for h in r.hits] == [1, 2]


def test_reranker_is_not_called_when_rerank_is_false(multi_doc):
    r = multi_doc.retriever(reranker=lambda q, t: pytest.fail("called")).retrieve("dataset", rerank=False)
    assert not r.reranked and r.latency_ms["rerank"] == 0.0


def test_rerank_without_reranker_is_an_error(multi_doc):
    with pytest.raises(ValueError, match="no reranker"):
        multi_doc.retriever().retrieve("dataset", rerank=True)


def test_rerank_k_smaller_than_top_k_is_an_error(multi_doc):
    with pytest.raises(ValueError, match="rerank_k"):
        multi_doc.retriever(reranker=lambda q, t: [0.0] * len(t)).retrieve("dataset", rerank=True, top_k=5, rerank_k=2)


def test_reranker_wrong_length_is_an_error(multi_doc):
    with pytest.raises(ValueError, match="scores"):
        multi_doc.retriever(reranker=lambda q, t: [1.0]).retrieve("dataset", rerank=True, top_k=2, rerank_k=4)


def test_rerank_ties_break_by_chunk_id(multi_doc):
    r = multi_doc.retriever(reranker=lambda q, t: [1.0] * len(t)).retrieve("dataset model", rerank=True, top_k=5, rerank_k=8)
    ids = [h.chunk_id for h in r.hits]
    assert ids == sorted(ids)


# ---- both legs found it: one hit, two ranks ------------------------------------------------------

def test_chunk_found_by_both_legs_appears_once_with_both_ranks(multi_doc):
    target = next(c for c in multi_doc.chunks.values() if c.doc_id == "docB" and c.chunk_idx == 2)
    r = multi_doc.retriever().retrieve(target.text, top_k=10, candidate_k=24)
    ids = [h.chunk_id for h in r.hits]
    assert len(ids) == len(set(ids))
    hit = r.hits[0]
    assert hit.chunk_id == target.chunk_id
    assert set(hit.legs) == {"vector", "lexical"}
    assert hit.legs["vector"].rank >= 1 and hit.legs["lexical"].rank >= 1
    expected = 1 / (60 + hit.legs["vector"].rank) + 0.7 / (60 + hit.legs["lexical"].rank)
    assert hit.score == pytest.approx(expected)


# ---- single-leg modes never touch the other leg -----------------------------------------------------

class Exploding:
    def __getattr__(self, name):
        raise AssertionError(f"the other leg was touched: {name}")


def test_vector_mode_never_touches_the_lexical_index(multi_doc):
    r = multi_doc.retriever(lexical=Exploding()).retrieve("dataset model", mode="vector", top_k=3)
    assert r.hits and all(set(h.legs) == {"vector"} for h in r.hits)
    assert r.latency_ms["lexical"] == 0.0


def test_lexical_mode_never_embeds_or_queries_the_vector_store(multi_doc, monkeypatch):
    monkeypatch.setattr(multi_doc.store, "query", lambda *a, **k: pytest.fail("vector query ran"))
    r = multi_doc.retriever(embedder=Exploding()).retrieve("dataset model", mode="lexical", top_k=3)
    assert r.hits and all(set(h.legs) == {"lexical"} for h in r.hits)
    assert r.latency_ms["embed"] == 0.0 and r.latency_ms["vector"] == 0.0
    assert all(h.text and h.doc_id for h in r.hits)  # hydrated from the store by id


# ---- determinism ----------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["hybrid", "vector", "lexical"])
@pytest.mark.parametrize("fusion", ["rrf", "naive"])
def test_identical_chunks_tie_and_are_ordered_by_chunk_id(world, mode, fusion):
    world.add_many((f"dup{d}", 0, "identical duplicated sentence for the tie test") for d in range(6))
    runs = [[h.chunk_id for h in world.retriever().retrieve("identical duplicated sentence", mode=mode, fusion=fusion, top_k=6).hits]
            for _ in range(3)]
    assert runs[0] == runs[1] == runs[2] == sorted(runs[0]) and len(runs[0]) == 6


def test_equal_scores_are_ordered_by_chunk_id(world):
    world.add_many((f"dup{d}", 0, "identical duplicated sentence for the tie test") for d in range(5))
    r = world.retriever().retrieve("identical duplicated sentence", mode="vector", top_k=5)
    scores = [h.score for h in r.hits]
    assert len(set(scores)) == 1
    assert [h.chunk_id for h in r.hits] == sorted(h.chunk_id for h in r.hits)


# ---- latency is measured per stage ------------------------------------------------------------------

class SleepyEmbedder(HashEmbedder):
    def encode(self, texts, *a, **k):
        time.sleep(0.03)
        return super().encode(texts, *a, **k)


def test_every_stage_latency_is_measured(multi_doc, monkeypatch):
    real_q, real_s = multi_doc.store.query, multi_doc.lexical.search
    monkeypatch.setattr(multi_doc.store, "query", lambda *a, **k: (time.sleep(0.03), real_q(*a, **k))[1])
    monkeypatch.setattr(multi_doc.lexical, "search", lambda *a, **k: (time.sleep(0.03), real_s(*a, **k))[1])

    def reranker(q, texts):
        time.sleep(0.03)
        return [0.0] * len(texts)

    r = multi_doc.retriever(embedder=SleepyEmbedder(), reranker=reranker).retrieve(
        "dataset model", rerank=True, top_k=2, rerank_k=4)
    lat = r.latency_ms
    assert {"embed", "vector", "lexical", "fuse", "rerank", "total"} <= set(lat)
    for stage in ("embed", "vector", "lexical", "rerank"):
        assert lat[stage] >= 29.0, f"{stage} was not measured: {lat}"
    assert lat["fuse"] >= 0.0
    assert lat["total"] >= sum(lat[s] for s in ("embed", "vector", "lexical", "fuse", "rerank")) - 1.0


def test_skipped_stages_report_zero(multi_doc):
    lat = multi_doc.retriever().retrieve("dataset", mode="lexical").latency_ms
    assert lat["embed"] == lat["vector"] == lat["rerank"] == 0.0 and lat["lexical"] > 0.0


# ---- misc ---------------------------------------------------------------------------------------------

def test_result_shape(multi_doc):
    r = multi_doc.retriever().retrieve("dataset model", top_k=3)
    assert isinstance(r, RetrievalResult) and r.mode == "hybrid" and r.fusion == "rrf"
    assert [h.rank for h in r.hits] == [1, 2, 3]
    assert all(h.text and h.doc_id and h.legs for h in r.hits)
    assert [h.score for h in r.hits] == sorted((h.score for h in r.hits), reverse=True)


@pytest.mark.parametrize("kw", [{"mode": "nope"}, {"fusion": "nope"}, {"top_k": 0}, {"candidate_k": 0}])
def test_invalid_arguments_raise(multi_doc, kw):
    with pytest.raises(ValueError):
        multi_doc.retriever().retrieve("dataset", **kw)


def test_defaults_come_from_config(multi_doc):
    r = multi_doc.retriever().retrieve("dataset model")
    assert r.params["top_k"] == cfg.pipeline.top_k and r.params["candidate_k"] == cfg.pipeline.candidate_k
    assert r.params["rerank_k"] == cfg.pipeline.rerank_k
    assert len(r.hits) <= cfg.pipeline.top_k


def test_query_with_fts_syntax_does_not_break_the_hybrid_path(multi_doc):
    r = multi_doc.retriever().retrieve('"dataset" AND (model OR NOT x) col:val -y*', top_k=3)
    assert r.hits


def test_no_lexical_match_still_returns_vector_hits(multi_doc):
    r = multi_doc.retriever().retrieve("qqqqqq wwwwww", top_k=3)
    assert r.hits and all(set(h.legs) == {"vector"} for h in r.hits)


@pytest.mark.parametrize("mode", ["hybrid", "vector", "lexical"])
def test_hits_carry_chunk_provenance(multi_doc, mode):
    """Vector-leg hits get it from metadata, lexical-only hits from the hydration fetch: both must match the chunk."""
    r = multi_doc.retriever().retrieve("dataset model paper details", mode=mode, top_k=6, candidate_k=24)
    assert r.hits
    for h in r.hits:
        c = multi_doc.chunks[h.chunk_id]
        assert (h.section, h.char_start, h.char_end) == (c.section, c.char_start, c.char_end)
        assert h.char_end > h.char_start >= 0


def test_lexical_only_hit_provenance_comes_from_the_store(needle):
    r = needle.retriever().retrieve(needle.query, mode="hybrid", fusion="rrf", candidate_k=3, top_k=5)
    hit = next(h for h in r.hits if h.chunk_id == needle.rare.chunk_id)
    assert set(hit.legs) == {"lexical"}
    assert (hit.section, hit.char_start, hit.char_end) == (needle.rare.section, needle.rare.char_start, needle.rare.char_end)


def test_rrf_weights_and_k_are_per_call_settings_and_are_recorded(multi_doc):
    rt = multi_doc.retriever()
    r = rt.retrieve("dataset model", mode="hybrid", fusion="rrf", rrf_k=10, vector_weight=2.0, lexical_weight=0.25)
    assert (r.params["rrf_k"], r.params["vector_weight"], r.params["lexical_weight"]) == (10, 2.0, 0.25)
    both = next(h for h in r.hits if set(h.legs) == {"vector", "lexical"})
    expected = 2.0 / (10 + both.legs["vector"].rank) + 0.25 / (10 + both.legs["lexical"].rank)
    assert both.score == pytest.approx(expected)
    default = rt.retrieve("dataset model")
    assert (default.params["rrf_k"], default.params["vector_weight"], default.params["lexical_weight"]) == (
        cfg.pipeline.rrf_k, cfg.pipeline.vector_weight, cfg.pipeline.bm25_weight)
