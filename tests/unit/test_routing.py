"""Paper router, the two-stage Retriever entry point, and ingestion with a separate vector model. Hash embedder, tmp Chroma, tmp FTS."""

import inspect

import chromadb
import numpy as np
import pytest
from chromadb.config import Settings

from config import cfg
from db.store import build_store
from rag.embedders import HashEmbedder
from rag.gemini_embedder import QuotaExhausted
from rag.ingestion import ingest_corpus, ingest_signature
from rag.lexical import SqliteFtsIndex
from rag.retriever import Retriever
from rag.router_docs import PaperRecord, PaperRouter, papers_from_corpus

TOPICS = ["alpha model training dataset", "network latency server cache", "protein cell enzyme sequence",
          "river delta soil water", "market price trade supply"]
EMB = HashEmbedder()


def _sentences(topic, n, tag):
    return " ".join(f"{topic} {tag} number {i} appears in this sentence." for i in range(n))


def _paper(i):
    t = TOPICS[i]
    return {"title": f"Study of {t}", "abstract": _sentences(t, 4, f"abs{i}"),
            "qas": [{"question": "GOLD-QUESTION-SENTINEL", "evidence": "GOLD-EVIDENCE-SENTINEL"}],
            "full_text": [{"section_name": "Introduction", "paragraphs": [_sentences(t, 14, f"intro{i}")]},
                          {"section_name": "Methods", "paragraphs": [_sentences(t, 30, f"meth{i}")]}]}


CORPUS = {f"1234.0000{i}": _paper(i) for i in range(5)}


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("routing")
    client = chromadb.PersistentClient(path=str(tmp / "chroma"), settings=Settings(anonymized_telemetry=False))
    store = build_store(cfg, "recursive", client=client)
    lex = SqliteFtsIndex(tmp / "lexical_recursive.sqlite3")
    ingest_corpus(CORPUS, "recursive", store, EMB, cfg, lexical=lex, log=lambda *_: None)
    router = PaperRouter(papers_from_corpus(CORPUS), EMB)
    yield {"retriever": Retriever(store, EMB, lex, cfg=cfg), "router": router, "store": store, "lex": lex}
    lex.close()
    router.close()


# ---- router -----------------------------------------------------------------------------------------------------

def test_route_returns_the_matching_paper_first_and_honours_m(stack):
    r = stack["router"]
    assert r.route("protein cell enzyme", 1) == ["1234.00002"]
    top3 = r.route("protein cell enzyme", 3)
    assert len(top3) == 3 and top3[0] == "1234.00002" and len(set(top3)) == 3
    assert r.route("protein cell enzyme", 3)[:1] == r.route("protein cell enzyme", 1)  # prefix property of one ranking
    assert len(r.route("anything", 50)) == 5  # never more than the number of papers
    with pytest.raises(ValueError):
        r.route("x", 0)


def test_route_is_deterministic_and_breaks_ties_by_doc_id(stack):
    assert stack["router"].route("zzz unseen words", 5) == stack["router"].route("zzz unseen words", 5)
    twins = PaperRouter([PaperRecord("b", "same title", "same abstract"), PaperRecord("a", "same title", "same abstract")], EMB)
    assert twins.route("same title", 2) == ["a", "b"]


def test_the_lexical_leg_matters_when_the_vector_leg_cannot_tell_papers_apart(stack):
    class Constant(HashEmbedder):  # every text gets the same vector: only the FTS leg can discriminate
        def encode(self, texts, show_progress_bar=False, **kw):
            one = np.ones(self.dim, dtype=np.float32) / np.sqrt(self.dim)
            return one if isinstance(texts, str) else np.stack([one] * len(texts))

    r = PaperRouter(papers_from_corpus(CORPUS), Constant())
    assert r.route("protein enzyme", 1) == ["1234.00002"]


def test_the_router_is_built_from_titles_and_abstracts_only_and_never_sees_gold(monkeypatch):
    seen = []
    orig = EMB.encode

    def spy(texts, *a, **k):
        seen.append(texts)
        return orig(texts, *a, **k)

    monkeypatch.setattr(EMB, "encode", spy)
    router = PaperRouter(papers_from_corpus(CORPUS), EMB)
    router.route("protein cell enzyme", 2)
    docs = [t for call in seen if isinstance(call, list) for t in call]
    assert len(docs) == 5 and all(t.startswith("Study of ") for t in docs)  # title + abstract, nothing from the body or the questions
    assert not any("SENTINEL" in t or "Methods" in t for t in docs)
    assert [c for c in seen if isinstance(c, str)] == ["protein cell enzyme"]  # the only other input is the query text
    sig = inspect.signature(PaperRecord)
    assert list(sig.parameters) == ["doc_id", "title", "abstract"]
    assert list(inspect.signature(PaperRouter.route).parameters) == ["self", "query", "m"]
    router.close()


def test_duplicate_or_missing_papers_are_refused():
    with pytest.raises(ValueError, match="duplicate"):
        PaperRouter([PaperRecord("a", "t", "x"), PaperRecord("a", "u", "y")], EMB)
    with pytest.raises(ValueError, match="no papers"):
        PaperRouter([], EMB)


# ---- two-stage retrieval ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("m", [1, 2, 3])
def test_the_second_stage_never_returns_chunks_outside_the_routed_papers(stack, m):
    q = "network latency server cache"
    res = stack["retriever"].retrieve_routed(q, stack["router"], route_m=m, mode="hybrid", top_k=10, candidate_k=20)
    assert res.routed_doc_ids == stack["router"].route(q, m) and len(res.routed_doc_ids) == m
    assert res.hits and {h.doc_id for h in res.hits} <= set(res.routed_doc_ids)
    assert res.params["route_m"] == m and res.latency_ms["route"] > 0
    assert res.latency_ms["total"] >= res.latency_ms["route"]


@pytest.mark.parametrize("mode", ["hybrid", "vector", "lexical"])
def test_the_routing_filter_applies_to_every_leg(stack, mode):
    class Fixed:
        def route(self, query, m):
            return ["1234.00001"]

    res = stack["retriever"].retrieve_routed("number appears sentence protein", Fixed(), route_m=1, mode=mode)  # words shared by every paper
    assert res.hits and {h.doc_id for h in res.hits} == {"1234.00001"}


def test_the_router_gets_the_query_and_m_only_and_a_long_answer_is_cut_to_m(stack):
    calls = []

    class Spy:
        def route(self, *args, **kwargs):
            calls.append((args, kwargs))
            return ["1234.00000", "1234.00001", "1234.00002"]

    res = stack["retriever"].retrieve_routed("some question", Spy(), route_m=2)
    assert calls == [(("some question", 2), {})] and res.routed_doc_ids == ["1234.00000", "1234.00001"]


def test_an_empty_route_returns_no_hits_and_doc_ids_cannot_be_passed(stack):
    class Nothing:
        def route(self, query, m):
            return []

    res = stack["retriever"].retrieve_routed("q", Nothing(), route_m=3)
    assert res.hits == [] and res.routed_doc_ids == []
    with pytest.raises(ValueError, match="doc_ids"):
        stack["retriever"].retrieve_routed("q", stack["router"], route_m=3, doc_ids=["1234.00000"])
    with pytest.raises(ValueError):
        stack["retriever"].retrieve_routed("q", stack["router"], route_m=0)


def test_plain_retrieve_is_unchanged_and_reports_no_route(stack):
    res = stack["retriever"].retrieve("protein cell enzyme", mode="hybrid", top_k=5)
    assert res.routed_doc_ids is None and res.latency_ms["route"] == 0.0


# ---- ingestion with a separate vector model ---------------------------------------------------------------------

class AsymmetricFake:
    """Vector model different from the chunking embedder; fails with a quota error on demand."""

    name, revision, dims = "fake-asym", "1/dims16", 16

    def __init__(self, fail_on_call=None):
        self.inner, self.calls, self.fail_on_call, self.doc_calls, self.query_calls = HashEmbedder(dim=16), 0, fail_on_call, 0, 0

    def encode_documents(self, texts):
        self.calls += 1
        self.doc_calls += 1
        if self.fail_on_call == self.calls:
            raise QuotaExhausted("daily quota exhausted (test)")
        return self.inner.encode(list(texts))

    def encode_query(self, text):
        self.query_calls += 1
        return self.inner.encode(text)

    def stats(self):
        return {"calls": self.calls}


def _store(tmp_path, name):
    client = chromadb.PersistentClient(path=str(tmp_path / name), settings=Settings(anonymized_telemetry=False))
    return build_store(cfg, "recursive", client=client), SqliteFtsIndex(tmp_path / f"{name}.sqlite3")


def test_a_separate_vector_model_keeps_the_chunks_identical_and_changes_only_the_vectors(tmp_path):
    (plain, lp), (gem, lg) = _store(tmp_path, "plain"), _store(tmp_path, "gem")
    ingest_corpus(CORPUS, "recursive", plain, EMB, cfg, lexical=lp, log=lambda *_: None)
    fake = AsymmetricFake()
    manifest = ingest_corpus(CORPUS, "recursive", gem, EMB, cfg, lexical=lg, log=lambda *_: None, vector_embedder=fake)
    a, b = {c.chunk_id: c for c in plain.get_all()}, {c.chunk_id: c for c in gem.get_all()}
    assert a.keys() == b.keys() and all((a[i].text, a[i].section, a[i].char_start, a[i].char_end) == (b[i].text, b[i].section, b[i].char_start, b[i].char_end) for i in a)
    assert lp.chunk_ids() == lg.chunk_ids() == set(a)  # lexical index covers the same chunks
    assert plain.get_signature() != gem.get_signature()
    assert plain.get_signature() == ingest_signature("recursive", EMB, cfg)  # the historical signature is unchanged
    assert manifest["vector_embedder"]["name"] == "fake-asym" and manifest["verification"]["ok"] and fake.doc_calls == 5
    hits = gem.query(fake.encode_query(TOPICS[1]), k=3)
    assert hits and hits[0]["metadata"]["doc_id"] == "1234.00001"
    lp.close()
    lg.close()


def test_a_quota_stop_keeps_finished_documents_and_the_next_run_resumes_then_skips_everything(tmp_path):
    gem, lg = _store(tmp_path, "gem")
    with pytest.raises(QuotaExhausted):
        ingest_corpus(CORPUS, "recursive", gem, EMB, cfg, lexical=lg, log=lambda *_: None, vector_embedder=AsymmetricFake(fail_on_call=3))
    assert gem.doc_ids() == {"1234.00000", "1234.00001"} and lg.doc_ids() == gem.doc_ids()  # nothing of doc 3 was written
    resumed = ingest_corpus(CORPUS, "recursive", gem, EMB, cfg, lexical=lg, log=lambda *_: None, vector_embedder=AsymmetricFake())
    assert resumed["docs"]["ingested"] == 3 and resumed["docs"]["skipped_unchanged"] == 2 and resumed["verification"]["ok"]
    again = ingest_corpus(CORPUS, "recursive", gem, EMB, cfg, lexical=lg, log=lambda *_: None, vector_embedder=AsymmetricFake())
    assert again["docs"]["ingested"] == 0 and again["docs"]["skipped_unchanged"] == 5 and again["chunks"]["written_this_run"] == 0
    lg.close()


def test_a_store_built_with_one_vector_model_refuses_another(tmp_path):
    from rag.ingestion import IngestConfigMismatch

    gem, lg = _store(tmp_path, "gem")
    ingest_corpus(CORPUS, "recursive", gem, EMB, cfg, lexical=lg, log=lambda *_: None, vector_embedder=AsymmetricFake())
    with pytest.raises(IngestConfigMismatch):
        ingest_corpus(CORPUS, "recursive", gem, EMB, cfg, lexical=lg, log=lambda *_: None)  # MiniLM-style vectors into a Gemini store
    lg.close()


def test_the_retriever_encodes_queries_with_the_query_entry_point(tmp_path):
    gem, lg = _store(tmp_path, "gem")
    fake = AsymmetricFake()
    ingest_corpus(CORPUS, "recursive", gem, EMB, cfg, lexical=lg, log=lambda *_: None, vector_embedder=fake)
    res = Retriever(gem, fake, lg, cfg=cfg).retrieve("protein cell enzyme", mode="vector", top_k=3)
    assert fake.query_calls == 1 and res.hits[0].doc_id == "1234.00002"
    lg.close()
