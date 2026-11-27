"""RagService with the REAL Retriever (tmp Chroma, tmp FTS5, hash embedder) and a scripted fake provider. No network, no keys."""

import chromadb
import pytest
from chromadb.config import Settings

from config import cfg
from db.store import build_store
from rag.embedders import HashEmbedder
from rag.ingestion import ingest_corpus
from rag.lexical import SqliteFtsIndex
from rag.retriever import Retriever
from rag_core.cache import InMemoryTTLCache
from rag_core.config import RagConfig
from rag_core.gateway import LLMGateway
from rag_core.service import RagService, index_version_for
from tests.unit.rag_core_fakes import FakeClock, ScriptedProvider, answer_json, ok

TOPICS = ["alpha model training dataset", "network latency server cache", "protein cell enzyme sequence", "river delta soil water", "market price trade supply"]
EMB = HashEmbedder()


def _sentences(topic, n, tag):
    return " ".join(f"{topic} {tag} number {i} appears in this sentence." for i in range(n))


def _paper(i):
    t = TOPICS[i]
    return {"title": f"Paper {i}", "abstract": _sentences(t, 4, f"abs{i}"),
            "full_text": [{"section_name": "Introduction", "paragraphs": [_sentences(t, 14, f"intro{i}")]},
                          {"section_name": "Methods", "paragraphs": [_sentences(t, 30, f"meth{i}")]}]}


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rag_core_integration")
    client = chromadb.PersistentClient(path=str(tmp / "chroma"), settings=Settings(anonymized_telemetry=False))
    store = build_store(cfg, "recursive", client=client)
    lex = SqliteFtsIndex(tmp / "lexical_recursive.sqlite3")
    ingest_corpus({f"1234.0000{i}": _paper(i) for i in range(5)}, "recursive", store, EMB, cfg, lexical=lex, log=lambda *_: None)
    yield Retriever(store, EMB, lex, cfg=cfg), store
    lex.close()


def service(stack, script, **kw):
    retriever, store = stack
    clock = FakeClock()
    provider = ScriptedProvider("groq", "model-a", script)
    gw = LLMGateway([provider], sleep=clock.sleep, clock=clock)
    return RagService(retriever, gw, InMemoryTTLCache(clock=clock), RagConfig(), index_version=index_version_for(store), **kw), provider


def test_end_to_end_with_the_real_retriever_the_context_comes_from_the_matching_paper(stack):
    svc, provider = service(stack, [ok(answer_json(("The enzyme work is described.", [("S1", "unverifiable quote here")])))])
    evs = list(svc.answer_stream("protein cell enzyme sequence"))
    r = evs[-1]["result"]
    assert len(r.used_sources) == 5 and r.used_sources[0].doc_id == "1234.00002" and r.citations == ["S1"] and not r.degraded
    assert all(s.text and s.section and s.rank >= 1 for s in r.used_sources) and [s.label for s in r.used_sources] == ["S1", "S2", "S3", "S4", "S5"]
    user = provider.calls[0]["messages"][1]["content"]
    assert all(s.text in user for s in r.used_sources if "<" not in s.text)


def test_the_document_filter_restricts_the_context_to_that_paper(stack):
    svc, provider = service(stack, [ok(answer_json(("Answer.", ["S1"])))])
    r = list(svc.answer_stream("protein cell enzyme sequence", doc_ids=["1234.00001"]))[-1]["result"]
    assert r.used_sources and {s.doc_id for s in r.used_sources} == {"1234.00001"} and {s.doc_id for s in r.retrieved_sources} == {"1234.00001"}


def test_an_empty_document_filter_retrieves_nothing_and_abstains_without_calling_the_model(stack):
    svc, provider = service(stack, [ok(answer_json(("never used", ["S1"])))])
    r = list(svc.answer_stream("protein cell enzyme sequence", doc_ids=[]))[-1]["result"]
    assert r.abstained and r.used_sources == [] and provider.calls == []


def test_the_index_version_starts_with_the_chunk_count_of_the_store(stack):
    retriever, store = stack
    before = index_version_for(store)
    assert before.startswith(f"{store.count()}:")


def test_a_second_identical_question_is_a_cache_hit_with_the_real_retriever(stack):
    svc, provider = service(stack, [ok(answer_json(("Cached answer.", ["S1"])))])
    first = list(svc.answer_stream("network latency server cache"))[-1]["result"]
    second = list(svc.answer_stream("network latency server cache"))[-1]["result"]
    assert not first.cached and second.cached and len(provider.calls) == 1 and second.answer == first.answer
