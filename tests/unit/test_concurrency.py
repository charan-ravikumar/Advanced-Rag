"""Concurrent retrieval: parallel legs, bounded pools, the SQLite connection pool, context propagation and the API's queue limit.
HashEmbedder, a tmp local Chroma and tmp SQLite; the API tests use the in-process app with a fake retriever. No network, no model."""

import asyncio
import contextvars
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from opentelemetry import context as otel_context

from rag.concurrency import AdmissionGate, RetrievalExecutors, run_chain
from rag.embedders import HashEmbedder
from rag.lexical import PooledFtsIndex
from rag.retriever import Retriever
from tests.unit import test_api
from tests.unit.test_api import Q, code_of, make, stream_events
from tests.unit.test_retriever import World

api = test_api.api  # the fixture of the API tests: builds a client around a fake service
WAIT = 5.0  # seconds: only reached when something is broken (a test never sleeps this long when it passes)
QUERIES = ["dataset model paper details", "training corpus annotation", "evaluation result section 3", "label sentence about B2", "zzz nothing"]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    w = World(tmp_path_factory.mktemp("concurrency"))
    w.add_many((f"doc{d}", i, f"dataset model training corpus annotation label evaluation result paper {d} section {i} sentence about {d}{i} details")
               for d in "ABCD" for i in range(6))
    yield w
    w.lexical.close()


@pytest.fixture
def pools():
    made = []

    def build(**kw):
        made.append(RetrievalExecutors(**kw))
        return made[-1]

    yield build
    for executors in made:
        executors.shutdown()


def parallel(world, executors, **kw):
    pool = PooledFtsIndex(world.lexical.path, size=executors.workers["db"])
    return Retriever(world.store, kw.pop("embedder", HashEmbedder()), kw.pop("lexical", pool), cfg=world.retriever().cfg, executors=executors, **kw)


def fingerprint(result):
    return [(h.chunk_id, h.doc_id, h.text, h.score, h.rank, h.legs, h.rerank_score, h.section, h.char_start) for h in result.hits], result.fused_candidates


# ---- the same answers ------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("options", [dict(), dict(mode="vector"), dict(mode="lexical"), dict(fusion="naive"), dict(doc_ids=["docB", "docD"]), dict(doc_ids=[]),
                                     dict(rerank=True, rerank_k=8, top_k=3)])
@pytest.mark.parametrize("workers", [1, 4])
def test_parallel_and_async_retrieval_equal_sequential_retrieval_under_load(world, pools, options, workers):
    reranker = lambda query, texts: [float(len(t) % 7) for t in texts]  # noqa: E731
    sequential = world.retriever(reranker=reranker)
    concurrent = parallel(world, pools(embed_workers=workers, db_workers=workers, rerank_workers=workers), reranker=reranker)
    expected = {q: fingerprint(sequential.retrieve(q, **options)) for q in QUERIES}
    with ThreadPoolExecutor(8) as callers:
        threaded = list(callers.map(lambda q: (q, fingerprint(concurrent.retrieve(q, **options))), QUERIES * 6))

    async def many():
        return await asyncio.gather(*(concurrent.aretrieve(q, **options) for q in QUERIES * 6))

    results = asyncio.run(many())
    assert all(got == expected[q] for q, got in threaded)
    assert all(fingerprint(r) == expected[r.query] for r in results)
    assert set(results[0].latency_ms) == set(sequential.retrieve(QUERIES[0], **options).latency_ms)


def test_without_executors_the_async_form_still_works(world):
    pooled = world.retriever(lexical=PooledFtsIndex(world.lexical.path, size=1))
    assert fingerprint(asyncio.run(pooled.aretrieve(QUERIES[0]))) == fingerprint(world.retriever().retrieve(QUERIES[0]))


# ---- really at the same time, in the right pools, with the caller's context ---------------------------------------------------------------

class Probe:
    """Wraps the embedder and the lexical index: records the thread, a context variable and the OTel context each call sees."""

    marker: contextvars.ContextVar = contextvars.ContextVar("marker", default="unset")

    def __init__(self, world, barrier=None):
        self.world, self.barrier, self.seen = world, barrier, {}

    def _note(self, name):
        self.seen[name] = (threading.current_thread().name, self.marker.get(), otel_context.get_value("probe"))
        if self.barrier is not None:
            self.barrier.wait(WAIT)  # BrokenBarrierError unless the other leg is running at the same moment

    def encode_query(self, text):
        self._note("embed")
        return HashEmbedder().encode(text)

    def search(self, query, k, doc_ids=None):
        self._note("lexical")
        return [(next(iter(self.world.chunks)), 1.0)]


def call(retriever, how):
    return retriever.retrieve(QUERIES[0]) if how == "sync" else asyncio.run(retriever.aretrieve(QUERIES[0]))


@pytest.mark.parametrize("how", ["sync", "async"])
def test_the_two_legs_overlap_and_run_in_their_own_pools_with_the_callers_context(world, pools, how):
    probe = Probe(world, threading.Barrier(2))

    class Store:  # the second stage of the vector leg and the hydration: also tasks, also with the context
        def query(self, *a, **k):
            probe.seen["vector"] = (threading.current_thread().name, Probe.marker.get(), otel_context.get_value("probe"))
            return []

        def get_chunks(self, ids):
            probe.seen["hydrate"] = (threading.current_thread().name, Probe.marker.get(), otel_context.get_value("probe"))
            return world.store.get_chunks(ids)

    retriever = Retriever(Store(), probe, probe, executors=pools(embed_workers=1, db_workers=2))
    token = otel_context.attach(otel_context.set_value("probe", "span-ctx"))
    reset = Probe.marker.set("request-1")
    try:
        assert call(retriever, how).hits
    finally:
        Probe.marker.reset(reset)
        otel_context.detach(token)
    assert set(probe.seen) == {"embed", "lexical", "vector", "hydrate"}
    assert probe.seen["embed"][0].startswith("rag-embed") and all(probe.seen[s][0].startswith("rag-db") for s in ("lexical", "vector", "hydrate"))
    assert all(seen[1:] == ("request-1", "span-ctx") for seen in probe.seen.values())


@pytest.mark.parametrize("how", ["sync", "async"])
def test_a_failing_leg_is_raised_after_the_other_leg_has_finished(world, pools, how):
    finished = []

    class Broken:
        def encode_query(self, text):
            raise OSError("model files missing")

    class Slow:
        def search(self, *a, **k):
            threading.Event().wait(0.05)
            finished.append(True)
            return []

    retriever = Retriever(world.store, Broken(), Slow(), executors=pools())
    with pytest.raises(OSError, match="model files missing"):
        call(retriever, how)
    assert finished == [True]


def test_no_pool_runs_more_tasks_than_it_has_workers_and_one_worker_each_cannot_deadlock(world, pools):
    lock, active, peak = threading.Lock(), {"embed": 0, "db": 0}, {"embed": 0, "db": 0}

    def task(kind):
        def run(_):
            with lock:
                active[kind] += 1
                peak[kind] = max(peak[kind], active[kind])
            threading.Event().wait(0.005)
            with lock:
                active[kind] -= 1
            return kind

        return run

    executors = pools(embed_workers=2, db_workers=3)
    chains = [run_chain(executors, [("embed", task("embed")), ("db", task("db")), ("embed", task("embed"))]) for _ in range(40)]
    assert [c.result(WAIT) for c in chains] == ["embed"] * 40 and peak["embed"] <= 2 and peak["db"] <= 3
    retriever = parallel(world, pools(embed_workers=1, db_workers=1))
    with ThreadPoolExecutor(8) as callers:
        assert len(list(callers.map(retriever.retrieve, QUERIES * 8))) == 40


def test_pool_sizes_below_one_are_refused_and_a_shut_down_pool_fails_the_request():
    with pytest.raises(ValueError):
        RetrievalExecutors(db_workers=0)
    executors = RetrievalExecutors()
    executors.shutdown()
    with pytest.raises(RuntimeError):
        run_chain(executors, [("db", lambda _: 1)]).result(WAIT)


# ---- the SQLite connection pool ----------------------------------------------------------------------------------------------------------

def test_each_borrower_gets_its_own_connection_and_the_next_one_waits(world):
    pool = PooledFtsIndex(world.lexical.path, size=2)
    got = threading.Event()
    with pool.connection() as first, pool.connection() as second:
        assert first is not second
        waiter = threading.Thread(target=lambda: (pool.count(), got.set()))
        waiter.start()
        assert not got.wait(0.1)  # both connections are busy
    waiter.join(WAIT)
    assert got.is_set() and pool.count() == len(world.chunks) and pool.chunk_ids() == set(world.chunks)
    assert pool.doc_chunk_ids("docA") == world.lexical.doc_chunk_ids("docA")
    pool.close()


def test_a_pooled_connection_cannot_write_and_a_failed_search_returns_it(world):
    pool = PooledFtsIndex(world.lexical.path, size=1)
    with pool.connection() as index, pytest.raises(sqlite3.OperationalError, match="readonly"):
        index.delete_by_doc("docA")
    with pytest.raises(ValueError):
        pool.search("dataset", 5, doc_ids=[str(i) for i in range(30001)])
    assert pool.search("dataset", 5) == world.lexical.search("dataset", 5)
    with pytest.raises(ValueError):
        PooledFtsIndex(world.lexical.path, size=0)
    pool.close()


# ---- the queue limit ----------------------------------------------------------------------------------------------------------------------

def test_the_gate_admits_up_to_its_limit_and_a_ticket_is_released_once():
    gate = AdmissionGate(2)
    a, b = gate.try_acquire(), gate.try_acquire()
    assert a and b and gate.try_acquire() is None and (gate.active, gate.refused) == (2, 1)
    a.release()
    a.release()
    assert gate.active == 1 and gate.try_acquire() is not None
    with pytest.raises(ValueError):
        AdmissionGate(0)


@pytest.mark.parametrize("path", ["/v1/retrieve", "/v1/ask", "/v1/answer"])
def test_a_full_queue_is_a_503_with_retry_after_and_nothing_is_spent(api, path):
    client = api()
    state = client.app.state.api
    state.gate = AdmissionGate(1)
    held = state.gate.try_acquire()
    response = client.post(path, json={"question": Q})
    assert response.status_code == 503 and code_of(response) == "overloaded" and response.headers["Retry-After"] == "1"
    assert state.resources.daily.used == 0 and state.resources.service.retriever.calls == []
    held.release()
    assert client.post(path, json={"question": Q}).status_code == 200 and state.gate.active == 0


@pytest.mark.parametrize("path", ["/v1/retrieve", "/v1/ask", "/v1/answer"])
def test_the_place_in_the_queue_is_given_back_also_when_retrieval_fails_or_the_budget_is_used_up(api, path):
    service, retriever, *_ = make()
    client = api(service, daily_cap=0)
    state = client.app.state.api
    state.gate = AdmissionGate(1)
    for _ in range(2):
        assert client.post(path, json={"question": Q}).status_code == 200 and state.gate.active == 0
    retriever.retrieve = lambda *a, **k: (_ for _ in ()).throw(OSError("index gone"))
    for _ in range(2):
        assert client.post(path, json={"question": Q}).status_code == (503 if path == "/v1/retrieve" else 200) and state.gate.active == 0


def test_the_place_is_given_back_when_the_passages_are_found_not_when_the_answer_ends(api):
    service, *_ = make()
    client = api(service)
    state = client.app.state.api
    state.gate = AdmissionGate(1)
    seen = []
    stream = service.gateway.stream

    def spying(*args, **kwargs):  # the model call: retrieval is over, so the queue must be free again
        seen.append(state.gate.active)
        return stream(*args, **kwargs)

    service.gateway.stream = spying
    assert stream_events(client, {"question": Q})[-1]["type"] == "done"
    assert client.post("/v1/answer", json={"question": Q + " Again?"}).status_code == 200
    assert seen == [0, 0]
