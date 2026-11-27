"""GeminiEmbedder with a fake transport: task types, cache, batching, retries, quota exits, key hygiene. No network, no real key."""

import hashlib
import json

import numpy as np
import pytest

from rag.embedders import HashEmbedder, embed_documents, embed_query
from rag.gemini_embedder import (
    KEY_ENV,
    TASK_DOCUMENT,
    TASK_QUERY,
    EmbeddingCache,
    GeminiEmbedder,
    GeminiError,
    QuotaExhausted,
    default_cache_path,
)

SECRET = "AIzaSy-THIS-IS-A-FAKE-TEST-KEY-1234567890"
DIMS = 8


def _vec(task, text, dims=DIMS):
    seed = int(hashlib.sha256(f"{task}|{text}".encode()).hexdigest()[:8], 16)
    return (np.random.default_rng(seed).normal(size=dims) * 3).tolist()


class FakeTransport:
    """Behaves like the API; ``script`` entries (status, json) are consumed first, one per HTTP attempt."""

    def __init__(self, script=None, dims=DIMS):
        self.calls, self.script, self.dims = [], list(script or []), dims

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append({"method": method, "url": url, "headers": dict(headers), "body": body})
        if self.script:
            status, data = self.script.pop(0)
            if status is not None:
                return status, data, {}
        if url.endswith(":batchEmbedContents"):
            return 200, {"embeddings": [{"values": _vec(r["taskType"], r["content"]["parts"][0]["text"], self.dims)}
                                        for r in body["requests"]]}, {}
        if url.endswith(":countTokens"):
            return 200, {"totalTokens": sum(len(c["parts"][0]["text"].split()) for c in body["contents"])}, {}
        if "/models?" in url:
            return 200, {"models": [{"name": "models/gemini-embedding-001", "version": "001", "displayName": "Gemini Embedding",
                                     "inputTokenLimit": 2048, "supportedGenerationMethods": ["embedContent", "batchEmbedContents"]},
                                    {"name": "models/gemini-x", "supportedGenerationMethods": ["generateContent"]}]}, {}
        raise AssertionError(url)


def make(tmp_path, transport=None, **kw):
    sleeps = []
    emb = GeminiEmbedder(dims=DIMS, cache=EmbeddingCache(tmp_path / "emb.sqlite3"), transport=transport or FakeTransport(),
                         sleep=sleeps.append, **kw)
    return emb, sleeps


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv(KEY_ENV, SECRET)


def test_documents_and_queries_use_the_right_task_types_dimensions_and_model(tmp_path):
    t = FakeTransport()
    emb, _ = make(tmp_path, t)
    emb.encode_documents(["a passage"])
    emb.encode_query("a question")
    doc, qry = (c["body"]["requests"][0] for c in t.calls)
    assert doc["taskType"] == TASK_DOCUMENT == "RETRIEVAL_DOCUMENT" and qry["taskType"] == TASK_QUERY == "RETRIEVAL_QUERY"
    assert doc["outputDimensionality"] == DIMS and doc["model"] == "models/gemini-embedding-001"
    assert t.calls[0]["url"].endswith("/models/gemini-embedding-001:batchEmbedContents")


def test_vectors_are_l2_normalised_and_asymmetric(tmp_path):
    emb, _ = make(tmp_path)
    d = emb.encode_documents(["same text"])[0]
    q = emb.encode_query("same text")
    assert np.isclose(np.linalg.norm(d), 1.0, atol=1e-5) and np.isclose(np.linalg.norm(q), 1.0, atol=1e-5)
    assert not np.allclose(d, q)  # the task type changes the vector: document and query encodings are different calls


def test_the_cache_is_keyed_by_task_and_text_and_a_repeat_makes_no_request(tmp_path):
    t = FakeTransport()
    emb, _ = make(tmp_path, t)
    first = emb.encode_documents(["one", "two"])
    assert len(t.calls) == 1
    again = emb.encode_documents(["two", "one"])
    assert len(t.calls) == 1 and np.allclose(again[0], first[1]) and np.allclose(again[1], first[0])
    emb.encode_query("one")  # same text, other task: a different cache entry, one more request
    assert len(t.calls) == 2
    s = emb.stats()
    assert (s["cache_hits"], s["cache_misses"], s["network_calls"]) == (2, 3, 2)


def test_a_new_process_with_the_same_cache_file_hits_the_cache(tmp_path):
    t1, t2 = FakeTransport(), FakeTransport()
    path = tmp_path / "emb.sqlite3"
    a = GeminiEmbedder(dims=DIMS, cache=EmbeddingCache(path), transport=t1)
    a.encode_documents(["persisted"])
    b = GeminiEmbedder(dims=DIMS, cache=EmbeddingCache(path), transport=t2)
    b.encode_documents(["persisted"])
    assert len(t1.calls) == 1 and len(t2.calls) == 0


def test_the_cache_key_depends_on_model_version_and_dims(tmp_path):
    cache = EmbeddingCache(tmp_path / "emb.sqlite3")
    keys = {GeminiEmbedder("m", "001", 8, cache=cache)._key(TASK_DOCUMENT, "x"), GeminiEmbedder("m2", "001", 8, cache=cache)._key(TASK_DOCUMENT, "x"),
            GeminiEmbedder("m", "002", 8, cache=cache)._key(TASK_DOCUMENT, "x"), GeminiEmbedder("m", "001", 16, cache=cache)._key(TASK_DOCUMENT, "x"),
            GeminiEmbedder("m", "001", 8, cache=cache)._key(TASK_QUERY, "x")}
    assert len(keys) == 5


def test_requests_are_batched_at_100_and_duplicates_are_sent_once(tmp_path):
    t = FakeTransport()
    emb, _ = make(tmp_path, t)
    out = emb.encode_documents([f"text {i}" for i in range(250)] + ["text 0"])
    assert [len(c["body"]["requests"]) for c in t.calls] == [100, 100, 50] and out.shape == (251, DIMS)
    assert np.allclose(out[0], out[250])
    with pytest.raises(ValueError):
        GeminiEmbedder(batch_size=101)


def test_a_rate_limit_is_retried_after_the_servers_delay_then_succeeds(tmp_path):
    limited = {"error": {"code": 429, "message": "slow down", "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "34s"}]}}
    t = FakeTransport(script=[(429, limited), (429, limited)])
    emb, sleeps = make(tmp_path, t)
    assert emb.encode_documents(["x"]).shape == (1, DIMS)
    assert len(t.calls) == 3 and sleeps == [34.0, 34.0]
    s = emb.stats()
    assert (s["http_429"], s["retries"]) == (2, 2)


def test_backoff_grows_when_the_server_gives_no_delay(tmp_path):
    t = FakeTransport(script=[(503, {}), (503, {}), (503, {})])
    emb, sleeps = make(tmp_path, t)
    emb.encode_documents(["x"])
    assert sleeps == [2.0, 4.0, 8.0]


def test_a_daily_quota_stops_at_once_and_keeps_what_was_already_embedded(tmp_path):
    daily = {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED",
                       "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                                    "violations": [{"quotaId": "EmbedContentRequestsPerDayPerProjectPerModel-FreeTier"}]}]}}
    t = FakeTransport(script=[(None, None), (429, daily)])  # first batch fine, second batch over the daily quota
    emb, sleeps = make(tmp_path, t)
    with pytest.raises(QuotaExhausted, match="daily quota"):
        emb.encode_documents([f"t{i}" for i in range(150)])
    assert sleeps == [] and len(t.calls) == 2
    t2 = FakeTransport()
    emb2 = GeminiEmbedder(dims=DIMS, cache=emb.cache, transport=t2)
    emb2.encode_documents([f"t{i}" for i in range(150)])  # resume: only the unfinished batch is requested
    assert [len(c["body"]["requests"]) for c in t2.calls] == [50]


def test_repeated_rate_limits_end_in_quota_exhausted_not_an_infinite_loop(tmp_path):
    limited = {"error": {"code": 429, "message": "slow down"}}
    t = FakeTransport(script=[(429, limited)] * 20)
    emb, sleeps = make(tmp_path, t, max_retries=3)
    with pytest.raises(QuotaExhausted, match="gave up after 3 retries"):
        emb.encode_documents(["x"])
    assert len(t.calls) == 4 and len(sleeps) == 3


def test_a_client_error_is_not_retried(tmp_path):
    t = FakeTransport(script=[(400, {"error": {"message": "bad request"}})])
    emb, sleeps = make(tmp_path, t)
    with pytest.raises(GeminiError, match="HTTP 400"):
        emb.encode_documents(["x"])
    assert len(t.calls) == 1 and sleeps == []


def test_the_key_goes_only_in_the_header_and_never_into_messages_or_stats(tmp_path):
    t = FakeTransport(script=[(400, {"error": {"message": f"API key {SECRET} is invalid"}})])
    emb, _ = make(tmp_path, t)
    with pytest.raises(GeminiError) as exc:
        emb.encode_documents(["x"])
    assert SECRET not in str(exc.value) and "[redacted]" in str(exc.value)
    emb.encode_documents(["y"])
    for call in t.calls:
        assert call["headers"]["x-goog-api-key"] == SECRET and SECRET not in call["url"] and SECRET not in json.dumps(call["body"])
    for dump in (json.dumps(emb.stats()), json.dumps(emb.describe()), repr(emb.__dict__.keys()), emb.name, emb.revision):
        assert SECRET not in dump
    assert SECRET not in repr(vars(emb)).replace("environ", "")  # not stored on the object


def test_a_missing_key_fails_before_any_request(tmp_path, monkeypatch):
    monkeypatch.delenv(KEY_ENV)
    t = FakeTransport()
    emb, _ = make(tmp_path, t)
    with pytest.raises(GeminiError, match=f"{KEY_ENV} is not set"):
        emb.encode_query("x")
    assert t.calls == []
    assert emb.encode_documents([]).shape == (0, DIMS)  # nothing to embed needs no key


def test_a_response_with_the_wrong_shape_is_an_error_not_silently_accepted(tmp_path):
    emb, _ = make(tmp_path, FakeTransport(dims=DIMS + 1))
    with pytest.raises(GeminiError, match="unexpected response"):
        emb.encode_documents(["x"])
    assert emb.cache.count() == 0


def test_construction_is_import_safe_and_opens_no_cache_file(tmp_path):
    path = tmp_path / "never" / "emb.sqlite3"
    GeminiEmbedder(cache_path=path)
    assert not path.exists()


def test_the_cache_refuses_a_onedrive_path(tmp_path):
    with pytest.raises(RuntimeError, match="OneDrive"):
        EmbeddingCache(tmp_path / "OneDrive - Corp" / "emb.sqlite3")
    assert "onedrive" not in str(default_cache_path()).lower()


def test_discovery_lists_only_embedding_models_and_counts_tokens(tmp_path):
    emb, _ = make(tmp_path)
    assert emb.list_embedding_models() == [{"id": "gemini-embedding-001", "version": "001", "displayName": "Gemini Embedding",
                                            "inputTokenLimit": 2048, "methods": ["embedContent", "batchEmbedContents"]}]
    assert emb.remote_token_count(["two words", "three little words"]) == 5


def test_latency_of_network_calls_is_recorded_separately_from_cache_hits(tmp_path):
    ticks = iter(np.arange(0, 100, 0.25))
    emb, _ = make(tmp_path, clock=lambda: float(next(ticks)))
    emb.encode_query("q")
    emb.encode_query("q")  # cache hit: no network sample
    lat = emb.stats()["network_latency_ms"]
    assert lat["n"] == 1 and lat["p50"] == 250.0


# ---- the asymmetric API of the other embedders ------------------------------------------------------------------

def test_symmetric_embedders_implement_both_entry_points_as_the_same_call():
    e = HashEmbedder()
    assert np.allclose(e.encode_query("hello world"), e.encode("hello world"))
    assert np.allclose(e.encode_documents(["hello world", "bye"]), e.encode(["hello world", "bye"]))


def test_helpers_fall_back_to_encode_for_embedders_without_the_asymmetric_api():
    class Old:
        def encode(self, texts, show_progress_bar=False, **kw):
            return np.ones(3) if isinstance(texts, str) else np.ones((len(texts), 3))

    assert embed_query(Old(), "q").shape == (3,) and embed_documents(Old(), ["a", "b"]).shape == (2, 3)
