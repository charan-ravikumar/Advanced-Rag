"""The HTTP API (api/): streaming, validation, per-key rate limiting (slowapi), the daily budget, authentication, readiness and the OpenAPI document.

A real ``RagService`` with a fake retriever and scripted providers behind the real FastAPI app (``TestClient``, in-process: no socket, no network, no key).
"""

import json

import pytest
from fastapi.testclient import TestClient

from api import limits
from api.main import create_app
from api.sse import decode_service_event, iter_sse
from app.bootstrap import Check, Resources, SelfCheck
from app.guards import AppConfig, DailyBudget
from rag_core.scores import FeedbackResult
from tests.unit.rag_core_fakes import make_hit, ok
from tests.unit.test_rag_core_service import Q, make


def make_resources(service, *, daily_cap=1000, rate=1000, per_day=100000, ready=True, feedback=None):
    check = SelfCheck([Check("Index directory", ready, "set and found" if ready else "not set")])
    return Resources(settings=AppConfig(rate_limit_requests=rate, rate_limit_window_seconds=60.0, session_request_cap=per_day), selfcheck=check,
                     service=service if ready else None, daily=DailyBudget(daily_cap), version_info={"chunk_count": 3, "index_dir": "C:\\private\\index"},
                     record_feedback=feedback or (lambda request_id, trace_id, value, comment=None: FeedbackResult(True, False)))


@pytest.fixture
def api():
    """``build(service, **resources_kwargs) -> TestClient`` (the lifespan has run); every app starts with empty rate-limit counters."""
    clients = []

    def build(service=None, *, token="", **kw):
        limits.limiter.reset()
        service = service or make()[0]
        client = TestClient(create_app(lambda: make_resources(service, **kw), token=token))
        client.__enter__()
        clients.append(client)
        return client

    yield build
    for client in clients:
        client.__exit__(None, None, None)
    limits.limiter.reset()


def stream_events(client, body, headers=None):
    with client.stream("POST", "/v1/ask", json=body, headers=headers) as response:
        assert response.status_code == 200, response.read()
        assert response.headers["content-type"].startswith("text/event-stream")
        return [e for e in (decode_service_event(n, d) for n, d in iter_sse(response.iter_lines())) if e is not None]


def code_of(response):
    return response.json()["error"]["code"]


# ---- the happy paths ---------------------------------------------------------------------------------------------------------------------

def test_ask_streams_sources_then_tokens_then_exactly_one_done(api):
    client = api(make([ok("Attention weighs ", "the tokens [S1].")])[0])
    events = stream_events(client, {"question": Q})
    kinds = [e["type"] for e in events]
    assert kinds[0] == "sources" and kinds[-1] == "done" and kinds.count("done") == 1 and set(kinds[1:-1]) == {"token"}
    result = events[-1]["result"]
    assert "".join(e["text"] for e in events if e["type"] == "token") == result.answer == "Attention weighs the tokens [S1]."
    assert result.citations == ["S1"] and [s.label for s in events[0]["sources"]][:2] == ["S1", "S2"]


def test_answer_returns_the_same_result_as_one_json_body(api):
    client = api(make([ok("Because [S2].")])[0])
    response = client.post("/v1/answer", json={"question": Q})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Because [S2]." and body["citations"] == ["S2"] and body["request_id"] and body["degraded"] is False


def test_retrieve_returns_passages_and_never_calls_the_model(api):
    svc, retriever, provider, *_ = make()
    client = api(svc)
    response = client.post("/v1/retrieve", json={"question": Q, "doc_ids": ["1234.00001"], "rerank": True})
    assert response.status_code == 200 and [s["label"] for s in response.json()["sources"]] == ["S1", "S2", "S3", "S4", "S5"]
    assert provider.calls == [] and retriever.calls[-1]["doc_ids"] == ["1234.00001"] and retriever.calls[-1]["rerank"] is True


def test_feedback_is_recorded_and_the_meta_never_shows_the_server_path(api):
    client = api()
    assert client.post("/v1/feedback", json={"request_id": "abc123", "value": 1, "comment": "good"}).json() == {"saved": True, "forwarded": False}
    meta = client.get("/v1/meta").json()
    assert meta["version_info"] == {"chunk_count": 3} and meta["limits"]["max_question_chars"] == 500 and meta["budget"]["remaining"] == 1000


# ---- input validation --------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", ["/v1/ask", "/v1/answer", "/v1/retrieve"])
def test_every_query_endpoint_rejects_unknown_fields_wrong_types_and_non_objects(api, path):
    client = api()
    for body in ({"question": Q, "model": "gpt-9"}, {"question": 5}, {"question": Q, "rerank": "maybe"}, {}, [Q], "just text"):
        response = client.post(path, json=body)
        assert response.status_code == 422 and code_of(response) == "invalid_request", body
    assert client.post(path, content=b"{not json", headers={"Content-Type": "application/json"}).status_code == 422


def test_question_rules_come_back_as_typed_422_codes(api):
    client = api()
    assert code_of(client.post("/v1/answer", json={"question": "   \n\t "})) == "empty"
    assert code_of(client.post("/v1/answer", json={"question": "x" * 501})) == "too_long"
    assert code_of(client.post("/v1/answer", json={"question": "x" * 5001})) == "invalid_request"  # over the payload guard: refused by the schema
    assert client.post("/v1/answer", json={"question": "x" * 500}).status_code == 200


def test_control_and_invisible_characters_are_removed_before_retrieval(api):
    svc, retriever, *_ = make()
    api(svc).post("/v1/retrieve", json={"question": "What\x00 is\u202e the\u200b loss?"})
    assert retriever.calls[-1]["query"] == "What is the loss?"


@pytest.mark.parametrize("doc_ids", [["../etc/passwd"], ["a b"], ["1234.0001\"; DROP TABLE x"], ["x" * 65], [""], ["a/b"], ["id\n2"], ["1234"] * 51, [5]])
def test_doc_ids_must_look_like_paper_ids(api, doc_ids):
    response = api().post("/v1/retrieve", json={"question": Q, "doc_ids": doc_ids})
    assert response.status_code == 422 and code_of(response) == "invalid_request"


def test_duplicate_doc_ids_are_collapsed(api):
    svc, retriever, *_ = make()
    api(svc).post("/v1/retrieve", json={"question": Q, "doc_ids": ["1601.02403", "1601.02403", "1904.07094"]})
    assert retriever.calls[-1]["doc_ids"] == ["1601.02403", "1904.07094"]


@pytest.mark.parametrize("bad", ["has space", "semi;colon", "a" * 65, "", "naïve", "../x"])
def test_session_ids_and_feedback_ids_are_opaque_tokens_only(api, bad):
    client = api()
    assert client.post("/v1/retrieve", json={"question": Q, "session_id": bad}).status_code == 422
    assert client.post("/v1/feedback", json={"request_id": bad, "value": 1}).status_code == 422
    assert client.post("/v1/feedback", json={"request_id": "ok1", "trace_id": bad, "value": 1}).status_code == 422


def test_feedback_value_comment_and_days_are_bounded(api):
    client = api()
    for body in ({"request_id": "r1", "value": 2}, {"request_id": "r1", "value": -1}, {"request_id": "r1", "value": 1, "comment": "x" * 301}, {"request_id": "r1", "value": "up"}):
        assert client.post("/v1/feedback", json=body).status_code == 422, body
    assert client.get("/v1/observability", params={"days": 0}).status_code == 422
    assert client.get("/v1/observability", params={"days": 91}).status_code == 422


def test_an_oversized_body_is_refused_before_it_is_read_and_a_missing_length_is_411(api):
    client = api()
    big = client.post("/v1/ask", content=b'{"question": "' + b"x" * 20000 + b'"}', headers={"Content-Type": "application/json"})
    assert big.status_code == 413 and code_of(big) == "payload_too_large"
    chunked = client.post("/v1/ask", content=iter([b'{"question": "hi"}']), headers={"Content-Type": "application/json"})
    assert chunked.status_code == 411 and code_of(chunked) == "length_required"


# ---- per-key rate limiting (slowapi) -----------------------------------------------------------------------------------------------------

def test_the_request_over_the_limit_gets_429_with_retry_after_and_the_error_shape(api):
    client = api(rate=3)
    assert [client.post("/v1/retrieve", json={"question": Q}).status_code for _ in range(3)] == [200, 200, 200]
    blocked = client.post("/v1/retrieve", json={"question": Q})
    assert blocked.status_code == 429 and code_of(blocked) == "rate_limited" and int(blocked.headers["Retry-After"]) == 60
    assert set(blocked.json()) == {"error"} and "wait" in blocked.json()["error"]["message"]


def test_ask_answer_and_retrieve_share_one_bucket_per_key(api):
    client = api(rate=3)
    h = {"X-Session-Id": "alice"}
    assert client.post("/v1/retrieve", json={"question": Q}, headers=h).status_code == 200
    assert client.post("/v1/answer", json={"question": Q}, headers=h).status_code == 200
    assert len(stream_events(client, {"question": "Another question about dropout?"}, headers=h)) > 0
    assert client.post("/v1/retrieve", json={"question": Q}, headers=h).status_code == 429


def test_each_session_key_has_its_own_allowance(api):
    client = api(rate=2)
    a, b = {"X-Session-Id": "alice"}, {"X-Session-Id": "bob"}
    for _ in range(2):
        assert client.post("/v1/retrieve", json={"question": Q}, headers=a).status_code == 200
    assert client.post("/v1/retrieve", json={"question": Q}, headers=a).status_code == 429
    assert client.post("/v1/retrieve", json={"question": Q}, headers=b).status_code == 200  # bob is not punished for alice


def test_a_malformed_session_header_is_not_trusted_it_falls_back_to_the_client_address(api):
    client = api(rate=2)
    for header in ("not valid!", "../../etc", "x" * 65):  # rotating junk values must NOT mint fresh allowances
        client.post("/v1/retrieve", json={"question": Q}, headers={"X-Session-Id": header})
    assert client.post("/v1/retrieve", json={"question": Q}, headers={"X-Session-Id": "another junk!"}).status_code == 429
    assert client.post("/v1/retrieve", json={"question": Q}).status_code == 429  # no header at all: the same address bucket


def test_a_daily_cap_per_key_is_reported_as_session_cap(api):
    client = api(rate=100, per_day=2)
    h = {"X-Session-Id": "carol"}
    assert [client.post("/v1/retrieve", json={"question": Q}, headers=h).status_code for _ in range(2)] == [200, 200]
    blocked = client.post("/v1/retrieve", json={"question": Q}, headers=h)
    assert blocked.status_code == 429 and code_of(blocked) == "session_cap"


def test_feedback_has_its_own_bucket_and_invalid_or_unauthenticated_requests_do_not_spend_the_allowance(api):
    client = api(rate=2, token="s3cret")
    auth = {"Authorization": "Bearer s3cret"}
    for _ in range(5):
        assert client.post("/v1/retrieve", json={"question": Q}).status_code == 401  # refused before the limiter
    assert [client.post("/v1/retrieve", json={"question": Q, "extra": 1}, headers=auth).status_code for _ in range(5)] == [422] * 5  # schema errors never reach the limiter
    assert [client.post("/v1/retrieve", json={"question": Q}, headers=auth).status_code for _ in range(2)] == [200, 200]
    assert client.post("/v1/retrieve", json={"question": Q}, headers=auth).status_code == 429
    assert client.post("/v1/feedback", json={"request_id": "r1", "value": 1}, headers=auth).status_code == 200


# ---- the daily budget --------------------------------------------------------------------------------------------------------------------

def test_over_the_daily_budget_the_answer_is_the_passages_with_a_notice_and_no_model_call(api):
    svc, _, provider, *_ = make()
    client = api(svc, daily_cap=1)
    assert client.post("/v1/answer", json={"question": Q}).json()["degraded"] is False and len(provider.calls) == 1
    events = stream_events(client, {"question": "A different question about regularisation?"})
    assert [e["type"] for e in events] == ["error", "done"] and events[0]["error"].code == "daily_budget_reached"
    assert len(events[-1]["result"].retrieved_sources) == 5 and len(provider.calls) == 1  # nothing was sent to the model
    assert client.post("/v1/answer", json={"question": "Yet another question about attention heads?"}).json()["errors"][0]["code"] == "daily_budget_reached"


def test_a_cache_hit_gives_its_unit_of_the_budget_back(api):
    svc, _, provider, *_ = make()
    client = api(svc, daily_cap=2)
    assert client.post("/v1/answer", json={"question": Q}).status_code == 200
    again = client.post("/v1/answer", json={"question": Q}).json()
    assert again["cached"] is True and again["degraded"] is False and len(provider.calls) == 1
    assert client.get("/v1/meta").json()["budget"] == {"cap": 2, "used": 1, "remaining": 1}  # the cache hit took a unit and gave it back


# ---- authentication and readiness --------------------------------------------------------------------------------------------------------

def test_with_a_token_everything_but_health_needs_the_bearer_header(api):
    client = api(token="s3cret")
    assert client.get("/v1/health").status_code == 200
    for method, path in (("get", "/v1/ready"), ("get", "/v1/meta"), ("get", "/v1/observability")):
        denied = getattr(client, method)(path)
        assert denied.status_code == 401 and code_of(denied) == "unauthorized" and denied.headers["WWW-Authenticate"] == "Bearer"
    assert client.post("/v1/ask", json={"question": Q}, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/v1/ask", json={"question": Q}, headers={"Authorization": "Basic s3cret"}).status_code == 401
    assert client.get("/v1/meta", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_without_a_token_the_api_is_open(api):
    assert api(token="").get("/v1/meta").status_code == 200


def test_a_failed_self_check_gives_503_with_the_checks_and_health_stays_up(api):
    client = api(ready=False)
    ready = client.get("/v1/ready")
    assert ready.status_code == 503 and ready.json() == {"ready": False, "checks": [{"name": "Index directory", "ok": False, "detail": "not set", "fatal": True}]}
    assert client.get("/v1/health").status_code == 200
    for response in (client.post("/v1/ask", json={"question": Q}), client.post("/v1/answer", json={"question": Q}), client.get("/v1/meta")):
        assert response.status_code == 503 and code_of(response) == "not_ready"


def test_a_broken_resources_factory_does_not_stop_the_server():
    def explode():
        raise RuntimeError("boom with a secret path C:\\private")

    with TestClient(create_app(explode, token="")) as client:
        body = client.get("/v1/ready").json()
        assert client.get("/v1/ready").status_code == 503 and body["ready"] is False and "RuntimeError" in body["checks"][0]["detail"] and "private" not in json.dumps(body)


def test_an_unexpected_server_error_leaks_nothing(api):
    svc = make()[0]

    def explode(*args, **kwargs):
        raise RuntimeError("secret: C:\\private\\key.txt")

    svc.retrieve_sources = explode
    response = api(svc).post("/v1/retrieve", json={"question": Q})
    assert response.status_code == 503 and code_of(response) == "retrieval_failed" and "private" not in response.text and "RuntimeError" in response.text


# ---- the OpenAPI document ----------------------------------------------------------------------------------------------------------------

def test_the_openapi_document_lists_every_endpoint_the_sse_events_and_the_bearer_scheme(api):
    spec = api().get("/openapi.json").json()
    assert set(spec["paths"]) == {"/v1/ask", "/v1/answer", "/v1/retrieve", "/v1/feedback", "/v1/health", "/v1/ready", "/v1/meta", "/v1/observability"}
    assert {"SourcesEvent", "TokenEvent", "ErrorEvent", "DoneEvent", "AnswerResult", "ErrorResponse"} <= set(spec["components"]["schemas"])
    assert "text/event-stream" in spec["paths"]["/v1/ask"]["post"]["responses"]["200"]["content"] and "HTTPBearer" in spec["components"]["securitySchemes"]
    assert {"401", "422", "429", "503"} <= set(spec["paths"]["/v1/ask"]["post"]["responses"])
    assert api().get("/docs").status_code == 200
