"""rag_core.scores (feedback) and rag_core.observability_reader (Metrics API v2 read side). Scripted HTTP, fake clock, FakeLangfuse: no network, no keys."""

import json
from datetime import datetime, timezone

import pytest

from rag_core.observability_reader import DAILY_QUERY_BUDGET, ERROR_TTL_S, ROW_LIMIT, TTL_S, ObservabilityReader, Panel, sample_data
from rag_core.scores import (ANSWER_F1, COMMENT_MAX_CHARS, JUDGE_AGREEMENT, RAGAS_ANSWER_RELEVANCY, RAGAS_FAITHFULNESS, SCORES, USER_FEEDBACK, submit_feedback,
                             submit_score, validate_score)
from tests.unit.rag_core_fakes import FakeClock, FakeHttp, FakeLangfuse

ENV = {"LANGFUSE_PUBLIC_KEY": "pk-lf-test", "LANGFUSE_SECRET_KEY": "sk-lf-test", "LANGFUSE_BASE_URL": "https://lf.example.test/"}
PANELS = ["request_volume_over_time", "latency_percentiles_by_stage", "time_to_first_token", "token_usage", "cost", "error_and_degraded_rates", "cache_hit_rate",
          "feedback_summary", "prompt_version_breakdown"]


# ---- scores ----------------------------------------------------------------------------------------------------------------------

def test_the_score_schema_is_fixed_up_front_with_one_written_score_and_the_rest_reserved():
    assert set(SCORES) == {USER_FEEDBACK, RAGAS_FAITHFULNESS, RAGAS_ANSWER_RELEVANCY, ANSWER_F1, JUDGE_AGREEMENT}
    assert [n for n, s in SCORES.items() if not s.reserved] == [USER_FEEDBACK] and SCORES[USER_FEEDBACK].data_type == "BOOLEAN"
    assert all(s.data_type == "NUMERIC" and (s.minimum, s.maximum) == (0, 1) for n, s in SCORES.items() if n != USER_FEEDBACK)


@pytest.mark.parametrize("name,value,expected", [(USER_FEEDBACK, 1, (1.0, "BOOLEAN")), (USER_FEEDBACK, 0, (0.0, "BOOLEAN")), (USER_FEEDBACK, True, (1.0, "BOOLEAN")),
                                                  (RAGAS_FAITHFULNESS, 0.83, (0.83, "NUMERIC")), (ANSWER_F1, 0, (0.0, "NUMERIC")), (ANSWER_F1, 1, (1.0, "NUMERIC"))])
def test_validate_score_accepts_in_range_values(name, value, expected):
    assert validate_score(name, value) == expected


@pytest.mark.parametrize("name,value", [("nope", 1), (USER_FEEDBACK, 2), (USER_FEEDBACK, 0.5), (USER_FEEDBACK, -1), (RAGAS_FAITHFULNESS, 1.01), (RAGAS_FAITHFULNESS, -0.01),
                                        (ANSWER_F1, "0.5"), (ANSWER_F1, None), (ANSWER_F1, float("nan")), (ANSWER_F1, [1])])
def test_validate_score_rejects_unknown_names_non_numbers_and_out_of_range_values(name, value):
    with pytest.raises(ValueError):
        validate_score(name, value)


def test_feedback_is_sent_with_a_deterministic_score_id_so_a_changed_mind_updates_instead_of_adding_a_unit():
    client = FakeLangfuse()
    assert submit_feedback("trace-1", 1, client_factory=lambda: client) is True
    assert submit_feedback("trace-1", False, client_factory=lambda: client) is True
    assert [s["score_id"] for s in client.scores] == ["trace-1-user_feedback"] * 2
    assert client.scores[0] == {"name": "user_feedback", "value": 1.0, "trace_id": "trace-1", "data_type": "BOOLEAN", "score_id": "trace-1-user_feedback"}
    assert client.scores[1]["value"] == 0.0


def test_the_comment_is_masked_and_truncated_before_it_leaves_the_process():
    client = FakeLangfuse()
    submit_feedback("t", 0, "wrong, ask bob@corp.example.com key gsk_abcdefghijklmnop1234 " + "word " * 300, client_factory=lambda: client)
    comment = client.scores[0]["comment"]
    assert "bob@" not in comment and "gsk_" not in comment and "[EMAIL]" in comment and len(comment) <= COMMENT_MAX_CHARS + 1


@pytest.mark.parametrize("trace_id", [None, ""])
def test_no_trace_id_means_the_request_was_not_traced_and_nothing_is_sent(trace_id):
    client = FakeLangfuse()
    assert submit_feedback(trace_id, 1, client_factory=lambda: client) is False and client.scores == []


def test_invalid_scores_missing_clients_and_sdk_failures_return_false_and_never_raise():
    client = FakeLangfuse()
    assert submit_score("t", "unknown", 1, client_factory=lambda: client) is False
    assert submit_score("t", ANSWER_F1, 3, client_factory=lambda: client) is False
    assert submit_feedback("t", 1, client_factory=lambda: None) is False and client.scores == []

    class Boom(FakeLangfuse):
        def create_score(self, **kw):
            raise RuntimeError("network")

    assert submit_feedback("t", 1, client_factory=Boom) is False

    def broken_factory():
        raise RuntimeError("cannot build")

    assert submit_feedback("t", 1, client_factory=broken_factory) is False


def test_numeric_reserved_scores_can_be_submitted_for_the_offline_phase():
    client = FakeLangfuse()
    assert submit_score("t", RAGAS_FAITHFULNESS, 0.5, client_factory=lambda: client) and client.scores[0]["data_type"] == "NUMERIC"


# ---- reader ----------------------------------------------------------------------------------------------------------------------

def ok_rows(query):
    """Plausible Metrics API rows for each query the reader issues (field names as documented)."""
    view, measures = query["view"], [m["measure"] for m in query["metrics"]]
    dims = [d["field"] for d in query.get("dimensions", [])]
    if view == "scores-boolean":
        return [{"name": "user_feedback", "count_count": 4, "avg_value": 0.75}]
    if "timeDimension" in query:
        return [{"time_dimension": "2026-11-20T00:00:00Z", "count_count": 3}, {"time_dimension": "2026-11-21T00:00:00Z", "count_count": 5}]
    if "latency" in measures and dims == ["name"]:
        return [{"name": "generation", "p50_latency": 800, "p95_latency": 1700, "p99_latency": 2600}, {"name": "retrieve", "p50_latency": 90, "p95_latency": 200, "p99_latency": 250}]
    if "latency" in measures:
        return [{"promptName": "rag-answer-long", "promptVersion": 2, "count_count": 6, "p95_latency": 1900}, {"promptName": None, "promptVersion": None, "count_count": 1, "p95_latency": 100}]
    if "timeToFirstToken" in measures:
        return [{"p50_timeToFirstToken": 400, "p95_timeToFirstToken": 900, "count_count": 7}]
    if "inputTokens" in measures:
        return [{"providedModelName": "m", "sum_inputTokens": 100, "sum_outputTokens": 40, "sum_totalTokens": 140}, {"providedModelName": None, "sum_totalTokens": 5}]
    if "totalCost" in measures:
        return [{"providedModelName": "m", "sum_totalCost": 0.25}]
    if dims == ["level"]:
        return [{"level": "DEFAULT", "count_count": 7}, {"level": "WARNING", "count_count": 1}, {"level": "ERROR", "count_count": 2}]
    if dims == ["name"]:
        return [{"name": "cache.hit", "count_count": 3}, {"name": "cache.miss", "count_count": 9}]
    raise AssertionError(f"unexpected query {query}")


def reader(responder=None, env=ENV, clock=None, **kw):
    clock = clock or FakeClock()
    if clock.t == 0.0:
        clock.t = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc).timestamp()
    http = FakeHttp(responder or (lambda q: (200, {"data": ok_rows(q)})))
    return ObservabilityReader(http_get=http, env=env, clock=clock, **kw), http, clock


def shape(value):
    if isinstance(value, dict):
        return {k: shape(v) for k, v in value.items()}
    if isinstance(value, list):
        return [shape(value[0])] if value else []
    return "scalar"


def test_every_panel_is_ok_with_the_documented_shape_and_values():
    r, http, _ = reader()
    snap = r.snapshot()
    assert set(snap) == set(PANELS) and all(p.status == "ok" and p.message is None and p.sample is False and p.as_of for p in snap.values())
    assert snap["request_volume_over_time"].data == {"granularity": "day", "points": [{"time": "2026-11-20T00:00:00Z", "requests": 3}, {"time": "2026-11-21T00:00:00Z", "requests": 5}]}
    assert snap["latency_percentiles_by_stage"].data["stages"][0] == {"stage": "generation", "p50_ms": 800.0, "p95_ms": 1700.0, "p99_ms": 2600.0}
    assert snap["time_to_first_token"].data == {"p50_ms": 400.0, "p95_ms": 900.0, "generations": 7}
    assert snap["token_usage"].data == {"models": [{"model": "m", "input_tokens": 100, "output_tokens": 40, "total_tokens": 140}], "total_tokens": 140}  # rows without a model are dropped
    assert snap["cost"].data["models"] == [{"model": "m", "cost_usd": 0.25}] and snap["cost"].data["total_cost_usd"] == 0.25
    assert snap["error_and_degraded_rates"].data == {"requests": 10, "errors": 2, "degraded": 1, "error_rate": 0.2, "degraded_rate": 0.1}
    assert snap["cache_hit_rate"].data == {"hits": 3, "misses": 9, "lookups": 12, "hit_rate": 0.25}
    assert snap["feedback_summary"].data == {"count": 4, "positive_rate": 0.75, "positive": 3, "negative": 1}
    assert snap["prompt_version_breakdown"].data["prompts"][1]["name"] == "(local template: not linked)" and snap["prompt_version_breakdown"].data["prompts"][0]["version"] == 2


def test_the_sample_data_has_exactly_the_shape_of_real_data_for_every_panel():
    real = reader()[0].snapshot()
    sample = sample_data()
    assert set(sample) == set(real)
    for name in PANELS:
        assert sample[name].sample is True and sample[name].status == "ok" and "SAMPLE" in sample[name].message
        assert shape(sample[name].data) == shape(real[name].data), name


def test_no_data_returns_the_explicit_empty_structure_for_every_panel_and_the_empty_shape_matches():
    empty, _, _ = reader(lambda q: (200, {"data": []}))
    real = reader()[0].snapshot()
    for name, panel in empty.snapshot().items():
        assert panel.status == "no_data" and panel.message and panel.data is not None, name
        assert set(panel.data) == set(real[name].data), name  # same keys as real data, so the UI needs no None checks


def test_no_keys_means_no_request_at_all():
    for env in ({}, {"LANGFUSE_PUBLIC_KEY": "a"}, {"LANGFUSE_SECRET_KEY": "b"}, {"LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": ""}):
        r, http, _ = reader(env=env)
        snap = r.snapshot()
        assert all(p.status == "no_keys" and p.data is not None for p in snap.values()) and http.calls == []


@pytest.mark.parametrize("response,fragment", [((429, {"message": "slow down"}), "rate limited"), ((401, None), "credentials"), ((403, {}), "credentials"),
                                               ((500, {"message": "boom"}), "HTTP 500: boom"), ((400, None), "HTTP 400"), ((404, "not json object"), "HTTP 404"),
                                               ((200, None), "HTTP 200"), ((200, "text"), "HTTP 200"), ((200, []), "HTTP 200")])
def test_http_failures_are_statuses_not_exceptions(response, fragment):
    r, _, _ = reader(lambda q: response)
    for name in PANELS:
        panel = getattr(r, name)()
        assert panel.status == "unavailable" and fragment in panel.message and panel.data is not None, (name, panel)


def test_a_network_exception_is_a_status_and_never_leaks_its_message():
    r, _, _ = reader(lambda q: RuntimeError("connect to secret-host failed"))
    panel = r.cost()
    assert panel.status == "unavailable" and panel.message == "request failed (RuntimeError)" and "secret-host" not in panel.message


@pytest.mark.parametrize("body", [{"data": [1, 2]}, {"data": ["x"]}, {"data": [{"count_count": "NaN-ish"}]}, {"data": "abc"}, {"data": [[1]]}])
def test_an_unexpected_response_shape_is_unavailable_and_never_raises(body):
    r, _, _ = reader(lambda q: (200, body))
    for name in PANELS:
        panel = getattr(r, name)()
        assert isinstance(panel, Panel) and panel.status in {"ok", "no_data", "unavailable"} and panel.data is not None


def test_errors_are_root_observations_at_level_error_and_degraded_ones_at_level_warning():
    rows = [{"level": "DEFAULT", "count_count": 6}, {"level": "WARNING", "count_count": 3}, {"level": "ERROR", "count_count": 1}, {"level": "DEBUG", "count_count": 0}]
    p = reader(lambda q: (200, {"data": rows}))[0].error_and_degraded_rates()
    assert p.status == "ok" and p.data == {"requests": 10, "errors": 1, "degraded": 3, "error_rate": 0.1, "degraded_rate": 0.3}
    only_ok = reader(lambda q: (200, {"data": [{"level": "DEFAULT", "count_count": 4}]}))[0].error_and_degraded_rates()
    assert only_ok.data["errors"] == 0 and only_ok.data["degraded"] == 0 and only_ok.data["error_rate"] == 0.0


# What the live Metrics API v2 reported as valid dimensions for the observations view (its HTTP 400 message lists them). The reader once used a
# column that is not in this list (statusMessage) and silently showed zero degraded requests.
VALID_OBSERVATION_COLUMNS = {"id", "evaluatorId", "isEvaluatorTest", "traceId", "environment", "parentObservationId", "isRootObservation", "type", "name", "level",
                             "version", "userId", "sessionId", "tags", "release", "traceName", "traceRelease", "traceVersion", "providedModelName", "promptName",
                             "promptVersion", "startTimeMonth", "toolNames", "calledToolNames", "costType", "usageType"}


def test_every_dimension_and_filter_column_the_reader_sends_is_one_the_live_api_accepts():
    r, http, _ = reader()
    r.snapshot()
    for call in http.calls:
        q = call["query"]
        if q["view"] != "observations":
            continue
        used = {d["field"] for d in q.get("dimensions", [])} | {f["column"] for f in q.get("filters", [])}
        assert used <= VALID_OBSERVATION_COLUMNS, (q, used - VALID_OBSERVATION_COLUMNS)


def test_requests_use_basic_auth_the_configured_base_url_and_a_bounded_row_limit():
    r, http, _ = reader()
    r.snapshot()
    assert http.calls and all(c["auth"] == ("pk-lf-test", "sk-lf-test") and c["url"] == "https://lf.example.test/api/public/v2/metrics" for c in http.calls)
    assert all(c["query"]["config"]["row_limit"] <= ROW_LIMIT for c in http.calls) and all(c["timeout"] == 20.0 for c in http.calls)
    for c in http.calls:  # a seven-day window ending now
        start, end = datetime.fromisoformat(c["query"]["fromTimestamp"]), datetime.fromisoformat(c["query"]["toTimestamp"])
        assert (end - start).days == 7


def test_the_base_url_falls_back_to_the_legacy_host_variable_then_to_the_cloud_default():
    env = {"LANGFUSE_PUBLIC_KEY": "a", "LANGFUSE_SECRET_KEY": "b", "LANGFUSE_HOST": "https://legacy.example.test"}
    r, http, _ = reader(env=env)
    r.cost()
    assert http.calls[0]["url"] == "https://legacy.example.test/api/public/v2/metrics"
    r, http, _ = reader(env={"LANGFUSE_PUBLIC_KEY": "a", "LANGFUSE_SECRET_KEY": "b"})
    r.cost()
    assert http.calls[0]["url"] == "https://cloud.langfuse.com/api/public/v2/metrics"


def test_volume_counts_root_observations_only_and_the_other_panels_use_the_documented_views():
    r, http, _ = reader()
    r.request_volume_over_time(days=3, granularity="hour")
    q = http.calls[0]["query"]
    assert q["filters"] == [ObservabilityReader.ROOT] and q["timeDimension"] == {"granularity": "hour"} and ObservabilityReader.ROOT["column"] == "isRootObservation"
    assert (datetime.fromisoformat(q["toTimestamp"]) - datetime.fromisoformat(q["fromTimestamp"])).days == 3
    r.feedback_summary()
    assert http.calls[1]["query"]["view"] == "scores-boolean"


# ---- TTL, error TTL, budget ------------------------------------------------------------------------------------------------------

def test_a_second_identical_read_inside_the_ttl_is_served_from_the_cache_and_expires_after_it():
    r, http, clock = reader()
    first = r.cost()
    clock.advance(TTL_S - 1)
    second = r.cost()
    assert len(http.calls) == 1 and second.data == first.data
    clock.advance(2)
    r.cost()
    assert len(http.calls) == 2


def test_failures_are_cached_briefly_so_a_down_api_is_not_hammered_and_then_retried():
    state = {"up": False}
    r, http, clock = reader(lambda q: (200, {"data": ok_rows(q)}) if state["up"] else (500, None))
    assert r.cost().status == "unavailable" and r.cost().status == "unavailable" and len(http.calls) == 1
    state["up"] = True
    assert r.cost().status == "unavailable" and len(http.calls) == 1  # still inside the error window
    clock.advance(ERROR_TTL_S + 1)
    assert r.cost().status == "ok" and len(http.calls) == 2


def test_different_windows_are_different_cache_entries():
    r, http, _ = reader()
    r.cost(days=7)
    r.cost(days=1)
    assert len(http.calls) == 2


def test_the_local_daily_budget_stops_queries_before_the_plan_limit_and_resets_the_next_utc_day():
    assert DAILY_QUERY_BUDGET < 100  # the free plan's Metrics API allowance
    r, http, clock = reader(daily_budget=2)
    clock.t = datetime(2026, 11, 22, 23, 30, tzinfo=timezone.utc).timestamp()
    assert r.cost(days=1).status == "ok" and r.cost(days=2).status == "ok"
    over = r.cost(days=3)
    assert over.status == "unavailable" and "daily query budget" in over.message and len(http.calls) == 2 and r.queries_sent == 2
    assert r.cost(days=1).status == "ok"  # cached results stay available when the budget is gone
    clock.advance(3600)  # past midnight UTC
    assert r.cost(days=3).status == "ok" and len(http.calls) == 3


def test_a_full_snapshot_stays_far_below_the_daily_budget():
    r, http, _ = reader()
    r.snapshot()
    r.snapshot()  # second call is entirely cached
    assert len(http.calls) == 9 and len(http.calls) < DAILY_QUERY_BUDGET // 4  # one query per panel; the second snapshot is entirely cached


def test_the_reader_never_raises_even_when_the_clock_or_the_http_layer_misbehave():
    def bad_http(url, params, auth, timeout_s):
        raise OSError("boom")

    r = ObservabilityReader(http_get=bad_http, env=ENV)
    assert all(p.status == "unavailable" for p in r.snapshot().values())
