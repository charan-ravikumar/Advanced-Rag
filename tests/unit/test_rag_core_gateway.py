"""rag_core.gateway: timeout, retry with backoff + jitter, circuit breaker, failover only before the first token, attempt records,
and the two thin SDK wrappers (with fake clients and the REAL SDK exception classes). No network, no real keys."""

from types import SimpleNamespace

import pytest

from rag_core.errors import AllProvidersFailed, MidStreamError, ProviderError
from rag_core.gateway import (
    CircuitBreaker,
    Delta,
    Final,
    GeminiProvider,
    LLMGateway,
    StreamDone,
    as_provider_error,
)
from tests.unit.rag_core_fakes import FakeClock, ScriptedProvider, fail, ok

MSGS = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


def gateway(*providers, clock=None, **kw):
    clock = clock or FakeClock()
    kw.setdefault("rng", lambda: 1.0)  # no jitter reduction unless a test asks for it
    return LLMGateway(providers, sleep=clock.sleep, clock=clock, **kw), clock


def run(gw):
    """(text, StreamDone) of one full stream."""
    items = list(gw.stream(MSGS))
    return "".join(i.text for i in items if isinstance(i, Delta)), items[-1]


# ---- the happy path and the per-call timeout ----------------------------------------------------------------------------

def test_a_successful_call_streams_deltas_then_one_done_with_usage_and_an_attempt_record():
    p = ScriptedProvider("groq", "m1", [ok("Hel", "lo", prompt_tokens=5, completion_tokens=2)])
    gw, _ = gateway(p)
    items = list(gw.stream(MSGS))
    assert [type(i) for i in items] == [Delta, Delta, StreamDone]
    done = items[-1]
    assert (done.provider, done.model, done.finish_reason, done.prompt_tokens, done.completion_tokens) == ("groq", "m1", "stop", 5, 2)
    (a,) = done.attempts
    assert (a.provider, a.model, a.attempt, a.outcome, a.error_class) == ("groq", "m1", 1, "ok", None) and a.latency_ms >= 0


def test_the_per_call_timeout_and_generation_limits_are_passed_to_the_provider():
    p = ScriptedProvider("groq", "m1", [ok("x")])
    gw, _ = gateway(p, timeout_s=12.5)
    list(gw.stream(MSGS, max_output_tokens=33, temperature=0.2))
    assert p.calls[0]["timeout_s"] == 12.5 and p.calls[0]["max_output_tokens"] == 33 and p.calls[0]["temperature"] == 0.2 and p.calls[0]["messages"] == MSGS


def test_a_timeout_is_retried_like_a_rate_limit():
    p = ScriptedProvider("groq", "m1", [[fail("timeout")], ok("done")])
    gw, clock = gateway(p)
    text, done = run(gw)
    assert text == "done" and len(p.calls) == 2 and len(clock.sleeps) == 1
    assert [a.outcome for a in done.attempts] == ["error", "ok"] and done.attempts[0].error_kind == "timeout"


# ---- retry and backoff ------------------------------------------------------------------------------------------------

def test_429_is_retried_with_exponential_backoff_then_succeeds():
    p = ScriptedProvider("groq", "m1", [[fail("rate_limit", 429)], [fail("server", 503)], ok("fine")])
    gw, clock = gateway(p, base_delay_s=1.0, max_delay_s=60.0)
    text, done = run(gw)
    assert text == "fine" and len(p.calls) == 3 and clock.sleeps == [1.0, 2.0]
    assert [(a.attempt, a.outcome, a.error_kind, a.status) for a in done.attempts] == [(1, "error", "rate_limit", 429), (2, "error", "server", 503), (3, "ok", None, None)]


def test_retries_are_bounded_and_then_the_next_provider_is_tried():
    a = ScriptedProvider("groq", "m1", [[fail("server", 500)]])
    b = ScriptedProvider("gemini", "m2", [ok("from b")])
    gw, clock = gateway(a, b, max_attempts=3)
    text, done = run(gw)
    assert len(a.calls) == 3 and len(clock.sleeps) == 2 and len(b.calls) == 1  # no sleep after the last attempt of a
    assert text == "from b" and done.provider == "gemini"
    assert [(x.provider, x.attempt, x.outcome) for x in done.attempts] == [("groq", 1, "error"), ("groq", 2, "error"), ("groq", 3, "error"), ("gemini", 1, "ok")]


@pytest.mark.parametrize("kind,status", [("auth", 401), ("auth", 403), ("bad_request", 400), ("bad_request", 422), ("not_found", 404)])
def test_4xx_errors_are_not_retried_on_the_same_provider(kind, status):
    a = ScriptedProvider("groq", "m1", [[fail(kind, status)]])
    b = ScriptedProvider("gemini", "m2", [ok("fallback")])
    gw, clock = gateway(a, b)
    text, done = run(gw)
    assert len(a.calls) == 1 and clock.sleeps == [] and text == "fallback"  # one attempt, no backoff, then failover
    assert done.attempts[0].error_kind == kind and done.attempts[0].status == status


def test_no_retry_and_no_failover_target_means_all_providers_failed_after_a_single_attempt():
    a = ScriptedProvider("groq", "m1", [[fail("auth", 401)]])
    gw, clock = gateway(a)
    with pytest.raises(AllProvidersFailed) as exc:
        list(gw.stream(MSGS))
    assert len(a.calls) == 1 and clock.sleeps == [] and len(exc.value.attempts) == 1


def test_a_server_retry_after_is_honoured_but_capped():
    p = ScriptedProvider("groq", "m1", [[fail("rate_limit", 429, retry_after=5.0)], [fail("rate_limit", 429, retry_after=500.0)], ok("x")])
    gw, clock = gateway(p, base_delay_s=0.5, max_delay_s=8.0)
    run(gw)
    assert clock.sleeps == [5.0, 8.0]


def test_backoff_has_exponential_growth_a_cap_and_jitter_between_half_and_full():
    gw, _ = gateway(ScriptedProvider("a", "m", [ok("x")]), base_delay_s=1.0, max_delay_s=10.0, jitter=0.5, rng=lambda: 1.0)
    assert [gw.backoff_s(n) for n in (1, 2, 3, 4, 5)] == [1.0, 2.0, 4.0, 8.0, 10.0]
    gw.__dict__["_rng"] = lambda: 0.0
    assert [gw.backoff_s(n) for n in (1, 2, 3)] == [0.5, 1.0, 2.0]
    gw.__dict__["_rng"] = lambda: 0.5
    assert gw.backoff_s(2) == pytest.approx(1.5)


def test_the_total_deadline_stops_new_attempts():
    p = ScriptedProvider("groq", "m1", [[fail("server", 500)]])
    gw, clock = gateway(p, max_attempts=10, base_delay_s=4.0, max_delay_s=100.0, deadline_s=10.0)
    with pytest.raises(AllProvidersFailed) as exc:
        list(gw.stream(MSGS))
    assert len(p.calls) < 10 and exc.value.attempts[-1].outcome == "skipped_deadline"


# ---- circuit breaker ----------------------------------------------------------------------------------------------------

def test_the_breaker_opens_after_n_consecutive_failed_calls_and_stops_calling_the_provider():
    a = ScriptedProvider("groq", "m1", [[fail("server", 500)]])
    b = ScriptedProvider("gemini", "m2", [ok("b")])
    gw, clock = gateway(a, b, max_attempts=1, breaker_threshold=3, breaker_cooldown_s=30.0)
    for _ in range(3):
        run(gw)
    assert len(a.calls) == 3 and gw.breakers["groq"].state == "open"
    text, done = run(gw)  # the 4th request does not even try groq
    assert len(a.calls) == 3 and text == "b"
    assert (done.attempts[0].provider, done.attempts[0].outcome, done.attempts[0].error_class) == ("groq", "skipped_circuit_open", "CircuitOpen")


def test_a_success_resets_the_consecutive_failure_count():
    a = ScriptedProvider("groq", "m1", [[fail("server", 500)], [fail("server", 500)], ok("fine"), [fail("server", 500)], [fail("server", 500)], ok("fine")])
    b = ScriptedProvider("gemini", "m2", [ok("b")])
    gw, _ = gateway(a, b, max_attempts=1, breaker_threshold=3)
    for _ in range(6):
        run(gw)
    assert gw.breakers["groq"].state == "closed" and len(a.calls) == 6


def test_half_open_after_the_cooldown_one_trial_call_then_closed_on_success():
    a = ScriptedProvider("groq", "m1", [[fail("server", 500)], [fail("server", 500)], ok("recovered")])
    b = ScriptedProvider("gemini", "m2", [ok("b")])
    gw, clock = gateway(a, b, max_attempts=1, breaker_threshold=2, breaker_cooldown_s=30.0)
    run(gw), run(gw)
    assert gw.breakers["groq"].state == "open"
    clock.advance(29.0)
    run(gw)
    assert len(a.calls) == 2  # still open
    clock.advance(2.0)
    assert gw.breakers["groq"].state == "half_open"
    text, done = run(gw)
    assert text == "recovered" and done.provider == "groq" and len(a.calls) == 3 and gw.breakers["groq"].state == "closed"


def test_a_failed_half_open_trial_re_opens_the_breaker_for_a_full_cooldown():
    a = ScriptedProvider("groq", "m1", [[fail("server", 500)]])
    b = ScriptedProvider("gemini", "m2", [ok("b")])
    gw, clock = gateway(a, b, max_attempts=1, breaker_threshold=2, breaker_cooldown_s=30.0)
    run(gw), run(gw)
    clock.advance(31.0)
    run(gw)  # trial fails
    assert len(a.calls) == 3 and gw.breakers["groq"].state == "open"
    clock.advance(29.0)
    run(gw)
    assert len(a.calls) == 3  # cooldown restarted at the failed trial
    clock.advance(2.0)
    run(gw)
    assert len(a.calls) == 4


def test_the_breaker_unit_allows_exactly_one_half_open_trial_at_a_time():
    clock = FakeClock()
    br = CircuitBreaker(threshold=1, cooldown_s=10.0, clock=clock)
    assert br.allow() and br.state == "closed"
    br.record_failure()
    assert br.state == "open" and not br.allow()
    clock.advance(10.0)
    assert br.state == "half_open" and br.allow() and not br.allow()  # the trial slot is taken
    br.release()
    assert br.allow()
    br.record_success()
    assert br.state == "closed" and br.allow() and br.allow()
    with pytest.raises(ValueError):
        CircuitBreaker(threshold=0)


def test_a_bad_request_is_about_the_request_not_the_provider_so_it_never_opens_the_breaker():
    a = ScriptedProvider("groq", "m1", [[fail("bad_request", 400)]])
    b = ScriptedProvider("gemini", "m2", [ok("b")])
    gw, _ = gateway(a, b, max_attempts=1, breaker_threshold=2)
    for _ in range(5):
        run(gw)
    assert len(a.calls) == 5 and gw.breakers["groq"].state == "closed"


def test_auth_failures_do_open_the_breaker():
    a = ScriptedProvider("groq", "m1", [[fail("auth", 401)]])
    b = ScriptedProvider("gemini", "m2", [ok("b")])
    gw, _ = gateway(a, b, breaker_threshold=2)
    run(gw), run(gw), run(gw)
    assert len(a.calls) == 2 and gw.breakers["groq"].state == "open"


def test_every_breaker_open_means_all_providers_failed_without_any_call():
    a, b = ScriptedProvider("a", "m1", [[fail("server", 500)]]), ScriptedProvider("b", "m2", [[fail("server", 500)]])
    gw, _ = gateway(a, b, max_attempts=1, breaker_threshold=1)
    with pytest.raises(AllProvidersFailed):
        list(gw.stream(MSGS))
    n = (len(a.calls), len(b.calls))
    with pytest.raises(AllProvidersFailed) as exc:
        list(gw.stream(MSGS))
    assert (len(a.calls), len(b.calls)) == n and [x.outcome for x in exc.value.attempts] == ["skipped_circuit_open"] * 2


# ---- failover: only before the first token -----------------------------------------------------------------------------

def test_failover_before_the_first_token_primary_fails_immediately_secondary_answers():
    a = ScriptedProvider("groq", "m1", [[fail("auth", 401)]])
    b = ScriptedProvider("gemini", "m2", [ok("Answer from B.")])
    gw, _ = gateway(a, b)
    text, done = run(gw)
    assert text == "Answer from B." and done.provider == "gemini" and done.model == "m2" and len(a.calls) == 1 and len(b.calls) == 1


def test_failover_after_retries_are_exhausted_also_counts_as_before_the_first_token():
    a = ScriptedProvider("groq", "m1", [[fail("rate_limit", 429)]])
    b = ScriptedProvider("gemini", "m2", [ok("B")])
    gw, _ = gateway(a, b, max_attempts=2)
    assert run(gw)[0] == "B" and len(a.calls) == 2


def test_an_empty_response_is_a_failure_that_fails_over_not_a_blank_answer():
    a = ScriptedProvider("groq", "m1", [ok()])  # only a Final, no text
    b = ScriptedProvider("gemini", "m2", [ok("real answer")])
    gw, _ = gateway(a, b)
    text, done = run(gw)
    assert text == "real answer" and done.attempts[0].error_class == "EmptyResponse"


def test_three_streams_in_a_row_that_stop_with_length_and_no_text_do_not_open_the_breaker():
    from rag_core.errors import RagError

    p = ScriptedProvider("groq", "m1", [ok(finish="length")])  # no visible text: the whole output budget went into reasoning
    gw, _ = gateway(p, max_attempts=1, breaker_threshold=3)
    for _ in range(3):
        with pytest.raises(RagError):
            list(gw.stream(MSGS))
    assert gw.breakers["groq"].state == "closed"  # a hard question says nothing about the provider's health
    with pytest.raises(RagError):
        list(gw.stream(MSGS))
    assert len(p.calls) == 4  # the 4th request still reached the (healthy) provider


def test_a_length_stop_without_text_is_retried_once_with_a_higher_cap_and_can_succeed():
    p = ScriptedProvider("groq", "m1", [ok(finish="length"), ok("the answer")])
    gw, clock = gateway(p)
    items = list(gw.stream(MSGS, max_output_tokens=512))
    assert "".join(i.text for i in items if isinstance(i, Delta)) == "the answer"
    assert [c["max_output_tokens"] for c in p.calls] == [512, 1024] and clock.sleeps == []  # a higher cap, and no backoff delay
    assert [(a.outcome, a.error_kind) for a in items[-1].attempts] == [("error", "length"), ("ok", None)]
    assert gw.breakers["groq"].state == "closed"


def test_two_length_stops_in_a_row_end_in_a_typed_answer_too_long_error():
    from rag_core.errors import AnswerTooLong

    p = ScriptedProvider("groq", "m1", [ok(finish="length")])
    gw, _ = gateway(p)
    with pytest.raises(AnswerTooLong) as exc:
        list(gw.stream(MSGS, max_output_tokens=512))
    assert [c["max_output_tokens"] for c in p.calls] == [512, 1024]
    assert [(a.outcome, a.error_kind) for a in exc.value.attempts] == [("error", "length"), ("error", "length")]
    assert gw.breakers["groq"].state == "closed"


def test_no_second_try_when_the_cap_is_already_at_the_ceiling():
    from rag_core.errors import AnswerTooLong

    p = ScriptedProvider("groq", "m1", [ok(finish="length")])
    gw, _ = gateway(p)
    with pytest.raises(AnswerTooLong):
        list(gw.stream(MSGS, max_output_tokens=4096))
    assert len(p.calls) == 1


def test_a_length_failure_still_lets_the_next_provider_answer_and_opens_no_breaker():
    a = ScriptedProvider("groq", "m1", [ok(finish="length")])
    b = ScriptedProvider("gemini", "m2", [ok("B answers")])
    gw, _ = gateway(a, b)
    text, done = run(gw)
    assert text == "B answers" and done.provider == "gemini" and gw.breakers["groq"].state == "closed"


class SlowDrip:
    """A provider whose every delta takes ``step_s`` seconds on the fake clock; records whether its generator was closed."""

    def __init__(self, clock, texts, *, step_s, name="groq", model="m1"):
        self.name, self.model, self.clock, self.texts, self.step_s = name, model, clock, list(texts), step_s
        self.calls, self.closed = 0, False

    def stream(self, messages, *, timeout_s, max_output_tokens, temperature, response_schema=None):
        self.calls += 1
        try:
            for text in self.texts:
                self.clock.advance(self.step_s)
                yield Delta(text)
            yield Final("stop", 1, 1)
        finally:
            self.closed = True


def test_a_slow_drip_stream_is_cut_at_the_total_deadline_and_ends_with_the_partial_text():
    clock = FakeClock()
    slow = SlowDrip(clock, ["a", "b", "c", "d", "e"], step_s=4.0)  # t = 4, 8, 12 ...: the third delta arrives after the 10 s deadline
    other = ScriptedProvider("gemini", "m2", [ok("SPLICED")])
    gw, _ = gateway(slow, other, clock=clock, deadline_s=10.0, max_attempts=3)
    seen = []
    with pytest.raises(MidStreamError) as exc:
        for item in gw.stream(MSGS):
            seen.append(item)
    e = exc.value
    assert e.partial_text == "ab" and "".join(i.text for i in seen) == "ab"  # the delta that arrived late is not yielded
    assert e.cause.kind == "timeout" and e.cause.error_class == "DeadlineExceeded"
    assert slow.calls == 1 and slow.closed and len(other.calls) == 0  # no retry, no failover, the provider's stream (its connection) is closed
    assert e.attempts[-1].first_token_seen is True and e.attempts[-1].error_class == "DeadlineExceeded"
    assert clock.sleeps == []  # no backoff wait on a deadline


def test_a_stream_that_never_shows_text_before_the_deadline_fails_without_waiting_or_retrying():
    clock = FakeClock()
    slow = SlowDrip(clock, ["", "", "", ""], step_s=4.0)  # reasoning-only deltas: no visible text
    gw, _ = gateway(slow, clock=clock, deadline_s=10.0, max_attempts=3)
    with pytest.raises(AllProvidersFailed) as exc:
        list(gw.stream(MSGS))
    assert slow.calls == 1 and slow.closed and clock.sleeps == []
    assert exc.value.attempts[-1].error_class == "DeadlineExceeded" and exc.value.attempts[-1].first_token_seen is False


def test_a_stream_that_finishes_inside_the_deadline_is_not_affected():
    clock = FakeClock()
    fast = SlowDrip(clock, ["a", "b", "c"], step_s=3.0)  # 9 s in total, deadline 10 s
    gw, _ = gateway(fast, clock=clock, deadline_s=10.0)
    text, done = run(gw)
    assert text == "abc" and done.finish_reason == "stop" and fast.closed


def test_the_deadline_is_measured_from_the_start_of_the_request_not_of_the_attempt():
    clock = FakeClock()
    first = ScriptedProvider("groq", "m1", [[fail("server", 503)], [fail("server", 503)]])
    slow = SlowDrip(clock, ["a", "b", "c"], step_s=4.0, name="gemini", model="m2")
    gw, _ = gateway(first, slow, clock=clock, deadline_s=10.0, max_attempts=2, base_delay_s=3.0, max_delay_s=3.0)
    with pytest.raises(MidStreamError) as exc:  # 3 s of backoff were already spent: the second delta (3 + 4 + 4 = 11 s) is over the deadline
        list(gw.stream(MSGS))
    assert exc.value.partial_text == "a" and exc.value.provider == "gemini"


def test_MIDSTREAM_failure_raises_a_typed_error_and_does_not_retry_or_fail_over():
    a = ScriptedProvider("groq", "m1", [["The answer ", "is par", fail("server", 503)]])
    b = ScriptedProvider("gemini", "m2", [ok("SPLICED TEXT")])
    gw, clock = gateway(a, b, max_attempts=3)
    seen = []
    with pytest.raises(MidStreamError) as exc:
        for item in gw.stream(MSGS):
            seen.append(item)
    e = exc.value
    assert e.partial_text == "The answer is par" and "".join(i.text for i in seen) == "The answer is par"
    assert (e.provider, e.model, e.cause.kind) == ("groq", "m1", "server")
    assert len(a.calls) == 1 and clock.sleeps == []  # not retried
    assert len(b.calls) == 0  # not failed over: nothing of the second provider is ever spliced onto the partial text
    assert e.attempts[-1].first_token_seen is True and e.attempts[-1].outcome == "error"


def test_midstream_failure_of_an_unclassified_exception_is_still_typed():
    a = ScriptedProvider("groq", "m1", [["partial", RuntimeError("boom with secret-looking text")]])
    gw, _ = gateway(a, ScriptedProvider("g", "m2", [ok("x")]))
    with pytest.raises(MidStreamError) as exc:
        list(gw.stream(MSGS))
    assert exc.value.cause.kind == "other" and exc.value.cause.error_class == "RuntimeError" and "secret-looking" not in str(exc.value)


def test_a_midstream_failure_counts_towards_the_breaker():
    a = ScriptedProvider("groq", "m1", [["x", fail("connection")]])
    gw, _ = gateway(a, ScriptedProvider("g", "m2", [ok("y")]), breaker_threshold=2)
    for _ in range(2):
        with pytest.raises(MidStreamError):
            list(gw.stream(MSGS))
    assert gw.breakers["groq"].state == "open"


def test_an_unclassified_exception_before_any_text_is_not_retried_and_fails_over():
    a = ScriptedProvider("groq", "m1", [[ValueError("weird")]])
    b = ScriptedProvider("gemini", "m2", [ok("B")])
    gw, clock = gateway(a, b)
    text, done = run(gw)
    assert text == "B" and len(a.calls) == 1 and done.attempts[0].error_kind == "other" and done.attempts[0].error_class == "ValueError"


def test_all_providers_failing_raises_with_one_record_per_attempt():
    a, b = ScriptedProvider("groq", "m1", [[fail("server", 500)]]), ScriptedProvider("gemini", "m2", [[fail("timeout")]])
    gw, _ = gateway(a, b, max_attempts=2)
    with pytest.raises(AllProvidersFailed) as exc:
        list(gw.stream(MSGS))
    got = [(x.provider, x.attempt, x.error_kind) for x in exc.value.attempts]
    assert got == [("groq", 1, "server"), ("groq", 2, "server"), ("gemini", 1, "timeout"), ("gemini", 2, "timeout")]


def test_a_single_provider_that_succeeds_is_called_once():
    a = ScriptedProvider("groq", "m1", [ok("A")])
    gw, _ = gateway(a)
    assert run(gw)[0] == "A" and len(a.calls) == 1


def test_constructor_validation():
    p = ScriptedProvider("a", "m", [ok("x")])
    with pytest.raises(ValueError):
        LLMGateway([])
    with pytest.raises(ValueError):
        LLMGateway([p, ScriptedProvider("a", "m2", [ok("x")])])
    with pytest.raises(ValueError):
        LLMGateway([p], max_attempts=0)
    assert LLMGateway([p, ScriptedProvider("b", "m3", [ok("x")])]).primary_model == "m"


def test_as_provider_error_keeps_typed_errors_and_hides_the_details_of_others():
    e = fail("rate_limit", 429)
    assert as_provider_error(e) is e
    other = as_provider_error(KeyError("the-api-key-is-in-here"))
    assert other.kind == "other" and other.error_class == "KeyError" and "api-key" not in str(other)


# ---- Gemini provider (google-genai) --------------------------------------------------------------------------------

def gem_chunk(text=None, finish=None, usage=None, with_candidates=True):
    from google.genai import types

    cands = [SimpleNamespace(finish_reason=types.FinishReason[finish] if finish else None)] if with_candidates else []
    return SimpleNamespace(text=text, candidates=cands, usage_metadata=usage)


class FakeGenai:
    def __init__(self, chunks=None, exc=None):
        self.chunks, self.exc, self.kwargs = chunks or [], exc, None
        self.models = SimpleNamespace(generate_content_stream=self._stream)

    def _stream(self, **kw):
        self.kwargs = kw
        if self.exc:
            raise self.exc
        return iter(self.chunks)


def gemini(client, monkeypatch, key="AIza-FAKE-KEY-for-tests"):
    monkeypatch.setenv("GEMINI_API_KEY", key)
    return GeminiProvider("gemini-test", client_factory=lambda k: client), key


def test_the_old_gemini_bug_a_timeout_inside_the_config_dict_is_rejected_by_pydantic():
    from google.genai import types
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        types.GenerateContentConfig(**{"timeout": 30})


def test_gemini_timeout_goes_into_http_options_in_milliseconds_with_sdk_retries_off():
    p = GeminiProvider("gemini-test")
    cfg = p.build_config("system text", timeout_s=12.5, max_output_tokens=77, temperature=0.1)
    assert cfg.http_options.timeout == 12500 and cfg.http_options.retry_options.attempts == 1
    assert cfg.system_instruction == "system text" and cfg.max_output_tokens == 77 and cfg.temperature == 0.1
    assert GeminiProvider("m", thinking_budget=0).build_config(None, 1, 1, 0).thinking_config.thinking_budget == 0


def test_gemini_streams_text_finish_reason_and_usage_and_splits_the_system_prompt(monkeypatch):
    usage = SimpleNamespace(prompt_token_count=20, candidates_token_count=4)
    client = FakeGenai([gem_chunk("Hel"), gem_chunk("lo", finish="STOP", usage=usage)])
    p, _ = gemini(client, monkeypatch)
    items = list(p.stream(MSGS, timeout_s=5.0, max_output_tokens=40, temperature=0.0))
    assert items == [Delta("Hel"), Delta("lo"), Final("stop", 20, 4)]
    kw = client.kwargs
    assert kw["model"] == "gemini-test" and kw["config"].system_instruction == "s" and kw["config"].http_options.timeout == 5000
    assert [c.role for c in kw["contents"]] == ["user"] and kw["contents"][0].parts[0].text == "u"


def test_gemini_max_tokens_is_reported_as_length(monkeypatch):
    client = FakeGenai([gem_chunk("x", finish="MAX_TOKENS")])
    p, _ = gemini(client, monkeypatch)
    assert list(p.stream(MSGS, timeout_s=1, max_output_tokens=5, temperature=0))[-1].finish_reason == "length"


def test_gemini_ignores_chunks_without_text(monkeypatch):
    client = FakeGenai([gem_chunk(None), gem_chunk("a", with_candidates=False), gem_chunk("b")])
    p, _ = gemini(client, monkeypatch)
    assert [i for i in p.stream(MSGS, timeout_s=1, max_output_tokens=5, temperature=0) if isinstance(i, Delta)] == [Delta("b")]


def test_gemini_missing_key_is_a_typed_auth_error_and_construction_is_lazy(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    p = GeminiProvider("m", client_factory=lambda k: pytest.fail("no client without a key"))
    assert p._client is None
    with pytest.raises(ProviderError) as exc:
        list(p.stream(MSGS, timeout_s=1, max_output_tokens=5, temperature=0))
    assert exc.value.kind == "auth" and "GEMINI_API_KEY" in str(exc.value)


@pytest.mark.parametrize("code,kind", [(429, "rate_limit"), (500, "server"), (503, "server"), (401, "auth"), (403, "auth"), (404, "not_found"), (400, "bad_request")])
def test_gemini_maps_real_sdk_errors_to_kinds(monkeypatch, code, kind):
    from google.genai import errors

    cls = errors.ServerError if code >= 500 else errors.ClientError
    err = cls(code, {"error": {"code": code, "message": "nope", "status": "X"}}, None)
    p, _ = gemini(FakeGenai(exc=err), monkeypatch)
    with pytest.raises(ProviderError) as exc:
        list(p.stream(MSGS, timeout_s=1, max_output_tokens=5, temperature=0))
    assert exc.value.kind == kind and exc.value.status == code


def test_gemini_maps_timeouts_and_connection_failures_by_class(monkeypatch):
    class ReadTimeout(Exception):
        pass

    class ConnectError(Exception):
        pass

    for err, kind in ((ReadTimeout("slow"), "timeout"), (ConnectError("down"), "connection"), (TimeoutError("t"), "timeout"), (ConnectionResetError("r"), "connection"),
                      (RuntimeError("x"), "other")):
        p, _ = gemini(FakeGenai(exc=err), monkeypatch)
        with pytest.raises(ProviderError) as exc:
            list(p.stream(MSGS, timeout_s=1, max_output_tokens=5, temperature=0))
        assert exc.value.kind == kind, (type(err).__name__, exc.value.kind)


def test_gemini_errors_never_contain_the_api_key(monkeypatch):
    from google.genai import errors

    secret = "AIza-SUPER-SECRET-TEST-KEY"
    err = errors.ClientError(400, {"error": {"code": 400, "message": f"API key not valid: {secret}"}}, None)
    p, _ = gemini(FakeGenai(exc=err), monkeypatch, key=secret)
    with pytest.raises(ProviderError) as exc:
        list(p.stream(MSGS, timeout_s=1, max_output_tokens=5, temperature=0))
    assert secret not in str(exc.value)


def test_gemini_thought_tokens_are_added_to_the_completion_tokens_and_reported_separately(monkeypatch):
    usage = SimpleNamespace(prompt_token_count=20, candidates_token_count=4, thoughts_token_count=30)
    client = FakeGenai([gem_chunk("Hi", finish="STOP", usage=usage)])
    p, _ = gemini(client, monkeypatch)
    assert list(p.stream(MSGS, timeout_s=1, max_output_tokens=5, temperature=0))[-1] == Final("stop", 20, 34, 30)


def test_gemini_failure_after_some_text_surfaces_through_the_gateway_as_midstream(monkeypatch):
    class Breaks(FakeGenai):
        def _stream(self, **kw):
            def gen():
                yield gem_chunk("partial ")
                raise ConnectionResetError("reset")
            return gen()

    p, _ = gemini(Breaks(), monkeypatch)
    gw, _ = gateway(p)
    with pytest.raises(MidStreamError) as exc:
        list(gw.stream(MSGS))
    assert exc.value.partial_text == "partial "
