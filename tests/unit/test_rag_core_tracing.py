"""rag_core.tracing: masking, capture modes, cost, fail-open wrapper, sampling, daily cap, LangfuseTracer against a fake client. No SDK, no network."""

import os
import subprocess
import sys
import textwrap
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_core.config import TracingConfig
from rag_core.tracing import (ANSWER_MAX_CHARS, QUESTION_MAX_CHARS, Capture, DailyCap, FakeTracer, LangfuseTracer, NoopTracer, SafeTracer, build_tracer,
                              cost_details, mask_text, redact_deep, redact_keys)
from tests.unit.rag_core_fakes import FakeLangfuse

ROOT = Path(__file__).resolve().parents[2]
KEYS = ["gsk_abcdefghijklmnop1234", "AIzaSyA1234567890abcdefghijKLMNOP", "sk-lf-1234abcd-5678", "pk-lf-1234abcd-5678", "sk-proj_abcdefgh12345678",
        "ghp_abcdefghijkl1234", "x" * 45]


def src(i=1, text="secret passage text"):
    return SimpleNamespace(chunk_id=f"c{i}", doc_id="1234.1", section="Intro", rank=i, score=0.123456789, label=f"S{i}", text=text)


# ---- masking ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("key", KEYS)
def test_key_like_strings_are_redacted_in_every_mode(key):
    text = f"my key is {key} ok"
    for mode in Capture.MODES:
        c = Capture(mode)
        for out in (c.question(text), c.answer(text)):
            assert out is None or key not in out
    assert key not in redact_keys(text) and key not in mask_text(text)
    assert key not in redact_deep({"a": [text, {"b": (text,)}]}).__repr__()


def test_masked_text_redacts_urls_emails_and_long_numbers_but_keeps_short_numbers():
    out = mask_text("see https://x.org/a?b=1 and www.foo.com mail bob@corp.example.com tel +1 (555) 123-4567 table 3 in 2019 with 42 items")
    assert "x.org" not in out and "foo.com" not in out and "bob@" not in out and "555" not in out
    assert "[URL]" in out and "[EMAIL]" in out and "[NUMBER]" in out
    assert "table 3 in 2019 with 42 items" in out


def test_masking_handles_empty_none_and_unicode():
    assert mask_text(None) == "" and mask_text("") == "" and mask_text("  ") == "  "
    assert mask_text("Zażółć gęślą jaźń – 漢字 é") == "Zażółć gęślą jaźń – 漢字 é"


def test_truncation_happens_after_redaction_so_a_cut_can_never_split_a_secret():
    key = "gsk_" + "A" * 30
    text = "a" * (QUESTION_MAX_CHARS - 10) + key
    out = Capture("masked").question(text)
    assert "gsk_" not in out and "AAAA" not in out and len(out) <= QUESTION_MAX_CHARS + 1


def test_masked_limits():
    c = Capture("masked")
    q, a = c.question("word " * 200), c.answer("word " * 400)
    assert len(q) <= QUESTION_MAX_CHARS + 1 and q.endswith("…") and len(a) <= ANSWER_MAX_CHARS + 1 and a.endswith("…")
    assert c.question("short") == "short"


# ---- capture modes ---------------------------------------------------------------------------------------------------------------

def test_masked_mode_never_includes_passage_text_or_the_compiled_prompt():
    c = Capture("masked")
    rows = c.sources([src(1), src(2)])
    assert all("text" not in r for r in rows) and rows[0] == {"chunk_id": "c1", "doc_id": "1234.1", "section": "Intro", "rank": 1, "score": 0.123457, "label": "S1"}
    assert c.messages([{"role": "user", "content": "x"}]) is None
    gi = c.generation_input([{"role": "user", "content": "PASSAGE"}], "Why? bob@x.org", [src(1)])
    assert gi == {"source_ids": ["c1"], "question": "Why? [EMAIL]"} and "PASSAGE" not in repr(gi)


def test_full_mode_includes_text_and_prompt_but_still_redacts_keys():
    c = Capture("full")
    rows = c.sources([src(1, "text with gsk_abcdefghijklmnop1234 inside")])
    assert rows[0]["text"] == "text with [REDACTED_KEY] inside"
    msgs = c.messages([{"role": "system", "content": "s"}, {"role": "user", "content": "u gsk_abcdefghijklmnop1234"}])
    assert msgs[1]["content"] == "u [REDACTED_KEY]"
    assert c.generation_input([{"role": "user", "content": "u"}], "q", [src(1)]) == [{"role": "user", "content": "u"}]
    assert c.question("word " * 200) == "word " * 200  # full: untruncated
    assert c.question("q" * 1000) == "[REDACTED_KEY]"  # a single 1000-character opaque token is treated as a key even in full mode


def test_none_mode_records_no_question_or_answer_text_at_all():
    c = Capture("none")
    assert c.question("anything") is None and c.answer("anything") is None and c.messages([{"role": "user", "content": "x"}]) is None
    assert c.generation_input([], "secret question", [src(1)]) == {"source_ids": ["c1"]}


def test_an_unknown_capture_mode_is_refused():
    with pytest.raises(ValueError, match="tracing.capture"):
        Capture("verbose")


def test_capture_meta_redacts_nested_keys():
    assert Capture("masked").meta({"x": {"y": ["gsk_abcdefghijklmnop1234"]}, "n": 3}) == {"x": {"y": ["[REDACTED_KEY]"]}, "n": 3}


# ---- cost ------------------------------------------------------------------------------------------------------------------------

def test_cost_details_only_from_configured_prices():
    prices = {"m": {"input": 2.0, "output": 10.0}}
    assert cost_details(prices, "m", 1_000_000, 500_000) == {"input": 2.0, "output": 5.0}
    assert cost_details({}, "m", 10, 10) is None and cost_details(None, "m", 10, 10) is None  # never invents a price
    assert cost_details(prices, "other", 10, 10) is None and cost_details(prices, None, 10, 10) is None
    assert cost_details(prices, "m", None, 5) is None and cost_details(prices, "m", 5, None) is None
    assert cost_details({"m": {"input": 1}}, "m", 1, 1) is None and cost_details({"m": {"input": "x", "output": 1}}, "m", 1, 1) is None


# ---- fail open -------------------------------------------------------------------------------------------------------------------

def drive(tracer):
    t = tracer.start_request("rag.request", session_id="s", input={"q": 1}, metadata={"a": 1})
    with t.span("route", input="x") as h:
        h.update(output="y")
    with t.generation("generation", model="m") as g:
        g.update(usage={"prompt_tokens": 1, "completion_tokens": 2})
    t.event("cache.miss", level="DEFAULT", metadata={})
    t.set_error("code", "msg")
    t.end(output="done", metadata={"k": 1})
    tracer.flush()
    return t.trace_id


def test_a_tracer_that_raises_on_every_call_is_swallowed_and_counted():
    safe = SafeTracer(FakeTracer(raise_on_call=True))
    assert drive(safe) is None and safe.stats.errors == 2  # start_request and flush; after a failed start every call is a no-op


class Exploding:
    """Raises inside every context manager entry, exit, update and attribute read."""

    enabled = True

    def start_request(self, *a, **k):
        return self

    @property
    def trace_id(self):
        raise RuntimeError("no id")

    @contextmanager
    def span(self, *a, **k):
        raise RuntimeError("enter")
        yield

    generation = span

    def event(self, *a, **k):
        raise RuntimeError("event")

    def set_error(self, *a, **k):
        raise RuntimeError("err")

    def end(self, *a, **k):
        raise RuntimeError("end")

    def flush(self, *a, **k):
        raise RuntimeError("flush")


def test_context_managers_that_fail_on_enter_still_run_the_body_exactly_once():
    safe = SafeTracer(Exploding())
    t = safe.start_request("r")
    ran = []
    with t.span("s") as h:
        ran.append(1)
        h.update(x=1)
    with t.generation("g") as g:
        ran.append(2)
        g.update(x=1)
    assert ran == [1, 2] and safe.stats.errors >= 3


class ExitFails:
    enabled = True

    def start_request(self, *a, **k):
        return self

    trace_id = "t"

    @contextmanager
    def span(self, *a, **k):
        yield SimpleNamespace(update=lambda **kw: (_ for _ in ()).throw(RuntimeError("update")))
        raise RuntimeError("exit")

    generation = span


def test_a_failing_exit_or_update_does_not_leak_and_body_exceptions_still_propagate():
    safe = SafeTracer(ExitFails())
    t = safe.start_request("r")
    with t.span("s") as h:
        h.update(a=1)
    with pytest.raises(KeyError):
        with t.span("s"):
            raise KeyError("body error is the caller's, not swallowed")
    assert safe.stats.errors >= 2


def test_the_safe_wrapper_passes_a_healthy_tracer_through_and_reports_enabled():
    inner = FakeTracer()
    safe = SafeTracer(inner)
    assert drive(safe) == "fake-trace-1" and inner.traces[0].names() == ["rag.request", "route", "generation", "cache.miss"] and inner.flushes == 1
    assert safe.enabled and SafeTracer().enabled is False and SafeTracer(NoopTracer()).start_request("x").trace_id is None


# ---- build_tracer / sampling / cap -----------------------------------------------------------------------------------------------

def test_build_tracer_is_a_noop_when_disabled_or_without_keys():
    assert isinstance(build_tracer(TracingConfig(enabled=False), env={"LANGFUSE_PUBLIC_KEY": "a", "LANGFUSE_SECRET_KEY": "b"}), NoopTracer)
    assert isinstance(build_tracer(TracingConfig(enabled=True), env={}), NoopTracer)
    assert isinstance(build_tracer(TracingConfig(enabled=True), env={"LANGFUSE_PUBLIC_KEY": "a"}), NoopTracer)
    assert isinstance(build_tracer(TracingConfig(enabled=True), env={"LANGFUSE_PUBLIC_KEY": "a", "LANGFUSE_SECRET_KEY": "b"}), LangfuseTracer)
    assert isinstance(build_tracer(TracingConfig(enabled=True), client_factory=FakeLangfuse), LangfuseTracer)


def test_the_noop_tracer_never_yields_a_trace_id():
    t = NoopTracer().start_request("x", input=1)
    with t.span("a") as h:
        h.update(x=1)
    assert t.trace_id is None and NoopTracer().flush() is None


@pytest.mark.slow
def test_building_a_tracer_without_keys_never_imports_the_sdk():
    env = {k: v for k, v in os.environ.items() if not k.startswith("LANGFUSE_")}
    env["PYTHONPATH"] = str(ROOT)
    code = textwrap.dedent("""
        import sys
        from rag_core.config import TracingConfig
        from rag_core.tracing import build_tracer, SafeTracer
        t = SafeTracer(build_tracer(TracingConfig(enabled=True)))
        tr = t.start_request("x"); tr.end(); t.flush()
        print(type(build_tracer(TracingConfig(enabled=True))).__name__, "langfuse" in sys.modules)
    """)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, env=env, timeout=120)
    assert proc.returncode == 0, proc.stderr[-500:]
    assert proc.stdout.strip() == "NoopTracer False"


def lf(cfg=None, rng=lambda: 0.0, clock=None, client=None):
    client = client or FakeLangfuse()
    props = []

    def propagate(**kw):
        props.append(kw)
        return _null()

    kwargs = {"clock": clock} if clock else {}
    return LangfuseTracer(cfg or TracingConfig(enabled=True), client_factory=lambda: client, propagate_factory=propagate, rng=rng, **kwargs), client, props


@contextmanager
def _null():
    yield


def test_sample_rate_zero_traces_nothing_and_one_traces_everything():
    tracer, client, _ = lf(TracingConfig(enabled=True, sample_rate=0.0), rng=lambda: 0.0)
    assert tracer.start_request("x").trace_id is None and not client.roots and tracer.stats.sampled_out == 1
    tracer, client, _ = lf(TracingConfig(enabled=True, sample_rate=1.0), rng=lambda: 0.999999)
    assert tracer.start_request("x").trace_id == "lf-trace-1" and len(client.roots) == 1


def test_sample_rate_is_applied_per_request():
    draws = iter([0.1, 0.6, 0.4, 0.9])
    tracer, client, _ = lf(TracingConfig(enabled=True, sample_rate=0.5), rng=lambda: next(draws))
    for _ in range(4):
        tracer.start_request("x")
    assert len(client.roots) == 2 and tracer.stats.sampled_out == 2


def test_the_daily_cap_blocks_after_the_limit_and_resets_on_the_next_utc_day():
    now = [datetime(2026, 11, 22, 23, 0, tzinfo=timezone.utc).timestamp()]
    cap = DailyCap(2, clock=lambda: now[0])
    assert [cap.allow() for _ in range(4)] == [True, True, False, False]
    now[0] += 2 * 3600  # past midnight UTC
    assert cap.allow() is True


def test_requests_over_the_cap_are_not_traced_and_still_counted():
    now = [datetime(2026, 11, 22, 10, 0, tzinfo=timezone.utc).timestamp()]
    tracer, client, _ = lf(TracingConfig(enabled=True, daily_request_cap=2), clock=lambda: now[0])
    ids = [tracer.start_request("x").trace_id for _ in range(3)]
    assert ids == ["lf-trace-1", "lf-trace-1", None] and len(client.roots) == 2 and tracer.stats.capped == 1 and tracer.stats.traced == 2


def test_a_sampled_out_request_does_not_consume_the_cap():
    tracer, client, _ = lf(TracingConfig(enabled=True, sample_rate=0.5, daily_request_cap=1), rng=iter([0.9, 0.1, 0.1]).__next__)
    tracer.start_request("x")  # sampled out
    assert tracer.start_request("x").trace_id and tracer.start_request("x").trace_id is None


# ---- LangfuseTracer against the fake client ---------------------------------------------------------------------------------------

def test_the_root_observation_carries_name_input_metadata_and_the_session_via_propagate_attributes():
    tracer, client, props = lf()
    tracer.start_request("rag.request", session_id="sess-1", input={"question": "q"}, metadata={"mode": "long"})
    assert props == [{"trace_name": "rag.request", "session_id": "sess-1"}]
    (root,) = client.roots
    assert (root.name, root.as_type, root.kwargs) == ("rag.request", "span", {"input": {"question": "q"}, "metadata": {"mode": "long"}})
    tracer.start_request("rag.request")
    assert props[1] == {"trace_name": "rag.request"}  # no session -> no session attribute


def test_children_are_parented_to_the_root_and_every_observation_ends():
    tracer, client, _ = lf()
    t = tracer.start_request("rag.request")
    with t.span("route", input="q", extra=1) as h:
        h.update(output={"m": 1}, other=2)
    with t.span("retrieve", as_type="retriever") as h:
        pass
    t.event("cache.miss", metadata={"k": 1})
    t.end(output="out", metadata={"m": 2})
    root = client.roots[0]
    assert [(c.name, c.as_type, c.parent is root) for c in root.children] == [("route", "span", True), ("retrieve", "retriever", True), ("cache.miss", "event", True)]
    route = root.children[0]
    assert route.ended and route.kwargs == {"input": "q", "metadata": {"extra": 1}} and route.updates == [{"output": {"m": 1}, "metadata": {"other": 2}}]
    assert root.children[1].ended and root.ended and root.updates == [{"output": "out", "metadata": {"m": 2}}]


def test_the_generation_maps_usage_first_token_model_cost_and_the_linked_prompt():
    tracer, client, _ = lf()
    t = tracer.start_request("rag.request")
    prompt_obj = object()
    first = datetime(2026, 11, 22, tzinfo=timezone.utc)
    with t.generation("generation", model="openai/gpt-oss-120b", prompt=prompt_obj, input={"source_ids": ["c1"]}, provider="groq") as g:
        g.update(completion_start_time=first, usage={"prompt_tokens": 120, "completion_tokens": 30}, output="ans", cost_details={"input": 0.1, "output": 0.2},
                 finish_reason="stop", attempts=1, level="DEFAULT")
    gen = client.roots[0].children[0]
    assert gen.as_type == "generation" and gen.ended
    assert gen.kwargs == {"model": "openai/gpt-oss-120b", "input": {"source_ids": ["c1"]}, "metadata": {"provider": "groq"}, "prompt": prompt_obj}
    (upd,) = gen.updates
    assert upd["usage_details"] == {"input": 120, "output": 30} and upd["completion_start_time"] == first and upd["output"] == "ans"
    assert upd["cost_details"] == {"input": 0.1, "output": 0.2} and upd["metadata"] == {"finish_reason": "stop", "attempts": 1} and upd["level"] == "DEFAULT"


def test_a_generation_without_a_linked_prompt_does_not_pass_one_and_missing_usage_is_omitted():
    tracer, client, _ = lf()
    t = tracer.start_request("r")
    with t.generation("generation", model="m") as g:
        g.update(usage={"prompt_tokens": None, "completion_tokens": 5})
    gen = client.roots[0].children[0]
    assert "prompt" not in gen.kwargs and gen.updates[0]["usage_details"] == {"output": 5}


def test_usage_on_a_plain_span_is_not_sent_as_usage_details():
    tracer, client, _ = lf()
    t = tracer.start_request("r")
    with t.span("route") as h:
        h.update(usage={"prompt_tokens": 1, "completion_tokens": 2})
    assert client.roots[0].children[0].updates == []  # usage is dropped on non-generation observations, nothing else to send


def test_set_error_marks_the_root_with_level_error_and_a_bounded_message():
    tracer, client, _ = lf()
    t = tracer.start_request("r")
    t.set_error("invalid_question", "x" * 2000)
    (upd,) = client.roots[0].updates
    assert upd["level"] == "ERROR" and upd["metadata"] == {"error_code": "invalid_question"} and len(upd["status_message"]) == 500 and upd["status_message"].startswith("invalid_question: ")


def test_set_error_can_mark_the_root_as_a_warning_for_degraded_requests():
    tracer, client, _ = lf()
    tracer.start_request("r").set_error("all_providers_failed", "m", "WARNING")
    assert client.roots[0].updates[0]["level"] == "WARNING"
    fake = FakeTracer()
    trace = fake.start_request("r")
    trace.set_error("c", "m", "WARNING")
    assert trace.observations[0]["level"] == "WARNING"
    safe = SafeTracer(FakeTracer())
    st = safe.start_request("r")
    st.set_error("c", "m", "WARNING")
    assert st._inner.observations[0]["level"] == "WARNING"



def test_events_carry_level_and_message():
    tracer, client, _ = lf()
    tracer.start_request("r").event("prompt.fallback", level="WARNING", status_message="no_keys", metadata={"prompt": "p"})
    ev = client.roots[0].children[0]
    assert ev.as_type == "event" and ev.kwargs == {"level": "WARNING", "status_message": "no_keys", "metadata": {"prompt": "p"}}


def test_no_client_means_no_trace_without_error():
    tracer = LangfuseTracer(TracingConfig(enabled=True), client_factory=lambda: None)
    assert tracer.start_request("x").trace_id is None
    tracer.flush()  # no client: returns


def test_flush_calls_the_client_and_gives_up_after_the_bounded_timeout():
    tracer, client, _ = lf(TracingConfig(enabled=True, flush_timeout_seconds=3))
    tracer.flush()
    assert client.flushes == 1 and tracer.stats.flush_timeouts == 0
    client.flush_blocks = 0.6
    tracer.flush(timeout_s=0.05)
    assert tracer.stats.flush_timeouts == 1  # returned after ~50 ms although the flush needs 600 ms


def test_a_flush_that_raises_is_counted_not_raised():
    class Bad(FakeLangfuse):
        def flush(self):
            raise RuntimeError("network")

    tracer, _, _ = lf(client=Bad())
    tracer.flush()
    import time

    time.sleep(0.05)
    assert tracer.stats.errors == 1


def test_the_real_tracer_behind_the_safe_wrapper_survives_a_client_that_raises_at_every_step():
    class Boom:
        def start_observation(self, **kw):
            raise RuntimeError("sdk broke")

    tracer = SafeTracer(LangfuseTracer(TracingConfig(enabled=True), client_factory=Boom, propagate_factory=lambda **kw: _null()))
    t = tracer.start_request("x", input=1)
    with t.span("a"):
        pass
    t.end(output="o")
    assert t.trace_id is None and tracer.stats.errors >= 1
