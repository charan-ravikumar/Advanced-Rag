"""rag_core.tracing and rag_core.otel_tracing: masking, capture modes, cost, fail-open wrapper, sampling, daily cap, and the OpenTelemetry tracer against an
IN-MEMORY span exporter (the real OpenTelemetry SDK, no network). The Langfuse attribute encoding is compared with the installed Langfuse SDK's own."""

import json
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
from rag_core.tracing import (
    ANSWER_MAX_CHARS,
    QUESTION_MAX_CHARS,
    Capture,
    DailyCap,
    FakeTracer,
    NoopTracer,
    SafeTracer,
    build_tracer,
    configured_exporters,
    cost_details,
    mask_text,
    redact_deep,
    redact_keys,
)

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
    assert isinstance(build_tracer(TracingConfig(enabled=True, exporters=[]), env={"LANGFUSE_PUBLIC_KEY": "a", "LANGFUSE_SECRET_KEY": "b"}), NoopTracer)
    assert isinstance(build_tracer(TracingConfig(enabled=True), exporters=[]), NoopTracer)


def test_with_both_langfuse_keys_the_tracer_is_opentelemetry_exporting_to_langfuses_otlp_endpoint_with_basic_auth(monkeypatch):
    import base64

    import opentelemetry.exporter.otlp.proto.http.trace_exporter as otlp_http

    from rag_core.otel_tracing import OtelTracer

    made = []

    class Recorder:
        def __init__(self, **kw):
            made.append(kw)

    monkeypatch.setattr(otlp_http, "OTLPSpanExporter", Recorder)
    env = {"LANGFUSE_PUBLIC_KEY": "pk-lf-a", "LANGFUSE_SECRET_KEY": "sk-lf-b", "LANGFUSE_BASE_URL": "https://lf.example.org/"}
    (exporter,) = configured_exporters(TracingConfig(enabled=True), env)
    assert isinstance(exporter, Recorder) and made[0]["endpoint"] == "https://lf.example.org/api/public/otel/v1/traces"
    assert made[0]["headers"] == {"Authorization": "Basic " + base64.b64encode(b"pk-lf-a:sk-lf-b").decode(), "x-langfuse-ingestion-version": "4"}
    configured_exporters(TracingConfig(enabled=True), {"LANGFUSE_PUBLIC_KEY": "a", "LANGFUSE_SECRET_KEY": "b"})
    assert made[1]["endpoint"] == "https://cloud.langfuse.com/api/public/otel/v1/traces"
    monkeypatch.undo()
    assert isinstance(build_tracer(TracingConfig(enabled=True), env=env), OtelTracer)


def test_the_otlp_exporter_needs_no_langfuse_keys_and_an_unknown_exporter_is_a_config_error():
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    (exporter,) = configured_exporters(TracingConfig(enabled=True, exporters=["otlp"]), {})
    assert isinstance(exporter, OTLPSpanExporter)  # its endpoint comes from the standard OTEL_EXPORTER_OTLP_* variables (read by the SDK)
    assert len(configured_exporters(TracingConfig(enabled=True, exporters=["langfuse", "otlp"]), {"LANGFUSE_PUBLIC_KEY": "a", "LANGFUSE_SECRET_KEY": "b"})) == 2
    with pytest.raises(ValueError, match="unknown exporter"):
        configured_exporters(TracingConfig(enabled=True, exporters=["jaeger"]), {})


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
        print(type(build_tracer(TracingConfig(enabled=True))).__name__, "langfuse" in sys.modules, "opentelemetry.sdk" in sys.modules)
    """)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, env=env, timeout=120)
    assert proc.returncode == 0, proc.stderr[-500:]
    assert proc.stdout.strip() == "NoopTracer False False"


def otel(cfg=None, rng=lambda: 0.0, clock=None):
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    mem = InMemorySpanExporter()
    kwargs = {"clock": clock} if clock else {}
    return build_tracer(cfg or TracingConfig(enabled=True), exporters=[mem], simple=True, rng=rng, **kwargs), mem


def finished(mem) -> dict:
    return {s.name: s for s in mem.get_finished_spans()}


def meta(span, key):
    return json.loads(span.attributes[f"langfuse.observation.metadata.{key}"])


def test_sample_rate_zero_traces_nothing_and_one_traces_everything():
    tracer, mem = otel(TracingConfig(enabled=True, sample_rate=0.0), rng=lambda: 0.0)
    t = tracer.start_request("x")
    t.end()
    assert t.trace_id is None and mem.get_finished_spans() == () and tracer.stats.sampled_out == 1
    tracer, mem = otel(TracingConfig(enabled=True, sample_rate=1.0), rng=lambda: 0.999999)
    t = tracer.start_request("x")
    t.end()
    assert len(t.trace_id) == 32 and len(mem.get_finished_spans()) == 1


def test_sample_rate_is_applied_per_request():
    draws = iter([0.1, 0.6, 0.4, 0.9])
    tracer, mem = otel(TracingConfig(enabled=True, sample_rate=0.5), rng=lambda: next(draws))
    for _ in range(4):
        tracer.start_request("x").end()
    assert len(mem.get_finished_spans()) == 2 and tracer.stats.sampled_out == 2


def test_the_daily_cap_blocks_after_the_limit_and_resets_on_the_next_utc_day():
    now = [datetime(2026, 11, 22, 23, 0, tzinfo=timezone.utc).timestamp()]
    cap = DailyCap(2, clock=lambda: now[0])
    assert [cap.allow() for _ in range(4)] == [True, True, False, False]
    now[0] += 2 * 3600  # past midnight UTC
    assert cap.allow() is True


def test_requests_over_the_cap_are_not_traced_and_still_counted():
    now = [datetime(2026, 11, 22, 10, 0, tzinfo=timezone.utc).timestamp()]
    tracer, mem = otel(TracingConfig(enabled=True, daily_request_cap=2), clock=lambda: now[0])
    traces = [tracer.start_request("x") for _ in range(3)]
    for t in traces:
        t.end()
    assert [t.trace_id is None for t in traces] == [False, False, True] and len(mem.get_finished_spans()) == 2
    assert tracer.stats.capped == 1 and tracer.stats.traced == 2 and traces[0].trace_id != traces[1].trace_id  # one trace per request


def test_a_sampled_out_request_does_not_consume_the_cap():
    tracer, _ = otel(TracingConfig(enabled=True, sample_rate=0.5, daily_request_cap=1), rng=iter([0.9, 0.1, 0.1]).__next__)
    tracer.start_request("x")  # sampled out
    assert tracer.start_request("x").trace_id and tracer.start_request("x").trace_id is None


# ---- the OpenTelemetry tracer ------------------------------------------------------------------------------------------------------

def test_the_root_span_starts_a_new_trace_and_carries_the_trace_name_session_input_and_metadata():
    tracer, mem = otel()
    t = tracer.start_request("rag.request", session_id="sess-1", input={"question": "q"}, metadata={"mode": "long", "skip": None})
    t.end(output="out", metadata={"cache_hit": False})
    root = finished(mem)["rag.request"]
    a = root.attributes
    assert root.parent is None and format(root.context.trace_id, "032x") == t.trace_id
    assert a["langfuse.trace.name"] == "rag.request" and a["session.id"] == "sess-1" and a["langfuse.observation.type"] == "span"
    assert json.loads(a["langfuse.observation.input"]) == {"question": "q"} and a["langfuse.observation.output"] == "out"
    assert meta(root, "mode") == "long" and meta(root, "cache_hit") is False and "langfuse.observation.metadata.skip" not in a  # None is not written
    tracer.start_request("rag.request").end()
    assert "session.id" not in mem.get_finished_spans()[-1].attributes  # no session -> no session attribute


def test_children_are_parented_to_the_root_by_explicit_context_and_every_span_ends():
    from opentelemetry import trace as otel_trace

    tracer, mem = otel()
    t = tracer.start_request("rag.request")
    with t.span("route", input="q", extra=1) as h:
        h.update(output={"m": 1}, other=2)
    with t.span("retrieve", as_type="retriever"):
        assert otel_trace.get_current_span() is otel_trace.INVALID_SPAN  # nothing ambient: safe inside a generator that yields
    t.event("cache.miss", metadata={"k": 1})
    t.end()
    spans = finished(mem)
    root = spans["rag.request"]
    for name, kind in [("route", "span"), ("retrieve", "retriever"), ("cache.miss", "event")]:
        s = spans[name]
        assert s.parent.span_id == root.context.span_id and s.context.trace_id == root.context.trace_id and s.attributes["langfuse.observation.type"] == kind
    route = spans["route"].attributes
    assert route["langfuse.observation.input"] == "q" and json.loads(route["langfuse.observation.output"]) == {"m": 1}
    assert meta(spans["route"], "extra") == 1 and meta(spans["route"], "other") == 2 and meta(spans["cache.miss"], "k") == 1
    assert spans["cache.miss"].start_time == spans["cache.miss"].end_time  # an event is a zero-length span: one observation, as before


def test_the_generation_carries_genai_semantic_conventions_and_langfuse_usage_model_cost_and_first_token():
    tracer, mem = otel()
    t = tracer.start_request("rag.request")
    first = datetime(2026, 11, 22, tzinfo=timezone.utc)
    with t.generation("generation", model="gemini-3.5-flash-lite", provider="gemini", input={"source_ids": ["c1"]},
                      model_parameters={"temperature": 0.0, "max_tokens": 1024}, prompt_name="rag-answer-long") as g:
        g.update(completion_start_time=first, usage={"prompt_tokens": 120, "completion_tokens": 30}, output="ans", cost_details={"input": 0.1, "output": 0.2},
                 finish_reason="stop", model="gemini-3.5-flash-lite", attempts=1, level="DEFAULT")
    t.end()
    gen = finished(mem)["generation"]
    a = gen.attributes
    assert (a["gen_ai.operation.name"], a["gen_ai.provider.name"], a["gen_ai.request.model"], a["gen_ai.request.temperature"], a["gen_ai.request.max_tokens"]) == \
        ("chat", "gemini", "gemini-3.5-flash-lite", 0.0, 1024)
    assert (a["gen_ai.response.model"], tuple(a["gen_ai.response.finish_reasons"]), a["gen_ai.usage.input_tokens"], a["gen_ai.usage.output_tokens"]) == \
        ("gemini-3.5-flash-lite", ("stop",), 120, 30)
    assert a["langfuse.observation.type"] == "generation" and a["langfuse.observation.model.name"] == "gemini-3.5-flash-lite"
    assert json.loads(a["langfuse.observation.usage_details"]) == {"input": 120, "output": 30} and json.loads(a["langfuse.observation.cost_details"]) == {"input": 0.1, "output": 0.2}
    assert json.loads(a["langfuse.observation.completion_start_time"]) == "2026-11-22T00:00:00Z" and a["langfuse.observation.output"] == "ans"
    assert meta(gen, "finish_reason") == "stop" and meta(gen, "attempts") == 1 and meta(gen, "prompt_name") == "rag-answer-long"
    assert "langfuse.observation.prompt.name" not in a  # a prompt is linked only when the Langfuse prompt object is passed


def test_a_langfuse_prompt_is_linked_by_name_and_version_but_a_fallback_prompt_is_not():
    tracer, mem = otel()
    t = tracer.start_request("r")
    with t.generation("generation", model="m", prompt=SimpleNamespace(name="rag-answer-long", version=4, is_fallback=False)):
        pass
    with t.generation("fallback", model="m", prompt=SimpleNamespace(name="rag-answer-long", version=4, is_fallback=True)):
        pass
    t.end()
    spans = finished(mem)
    assert spans["generation"].attributes["langfuse.observation.prompt.name"] == "rag-answer-long" and spans["generation"].attributes["langfuse.observation.prompt.version"] == 4
    assert "langfuse.observation.prompt.name" not in spans["fallback"].attributes


def test_missing_usage_is_omitted_and_usage_on_a_plain_span_is_not_sent_as_usage():
    tracer, mem = otel()
    t = tracer.start_request("r")
    with t.generation("generation", model="m") as g:
        g.update(usage={"prompt_tokens": None, "completion_tokens": 5})
    with t.span("route") as h:
        h.update(usage={"prompt_tokens": 1, "completion_tokens": 2})
    t.end()
    spans = finished(mem)
    assert json.loads(spans["generation"].attributes["langfuse.observation.usage_details"]) == {"output": 5} and "gen_ai.usage.input_tokens" not in spans["generation"].attributes
    assert not any("usage" in k for k in spans["route"].attributes)


def test_set_error_marks_the_root_with_level_error_a_bounded_message_and_an_error_status():
    from opentelemetry.trace import StatusCode

    tracer, mem = otel()
    t = tracer.start_request("r")
    t.set_error("invalid_question", "x" * 2000)
    t.end()
    root = finished(mem)["r"]
    msg = root.attributes["langfuse.observation.status_message"]
    assert root.attributes["langfuse.observation.level"] == "ERROR" and meta(root, "error_code") == "invalid_question" and len(msg) == 500
    assert msg.startswith("invalid_question: ") and root.status.status_code == StatusCode.ERROR


def test_set_error_can_mark_the_root_as_a_warning_for_degraded_requests_without_an_otel_error_status():
    from opentelemetry.trace import StatusCode

    tracer, mem = otel()
    t = tracer.start_request("r")
    t.set_error("all_providers_failed", "m", "WARNING")
    t.end()
    root = finished(mem)["r"]
    assert root.attributes["langfuse.observation.level"] == "WARNING" and root.status.status_code == StatusCode.UNSET
    fake = FakeTracer()
    trace = fake.start_request("r")
    trace.set_error("c", "m", "WARNING")
    assert trace.observations[0]["level"] == "WARNING"


def test_events_carry_level_and_message():
    tracer, mem = otel()
    t = tracer.start_request("r")
    t.event("prompt.fallback", level="WARNING", status_message="no_keys", metadata={"prompt": "p"})
    t.end()
    ev = finished(mem)["prompt.fallback"]
    assert (ev.attributes["langfuse.observation.level"], ev.attributes["langfuse.observation.status_message"], meta(ev, "prompt")) == ("WARNING", "no_keys", "p")


def test_the_attributes_match_what_the_installed_langfuse_sdk_would_send():
    """The names and the encoding are copied from the Langfuse SDK; if the SDK changes them, this fails (the SDK is used here only as a reference)."""
    from langfuse._client.attributes import LangfuseOtelSpanAttributes as A
    from langfuse._client.attributes import create_generation_attributes, create_span_attributes

    import rag_core.otel_tracing as ot

    for ours, theirs in [(ot.LF_TRACE_NAME, A.TRACE_NAME), (ot.LF_SESSION_ID, A.TRACE_SESSION_ID), (ot.LF_TYPE, A.OBSERVATION_TYPE), (ot.LF_METADATA, A.OBSERVATION_METADATA),
                         (ot.LF_LEVEL, A.OBSERVATION_LEVEL), (ot.LF_STATUS_MESSAGE, A.OBSERVATION_STATUS_MESSAGE), (ot.LF_INPUT, A.OBSERVATION_INPUT),
                         (ot.LF_OUTPUT, A.OBSERVATION_OUTPUT), (ot.LF_COMPLETION_START, A.OBSERVATION_COMPLETION_START_TIME), (ot.LF_MODEL, A.OBSERVATION_MODEL),
                         (ot.LF_MODEL_PARAMETERS, A.OBSERVATION_MODEL_PARAMETERS), (ot.LF_USAGE, A.OBSERVATION_USAGE_DETAILS), (ot.LF_COST, A.OBSERVATION_COST_DETAILS),
                         (ot.LF_PROMPT_NAME, A.OBSERVATION_PROMPT_NAME), (ot.LF_PROMPT_VERSION, A.OBSERVATION_PROMPT_VERSION)]:
        assert ours == theirs
    first = datetime(2026, 11, 22, 8, 30, tzinfo=timezone.utc)
    prompt = SimpleNamespace(name="rag-answer-long", version=3, is_fallback=False)
    common = dict(input={"source_ids": ["c1"], "question": "q\u00e9"}, output={"a": [1, 2]}, level="WARNING", status_message="m",
                  metadata={"mode": "long", "n": 3, "flag": True, "items": [1, "a"], "skip": None, "name": "caf\u00e9", "when": first})
    gen = dict(model="gemini-3.5-flash-lite", model_parameters={"temperature": 0.0, "max_tokens": 1024}, usage_details={"input": 12, "output": 3},
               cost_details={"input": 0.5}, completion_start_time=first)
    ours = ot.observation_attributes(as_type="generation", prompt_name=prompt.name, prompt_version=prompt.version, **common, **gen)
    assert ours == create_generation_attributes(prompt=prompt, **common, **gen)
    assert ot.observation_attributes(as_type="span", **common) == create_span_attributes(observation_type="span", **common)
    assert ot.observation_attributes(as_type="event", metadata={"k": "v"}) == create_span_attributes(observation_type="event", metadata={"k": "v"})


def test_flush_forces_the_export_and_gives_up_after_the_bounded_timeout():
    import time

    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

    class Slow(SpanExporter):
        def __init__(self, delay):
            self.delay, self.exported = delay, 0

        def export(self, spans):
            time.sleep(self.delay)
            self.exported += len(spans)
            return SpanExportResult.SUCCESS

    quick = Slow(0.0)
    tracer = build_tracer(TracingConfig(enabled=True, flush_timeout_seconds=3), exporters=[quick], rng=lambda: 0.0)  # batched, as in production
    tracer.start_request("x").end()
    tracer.flush()
    assert quick.exported == 1 and tracer.stats.flush_timeouts == 0
    slow = Slow(0.6)
    tracer = build_tracer(TracingConfig(enabled=True), exporters=[slow], rng=lambda: 0.0)
    tracer.start_request("x").end()
    tracer.flush(timeout_s=0.05)
    assert tracer.stats.flush_timeouts == 1  # returned after ~50 ms although the export needs 600 ms


def test_a_flush_that_raises_is_counted_not_raised():
    import time

    from opentelemetry import trace as otel_trace

    from rag_core.otel_tracing import OtelTracer

    class BadProvider:
        def get_tracer(self, name):
            return otel_trace.NoOpTracer()

        def force_flush(self, timeout_millis):
            raise RuntimeError("network")

    tracer = OtelTracer(TracingConfig(enabled=True), provider=BadProvider())
    tracer.flush()
    time.sleep(0.05)
    assert tracer.stats.errors == 1


def test_an_exporter_that_fails_never_reaches_the_answer():
    from opentelemetry.sdk.trace.export import SpanExporter

    from tests.unit.test_rag_core_service import done_of, events, make

    class Broken(SpanExporter):
        def export(self, spans):
            raise ConnectionError("backend down")

    tracer = build_tracer(TracingConfig(enabled=True), exporters=[Broken()], simple=True, rng=lambda: 0.0)
    svc, *_ = make(tracer=tracer, tracing=TracingConfig(enabled=True))
    r = done_of(events(svc))
    assert r.answer and not r.degraded and r.trace_id and len(r.trace_id) == 32


def test_the_real_tracer_behind_the_safe_wrapper_survives_an_sdk_that_raises_at_every_step():
    from rag_core.otel_tracing import OtelTracer

    class Boom:
        def start_span(self, *a, **k):
            raise RuntimeError("sdk broke")

    class Provider:
        def get_tracer(self, name):
            return Boom()

        def force_flush(self, timeout_millis):
            raise RuntimeError("sdk broke")

    tracer = SafeTracer(OtelTracer(TracingConfig(enabled=True), provider=Provider(), rng=lambda: 0.0))
    t = tracer.start_request("x", input=1)
    with t.span("a"):
        pass
    t.end(output="o")
    assert t.trace_id is None and tracer.stats.errors >= 1


def test_the_service_span_tree_is_the_same_under_opentelemetry_and_stays_within_eight_observations():
    from tests.unit.test_rag_core_service import done_of, events, make

    tracer, mem = otel()
    svc, *_ = make(tracer=tracer, tracing=TracingConfig(enabled=True))
    r = done_of(events(svc, session_id="s-1"))
    spans = mem.get_finished_spans()
    names = sorted(s.name for s in spans)
    assert names == sorted(["rag.request", "route", "cache.miss", "retrieve", "context", "generation"]) and len(spans) <= 8
    assert {format(s.context.trace_id, "032x") for s in spans} == {r.trace_id}
    root = finished(mem)["rag.request"]
    assert root.attributes["session.id"] == "s-1" and finished(mem)["retrieve"].attributes["langfuse.observation.type"] == "retriever"
