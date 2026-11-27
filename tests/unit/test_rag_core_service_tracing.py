"""RagService + tracing: span tree and attributes per scenario, fail-open, masking, one root per request. Fake tracer, fake providers: no SDK, no network."""

import json
import re
from datetime import datetime

import pytest

from rag_core.config import PromptsConfig, RagConfig, TracingConfig
from rag_core.errors import InvalidQuestion
from rag_core.prompt_store import LangfusePromptStore, LocalPromptStore
from rag_core.prompts import PROMPT_NAMES, PROMPT_VERSION
from rag_core.tracing import FakeTracer
from tests.unit.rag_core_fakes import ABSTAIN_JSON, FakePrompt, FakePromptClient, FakeRetriever, answer_json, chat_messages, fail, make_hit, ok
from tests.unit.test_rag_core_service import Q, done_of, events, make

KEY = "gsk_abcdefghijklmnop1234567890"


def run(svc, tracer, q=Q, **kw):
    evs = events(svc, q, **kw)
    return done_of(evs), tracer.traces[-1], evs


def tracing(**kw):
    return TracingConfig(enabled=True, **kw)


# ---- normal answer ---------------------------------------------------------------------------------------------------------------

def test_a_normal_answer_has_the_documented_span_tree_with_one_root_and_at_most_eight_observations():
    tracer = FakeTracer()
    svc, *_ = make(tracer=tracer, tracing=tracing())
    result, trace, _ = run(svc, tracer)
    assert trace.names() == ["rag.request", "route", "cache.miss", "retrieve", "context", "generation"]
    assert len(trace.observations) <= 8 and len(tracer.traces) == 1  # one trace per request, bounded unit usage
    assert [o["parent"] for o in trace.observations[1:]] == ["rag.request"] * 5 and all(o["ended"] for o in trace.observations)
    assert result.trace_id == trace.trace_id == "fake-trace-1"


def test_the_root_records_the_masked_question_the_session_and_request_metadata_then_the_masked_answer_and_outcome():
    tracer = FakeTracer()
    svc, *_ = make(tracer=tracer, tracing=tracing(), index_version="idx-9")
    result, trace, _ = run(svc, tracer, session_id="sess-7", q="Why does it work? ask bob@corp.example.com")
    root = trace.observations[0]["attrs"]
    assert root["input"] == "Why does it work? ask [EMAIL]" and root["session_id"] == "sess-7"
    assert root["mode"] == "long" and root["top_k"] == 5 and root["index_version"] == "idx-9" and root["capture"] == "masked" and root["rerank"] is False
    assert root["output"] == result.answer and root["provider"] == "groq" and root["model"] == "model-a" and root["cache_hit"] is False and root["degraded"] is False
    assert root["citations"] == 1 and root["invalid_citations"] == 0 and root["finish_reason"] == "stop" and root["error_codes"] == []
    assert root["prompt"]["name"] == PROMPT_NAMES["long"] and root["prompt"]["source"] == "local" and root["prompt"]["version"] == PROMPT_VERSION


def test_the_generation_carries_model_provider_usage_first_token_time_the_prompt_identity_and_no_passage_text():
    tracer = FakeTracer()
    svc, retriever, a, *_ = make(tracer=tracer, tracing=tracing())
    result, trace, _ = run(svc, tracer)
    gen = trace.get("generation")
    assert gen["type"] == "generation" and gen["attrs"]["model"] == "model-a"
    attrs = gen["attrs"]
    assert attrs["provider"] == "groq" and attrs["prompt_name"] == PROMPT_NAMES["long"] and attrs["prompt_version"] == PROMPT_VERSION and attrs["prompt_source"] == "local"
    assert len(attrs["prompt_sha256"]) == 64 and attrs["model_parameters"] == {"temperature": RagConfig().temperature, "max_tokens": RagConfig().max_output_tokens, "reasoning_effort": None,
                                                                     "response_format": "claims-1"}  # the fake gateway has no effort
    assert attrs["usage"] == {"prompt_tokens": 11, "completion_tokens": 7} and attrs["finish_reason"] == "stop" and isinstance(attrs["completion_start_time"], datetime)
    assert attrs["attempts"] == [{"provider": "groq", "attempt": 1, "outcome": "ok", "error_kind": None, "latency_ms": attrs["attempts"][0]["latency_ms"]}]
    assert attrs["prompt"] is None  # a local prompt has no Langfuse object to link
    assert attrs["input"] == {"source_ids": [s.chunk_id for s in result.used_sources], "question": Q}
    for hit in retriever.hits:  # masked capture: no passage text anywhere in the trace
        assert hit.text not in json.dumps(trace.observations, default=str)


def test_no_cost_without_configured_prices_and_cost_when_prices_exist():
    tracer = FakeTracer()
    svc, *_ = make(tracer=tracer, tracing=tracing())
    assert "cost_details" not in run(svc, tracer)[1].get("generation")["attrs"]
    tracer = FakeTracer()
    svc, *_ = make(tracer=tracer, tracing=tracing(), cfg=RagConfig(prices={"model-a": {"input": 1.0, "output": 2.0}}))
    cost = run(svc, tracer)[1].get("generation")["attrs"]["cost_details"]
    assert cost == {"input": 11 / 1e6, "output": 14 / 1e6}


def test_the_retrieve_span_is_a_retriever_with_chunk_ids_and_the_context_span_lists_what_the_model_saw():
    tracer = FakeTracer()
    svc, *_ = make(tracer=tracer, tracing=tracing())
    result, trace, _ = run(svc, tracer)
    retrieve = trace.get("retrieve")
    assert retrieve["type"] == "retriever" and retrieve["attrs"]["output"]["chunk_ids"] == [f"c{i}" for i in range(1, 6)]
    ctx = trace.get("context")["attrs"]["output"]
    assert ctx["used_chunk_ids"] == [s.chunk_id for s in result.used_sources] and all("text" not in s for s in ctx["sources"]) and ctx["dropped"] == 0


def test_full_capture_mode_records_the_compiled_prompt_and_passages_but_still_redacts_keys():
    tracer = FakeTracer()
    hits = [make_hit(1, text=f"passage with {KEY} inside"), make_hit(2)]
    svc, *_ = make(tracer=tracer, tracing=tracing(capture="full"), hits=hits)
    _, trace, _ = run(svc, tracer, q=f"question {KEY}")
    blob = json.dumps(trace.observations, default=str)
    assert KEY not in blob and "[REDACTED_KEY]" in blob
    prompt_text = trace.get("generation")["attrs"]["input"][1]["content"]
    assert '<source id="S1"' in prompt_text and "passage with [REDACTED_KEY] inside" in prompt_text


def test_capture_none_records_no_question_or_answer_text():
    tracer = FakeTracer()
    svc, *_ = make(tracer=tracer, tracing=tracing(capture="none"))
    result, trace, _ = run(svc, tracer)
    blob = json.dumps(trace.observations, default=str)
    assert Q not in blob and result.answer not in blob and trace.observations[0]["attrs"]["input"] is None and trace.observations[0]["attrs"]["output"] is None


def test_no_key_like_string_reaches_the_tracer_in_any_mode_even_from_a_provider_answer():
    for mode in ("masked", "full", "none"):
        tracer = FakeTracer()
        svc, *_ = make([[answer_json((f"The key is {KEY}.", [("S1", KEY)])), *ok()[-1:]]], tracer=tracer, tracing=tracing(capture=mode),
                       hits=[make_hit(1, text=KEY)])
        _, trace, _ = run(svc, tracer, q=f"my key is {KEY}")
        assert KEY not in json.dumps(trace.observations, default=str), mode


# ---- cache hit -------------------------------------------------------------------------------------------------------------------

def test_a_cache_hit_is_traced_with_a_cache_hit_event_no_retrieval_and_no_generation():
    tracer = FakeTracer()
    svc, retriever, a, *_ = make(tracer=tracer, tracing=tracing())
    first, t1, _ = run(svc, tracer)
    second, t2, _ = run(svc, tracer)
    assert t1.names()[2] == "cache.miss" and t2.names() == ["rag.request", "route", "cache.hit"]
    assert second.cached and t2.observations[0]["attrs"]["cache_hit"] is True and t2.trace_id != t1.trace_id and second.trace_id == t2.trace_id
    assert len(retriever.calls) == 1 and len(a.calls) == 1


# ---- failures --------------------------------------------------------------------------------------------------------------------

def test_every_provider_failing_is_a_degraded_trace_with_an_error_generation_and_an_error_root():
    tracer = FakeTracer()
    svc, *_ = make([[fail("server", 500)]], [[fail("auth", 401)]], tracer=tracer, tracing=tracing())
    result, trace, _ = run(svc, tracer)
    gen, root = trace.get("generation"), trace.observations[0]
    assert result.degraded and gen["level"] == "ERROR" and gen["status_message"] == "all_providers_failed" and len(gen["attrs"]["attempts"]) >= 3
    assert root["level"] == "WARNING" and root["status_message"].startswith("all_providers_failed") and root["attrs"]["degraded"] is True  # degraded is a WARNING, not an ERROR
    assert root["attrs"]["error_codes"] == ["all_providers_failed"] and all(o["ended"] for o in trace.observations)


def test_a_mid_stream_failure_marks_the_generation_and_root_as_errors_and_records_the_partial_text():
    tracer = FakeTracer()
    svc, *_ = make([['{"status": "answered", "claims": [{"text": "Attention weighs ', 'the tok', fail("connection")]], tracer=tracer, tracing=tracing())
    result, trace, _ = run(svc, tracer)
    gen = trace.get("generation")
    assert result.partial and gen["level"] == "ERROR" and gen["status_message"] == "mid_stream" and gen["attrs"]["output"] == "Attention weighs the tok"
    assert isinstance(gen["attrs"]["completion_start_time"], datetime) and trace.observations[0]["attrs"]["error_codes"] == ["mid_stream"]
    assert trace.observations[0]["level"] == "ERROR" and trace.observations[0]["attrs"]["partial"] is True


def test_a_retrieval_failure_marks_the_retrieve_span_and_has_no_context_or_generation_span():
    tracer = FakeTracer()
    svc, *_ = make(retriever=FakeRetriever(error=RuntimeError("path secret")), tracer=tracer, tracing=tracing())
    result, trace, _ = run(svc, tracer)
    assert trace.names() == ["rag.request", "route", "cache.miss", "retrieve"] and trace.get("retrieve")["level"] == "ERROR"
    assert trace.observations[0]["level"] == "WARNING"  # retrieval failed, the user still gets a degraded result
    assert "secret" not in json.dumps(trace.observations, default=str) and trace.observations[0]["attrs"]["error_codes"] == ["retrieval_failed"]


def test_an_internal_error_in_generation_is_traced_by_class_name_only():
    tracer = FakeTracer()
    svc, _, _, _, _, gw = make(tracer=tracer, tracing=tracing())

    def boom(*a, **k):
        raise KeyError("detail-not-for-traces")
        yield

    gw.stream = boom
    _, trace, _ = run(svc, tracer)
    assert trace.get("generation")["status_message"] == "internal:KeyError" and "detail-not-for-traces" not in json.dumps(trace.observations, default=str)


def test_an_invalid_question_creates_a_rejected_trace_and_still_raises():
    tracer = FakeTracer()
    svc, retriever, a, *_ = make(tracer=tracer, tracing=tracing())
    with pytest.raises(InvalidQuestion):
        svc.answer_stream("x" * 501, session_id="s1")
    (trace,) = tracer.traces
    root = trace.observations[0]
    assert trace.names() == ["rag.request"] and root["level"] == "ERROR" and root["attrs"]["rejected"] is True and root["attrs"]["error_code"] == "too_long" and root["ended"]
    assert root["attrs"]["session_id"] == "s1" and retriever.calls == [] and a.calls == []
    with pytest.raises(InvalidQuestion):
        svc.answer_stream(None)  # a non-string input is traced without text
    assert tracer.traces[-1].observations[0]["attrs"]["input"] is None


def test_small_talk_is_traced_as_a_route_only_trace():
    tracer = FakeTracer()
    svc, retriever, a, *_ = make(tracer=tracer, tracing=tracing())
    result, trace, _ = run(svc, tracer, q="hello")
    assert result.mode == "direct" and trace.names() == ["rag.request", "route"] and trace.get("route")["attrs"]["output"] == "direct"
    assert retriever.calls == [] and trace.observations[0]["attrs"]["provider"] == "rules"


def test_an_abstention_and_a_no_results_answer_are_traced_and_not_errors():
    tracer = FakeTracer()
    svc, *_ = make([[ABSTAIN_JSON, *ok()[-1:]]], tracer=tracer, tracing=tracing())
    result, trace, _ = run(svc, tracer)
    assert result.abstained and trace.observations[0]["attrs"]["abstained"] is True and trace.observations[0]["level"] == "DEFAULT"
    assert trace.get("generation")["attrs"]["abstained"] is True
    tracer = FakeTracer()
    svc, *_ = make(hits=[], tracer=tracer, tracing=tracing())
    r2, t2, _ = run(svc, tracer)
    assert r2.abstained and t2.names() == ["rag.request", "route", "cache.miss", "retrieve", "context"]


# ---- prompt fallback events ------------------------------------------------------------------------------------------------------

def test_a_langfuse_prompt_fetch_failure_adds_a_warning_event_and_the_answer_still_succeeds():
    tracer = FakeTracer()
    prompts = LangfusePromptStore(PromptsConfig(mode="langfuse"), client_factory=lambda: FakePromptClient(error=TimeoutError("slow")))
    svc, *_ = make(tracer=tracer, tracing=tracing(), prompts=prompts)
    result, trace, _ = run(svc, tracer)
    ev = trace.get("prompt.fallback")
    assert ev["level"] == "WARNING" and ev["status_message"] == "fetch_failed:TimeoutError" and ev["attrs"]["metadata"]["kind"] == "fetch_error"
    assert result.answer and not result.degraded and result.prompt.source == "local_fallback" and trace.names().index("prompt.fallback") < trace.names().index("cache.miss")
    assert trace.get("generation")["attrs"]["prompt_source"] == "local_fallback"


def test_an_invalid_langfuse_prompt_is_an_error_level_event():
    tracer = FakeTracer()
    from rag_core.config import PromptsConfig

    bad = FakePromptClient(FakePrompt(chat_messages("no sentinel, no citations")))
    svc, *_ = make(tracer=tracer, tracing=tracing(), prompts=LangfusePromptStore(PromptsConfig(mode="langfuse"), client_factory=lambda: bad))
    result, trace, _ = run(svc, tracer)
    ev = trace.get("prompt.fallback")
    assert ev["level"] == "ERROR" and ev["status_message"].startswith("invalid:") and result.answer


def test_a_good_langfuse_prompt_links_the_generation_to_the_prompt_object_and_adds_no_fallback_event():
    from rag_core.config import PromptsConfig
    from rag_core.prompts import QUOTE_INSTRUCTION, STATUS_INSUFFICIENT

    prompt = FakePrompt(chat_messages(f"Answer only from sources. Quote each source in {QUOTE_INSTRUCTION}. Else {STATUS_INSUFFICIENT}."), version=4)
    tracer = FakeTracer()
    svc, _, a, *_ = make(tracer=tracer, tracing=tracing(), prompts=LangfusePromptStore(PromptsConfig(mode="langfuse"), client_factory=lambda: FakePromptClient(prompt)))
    result, trace, _ = run(svc, tracer)
    attrs = trace.get("generation")["attrs"]
    assert not trace.has("prompt.fallback") and attrs["prompt"] is prompt and attrs["prompt_version"] == "4" and attrs["prompt_source"] == "langfuse"
    assert a.calls[0]["messages"][0]["content"].startswith("Answer only from sources") and result.prompt.source == "langfuse"


def test_a_prompt_store_that_raises_is_handled_like_a_fallback():
    class Broken(LocalPromptStore):
        def resolve(self, name):
            raise RuntimeError("store bug")

    tracer = FakeTracer()
    svc, *_ = make(tracer=tracer, tracing=tracing(), prompts=Broken())
    result, trace, _ = run(svc, tracer)
    assert result.answer and result.prompt.fallback_reason == "resolve_failed:RuntimeError" and trace.get("prompt.fallback")["level"] == "WARNING"


# ---- fail open -------------------------------------------------------------------------------------------------------------------

SCENARIOS = {
    "normal": dict(),
    "degraded": dict(primary=[[fail("server", 500)]], secondary=[[fail("auth", 401)]]),
    "midstream": dict(primary=[["part ", fail("connection")]]),
    "retrieval": dict(retriever=FakeRetriever(error=RuntimeError("x"))),
}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_a_tracer_that_raises_on_every_call_leaves_events_and_the_result_unchanged(name):
    kw = SCENARIOS[name]
    plain, *_ = make(kw.get("primary"), kw.get("secondary"), retriever=kw.get("retriever"))
    broken, *_ = make(kw.get("primary"), kw.get("secondary"), retriever=kw.get("retriever"), tracer=FakeTracer(raise_on_call=True), tracing=tracing())
    want, got = events(plain), events(broken)

    def norm(evs):
        out = []
        for e in evs:
            if e["type"] == "done":
                d = e["result"].model_dump(mode="json")
                for k in ("latency_ms", "trace_id", "attempts", "request_id"):  # unique per request by design
                    d.pop(k)
                out.append(("done", json.dumps(d, sort_keys=True, default=str)))
            else:
                out.append((e["type"], json.dumps(e.get("text") or e.get("error") or [s.chunk_id for s in e.get("sources", [])], default=str)))
        return out

    assert norm(got) == norm(want)
    assert done_of(got).trace_id is None and broken.tracer.stats.errors >= 1


def test_invalid_input_with_a_broken_tracer_still_raises_the_typed_error_not_the_tracer_error():
    svc, *_ = make(tracer=FakeTracer(raise_on_call=True), tracing=tracing())
    with pytest.raises(InvalidQuestion):
        svc.answer_stream("")


def test_flush_traces_calls_the_tracer_and_never_raises():
    tracer = FakeTracer()
    svc, *_ = make(tracer=tracer, tracing=tracing())
    svc.flush_traces()
    assert tracer.flushes == 1
    broken, *_ = make(tracer=FakeTracer(raise_on_call=True), tracing=tracing())
    broken.flush_traces()


def test_an_aborted_consumer_still_ends_the_root_and_marks_it_aborted():
    tracer = FakeTracer()
    reply = answer_json(("First part.", ["S1"]), ("Second part.", ["S1"]))
    svc, *_ = make([[reply[:40], reply[40:]]], tracer=tracer, tracing=tracing())
    stream = svc.answer_stream(Q)
    for ev in stream:
        if ev["type"] == "token":
            break
    stream.close()
    root = tracer.traces[0].observations[0]
    assert root["ended"] and root["attrs"]["aborted"] is True and all(o["ended"] for o in tracer.traces[0].observations)


def test_the_default_service_has_no_tracer_and_no_trace_id_and_behaves_as_before():
    svc, *_ = make()
    result = done_of(events(svc))
    assert result.trace_id is None and result.answer and svc.tracer.enabled is False


# ---- evaluation path -------------------------------------------------------------------------------------------------------------

def test_answer_short_is_untraced_by_default_and_traced_only_when_trace_evaluation_is_on():
    tracer = FakeTracer()
    svc, *_ = make([ok("Yes")], tracer=tracer, tracing=tracing())
    r = svc.answer_short(Q)
    assert tracer.traces == [] and r.trace_id is None
    tracer = FakeTracer()
    svc, *_ = make([ok("Yes")], tracer=tracer, tracing=tracing(trace_evaluation=True))
    r = svc.answer_short(Q)
    assert len(tracer.traces) == 1 and tracer.traces[0].observations[0]["attrs"]["mode"] == "short" and r.trace_id == tracer.traces[0].trace_id
    assert tracer.traces[0].get("generation")["attrs"]["prompt_name"] == PROMPT_NAMES["short"]


def test_an_invalid_question_to_answer_short_is_traced_only_with_trace_evaluation():
    tracer = FakeTracer()
    svc, *_ = make(tracer=tracer, tracing=tracing())
    with pytest.raises(InvalidQuestion):
        svc.answer_short("")
    assert tracer.traces == []
    svc, *_ = make(tracer=tracer, tracing=tracing(trace_evaluation=True))
    with pytest.raises(InvalidQuestion):
        svc.answer_short("")
    assert len(tracer.traces) == 1


def test_describe_prompts_reports_both_prompts_for_an_evaluation_config():
    svc, *_ = make()
    d = svc.describe_prompts()
    assert set(d) == set(PROMPT_NAMES.values()) and all(re.fullmatch(r"[0-9a-f]{64}", v["sha256"]) for v in d.values())
