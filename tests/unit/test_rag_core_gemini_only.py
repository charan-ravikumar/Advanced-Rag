"""The answer path is Gemini only: build_gateway(RagConfig()) + RagService with a fake google-genai client. No network, no real key."""

from types import SimpleNamespace

import pytest
from google.genai import errors, types

from rag_core.cache import InMemoryTTLCache
from rag_core.config import RagConfig
from rag_core.gateway import GeminiProvider, build_gateway
from rag_core.service import DEGRADED_MESSAGE, RagService
from tests.unit.rag_core_fakes import FakeClock, FakeRetriever, answer_json
from tests.unit.test_rag_core_service import done_of, events

FAKE_KEY = "AIza_test_not_a_real_key_0123456789"
MODEL = "gemini-3.5-flash-lite"


def chunk(text=None, finish=None, usage=None):
    candidates = [SimpleNamespace(finish_reason=types.FinishReason[finish] if finish else None)]
    return SimpleNamespace(text=text, candidates=candidates, usage_metadata=usage)


class FakeGenai:
    """Just enough of ``genai.Client(...).models.generate_content_stream``."""

    def __init__(self, script):
        self.script, self.calls = script, []
        self.models = SimpleNamespace(generate_content_stream=self._stream)

    def _stream(self, **kw):
        self.calls.append(kw)
        step = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(step, BaseException):
            raise step
        return iter(step)


def gemini_service(script, monkeypatch, *, key=FAKE_KEY, cfg=None, **gateway_kw):
    cfg = cfg or RagConfig()
    if key is None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("GEMINI_API_KEY", key)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    gateway = build_gateway(cfg)
    client = FakeGenai(script)
    gateway.providers[0]._client_factory = lambda k: client
    clock = FakeClock()
    gateway._sleep, gateway._clock, gateway._rng = clock.sleep, clock, lambda: 1.0
    for name, value in gateway_kw.items():
        setattr(gateway, name, value)
    return RagService(FakeRetriever(), gateway, InMemoryTTLCache(clock=clock), cfg), gateway, client


def status_error(code):
    cls = errors.ServerError if code >= 500 else errors.ClientError
    return cls(code, {"error": {"code": code, "message": f"HTTP {code} {FAKE_KEY}", "status": "X"}}, None)


USAGE = SimpleNamespace(prompt_token_count=30, candidates_token_count=9)
REPLY = answer_json(("Attention weighs the tokens.", ["S1"]))
GOOD = [chunk(REPLY[:30]), chunk(REPLY[30:]), chunk(None, "STOP", USAGE)]


def test_the_default_path_builds_exactly_one_provider_gemini_with_the_configured_model():
    gw = build_gateway(RagConfig())
    assert [p.name for p in gw.providers] == ["gemini"] and isinstance(gw.providers[0], GeminiProvider)
    assert gw.primary_model == RagConfig().gemini_model == MODEL and gw.primary_reasoning_effort is None


def test_groq_is_not_a_provider_any_more():
    with pytest.raises(ValueError, match="unknown provider"):
        build_gateway(RagConfig(providers=["groq"]))


def test_a_gemini_only_service_answers_with_streamed_tokens_citations_usage_and_the_gemini_model(monkeypatch):
    svc, gw, client = gemini_service([GOOD], monkeypatch)
    evs = events(svc)
    r = done_of(evs)
    assert "".join(e["text"] for e in evs if e["type"] == "token") == r.answer == "Attention weighs the tokens. [S1]"
    assert (r.provider, r.model, r.finish_reason, r.citations, r.degraded) == ("gemini", MODEL, "stop", ["S1"], False)
    assert (r.usage.prompt_tokens, r.usage.completion_tokens) == (30, 9) and [a.provider for a in r.attempts] == ["gemini"]
    call = client.calls[0]
    assert call["model"] == MODEL and call["config"].system_instruction and "<source id=\"S1\"" in call["contents"][0].parts[0].text
    assert call["config"].response_mime_type == "application/json"  # structured output: the schema goes to Gemini with every long answer
    assert call["config"].response_json_schema["properties"]["claims"]["items"]["properties"]["citations"]["items"]["properties"]["source"]["enum"] == \
        ["S1", "S2", "S3", "S4", "S5"]


def test_a_short_answer_sends_no_response_schema_to_gemini(monkeypatch):
    svc, gw, client = gemini_service([[chunk("yes"), chunk(None, "STOP", USAGE)]], monkeypatch)
    assert svc.answer_short("Do they use attention?").short_answer == "yes"
    config = client.calls[0]["config"]
    assert config.response_mime_type is None and config.response_json_schema is None


def test_a_gemini_failure_gives_a_degraded_result_with_the_retrieved_passages_and_no_second_provider(monkeypatch):
    svc, gw, client = gemini_service([status_error(500)], monkeypatch, max_attempts=1)
    evs = events(svc)
    r = done_of(evs)
    assert r.degraded and r.answer == DEGRADED_MESSAGE and [e["error"].code for e in evs if e["type"] == "error"] == ["all_providers_failed"]
    assert len(r.retrieved_sources) == 5 and r.used_sources == [] and len(client.calls) == 1 and [a.provider for a in r.attempts] == ["gemini"]
    assert FAKE_KEY not in r.model_dump_json()  # the provider's error text (which contained the key) is scrubbed


def test_an_unknown_model_id_404_is_not_retried_and_degrades(monkeypatch):
    svc, gw, client = gemini_service([status_error(404)], monkeypatch)
    r = done_of(events(svc))
    assert r.degraded and len(client.calls) == 1  # a 4xx is not retried
    assert (r.attempts[0].outcome, r.attempts[0].error_kind) == ("error", "not_found")


def test_an_auth_error_is_not_retried_and_degrades_without_leaking_the_key(monkeypatch):
    svc, gw, client = gemini_service([status_error(401)], monkeypatch)
    r = done_of(events(svc))
    assert r.degraded and len(client.calls) == 1 and FAKE_KEY not in r.model_dump_json()


def test_a_transient_429_is_retried_once_then_succeeds(monkeypatch):
    svc, gw, client = gemini_service([status_error(429), GOOD], monkeypatch)
    r = done_of(events(svc))
    assert not r.degraded and r.citations == ["S1"] and len(client.calls) == 2 and [a.outcome for a in r.attempts] == ["error", "ok"]


def test_a_missing_key_degrades_instead_of_crashing_and_names_no_key(monkeypatch):
    svc, gw, client = gemini_service([GOOD], monkeypatch, key=None)
    r = done_of(events(svc))
    assert r.degraded and client.calls == [] and r.errors[0].code == "all_providers_failed"


def test_a_stream_that_breaks_after_tokens_is_a_mid_stream_error_not_a_second_provider(monkeypatch):
    def broken():
        yield chunk('{"status": "answered", "claims": [{"text": "Attention weighs ')
        raise ConnectionResetError("reset")

    svc, gw, client = gemini_service([broken()], monkeypatch)
    r = done_of(events(svc))
    assert r.partial and r.answer == "Attention weighs" and r.errors[0].code == "mid_stream" and len(client.calls) == 1 and not r.degraded
    assert len(svc.cache) == 0


def test_the_cache_key_carries_the_gemini_model_so_a_model_change_misses(monkeypatch):
    svc, gw, client = gemini_service([GOOD], monkeypatch)
    done_of(events(svc))
    assert done_of(events(svc)).cached is True and len(client.calls) == 1
    svc2, gw2, client2 = gemini_service([GOOD], monkeypatch, cfg=RagConfig(gemini_model="another-model"))
    svc2.cache = svc.cache  # same cache, different primary model
    assert done_of(events(svc2)).cached is False and len(client2.calls) == 1


def test_thought_tokens_are_part_of_the_completion_tokens_and_reported_separately(monkeypatch):
    usage = SimpleNamespace(prompt_token_count=30, candidates_token_count=9, thoughts_token_count=6)
    svc, gw, client = gemini_service([[chunk("Yes."), chunk(None, "STOP", usage)]], monkeypatch)
    r = done_of(events(svc))
    assert (r.usage.prompt_tokens, r.usage.completion_tokens, r.usage.reasoning_tokens) == (30, 15, 6)


def test_a_usage_without_thought_tokens_gives_none_for_reasoning_tokens(monkeypatch):
    svc, gw, client = gemini_service([GOOD], monkeypatch)
    assert done_of(events(svc)).usage.reasoning_tokens is None


def test_the_thinking_budget_is_passed_to_the_api_only_when_configured(monkeypatch):
    svc, gw, client = gemini_service([GOOD], monkeypatch)
    done_of(events(svc))
    assert client.calls[0]["config"].thinking_config is None
    svc2, gw2, client2 = gemini_service([GOOD], monkeypatch, cfg=RagConfig(gemini_thinking_budget=0))
    done_of(events(svc2))
    assert client2.calls[0]["config"].thinking_config.thinking_budget == 0


def test_config_yaml_uses_gemini_3_5_flash_lite_for_the_answers_and_gemini_3_8_flash_for_the_judge_and_invents_no_prices():
    from config import cfg

    gen = RagConfig.from_config(cfg)
    assert gen.providers == ["gemini"] and gen.gemini_model == MODEL and gen.reasoning_effort is None and gen.prices == {}
    assert gen.judge["provider"] == "gemini" and gen.judge["model"] == "gemini-3.8-flash" != MODEL
    assert gen.judge["thinking_budget"] == 1024 and gen.judge["max_output_tokens"] == 4096 and gen.judge["temperature"] == 0.0
    assert RagConfig().judge == gen.judge and RagConfig().gemini_model == gen.gemini_model  # the code defaults equal config.yaml
