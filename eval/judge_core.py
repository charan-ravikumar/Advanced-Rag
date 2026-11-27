"""The cached, quota-aware judge for the RAGAS scoring (EXPLORATORY: Phase 4 minimal scope). PURE: no ragas, no google-genai, no network at import or in the tests.

``CachingJudge`` is what the RAGAS metrics see as their LLM (through ``eval/ragas_adapter.py``): ``agenerate(prompt, response_model)`` returns the parsed object.
* EVERY call is cached on disk, keyed by the sha256 of (judge model, temperature, thinking budget, output limit, API version, METRIC NAME, ragas version, the prompt text and
  the JSON schema of the expected output). Changing any of them is a miss, so a cached verdict can never be served for another judge, metric, ragas version or prompt.
* Only a reply that parses into the expected model is cached. An empty, truncated or invalid reply is retried a few times and then raised as ``JudgeError``; it is never cached.
* A per-minute 429 (``JudgeRateLimited``) is waited out (bounded); a per-day 429 or the session call budget raises ``JudgeQuotaExhausted``: the caller exits cleanly ("quota
  exhausted, resume later") and a re-run reads everything finished from the cache.
* Token usage (prompt, output, thinking) is accumulated per metric from the replies' own usage metadata.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable

from eval.disk_cache import DiskCache

CACHE_SCHEMA = 1


@dataclass(frozen=True)
class JudgeSettings:
    model: str
    temperature: float = 0.0
    thinking_budget: int = 1024
    max_output_tokens: int = 4096
    api_version: str = "v1beta"

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_config(cls, judge: dict) -> "JudgeSettings":
        return cls(model=judge["model"], temperature=float(judge.get("temperature", 0.0)), thinking_budget=int(judge["thinking_budget"]),
                   max_output_tokens=int(judge["max_output_tokens"]))


class JudgeQuotaExhausted(Exception):
    """Stop the session cleanly: a daily quota or the session call budget is used up. Everything finished is cached."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class JudgeRateLimited(Exception):
    """A per-minute 429: wait ``retry_after_s`` and try again."""

    def __init__(self, retry_after_s: float | None = None):
        super().__init__("rate limited")
        self.retry_after_s = retry_after_s


class JudgeError(Exception):
    """The judge could not produce a usable reply (empty, truncated, invalid, or a non-retryable error). Class name and a short reason only."""


@dataclass
class RawJudgeResult:
    text: str
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None
    finish_reason: str | None = None


RawJudge = Callable[[str, Any], Awaitable[RawJudgeResult]]


def judge_cache_key(settings: JudgeSettings, metric: str, ragas_version: str, prompt: str, response_model: Any) -> str:
    payload = {"schema": CACHE_SCHEMA, "judge": settings.as_dict(), "metric": metric, "ragas": ragas_version, "prompt": prompt,
               "response_schema": json.dumps(response_model.model_json_schema(), sort_keys=True)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass
class MetricUsage:
    calls: int = 0  # real network calls (a retry counts)
    cache_hits: int = 0
    retries: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0


@dataclass
class JudgeStats:
    """Shared by every CachingJudge of one session: per-metric usage and the session-wide call budget."""

    max_calls: int | None = None
    by_metric: dict[str, MetricUsage] = field(default_factory=dict)

    def usage(self, metric: str) -> MetricUsage:
        return self.by_metric.setdefault(metric, MetricUsage())

    @property
    def total_calls(self) -> int:
        return sum(u.calls for u in self.by_metric.values())

    def as_dict(self) -> dict:
        return {m: asdict(u) for m, u in sorted(self.by_metric.items())} | {"_total_network_calls": self.total_calls, "_max_calls": self.max_calls}


def classify_http_error(code: int | None, message: str) -> Exception:
    """Map a provider HTTP error to the judge's exception types. PURE (the adapter calls it with the SDK exception's code and text).

    429 with a per-day quota text is ``JudgeQuotaExhausted`` (a re-run tomorrow); any other 429 is a per-minute limit (``JudgeRateLimited`` with the
    delay the API names, if any); 5xx are transient (wait and retry); other 4xx are ``JudgeError`` (never retried here)."""
    text = (message or "").lower()
    if code == 429:
        if "per day" in text or "perday" in text.replace(" ", "") or "daily" in text:
            return JudgeQuotaExhausted("the judge's daily quota is exhausted (HTTP 429, per-day limit): resume tomorrow")
        match = re.search(r"retry(?:delay)?\D{0,12}?(\d+(?:\.\d+)?)\s*s", text)
        return JudgeRateLimited(float(match.group(1)) if match else None)
    if code is not None and 500 <= code < 600:
        return JudgeRateLimited(10.0)
    return JudgeError(f"HTTP {code}")


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def parse_reply(text: str, response_model: Any):
    """The reply parsed into ``response_model``; raises ``JudgeError`` on empty text or invalid JSON for the schema."""
    cleaned = _FENCE.sub("", text or "").strip()
    if not cleaned:
        raise JudgeError("empty reply")
    try:
        return response_model.model_validate_json(cleaned)
    except Exception as exc:  # noqa: BLE001: pydantic ValidationError or JSON error; the class name is enough
        raise JudgeError(f"reply does not match {response_model.__name__} ({type(exc).__name__})") from None


class CachingJudge:
    """One judge for one metric. Create one per metric so that the metric name is part of every cache key."""

    def __init__(self, raw: RawJudge, cache: DiskCache, settings: JudgeSettings, metric: str, ragas_version: str, stats: JudgeStats, *, pacer=None,
                 sleep: Callable[[float], None] = time.sleep, max_attempts: int = 4, max_wait_s: float = 90.0):
        self.raw, self.cache, self.settings, self.metric, self.ragas_version = raw, cache, settings, metric, ragas_version
        self.stats, self.pacer, self._sleep, self.max_attempts, self.max_wait_s = stats, pacer, sleep, max_attempts, max_wait_s

    def key(self, prompt: str, response_model: Any) -> str:
        return judge_cache_key(self.settings, self.metric, self.ragas_version, prompt, response_model)

    async def agenerate(self, prompt: str, response_model: Any):
        usage = self.stats.usage(self.metric)
        key = self.key(prompt, response_model)
        hit = self.cache.get(key)
        if hit is not None:
            try:
                parsed = parse_reply(hit["text"], response_model)
                usage.cache_hits += 1
                return parsed
            except (JudgeError, KeyError, TypeError):
                pass  # a damaged cache entry is a miss
        last_problem = "no attempt made"
        for attempt in range(1, self.max_attempts + 1):
            if self.stats.max_calls is not None and self.stats.total_calls >= self.stats.max_calls:
                raise JudgeQuotaExhausted(f"session judge-call budget reached ({self.stats.max_calls} network calls)")
            if self.pacer is not None:
                self.pacer.wait(0)
            usage.calls += 1
            if attempt > 1:
                usage.retries += 1
            try:
                reply = await self.raw(prompt, response_model)
            except JudgeRateLimited as exc:
                last_problem = "rate limited (per minute)"
                if self.pacer is not None:
                    self.pacer.record(0)
                self._sleep(min(self.max_wait_s, (exc.retry_after_s or 15.0) + 1.0))
                continue
            usage.prompt_tokens += reply.prompt_tokens or 0
            usage.output_tokens += reply.output_tokens or 0
            usage.thinking_tokens += reply.thinking_tokens or 0
            if self.pacer is not None:
                self.pacer.record(0)
            try:
                parsed = parse_reply(reply.text, response_model)
            except JudgeError as exc:
                last_problem = f"{exc} (finish reason {reply.finish_reason})"
                continue
            self.cache.set(key, {"text": reply.text, "metric": self.metric, "judge_model": self.settings.model, "ragas_version": self.ragas_version,
                                 "usage": {"prompt_tokens": reply.prompt_tokens, "output_tokens": reply.output_tokens, "thinking_tokens": reply.thinking_tokens},
                                 "finish_reason": reply.finish_reason})
            return parsed
        if last_problem.startswith("rate limited"):
            raise JudgeQuotaExhausted("the judge stayed rate limited after the retries: quota exhausted for now")
        raise JudgeError(f"{self.metric}: no usable reply after {self.max_attempts} attempts: {last_problem}")

    def generate(self, prompt: str, response_model: Any):
        return asyncio.run(self.agenerate(prompt, response_model))
