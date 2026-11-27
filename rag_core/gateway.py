"""LLM gateway: timeouts, bounded retries with backoff + jitter, a circuit breaker per provider, and failover.

Contract of ``LLMGateway.stream(messages)`` (an iterator of ``Delta`` and one final ``StreamDone``):

* a provider call that fails BEFORE it yielded any text is retried on the same provider (only 429/5xx/timeouts/connection errors,
  at most ``max_attempts`` times, exponential backoff with jitter, honouring a server ``Retry-After`` up to ``max_delay_s``);
  4xx errors (auth, not found, bad request) are not retried; then the next provider is tried (failover);
* a provider that fails AFTER it yielded text is never retried and never replaced: ``MidStreamError`` is raised with the partial
  text. Splicing a second model's answer onto a half-written first one would produce text no model wrote;
* each provider has a circuit breaker: it opens after ``breaker_threshold`` consecutive failed calls, rejects calls for
  ``breaker_cooldown_s``, then lets ONE trial call through (half-open): success closes it, failure re-opens it;
* every attempt (provider, model, attempt number, latency, error class and kind, whether a first token had been seen) is recorded
  and returned with the result or attached to the raised error.

The per-call timeout is passed to the provider, which hands it to its SDK (httpx timeouts: connect and between-chunk read). The
gateway also stops starting new attempts after ``deadline_s`` in total. Import-safe: SDKs are imported inside the providers.
"""

from __future__ import annotations

import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator, Protocol, Sequence

from rag_core.errors import BREAKER_KINDS, AllProvidersFailed, AnswerTooLong, MidStreamError, ProviderError
from rag_core.schemas import AttemptRecord

# A reply that stops with finish_reason "length" and no visible text spent its whole output budget on reasoning. It is retried ONCE on the same
# provider with the cap multiplied (no backoff: nothing is wrong with the provider), up to the ceiling; a second one is "answer too long".
LENGTH_RETRY_FACTOR = 2
LENGTH_RETRY_MAX_OUTPUT_TOKENS = 4096


@dataclass(frozen=True)
class Delta:
    text: str


@dataclass(frozen=True)
class Final:
    """The last item a provider yields: why it stopped and the token usage it reported (None when not reported)."""

    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None  # visible answer tokens PLUS reasoning ("thought") tokens
    reasoning_tokens: int | None = None  # the part of completion_tokens spent on reasoning, when the API reports it separately


@dataclass(frozen=True)
class StreamDone:
    """The last item the GATEWAY yields."""

    provider: str
    model: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    attempts: list = field(default_factory=list)
    reasoning_tokens: int | None = None


class Provider(Protocol):
    name: str
    model: str

    def stream(self, messages: list[dict], *, timeout_s: float, max_output_tokens: int, temperature: float) -> Iterator[Delta | Final]:
        """Yield ``Delta`` text pieces, then one ``Final``. Raise ``ProviderError`` (typed) on any failure."""


class CircuitBreaker:
    """closed -> (threshold consecutive failures) -> open -> (cooldown) -> half-open (one trial) -> closed | open."""

    def __init__(self, threshold: int = 3, cooldown_s: float = 30.0, clock: Callable[[], float] = time.monotonic):
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        self.threshold, self.cooldown_s, self._clock = threshold, cooldown_s, clock
        self._lock = threading.Lock()
        self._failures, self._opened_at, self._trial = 0, None, False

    @property
    def state(self) -> str:
        with self._lock:
            if self._opened_at is None:
                return "closed"
            return "half_open" if self._clock() - self._opened_at >= self.cooldown_s else "open"

    def allow(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return True
            if self._clock() - self._opened_at < self.cooldown_s:
                return False
            if self._trial:  # half-open: exactly one trial call at a time
                return False
            self._trial = True
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures, self._opened_at, self._trial = 0, None, False

    def record_failure(self) -> None:
        with self._lock:
            if self._opened_at is not None:  # the half-open trial failed: open again for a full cooldown
                self._opened_at, self._trial = self._clock(), False
                return
            self._failures += 1
            if self._failures >= self.threshold:
                self._opened_at = self._clock()

    def release(self) -> None:
        """A call ended without saying anything about the provider's health (a 400 about this request): free the trial slot."""
        with self._lock:
            self._trial = False


def as_provider_error(exc: BaseException) -> ProviderError:
    if isinstance(exc, ProviderError):
        return exc
    return ProviderError("other", f"{type(exc).__name__}", error_class=type(exc).__name__)  # class name only: never leak details


class LLMGateway:
    def __init__(self, providers: Sequence[Provider], *, timeout_s: float = 30.0, max_attempts: int = 3, base_delay_s: float = 0.5,
                 max_delay_s: float = 8.0, jitter: float = 0.5, deadline_s: float = 90.0, breaker_threshold: int = 3,
                 breaker_cooldown_s: float = 30.0, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic, rng: Callable[[], float] = random.random):
        if not providers:
            raise ValueError("at least one provider is required")
        if len({p.name for p in providers}) != len(providers):
            raise ValueError("provider names must be unique")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.providers = list(providers)
        self.timeout_s, self.max_attempts, self.base_delay_s, self.max_delay_s = timeout_s, max_attempts, base_delay_s, max_delay_s
        self.jitter, self.deadline_s = jitter, deadline_s
        self._sleep, self._clock, self._rng = sleep, clock, rng
        self.breakers = {p.name: CircuitBreaker(breaker_threshold, breaker_cooldown_s, clock) for p in self.providers}

    @property
    def primary_model(self) -> str:
        """The model a request goes to when nothing is wrong (used in cache keys)."""
        return self.providers[0].model

    @property
    def primary_name(self) -> str:
        return self.providers[0].name

    @property
    def primary_reasoning_effort(self) -> str | None:
        """The reasoning effort the primary provider sends (None if it has none): recorded on every result and part of the cache key."""
        return getattr(self.providers[0], "reasoning_effort", None)

    def backoff_s(self, attempt: int, retry_after_s: float | None = None) -> float:
        """Delay before attempt ``attempt + 1``: base * 2^(attempt-1), capped, scaled by jitter in [1 - jitter, 1]."""
        delay = min(self.max_delay_s, self.base_delay_s * 2 ** (attempt - 1))
        delay *= (1.0 - self.jitter) + self.jitter * self._rng()
        if retry_after_s:
            delay = max(delay, retry_after_s)
        return min(delay, self.max_delay_s)

    def stream(self, messages: list[dict], *, max_output_tokens: int = 700, temperature: float = 0.0) -> Iterator[Delta | StreamDone]:
        attempts: list[AttemptRecord] = []
        started = self._clock()
        length_seen = other_failure = False
        for provider in self.providers:
            breaker = self.breakers[provider.name]
            if not breaker.allow():
                attempts.append(AttemptRecord(provider=provider.name, model=provider.model, attempt=0, latency_ms=0.0,
                                              outcome="skipped_circuit_open", error_class="CircuitOpen"))
                continue
            last_error: ProviderError | None = None
            cap, length_retried = max_output_tokens, False
            for attempt in range(1, self.max_attempts + 1):
                if self._clock() - started >= self.deadline_s:
                    attempts.append(AttemptRecord(provider=provider.name, model=provider.model, attempt=attempt, latency_ms=0.0,
                                                  outcome="skipped_deadline", error_class="DeadlineExceeded"))
                    break
                parts: list[str] = []
                final: Final | None = None
                t0 = self._clock()
                stream = None
                try:
                    stream = provider.stream(messages, timeout_s=self.timeout_s, max_output_tokens=cap, temperature=temperature)
                    for item in stream:
                        if isinstance(item, Delta):
                            if self._clock() - started >= self.deadline_s:  # a slow drip must not hold this thread past the total deadline
                                raise ProviderError("timeout", "the total deadline passed while the answer was streaming", error_class="DeadlineExceeded")
                            if item.text:
                                parts.append(item.text)
                                yield item
                        elif isinstance(item, Final):
                            final = item
                    if not parts:
                        if final is not None and final.finish_reason == "length":  # the budget went into reasoning: about the question, not the provider
                            raise ProviderError("length", "the output budget was spent before any text was written", error_class="LengthWithoutText")
                        raise ProviderError("other", "the provider returned no text", error_class="EmptyResponse")
                except Exception as exc:  # noqa: BLE001: classified below; GeneratorExit/KeyboardInterrupt pass through
                    err = as_provider_error(exc)
                    attempts.append(AttemptRecord(provider=provider.name, model=provider.model, attempt=attempt,
                                                  latency_ms=(self._clock() - t0) * 1000, outcome="error", error_class=err.error_class,
                                                  error_kind=err.kind, status=err.status, first_token_seen=bool(parts)))
                    if parts:  # text was already yielded: never retry, never fail over
                        self._count_failure(breaker, err)
                        raise MidStreamError(provider.name, provider.model, "".join(parts), err, attempts) from None
                    last_error = err
                    if err.kind == "length" and not length_retried and attempt < self.max_attempts and cap < LENGTH_RETRY_MAX_OUTPUT_TOKENS:
                        length_retried, cap = True, min(cap * LENGTH_RETRY_FACTOR, LENGTH_RETRY_MAX_OUTPUT_TOKENS)
                        continue
                    if err.retryable and attempt < self.max_attempts and err.error_class != "DeadlineExceeded":
                        self._sleep(self.backoff_s(attempt, err.retry_after_s))
                        continue
                    break
                finally:
                    close = getattr(stream, "close", None)  # release the provider's connection also when we stop early
                    if close is not None:
                        close()
                attempts.append(AttemptRecord(provider=provider.name, model=provider.model, attempt=attempt,
                                              latency_ms=(self._clock() - t0) * 1000, outcome="ok", first_token_seen=True))
                breaker.record_success()
                final = final or Final()
                yield StreamDone(provider.name, provider.model, final.finish_reason, final.prompt_tokens, final.completion_tokens, attempts,
                                 reasoning_tokens=final.reasoning_tokens)
                return
            if last_error is not None:
                self._count_failure(breaker, last_error)
                length_seen, other_failure = length_seen or last_error.kind == "length", other_failure or last_error.kind != "length"
            else:
                breaker.release()
                other_failure = True
        if length_seen and not other_failure:
            raise AnswerTooLong(attempts)
        raise AllProvidersFailed(attempts)

    @staticmethod
    def _count_failure(breaker: CircuitBreaker, err: ProviderError) -> None:
        if err.kind in BREAKER_KINDS:
            breaker.record_failure()
        else:
            breaker.release()


# ---- providers (thin SDK wrappers, created lazily) --------------------------------------------------------------------

def _read_key(env_name: str) -> str:
    key = os.environ.get(env_name)
    if not key:
        raise ProviderError("auth", f"{env_name} is not set in the environment", error_class="MissingApiKey")
    return key


def _scrub(text: str, key: str | None) -> str:
    text = str(text)[:300]
    return text.replace(key, "[redacted]") if key else text


def _retry_after(headers) -> float | None:
    try:
        value = headers.get("retry-after") if headers is not None else None
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _kind_for_status(status: int) -> str:
    if status == 429:
        return "rate_limit"
    if status >= 500:
        return "server"
    if status in (401, 403):
        return "auth"
    if status == 404:
        return "not_found"
    if status in (400, 413, 422):
        return "bad_request"
    return "other"


class GeminiProvider:
    """Gemini through ``google-genai``. The timeout goes into the SDK's HTTP options (``HttpOptions.timeout`` is in
    MILLISECONDS), per request, together with ``HttpRetryOptions(attempts=1)`` so the SDK does not retry behind the gateway's back.

    The old code passed ``config={"timeout": 30}``: ``GenerateContentConfig`` has no such field and pydantic rejects it with a
    ValidationError on every call. ``GEMINI_API_KEY`` is read when the first request is made."""

    name = "gemini"

    def __init__(self, model: str, *, api_key_env: str = "GEMINI_API_KEY", thinking_budget: int | None = None,
                 client_factory: Callable[[str], object] | None = None):
        self.model, self.api_key_env, self.thinking_budget = model, api_key_env, thinking_budget
        self._client_factory, self._client = client_factory, None

    def _get_client(self):
        if self._client is None:
            key = _read_key(self.api_key_env)
            if self._client_factory is not None:
                self._client = self._client_factory(key)
            else:
                from google import genai

                self._client = genai.Client(api_key=key)
        return self._client

    def build_config(self, system: str | None, timeout_s: float, max_output_tokens: int, temperature: float):
        from google.genai import types

        extra = {"thinking_config": types.ThinkingConfig(thinking_budget=self.thinking_budget)} if self.thinking_budget is not None else {}
        return types.GenerateContentConfig(
            system_instruction=system, temperature=temperature, max_output_tokens=max_output_tokens,
            http_options=types.HttpOptions(timeout=int(timeout_s * 1000), retry_options=types.HttpRetryOptions(attempts=1)), **extra)

    def stream(self, messages, *, timeout_s, max_output_tokens, temperature):
        key = os.environ.get(self.api_key_env)
        try:
            from google.genai import types

            system = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or None
            contents = [types.Content(role="model" if m["role"] == "assistant" else "user", parts=[types.Part(text=m["content"])])
                        for m in messages if m["role"] != "system"]
            client = self._get_client()
            config = self.build_config(system, timeout_s, max_output_tokens, temperature)
            finish, usage = None, None
            for chunk in client.models.generate_content_stream(model=self.model, contents=contents, config=config):
                if getattr(chunk, "usage_metadata", None) is not None:
                    usage = chunk.usage_metadata
                candidates = getattr(chunk, "candidates", None) or []
                if candidates and getattr(candidates[0], "finish_reason", None):
                    finish = candidates[0].finish_reason
                text = chunk.text if candidates else None
                if text:
                    yield Delta(text)
            thoughts = getattr(usage, "thoughts_token_count", None)
            answer_tokens = getattr(usage, "candidates_token_count", None)
            # Gemini reports thought tokens separately from the answer tokens; both count toward max_output_tokens
            total = answer_tokens if thoughts is None or answer_tokens is None else answer_tokens + thoughts
            yield Final(_normalise_finish(finish), getattr(usage, "prompt_token_count", None), total, thoughts)
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise self._convert(exc, key) from None

    @staticmethod
    def _convert(exc: BaseException, key: str | None) -> ProviderError:
        name = type(exc).__name__
        code = getattr(exc, "code", None)
        if isinstance(code, int) and not isinstance(exc, (TimeoutError, OSError)):
            message = getattr(exc, "message", None) or exc
            return ProviderError(_kind_for_status(code), _scrub(message, key), status=code, error_class=name)
        lowered = name.lower()
        if "timeout" in lowered or isinstance(exc, TimeoutError):
            return ProviderError("timeout", _scrub(exc, key), error_class=name)
        if any(w in lowered for w in ("connect", "network", "protocol", "remoteclose", "readerror")) or isinstance(exc, OSError):
            return ProviderError("connection", _scrub(exc, key), error_class=name)
        return ProviderError("other", name, error_class=name)


def _normalise_finish(reason) -> str | None:
    """'stop' | 'length' | other lower-case string | None, from OpenAI-style strings and Gemini enums alike."""
    if reason is None:
        return None
    text = str(getattr(reason, "name", reason)).lower()
    return {"max_tokens": "length", "finish_reason_unspecified": None}.get(text, text)


def build_gateway(cfg) -> LLMGateway:
    """The production gateway from a ``RagConfig``: providers in the configured order (first = primary), created lazily, keys from
    the environment at the first request. Nothing is contacted here."""
    makers = {"gemini": lambda: GeminiProvider(cfg.gemini_model, thinking_budget=getattr(cfg, "gemini_thinking_budget", None))}
    unknown = [n for n in cfg.providers if n not in makers]
    if unknown:
        raise ValueError(f"unknown provider(s) {unknown}; choose from {sorted(makers)}")
    return LLMGateway([makers[n]() for n in cfg.providers], timeout_s=cfg.timeout_s, max_attempts=cfg.max_attempts,
                      breaker_threshold=cfg.breaker_threshold, breaker_cooldown_s=cfg.breaker_cooldown_s)
