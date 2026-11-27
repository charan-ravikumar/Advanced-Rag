"""Tracing for the generation path. OPTIONAL and FAIL OPEN: an observability bug or outage never changes or breaks an answer.

Pieces
* ``Capture``: what is allowed to leave the process, decided BEFORE anything reaches a tracer. ``capture="masked"`` (default): the question is
  truncated to 200 characters with emails, URLs, long digit runs and key-like strings redacted; the answer is truncated to 500 characters (same
  redaction); sources are recorded as chunk_id, doc_id, section, rank, score only (no passage text). ``"full"``: untruncated question/answer
  and the compiled prompt (key-like strings are STILL redacted). ``"none"``: no question or answer text at all.
* ``Tracer`` / ``RequestTrace``: ``tracer.start_request(...)`` returns one per-request trace with ``span``, ``generation`` (context managers),
  ``event``, ``set_error``, ``trace_id``, ``end``; ``tracer.flush()``.
* ``NoopTracer`` (default), ``LangfuseTracer`` (SDK imported lazily; explicit parent/child observations, no ambient OpenTelemetry context, so it is
  safe inside a generator that yields), ``FakeTracer`` (tests, UI development), ``SafeTracer`` (the fail-open wrapper the service always uses).
* ``build_tracer``: a LangfuseTracer only if both keys are present AND ``tracing.enabled``; ``sample_rate`` and an in-memory daily request cap.
  Beyond the cap (or when sampled out) a request is simply not traced.

Import-safe: the Langfuse SDK is imported inside ``LangfuseTracer`` methods only.
"""

from __future__ import annotations

import random
import re
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, Protocol, Sequence

from rag_core.config import TracingConfig

QUESTION_MAX_CHARS = 200
ANSWER_MAX_CHARS = 500

# ---- masking ---------------------------------------------------------------------------------------------------------
_KEY_PATTERNS = [
    re.compile(r"\bgsk_[A-Za-z0-9]{8,}"),  # Groq
    re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}"),  # Google
    re.compile(r"\b(?:sk|pk)-lf-[A-Za-z0-9\-]{6,}"),  # Langfuse
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"),  # OpenAI-style
    re.compile(r"\bpk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\b(?:ghp|gho|ghs|xox[abp])_[A-Za-z0-9\-]{8,}"),
    re.compile(r"\b[A-Za-z0-9_\-]{40,}\b"),  # any other long opaque token
]
_URL = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.IGNORECASE)
_EMAIL = re.compile(r"[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+")
_LONG_NUMBER = re.compile(r"\+?\d(?:[\d\s().\-]{5,}\d)")  # 7+ digits, possibly with separators (phone numbers, ids, card numbers)


def redact_keys(text: str) -> str:
    """Remove key-like strings. Applied in EVERY capture mode."""
    for pattern in _KEY_PATTERNS:
        text = pattern.sub("[REDACTED_KEY]", text)
    return text


def mask_text(text: Any, limit: int | None = None) -> str:
    """Redact keys, URLs, emails and long digit runs, THEN truncate (so a cut can never split a secret). ``None`` and ``""`` give ``""``."""
    if text is None:
        return ""
    out = redact_keys(str(text))
    out = _URL.sub("[URL]", out)
    out = _EMAIL.sub("[EMAIL]", out)
    out = _LONG_NUMBER.sub(lambda m: "[NUMBER]" if sum(c.isdigit() for c in m.group(0)) >= 7 else m.group(0), out)
    if limit is not None and len(out) > limit:
        out = out[:limit].rstrip() + "…"
    return out


def redact_deep(value: Any) -> Any:
    """``redact_keys`` over every string in nested dicts/lists/tuples (metadata is arbitrary)."""
    if isinstance(value, str):
        return redact_keys(value)
    if isinstance(value, dict):
        return {k: redact_deep(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_deep(v) for v in value]
    return value


class Capture:
    """What may be sent to the tracer, by mode. The service passes everything through this object before calling a tracer."""

    MODES = ("masked", "full", "none")

    def __init__(self, mode: str = "masked"):
        if mode not in self.MODES:
            raise ValueError(f"tracing.capture must be one of {self.MODES}, got {mode!r}")
        self.mode = mode

    def question(self, text: str) -> str | None:
        if self.mode == "none":
            return None
        return redact_keys(text) if self.mode == "full" else mask_text(text, QUESTION_MAX_CHARS)

    def answer(self, text: str) -> str | None:
        if self.mode == "none":
            return None
        return redact_keys(text) if self.mode == "full" else mask_text(text, ANSWER_MAX_CHARS)

    def sources(self, sources: Sequence) -> list[dict]:
        """Ids, section, rank and score; the passage text only in ``full`` mode."""
        rows = [{"chunk_id": s.chunk_id, "doc_id": s.doc_id, "section": s.section, "rank": s.rank, "score": round(float(s.score), 6),
                 **({"label": s.label} if getattr(s, "label", "") else {})} for s in sources]
        if self.mode == "full":
            for row, s in zip(rows, sources):
                row["text"] = redact_keys(s.text)
        return rows

    def messages(self, messages: Sequence[dict]) -> list[dict] | None:
        """The compiled prompt: only in ``full`` mode (it contains passage text and the question)."""
        return [{"role": m["role"], "content": redact_keys(m["content"])} for m in messages] if self.mode == "full" else None

    def generation_input(self, messages: Sequence[dict], question: str, sources: Sequence) -> dict | list:
        """What a generation observation records as its input: the compiled prompt (``full``), else the masked question and the source ids."""
        if self.mode == "full":
            return self.messages(messages) or []
        out: dict[str, Any] = {"source_ids": [s.chunk_id for s in sources]}
        masked = self.question(question)
        if masked is not None:
            out["question"] = masked
        return out

    def meta(self, data: dict) -> dict:
        return redact_deep(data)


def cost_details(prices: dict | None, model: str | None, prompt_tokens: int | None, completion_tokens: int | None) -> dict | None:
    """USD cost ``{"input", "output"}`` from ``generation.prices`` (per MILLION tokens) or ``None``. Never invents a price: no entry for the
    model, or no usage reported, means no cost."""
    entry = (prices or {}).get(model or "")
    if not entry or prompt_tokens is None or completion_tokens is None:
        return None
    try:
        return {"input": prompt_tokens * float(entry["input"]) / 1_000_000, "output": completion_tokens * float(entry["output"]) / 1_000_000}
    except (KeyError, TypeError, ValueError):
        return None


# ---- handles and the protocol ----------------------------------------------------------------------------------------------
class SpanHandle(Protocol):
    def update(self, **attrs: Any) -> None: ...


class RequestTrace(Protocol):
    trace_id: str | None

    def span(self, name: str, **attrs: Any): ...  # context manager yielding a SpanHandle

    def generation(self, name: str, **attrs: Any): ...  # context manager yielding a SpanHandle

    def event(self, name: str, *, level: str = "DEFAULT", status_message: str | None = None, metadata: dict | None = None) -> None: ...

    def set_error(self, code: str, message: str, level: str = "ERROR") -> None: ...

    def end(self, *, output: Any = None, metadata: dict | None = None) -> None: ...


class Tracer(Protocol):
    def start_request(self, name: str, *, session_id: str | None = None, input: Any = None, metadata: dict | None = None) -> RequestTrace: ...

    def flush(self, timeout_s: float | None = None) -> None: ...


class _NoopHandle:
    def update(self, **attrs: Any) -> None:
        return None


class NoopTrace:
    trace_id = None

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[SpanHandle]:
        yield _NoopHandle()

    generation = span

    def event(self, name: str, **kw: Any) -> None:
        return None

    def set_error(self, code: str, message: str, level: str = "ERROR") -> None:
        return None

    def end(self, **kw: Any) -> None:
        return None


class NoopTracer:
    """The default: traces nothing, imports nothing."""

    enabled = False

    def start_request(self, name: str, **kw: Any) -> NoopTrace:
        return NoopTrace()

    def flush(self, timeout_s: float | None = None) -> None:
        return None


# ---- the fail-open wrapper -----------------------------------------------------------------------------------------------------
class TraceStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.errors = 0  # tracer calls that raised (swallowed)
        self.traced = 0
        self.sampled_out = 0
        self.capped = 0
        self.flush_timeouts = 0

    def bump(self, field: str) -> None:
        with self._lock:
            setattr(self, field, getattr(self, field) + 1)

    def as_dict(self) -> dict:
        return {"errors": self.errors, "traced": self.traced, "sampled_out": self.sampled_out, "capped": self.capped, "flush_timeouts": self.flush_timeouts}


class _SafeHandle:
    def __init__(self, inner: Any, stats: TraceStats):
        self._inner, self._stats = inner, stats

    def update(self, **attrs: Any) -> None:
        try:
            self._inner.update(**attrs)
        except Exception:  # noqa: BLE001: observability must never break an answer
            self._stats.bump("errors")


class SafeTrace:
    """Wraps one request trace: every call is guarded, exceptions are swallowed and counted."""

    def __init__(self, inner: Any, stats: TraceStats):
        self._inner, self._stats = inner, stats
        try:
            self.trace_id = inner.trace_id
        except Exception:  # noqa: BLE001
            self.trace_id = None
            stats.bump("errors")

    def _guard(self, fn: Callable[[], Any]) -> None:
        try:
            fn()
        except Exception:  # noqa: BLE001
            self._stats.bump("errors")

    def _scoped(self, opener: Callable[[], Any]):
        cm, handle = None, _NoopHandle()
        try:
            cm = opener()
            handle = cm.__enter__()
        except Exception:  # noqa: BLE001
            self._stats.bump("errors")
            cm, handle = None, _NoopHandle()
        return cm, _SafeHandle(handle, self._stats)

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[SpanHandle]:
        cm, handle = self._scoped(lambda: self._inner.span(name, **attrs))
        try:
            yield handle
        finally:
            if cm is not None:
                self._guard(lambda: cm.__exit__(None, None, None))

    @contextmanager
    def generation(self, name: str, **attrs: Any) -> Iterator[SpanHandle]:
        cm, handle = self._scoped(lambda: self._inner.generation(name, **attrs))
        try:
            yield handle
        finally:
            if cm is not None:
                self._guard(lambda: cm.__exit__(None, None, None))

    def event(self, name: str, *, level: str = "DEFAULT", status_message: str | None = None, metadata: dict | None = None) -> None:
        self._guard(lambda: self._inner.event(name, level=level, status_message=status_message, metadata=metadata))

    def set_error(self, code: str, message: str, level: str = "ERROR") -> None:
        self._guard(lambda: self._inner.set_error(code, message, level))

    def end(self, *, output: Any = None, metadata: dict | None = None) -> None:
        self._guard(lambda: self._inner.end(output=output, metadata=metadata))


class SafeTracer:
    """The wrapper ``RagService`` always uses. A tracer that raises on every call changes nothing about an answer."""

    def __init__(self, inner: Any | None = None):
        self._inner = inner if inner is not None else NoopTracer()
        self.stats = getattr(inner, "stats", None) or TraceStats()

    @property
    def enabled(self) -> bool:
        return bool(getattr(self._inner, "enabled", True))

    def start_request(self, name: str, *, session_id: str | None = None, input: Any = None, metadata: dict | None = None) -> SafeTrace:
        try:
            trace = self._inner.start_request(name, session_id=session_id, input=input, metadata=metadata)
        except Exception:  # noqa: BLE001
            self.stats.bump("errors")
            trace = NoopTrace()
        return SafeTrace(trace, self.stats)

    def flush(self, timeout_s: float | None = None) -> None:
        try:
            self._inner.flush(timeout_s)
        except Exception:  # noqa: BLE001
            self.stats.bump("errors")


# ---- the fake (tests and UI development) --------------------------------------------------------------------------------------------
class _FakeHandle:
    def __init__(self, obs: dict):
        self._obs = obs

    def update(self, **attrs: Any) -> None:
        if self._obs["_raise"]:
            raise RuntimeError("fake tracer failure")
        for key in ("level", "status_message", "model"):
            if key in attrs:
                self._obs[key] = attrs.pop(key)
        self._obs["attrs"].update(attrs)


class FakeTrace:
    def __init__(self, tracer: "FakeTracer", trace_id: str, root: dict):
        self._tracer, self.trace_id, self.observations = tracer, trace_id, [root]

    def _add(self, name: str, otype: str, parent: str, attrs: dict, **extra: Any) -> dict:
        if self._tracer.raise_on_call:
            raise RuntimeError("fake tracer failure")
        obs = {"name": name, "type": otype, "parent": parent, "attrs": dict(attrs), "level": extra.get("level", "DEFAULT"),
               "status_message": extra.get("status_message"), "ended": False, "_raise": self._tracer.raise_on_call}
        self.observations.append(obs)
        return obs

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[SpanHandle]:
        obs = self._add(name, attrs.pop("as_type", "span"), "rag.request", attrs)
        try:
            yield _FakeHandle(obs)
        finally:
            obs["ended"] = True

    @contextmanager
    def generation(self, name: str, **attrs: Any) -> Iterator[SpanHandle]:
        obs = self._add(name, "generation", "rag.request", attrs)
        try:
            yield _FakeHandle(obs)
        finally:
            obs["ended"] = True

    def event(self, name: str, *, level: str = "DEFAULT", status_message: str | None = None, metadata: dict | None = None) -> None:
        self._add(name, "event", "rag.request", {"metadata": metadata or {}}, level=level, status_message=status_message)["ended"] = True

    def set_error(self, code: str, message: str, level: str = "ERROR") -> None:
        if self._tracer.raise_on_call:
            raise RuntimeError("fake tracer failure")
        root = self.observations[0]
        root["level"], root["status_message"] = level, f"{code}: {message}"
        root["attrs"]["error_code"] = code

    def end(self, *, output: Any = None, metadata: dict | None = None) -> None:
        if self._tracer.raise_on_call:
            raise RuntimeError("fake tracer failure")
        root = self.observations[0]
        root["attrs"]["output"] = output
        root["attrs"].update(metadata or {})
        root["ended"] = True

    # convenience for assertions
    def names(self) -> list[str]:
        return [o["name"] for o in self.observations]

    def get(self, name: str) -> dict:
        return next(o for o in self.observations if o["name"] == name)

    def has(self, name: str) -> bool:
        return any(o["name"] == name for o in self.observations)


class FakeTracer:
    """Records everything it is given. ``raise_on_call=True`` makes every call raise (to prove the service fails open)."""

    enabled = True

    def __init__(self, *, raise_on_call: bool = False):
        self.raise_on_call, self.traces, self.flushes = raise_on_call, [], 0

    def start_request(self, name: str, *, session_id: str | None = None, input: Any = None, metadata: dict | None = None) -> FakeTrace:
        if self.raise_on_call:
            raise RuntimeError("fake tracer failure")
        root = {"name": name, "type": "span", "parent": None, "attrs": {"input": input, "session_id": session_id, **(metadata or {})},
                "level": "DEFAULT", "status_message": None, "ended": False, "_raise": self.raise_on_call}
        trace = FakeTrace(self, f"fake-trace-{len(self.traces) + 1}", root)
        self.traces.append(trace)
        return trace

    def flush(self, timeout_s: float | None = None) -> None:
        if self.raise_on_call:
            raise RuntimeError("fake tracer failure")
        self.flushes += 1


# ---- Langfuse ---------------------------------------------------------------------------------------------------------------------
class DailyCap:
    """At most ``limit`` requests per UTC day, counted in memory (a restart resets it; it is a cost guard, not an accounting system)."""

    def __init__(self, limit: int, clock: Callable[[], float] = time.time):
        self.limit, self._clock, self._day, self._count, self._lock = limit, clock, None, 0, threading.Lock()

    def allow(self) -> bool:
        day = datetime.fromtimestamp(self._clock(), tz=timezone.utc).date()
        with self._lock:
            if day != self._day:
                self._day, self._count = day, 0
            if self._count >= self.limit:
                return False
            self._count += 1
            return True


class _LangfuseHandle:
    def __init__(self, obs: Any, as_generation: bool, stats: TraceStats):
        self._obs, self._gen = obs, as_generation

    def update(self, **attrs: Any) -> None:
        kw: dict[str, Any] = {}
        usage = attrs.pop("usage", None)
        if usage is not None and self._gen:
            parts = {"input": usage.get("prompt_tokens"), "output": usage.get("completion_tokens")}
            kw["usage_details"] = {k: int(v) for k, v in parts.items() if v is not None}
        for key in ("output", "input", "level", "status_message", "completion_start_time", "cost_details", "model_parameters", "model"):
            if key in attrs:
                kw[key] = attrs.pop(key)
        if attrs:
            kw["metadata"] = attrs
        if kw:
            self._obs.update(**kw)


class LangfuseTrace:
    def __init__(self, root: Any, stats: TraceStats):
        self._root, self._stats = root, stats
        self.trace_id = getattr(root, "trace_id", None)

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[SpanHandle]:
        as_type = attrs.pop("as_type", "span")
        obs = self._root.start_observation(name=name, as_type=as_type, input=attrs.pop("input", None), metadata=attrs or None)
        try:
            yield _LangfuseHandle(obs, False, self._stats)
        finally:
            obs.end()

    @contextmanager
    def generation(self, name: str, **attrs: Any) -> Iterator[SpanHandle]:
        prompt = attrs.pop("prompt", None)
        kw = {k: attrs.pop(k) for k in ("model", "model_parameters", "input") if k in attrs}
        obs = self._root.start_observation(name=name, as_type="generation", metadata=attrs or None, **({"prompt": prompt} if prompt is not None else {}), **kw)
        try:
            yield _LangfuseHandle(obs, True, self._stats)
        finally:
            obs.end()

    def event(self, name: str, *, level: str = "DEFAULT", status_message: str | None = None, metadata: dict | None = None) -> None:
        obs = self._root.start_observation(name=name, as_type="event", level=level, status_message=status_message, metadata=metadata)
        if hasattr(obs, "end"):
            obs.end()

    def set_error(self, code: str, message: str, level: str = "ERROR") -> None:
        self._root.update(level=level, status_message=f"{code}: {message}"[:500], metadata={"error_code": code})

    def end(self, *, output: Any = None, metadata: dict | None = None) -> None:
        kw: dict[str, Any] = {}
        if output is not None:
            kw["output"] = output
        if metadata:
            kw["metadata"] = metadata
        if kw:
            self._root.update(**kw)
        self._root.end()


class LangfuseTracer:
    """Sends traces to Langfuse through explicit parent/child observations. Subject to ``sample_rate`` and the daily cap."""

    enabled = True

    def __init__(self, cfg: TracingConfig | None = None, *, client_factory: Callable[[], Any] | None = None,
                 propagate_factory: Callable[..., Any] | None = None, rng: Callable[[], float] = random.random,
                 clock: Callable[[], float] = time.time):
        self.cfg = cfg or TracingConfig(enabled=True)
        self._factory, self._propagate, self._rng = client_factory, propagate_factory, rng
        self.stats, self._cap = TraceStats(), DailyCap(self.cfg.daily_request_cap, clock)

    def _client(self) -> Any:
        if self._factory is not None:
            return self._factory()
        from rag_core.langfuse_client import get_langfuse_client

        return get_langfuse_client()

    def _propagate_cm(self, **kw: Any):
        if self._propagate is not None:
            return self._propagate(**kw)
        from langfuse import propagate_attributes

        return propagate_attributes(**kw)

    def start_request(self, name: str, *, session_id: str | None = None, input: Any = None, metadata: dict | None = None):
        if self._rng() >= self.cfg.sample_rate:
            self.stats.bump("sampled_out")
            return NoopTrace()
        if not self._cap.allow():
            self.stats.bump("capped")
            return NoopTrace()
        client = self._client()
        if client is None:
            return NoopTrace()
        kw = {"trace_name": name, **({"session_id": session_id} if session_id else {})}
        with self._propagate_cm(**kw):
            root = client.start_observation(name=name, as_type="span", input=input, metadata=metadata)
        self.stats.bump("traced")
        return LangfuseTrace(root, self.stats)

    def flush(self, timeout_s: float | None = None) -> None:
        """Flush in a helper thread and give up after ``timeout_s`` (default ``tracing.flush_timeout_seconds``): a slow Langfuse must not stall."""
        client = self._client()
        if client is None:
            return
        done = threading.Event()

        def run() -> None:
            try:
                client.flush()
            except Exception:  # noqa: BLE001
                self.stats.bump("errors")
            finally:
                done.set()

        threading.Thread(target=run, daemon=True, name="langfuse-flush").start()
        if not done.wait(self.cfg.flush_timeout_seconds if timeout_s is None else timeout_s):
            self.stats.bump("flush_timeouts")


def build_tracer(cfg: TracingConfig, *, env: dict | None = None, client_factory: Callable[[], Any] | None = None,
                 propagate_factory: Callable[..., Any] | None = None, rng: Callable[[], float] = random.random,
                 clock: Callable[[], float] = time.time) -> Tracer:
    """``LangfuseTracer`` only if ``cfg.enabled`` AND both Langfuse keys are present; otherwise ``NoopTracer`` (and the SDK is never imported)."""
    from rag_core.langfuse_client import keys_present

    if not cfg.enabled or not (keys_present(env) if client_factory is None else True):
        return NoopTracer()
    return LangfuseTracer(cfg, client_factory=client_factory, propagate_factory=propagate_factory, rng=rng, clock=clock)
