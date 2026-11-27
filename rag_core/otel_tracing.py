"""OpenTelemetry tracing: the request's span tree as OTel spans, exported over OTLP. OPTIONAL and FAIL OPEN, like every tracer.

The service talks to the ``Tracer`` protocol (``rag_core.tracing``); ``OtelTracer`` implements it with the OpenTelemetry SDK. Every span carries
two vocabularies:

* the OpenTelemetry GenAI semantic conventions on the generation span (``gen_ai.operation.name``, ``gen_ai.provider.name``,
  ``gen_ai.request.model``, ``gen_ai.request.temperature``, ``gen_ai.request.max_tokens``, ``gen_ai.response.model``,
  ``gen_ai.response.finish_reasons``, ``gen_ai.usage.input_tokens``, ``gen_ai.usage.output_tokens``), which any OTLP backend understands;
* Langfuse's OTel attributes (``langfuse.observation.*``, ``langfuse.trace.name``, ``session.id``), so that Langfuse, receiving the SAME
  spans at its OTLP endpoint, shows them as an LLM trace: observation types (span, retriever, generation, event), levels, input/output,
  usage, cost, prompt links and sessions. The names are copied from the Langfuse SDK (``langfuse/_client/attributes.py``, 5.2.0) and a test
  compares them with the installed SDK, so a rename there fails here.

Where spans go (``tracing.exporters``): ``langfuse`` (OTLP/HTTP to ``<LANGFUSE_BASE_URL>/api/public/otel/v1/traces`` with Basic auth; built in
``rag_core.langfuse_client``, the only module that reads the Langfuse keys) and ``otlp`` (a plain ``OTLPSpanExporter()``: the OpenTelemetry SDK
reads the standard ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`` / ``OTEL_EXPORTER_OTLP_ENDPOINT`` / ``..._HEADERS`` variables itself). Both receive
the same attributes, after ``tracing.Capture`` (masked by default) has decided what may leave the process.

Design rules (unchanged from the SDK tracer): explicit parent context for every child, never the ambient "current span" (the pipeline is a
generator that yields between spans, so a context-local current span would leak across requests); the tracer owns a PRIVATE
``TracerProvider`` (never the global one, so nothing else in the process exports through it and it exports nothing of anyone else's);
``sample_rate`` and the in-memory daily cap apply before a trace is started; an event is a zero-length child span (Langfuse counts it as one
observation, exactly as before); ``flush`` is bounded by ``tracing.flush_timeout_seconds``.

Import-safe: the OpenTelemetry SDK is imported inside the functions that build the provider and the exporters.
"""

from __future__ import annotations

import json
import random
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any

from rag_core.config import TracingConfig
from rag_core.tracing import DailyCap, NoopTrace, TraceStats

SERVICE_NAME = "advrag"
INSTRUMENTATION_SCOPE = "advrag.rag_core"

# Langfuse's OTel attribute names (see the module doc).
LF_TRACE_NAME = "langfuse.trace.name"
LF_SESSION_ID = "session.id"
LF_TYPE = "langfuse.observation.type"
LF_METADATA = "langfuse.observation.metadata"
LF_LEVEL = "langfuse.observation.level"
LF_STATUS_MESSAGE = "langfuse.observation.status_message"
LF_INPUT = "langfuse.observation.input"
LF_OUTPUT = "langfuse.observation.output"
LF_COMPLETION_START = "langfuse.observation.completion_start_time"
LF_MODEL = "langfuse.observation.model.name"
LF_MODEL_PARAMETERS = "langfuse.observation.model.parameters"
LF_USAGE = "langfuse.observation.usage_details"
LF_COST = "langfuse.observation.cost_details"
LF_PROMPT_NAME = "langfuse.observation.prompt.name"
LF_PROMPT_VERSION = "langfuse.observation.prompt.version"


def _default(value: Any) -> Any:
    """Datetimes as the Langfuse SDK writes them (ISO 8601, ``Z`` for UTC, a naive time taken as local time); anything else as its string."""
    if isinstance(value, datetime):
        zoned = value if value.tzinfo is not None else value.astimezone()
        text = zoned.isoformat()
        return text.replace("+00:00", "Z") if zoned.utcoffset() == timedelta(0) else text
    return str(value)


def _json(value: Any) -> str:
    """Input, output, usage, cost, model parameters and times: ``json.dumps`` with its default separators (the SDK's ``_serialize``)."""
    return json.dumps(value, default=_default)


def _meta_json(value: Any) -> str:
    """One metadata value: compact separators and raw non-ASCII (the SDK's ``_serialize_metadata_value``)."""
    return json.dumps(value, default=_default, separators=(",", ":"), ensure_ascii=False)


def _io(value: Any) -> str | None:
    """Input/output: a string as is, anything else as JSON (what the Langfuse SDK sends)."""
    return value if value is None or isinstance(value, str) else _json(value)


def metadata_attributes(metadata: dict | None) -> dict:
    """One attribute per key, every value JSON-encoded (strings included), None skipped: the encoding of the Langfuse v5 SDK."""
    return {f"{LF_METADATA}.{k}": _meta_json(v) for k, v in (metadata or {}).items() if v is not None}


def observation_attributes(*, as_type: str, input: Any = None, output: Any = None, level: str | None = None, status_message: str | None = None,
                           metadata: dict | None = None, model: str | None = None, model_parameters: dict | None = None,
                           usage_details: dict | None = None, cost_details: dict | None = None, completion_start_time: datetime | None = None,
                           prompt_name: str | None = None, prompt_version: Any = None) -> dict:
    """The Langfuse attributes of one observation. Pure (a test compares it with the SDK's own ``create_*_attributes``)."""
    attrs = {LF_TYPE: as_type, LF_LEVEL: level, LF_STATUS_MESSAGE: status_message, LF_INPUT: _io(input), LF_OUTPUT: _io(output),
             LF_MODEL: model, LF_PROMPT_NAME: prompt_name, LF_PROMPT_VERSION: prompt_version,
             LF_USAGE: None if usage_details is None else _json(usage_details), LF_COST: None if cost_details is None else _json(cost_details),
             LF_COMPLETION_START: None if completion_start_time is None else _json(completion_start_time),
             LF_MODEL_PARAMETERS: None if model_parameters is None else _json(model_parameters), **metadata_attributes(metadata)}
    return {k: v for k, v in attrs.items() if v is not None}


def genai_request_attributes(*, provider: str | None, model: str | None, model_parameters: dict | None) -> dict:
    params = model_parameters or {}
    attrs = {"gen_ai.operation.name": "chat", "gen_ai.provider.name": provider, "gen_ai.request.model": model,
             "gen_ai.request.temperature": params.get("temperature"), "gen_ai.request.max_tokens": params.get("max_tokens")}
    return {k: v for k, v in attrs.items() if v is not None}


def genai_response_attributes(*, model: str | None = None, finish_reason: str | None = None, usage: dict | None = None) -> dict:
    usage = usage or {}
    attrs = {"gen_ai.response.model": model, "gen_ai.response.finish_reasons": [finish_reason] if finish_reason else None,
             "gen_ai.usage.input_tokens": usage.get("prompt_tokens"), "gen_ai.usage.output_tokens": usage.get("completion_tokens")}
    return {k: v for k, v in attrs.items() if v is not None}


# ---- handles ---------------------------------------------------------------------------------------------------------------------
_PLAIN_KEYS = ("output", "input", "level", "status_message", "completion_start_time", "cost_details", "model_parameters", "model")


def _set_level(span: Any, level: str | None, message: str | None) -> None:
    if level == "ERROR":
        from opentelemetry.trace import Status, StatusCode

        span.set_status(Status(StatusCode.ERROR, message))


class _OtelHandle:
    def __init__(self, span: Any, as_generation: bool):
        self._span, self._gen = span, as_generation

    def update(self, **attrs: Any) -> None:
        usage = attrs.pop("usage", None)
        finish_reason = attrs.get("finish_reason") if self._gen else None
        plain = {k: attrs.pop(k) for k in _PLAIN_KEYS if k in attrs}
        usage_details = None
        if usage is not None and self._gen:
            parts = {"input": usage.get("prompt_tokens"), "output": usage.get("completion_tokens")}
            usage_details = {k: int(v) for k, v in parts.items() if v is not None}
        out = observation_attributes(as_type=None, usage_details=usage_details, metadata=attrs or None, **plain)  # type: ignore[arg-type]
        out.pop(LF_TYPE, None)
        if self._gen:
            out.update(genai_response_attributes(model=plain.get("model"), finish_reason=finish_reason, usage=usage))
        if out:
            self._span.set_attributes(out)
        _set_level(self._span, plain.get("level"), plain.get("status_message"))


class OtelTrace:
    def __init__(self, tracer: Any, root: Any):
        from opentelemetry import trace as otel_trace

        self._tracer, self._root = tracer, root
        self._ctx = otel_trace.set_span_in_context(root)
        self.trace_id = format(root.get_span_context().trace_id, "032x")  # the id Langfuse gives the trace (feedback scores use it)

    def _start(self, name: str, attributes: dict) -> Any:
        return self._tracer.start_span(name, context=self._ctx, attributes=attributes)

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[_OtelHandle]:
        as_type = attrs.pop("as_type", "span")
        span = self._start(name, observation_attributes(as_type=as_type, input=attrs.pop("input", None), metadata=attrs or None))
        try:
            yield _OtelHandle(span, False)
        finally:
            span.end()

    @contextmanager
    def generation(self, name: str, **attrs: Any) -> Iterator[_OtelHandle]:
        prompt = attrs.pop("prompt", None)  # a Langfuse prompt object: linked only when the prompt really came from Langfuse
        provider, model = attrs.pop("provider", None), attrs.pop("model", None)
        params, input_ = attrs.pop("model_parameters", None), attrs.pop("input", None)
        linked = prompt is not None and not getattr(prompt, "is_fallback", False)
        out = observation_attributes(as_type="generation", input=input_, model=model, model_parameters=params, metadata=attrs or None,
                                     prompt_name=getattr(prompt, "name", None) if linked else None,
                                     prompt_version=getattr(prompt, "version", None) if linked else None)
        out.update(genai_request_attributes(provider=provider, model=model, model_parameters=params))
        span = self._start(name, out)
        try:
            yield _OtelHandle(span, True)
        finally:
            span.end()

    def event(self, name: str, *, level: str = "DEFAULT", status_message: str | None = None, metadata: dict | None = None) -> None:
        span = self._start(name, observation_attributes(as_type="event", level=level, status_message=status_message, metadata=metadata))
        _set_level(span, level, status_message)
        span.end(end_time=getattr(span, "start_time", None))

    def set_error(self, code: str, message: str, level: str = "ERROR") -> None:
        text = f"{code}: {message}"[:500]
        self._root.set_attributes({LF_LEVEL: level, LF_STATUS_MESSAGE: text, **metadata_attributes({"error_code": code})})
        _set_level(self._root, level, text)

    def end(self, *, output: Any = None, metadata: dict | None = None) -> None:
        attrs = observation_attributes(as_type="span", output=output, metadata=metadata)
        attrs.pop(LF_TYPE, None)
        if attrs:
            self._root.set_attributes(attrs)
        self._root.end()


# ---- the tracer -------------------------------------------------------------------------------------------------------------------
def build_provider(exporters: Sequence[Any], *, simple: bool = False) -> Any:
    """A PRIVATE TracerProvider with one span processor per exporter (batched; ``simple=True`` exports synchronously, for tests)."""
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    for exporter in exporters:
        provider.add_span_processor(SimpleSpanProcessor(exporter) if simple else BatchSpanProcessor(exporter))
    return provider


class OtelTracer:
    """Sends the request's spans through a private OpenTelemetry provider. Subject to ``sample_rate`` and the daily cap."""

    enabled = True

    def __init__(self, cfg: TracingConfig | None = None, *, provider: Any, rng: Callable[[], float] = random.random,
                 clock: Callable[[], float] = time.time):
        self.cfg = cfg or TracingConfig(enabled=True)
        self.provider, self._rng = provider, rng
        self._tracer = provider.get_tracer(INSTRUMENTATION_SCOPE)
        self.stats, self._cap = TraceStats(), DailyCap(self.cfg.daily_request_cap, clock)

    def start_request(self, name: str, *, session_id: str | None = None, input: Any = None, metadata: dict | None = None):
        if self._rng() >= self.cfg.sample_rate:
            self.stats.bump("sampled_out")
            return NoopTrace()
        if not self._cap.allow():
            self.stats.bump("capped")
            return NoopTrace()
        from opentelemetry.context import Context

        attrs = {LF_TRACE_NAME: name, **({LF_SESSION_ID: session_id} if session_id else {}),
                 **observation_attributes(as_type="span", input=input, metadata=metadata)}
        root = self._tracer.start_span(name, context=Context(), attributes=attrs)  # an EMPTY context: always a new trace, never a child of anything
        self.stats.bump("traced")
        return OtelTrace(self._tracer, root)

    def flush(self, timeout_s: float | None = None) -> None:
        """``force_flush`` in a helper thread; give up after ``timeout_s`` (default ``tracing.flush_timeout_seconds``): a slow backend must not stall."""
        timeout = self.cfg.flush_timeout_seconds if timeout_s is None else timeout_s
        done = threading.Event()

        def run() -> None:
            try:
                self.provider.force_flush(int(timeout * 1000))
            except Exception:  # noqa: BLE001
                self.stats.bump("errors")
            finally:
                done.set()

        threading.Thread(target=run, daemon=True, name="otel-flush").start()
        if not done.wait(timeout):
            self.stats.bump("flush_timeouts")


def otlp_exporter() -> Any:
    """A plain OTLP/HTTP exporter configured by the standard OTEL_EXPORTER_OTLP_* variables (read by the SDK, not by this code)."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    return OTLPSpanExporter()
