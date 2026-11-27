"""Langfuse access: one lazily-created SDK client (prompts, scores) and the OTLP exporter that feeds Langfuse the OpenTelemetry traces.
Import-safe: the SDKs are imported inside the functions.

Keys come ONLY from ``os.environ`` (``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY``; the host from ``LANGFUSE_BASE_URL``, legacy ``LANGFUSE_HOST``).
Nothing here prints or returns a key (the exporter carries it in its Basic-auth header, which is never logged or returned). If either key is missing,
``get_langfuse_client`` and ``langfuse_span_exporter`` return ``None`` and import nothing. Traces no longer go through the Langfuse SDK: they are
OpenTelemetry spans (``rag_core.otel_tracing``) exported over OTLP/HTTP to Langfuse's OTLP endpoint, exactly as the SDK itself would send them.
"""

from __future__ import annotations

import base64
import os
import threading
from typing import Any, Mapping

PUBLIC_KEY_ENV = "LANGFUSE_PUBLIC_KEY"
SECRET_KEY_ENV = "LANGFUSE_SECRET_KEY"
BASE_URL_ENVS = ("LANGFUSE_BASE_URL", "LANGFUSE_HOST")
DEFAULT_TIMEOUT_S = 5
DEFAULT_BASE_URL = "https://cloud.langfuse.com"  # the SDK's default host
OTLP_TRACES_PATH = "/api/public/otel/v1/traces"  # Langfuse's OTLP/HTTP traces endpoint (the path the SDK's own exporter uses)

_lock = threading.Lock()
_clients: dict[tuple[str, str], Any] = {}


def keys_present(env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    return bool(env.get(PUBLIC_KEY_ENV)) and bool(env.get(SECRET_KEY_ENV))


def base_url(env: Mapping[str, str] | None = None) -> str | None:
    env = os.environ if env is None else env
    return next((env[name] for name in BASE_URL_ENVS if env.get(name)), None)


def get_langfuse_client(env: Mapping[str, str] | None = None, *, timeout_s: int = DEFAULT_TIMEOUT_S):
    """The shared ``langfuse.Langfuse`` client, or ``None`` when the keys are not present. Raises only if the SDK cannot be imported or built;
    callers wrap this call (every use of Langfuse in rag_core fails open)."""
    env = os.environ if env is None else env
    if not keys_present(env):
        return None
    cache_key = (env[PUBLIC_KEY_ENV], base_url(env) or "")
    with _lock:
        client = _clients.get(cache_key)
        if client is None:
            from langfuse import Langfuse

            client = Langfuse(public_key=env[PUBLIC_KEY_ENV], secret_key=env[SECRET_KEY_ENV], base_url=base_url(env), timeout=timeout_s)
            _clients[cache_key] = client
        return client


def otlp_endpoint(env: Mapping[str, str] | None = None) -> str:
    return (base_url(env) or DEFAULT_BASE_URL).rstrip("/") + OTLP_TRACES_PATH


def langfuse_span_exporter(env: Mapping[str, str] | None = None, *, timeout_s: int = DEFAULT_TIMEOUT_S):
    """An ``OTLPSpanExporter`` that sends OpenTelemetry spans to Langfuse (Basic auth with the key pair, as the SDK does), or ``None`` without keys."""
    env = os.environ if env is None else env
    if not keys_present(env):
        return None
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    auth = base64.b64encode(f"{env[PUBLIC_KEY_ENV]}:{env[SECRET_KEY_ENV]}".encode("utf-8")).decode("ascii")
    # ingestion version 4 = metadata values arrive JSON-encoded, as rag_core.otel_tracing writes them (the same header the v5 SDK sends)
    return OTLPSpanExporter(endpoint=otlp_endpoint(env), headers={"Authorization": f"Basic {auth}", "x-langfuse-ingestion-version": "4"}, timeout=timeout_s)


def reset_for_tests() -> None:
    with _lock:
        _clients.clear()
