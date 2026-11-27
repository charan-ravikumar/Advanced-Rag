"""One lazily-created Langfuse client, shared by the prompt store and the tracer. Import-safe: the SDK is imported inside the functions.

Keys come ONLY from ``os.environ`` (``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY``; the host from ``LANGFUSE_BASE_URL``, legacy ``LANGFUSE_HOST``).
Nothing here prints or returns a key. If either key is missing, ``get_langfuse_client`` returns ``None`` and imports nothing.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Mapping

PUBLIC_KEY_ENV = "LANGFUSE_PUBLIC_KEY"
SECRET_KEY_ENV = "LANGFUSE_SECRET_KEY"
BASE_URL_ENVS = ("LANGFUSE_BASE_URL", "LANGFUSE_HOST")
DEFAULT_TIMEOUT_S = 5

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


def reset_for_tests() -> None:
    with _lock:
        _clients.clear()
