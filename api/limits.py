"""Per-key rate limiting with ``slowapi`` (the ``limits`` library underneath, in-memory storage).

The KEY is the client's ``X-Session-Id`` header when it is a well-formed id (``[A-Za-z0-9_-]{1,64}``), else the client's network address. A malformed header is ignored, not trusted.
The query endpoints (ask, answer, retrieve) share ONE bucket per key, so a client cannot multiply its allowance by switching endpoints; feedback has its own, looser bucket.

Values come from ``config.yaml`` ``app.*`` (the same numbers the UI shows): ``rate_limit_requests`` per ``rate_limit_window_seconds``, and ``session_request_cap`` per day.
The limits are read on every request (callables), so ``configure`` at startup is all that is needed. The limiter is module-level because slowapi binds it at decoration time:
tests call ``configure`` and ``limiter.reset()``. In memory and per process: a restart resets it, and several replicas would each count on their own (a cost guard, not accounting).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from fastapi import Request
from slowapi import Limiter

from api.schemas import ID_PATTERN

SESSION_HEADER = "X-Session-Id"
FEEDBACK_LIMIT = "30 per 60 second"
_ID = re.compile(ID_PATTERN)


@dataclass
class RateLimitConfig:
    requests: int = 4
    window_s: float = 60.0
    per_day: int = 20


_config = RateLimitConfig()


def configure(requests: int, window_s: float, per_day: int) -> None:
    _config.requests, _config.window_s, _config.per_day = int(requests), float(window_s), int(per_day)


def rate_limit_key(request: Request) -> str:
    supplied = request.headers.get(SESSION_HEADER, "")
    if _ID.fullmatch(supplied):
        return f"session:{supplied}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


def query_limits() -> str:
    """Two limits for one bucket (``;`` separates them): a short window against bursts and a daily cap per key."""
    return f"{_config.requests} per {max(1, int(_config.window_s))} second;{_config.per_day} per day"


limiter = Limiter(key_func=rate_limit_key, storage_uri="memory://", headers_enabled=False)
