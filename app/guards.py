"""Abuse protection for a public demo. PURE: no streamlit, no network, injectable clocks. Every limit comes from ``config.yaml`` (``app.*``).

* ``AccessGate``: the access code (``APP_ACCESS_CODE``). Compared with ``hmac.compare_digest`` on UTF-8 bytes (constant time); unset means the app runs open
  ("demo mode"). A session gets ``access_code_attempts`` wrong codes, then is locked out for ``access_code_lockout_seconds``. The secret is never stored
  anywhere but in the gate's private attribute and is never part of any message, ``repr`` or exception text.
* ``SessionGuard``: one per browser session: a sliding-window rate limit and a cap on answers per session.
* ``DailyBudget``: ONE per process, shared by every session: at most ``daily_request_cap`` LLM-backed answers per UTC day. ``try_acquire`` is atomic across threads.
  Beyond the cap no LLM call is made; the UI shows the retrieved passages only.
* Question length: the cap is ``rag_core``'s (``generation.max_question_chars``, enforced by ``RagService.validate_question``); ``check_question`` calls it so the guard and
  the service can never disagree.
"""

from __future__ import annotations

import hmac
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from rag_core.config import _overlay  # the same overlay as RagConfig (an unknown key is an error, defaults kept for missing ones)


@dataclass
class AppConfig:
    """The ``app`` section of config.yaml."""

    strategy: str = "recursive"
    daily_request_cap: int = 80
    session_request_cap: int = 20
    rate_limit_requests: int = 4
    rate_limit_window_seconds: float = 60.0
    access_code_attempts: int = 5
    access_code_lockout_seconds: float = 60.0
    metrics_ttl_seconds: int = 3600
    results_path: str = "app_data/results.json"
    papers_path: str = "app_data/papers.json"
    manifest_dir: str = "runs/ingest"
    warmup_query: str = "warm up"
    index_version_ttl_seconds: float = 30.0  # how often the served index version (part of every cache key) is re-read, so a re-ingest is seen without a restart
    feedback_path: str = "%LOCALAPPDATA%/advrag/feedback.sqlite3"  # local SQLite file of the thumbs (kept outside OneDrive); created on the first vote

    @classmethod
    def from_config(cls, cfg) -> "AppConfig":
        return _overlay(cls(), getattr(cfg, "app", None), "app")


# ---- access code -----------------------------------------------------------------------------------------------------------------

class AccessGate:
    def __init__(self, secret: str | None, *, max_attempts: int = 5, lockout_s: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self._secret = (secret or "").strip() or None  # an empty or whitespace secret means "not set"
        self.max_attempts, self.lockout_s, self._clock = max_attempts, lockout_s, clock

    @property
    def required(self) -> bool:
        return self._secret is not None

    def __repr__(self) -> str:  # never show the secret, not even its length
        return f"AccessGate(required={self.required})"

    def new_state(self) -> dict:
        return {"unlocked": False, "failures": 0, "locked_until": 0.0}

    def seconds_locked(self, state: dict) -> float:
        return max(0.0, state["locked_until"] - self._clock())

    def is_open(self, state: dict) -> bool:
        """True in demo mode (no secret), or once the session has entered the right code."""
        return (not self.required) or bool(state["unlocked"])

    def attempt(self, state: dict, candidate: str | None) -> tuple[bool, str]:
        """``(ok, message)`` for one entered code. A locked session is refused WITHOUT looking at the code. The message never contains the code or the secret."""
        if self.is_open(state):
            return True, "ok"
        wait = self.seconds_locked(state)
        if wait > 0:
            return False, f"Too many wrong codes. Try again in {int(wait) + 1} seconds."
        entered = (candidate or "").encode("utf-8")
        if hmac.compare_digest(entered, self._secret.encode("utf-8")):  # type: ignore[union-attr]
            state.update(unlocked=True, failures=0, locked_until=0.0)
            return True, "ok"
        state["failures"] += 1
        if state["failures"] >= self.max_attempts:
            state.update(failures=0, locked_until=self._clock() + self.lockout_s)
            return False, f"Wrong code. Locked for {int(self.lockout_s)} seconds."
        return False, "Wrong code."


# ---- per-session limits ------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str | None = None  # "rate_limited" | "session_cap"
    message: str = ""
    retry_after_s: float = 0.0


class SessionGuard:
    """Sliding-window rate limit and a cap on answers, per browser session. ``check`` does not count; ``record`` counts (call it when the request proceeds)."""

    def __init__(self, *, max_requests: int, window_s: float, session_cap: int, clock: Callable[[], float] = time.monotonic):
        self.max_requests, self.window_s, self.session_cap, self._clock = max_requests, window_s, session_cap, clock
        self._times: deque[float] = deque()
        self.total = 0

    def _trim(self, now: float) -> None:
        while self._times and now - self._times[0] >= self.window_s:
            self._times.popleft()

    def check(self) -> Decision:
        if self.total >= self.session_cap:
            return Decision(False, "session_cap", f"This session has reached its limit of {self.session_cap} questions. Reload the page to start a new session.")
        now = self._clock()
        self._trim(now)
        if len(self._times) >= self.max_requests:
            wait = self.window_s - (now - self._times[0])
            return Decision(False, "rate_limited", f"Please wait {int(wait) + 1} seconds before asking again.", retry_after_s=wait)
        return Decision(True)

    def record(self) -> None:
        now = self._clock()
        self._trim(now)
        self._times.append(now)
        self.total += 1

    @property
    def remaining(self) -> int:
        return max(0, self.session_cap - self.total)


# ---- the process-wide daily cap -------------------------------------------------------------------------------------------------------

class DailyBudget:
    """At most ``cap`` LLM-backed answers per UTC day for the whole process (all sessions). In memory: a restart resets it (a cost guard, not an accounting system)."""

    def __init__(self, cap: int, clock: Callable[[], float] = time.time):
        self.cap, self._clock, self._day, self._used, self._lock = int(cap), clock, None, 0, threading.Lock()

    def _roll(self) -> None:
        day = datetime.fromtimestamp(self._clock(), tz=timezone.utc).date()
        if day != self._day:
            self._day, self._used = day, 0

    def try_acquire(self) -> bool:
        """Atomically take one unit; False (and nothing taken) once the cap is reached."""
        with self._lock:
            self._roll()
            if self._used >= self.cap:
                return False
            self._used += 1
            return True

    def release(self) -> None:
        """Give a unit back (the request ended up making no LLM call, for example an invalid question)."""
        with self._lock:
            self._roll()
            self._used = max(0, self._used - 1)

    @property
    def used(self) -> int:
        with self._lock:
            self._roll()
            return self._used

    @property
    def remaining(self) -> int:
        return max(0, self.cap - self.used)


def check_question(service: Any, question: Any) -> str:
    """The cleaned question, or ``rag_core.errors.InvalidQuestion`` (typed). Delegates to ``RagService.validate_question`` so the length cap is ``rag_core``'s."""
    return service.validate_question(question)
