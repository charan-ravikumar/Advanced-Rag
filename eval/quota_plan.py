"""Wall-time arithmetic for the generation-evaluation plan under per-minute and per-day request and token limits. Pure.

Each limit gives a lower bound on the time the plan needs; the plan needs the LARGEST of them (the binding limit). A daily limit is treated as a
continuous rate (total / limit days), which understates the real time slightly because a day's allowance cannot be spent faster than the
per-minute limits allow and unused allowance does not carry over: ``days_with_resets`` is the number of daily allowances the plan consumes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Limits:
    rpm: float | None = None  # requests per minute
    rpd: float | None = None  # requests per day
    tpm: float | None = None  # tokens per minute
    tpd: float | None = None  # tokens per day


def plan_wall_time(items: dict[str, tuple[int, float]], limits: Limits) -> dict:
    """``items``: label -> (number of calls, mean tokens per call, prompt + completion). Returns totals, the hours each limit alone would need,
    the binding limit and the number of daily allowances consumed."""
    for label, (n, mean) in items.items():
        if n < 0 or mean < 0:
            raise ValueError(f"{label}: negative count or token mean")
    calls = sum(n for n, _ in items.values())
    tokens = sum(n * mean for n, mean in items.values())
    hours: dict[str, float] = {}
    if limits.rpm:
        hours["rpm"] = calls / limits.rpm / 60.0
    if limits.tpm:
        hours["tpm"] = tokens / limits.tpm / 60.0
    if limits.rpd:
        hours["rpd"] = calls / limits.rpd * 24.0
    if limits.tpd:
        hours["tpd"] = tokens / limits.tpd * 24.0
    binding = max(hours, key=hours.get) if hours else None
    day_counts = [1]
    if limits.rpd:
        day_counts.append(math.ceil(calls / limits.rpd))
    if limits.tpd:
        day_counts.append(math.ceil(tokens / limits.tpd))
    return {"calls": calls, "tokens": tokens, "hours_by_limit": hours, "binding_limit": binding,
            "wall_hours": hours[binding] if binding else 0.0, "days_with_resets": max(day_counts)}


def scale_to_budget(items: dict[str, tuple[int, float]], limits: Limits, max_hours: float) -> float:
    """The largest fraction ``f`` in (0, 1] of every arm's calls such that the plan still fits in ``max_hours`` (1.0 if the whole plan fits;
    the wall time is proportional to the number of calls). 0.0 if the plan has no calls."""
    whole = plan_wall_time(items, limits)["wall_hours"]
    if whole <= 0:
        return 0.0 if not any(n for n, _ in items.values()) else 1.0
    return min(1.0, max_hours / whole)


class Pacer:
    """Sliding 60-second window over requests and tokens: ``wait(estimate)`` sleeps just long enough that one more call of about ``estimate``
    tokens stays within ``rpm`` and ``tpm`` (the caller then makes the call and reports the real usage with ``record``). Clock and sleep are
    injected so the tests need no real waiting."""

    def __init__(self, rpm: float | None, tpm: float | None, *, safety: float = 0.9, clock=None, sleep=None):
        import time

        self.rpm, self.tpm, self.safety = rpm, tpm, safety
        self._clock, self._sleep = clock or time.monotonic, sleep or time.sleep
        self._events: list[tuple[float, int]] = []  # (time, tokens)

    def _trim(self, now: float) -> None:
        self._events = [(t, k) for t, k in self._events if now - t < 60.0]

    def wait(self, estimate_tokens: int = 0) -> float:
        """Sleep until the call fits; returns the seconds slept."""
        slept = 0.0
        while True:
            now = self._clock()
            self._trim(now)
            used = sum(k for _, k in self._events)
            over_requests = self.rpm and len(self._events) + 1 > self.rpm * self.safety
            over_tokens = self.tpm and self._events and used + estimate_tokens > self.tpm * self.safety
            if not self._events or not (over_requests or over_tokens):
                return slept
            delay = max(0.05, 60.0 - (now - self._events[0][0]) + 0.05)
            self._sleep(delay)
            slept += delay

    def record(self, tokens: int) -> None:
        self._events.append((self._clock(), int(tokens)))
