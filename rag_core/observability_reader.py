"""Read side for the Observability tab: aggregates from the Langfuse Metrics API v2 (``GET /api/public/v2/metrics``).

Every function returns a ``Panel`` and NEVER raises: ``status`` is one of

* ``"ok"``          data returned;
* ``"no_data"``     the query worked but there is nothing in the window (``data`` is the explicit empty structure);
* ``"no_keys"``     Langfuse keys are not available (nothing was sent anywhere);
* ``"unavailable"`` the API failed, rate-limited us, rejected the query, or our own daily query budget is used up (``message`` says why).

``data`` always has the same shape for a panel (an explicit empty structure when there is nothing), so the UI never needs ``None`` checks.

Plan limits that shape this module (docs/langfuse_notes.md): on the free Hobby plan the Metrics API v2 allows 100 requests PER DAY, so
there is one TTL cache (default 15 minutes) over every query, failures are cached briefly (no hammering), and a local daily query budget stops
us before the API does. Data can lag by up to 15 minutes for old SDKs; each panel reports ``as_of``. ``sample_data()`` returns the same
structures filled with clearly-labelled FAKE values for UI development.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

HttpGet = Callable[[str, dict, tuple, float], tuple]  # (url, params, (user, password), timeout_s) -> (status_code, json_or_None)
DEFAULT_BASE_URL = "https://cloud.langfuse.com"
TTL_S = 900.0
ERROR_TTL_S = 60.0
DAILY_QUERY_BUDGET = 80  # below the free plan's 100/day, so a few manual checks still fit
ROW_LIMIT = 1000


@dataclass
class Panel:
    status: str
    data: Any
    as_of: str | None = None
    message: str | None = None
    sample: bool = False

    def as_dict(self) -> dict:
        return {"status": self.status, "data": self.data, "as_of": self.as_of, "message": self.message, "sample": self.sample}


def _httpx_get(url: str, params: dict, auth: tuple, timeout_s: float):
    import httpx  # lazy: importing this module never imports an HTTP client

    resp = httpx.get(url, params=params, auth=auth, timeout=timeout_s)
    try:
        body = resp.json()
    except ValueError:
        body = None
    return resp.status_code, body


class ObservabilityReader:
    def __init__(self, *, http_get: HttpGet | None = None, env: Mapping[str, str] | None = None, ttl_s: float = TTL_S, window_days: int = 7,
                 clock: Callable[[], float] = time.time, daily_budget: int = DAILY_QUERY_BUDGET, timeout_s: float = 20.0):
        self._get, self._env = http_get or _httpx_get, env
        self.ttl_s, self.window_days, self._clock, self.daily_budget, self.timeout_s = ttl_s, window_days, clock, daily_budget, timeout_s
        self._cache: dict[str, tuple[float, tuple]] = {}
        self._lock = threading.Lock()
        self._day: date | None = None
        self._used = 0
        self.queries_sent = 0

    # ---- plumbing ------------------------------------------------------------------------------------------------------
    def _environ(self) -> Mapping[str, str]:
        return os.environ if self._env is None else self._env

    def _now(self) -> datetime:
        return datetime.fromtimestamp(self._clock(), tz=timezone.utc)

    def _budget_left(self) -> bool:
        day = self._now().date()
        with self._lock:
            if day != self._day:
                self._day, self._used = day, 0
            return self._used < self.daily_budget

    def _run(self, query: dict) -> tuple[str, list | None, str | None]:
        """``(status, rows, message)`` for one Metrics API query, cached. Never raises."""
        env = self._environ()
        public, secret = env.get("LANGFUSE_PUBLIC_KEY"), env.get("LANGFUSE_SECRET_KEY")
        if not public or not secret:
            return "no_keys", None, "Langfuse keys are not available"
        # The window slides with the clock, so the cache key uses its LENGTH, not its timestamps (otherwise a repeat read would never hit).
        stable = {k: v for k, v in query.items() if k not in ("fromTimestamp", "toTimestamp")}
        try:
            span_s = (datetime.fromisoformat(query["toTimestamp"]) - datetime.fromisoformat(query["fromTimestamp"])).total_seconds()
        except (KeyError, ValueError):
            span_s = None
        key = json.dumps({"q": stable, "window_s": span_s}, sort_keys=True)
        now = self._clock()
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None and now - hit[0] < (self.ttl_s if hit[1][0] == "ok" else ERROR_TTL_S):
            return hit[1]
        if not self._budget_left():
            return "unavailable", None, "the local daily query budget for the Langfuse Metrics API is used up"
        base = (env.get("LANGFUSE_BASE_URL") or env.get("LANGFUSE_HOST") or DEFAULT_BASE_URL).rstrip("/")
        result: tuple[str, list | None, str | None]
        try:
            with self._lock:
                self._used += 1
                self.queries_sent += 1
            code, body = self._get(f"{base}/api/public/v2/metrics", {"query": json.dumps(query)}, (public, secret), self.timeout_s)
        except Exception as exc:  # noqa: BLE001: a network error is a status, not an exception
            result = ("unavailable", None, f"request failed ({type(exc).__name__})")
        else:
            if code == 200 and isinstance(body, dict):
                result = ("ok", list(body.get("data") or []), None)
            elif code == 429:
                result = ("unavailable", None, "rate limited by Langfuse (the free plan allows few Metrics API requests per day)")
            elif code in (401, 403):
                result = ("unavailable", None, "Langfuse rejected the credentials")
            else:
                reason = (body or {}).get("message") if isinstance(body, dict) else None
                result = ("unavailable", None, f"Langfuse returned HTTP {code}" + (f": {str(reason)[:120]}" if reason else ""))
        with self._lock:
            self._cache[key] = (now, result)
        return result

    def _window(self, days: int | None = None) -> dict:
        end = self._now()
        return {"fromTimestamp": (end - timedelta(days=days or self.window_days)).isoformat(), "toTimestamp": end.isoformat()}

    def _panel(self, query: dict, empty: Any, build: Callable[[list], Any]) -> Panel:
        status, rows, message = self._run(query)
        as_of = self._now().isoformat()
        if status != "ok":
            return Panel(status, empty, as_of, message)
        if not rows:
            return Panel("no_data", empty, as_of, "no observations in the selected window")
        try:
            return Panel("ok", build(rows), as_of)
        except Exception as exc:  # noqa: BLE001: an unexpected response shape is a status too
            return Panel("unavailable", empty, as_of, f"unexpected response shape ({type(exc).__name__})")

    @staticmethod
    def _num(row: dict, name: str) -> float:
        value = row.get(name)
        return float(value) if value not in (None, "") else 0.0

    # ---- panels --------------------------------------------------------------------------------------------------------
    ROOT = {"column": "isRootObservation", "operator": "=", "value": True, "type": "boolean"}

    def request_volume_over_time(self, days: int | None = None, granularity: str = "day") -> Panel:
        query = {"view": "observations", "metrics": [{"measure": "count", "aggregation": "count"}], "dimensions": [], "filters": [self.ROOT],
                 "timeDimension": {"granularity": granularity}, "orderBy": [{"field": "time_dimension", "direction": "asc"}],
                 "config": {"row_limit": ROW_LIMIT}, **self._window(days)}
        return self._panel(query, {"granularity": granularity, "points": []},
                           lambda rows: {"granularity": granularity, "points": [{"time": r.get("time_dimension"), "requests": int(self._num(r, "count_count"))} for r in rows]})

    def latency_percentiles_by_stage(self, days: int | None = None) -> Panel:
        query = {"view": "observations", "metrics": [{"measure": "latency", "aggregation": a} for a in ("p50", "p95", "p99")],
                 "dimensions": [{"field": "name"}], "filters": [], "orderBy": [{"field": "p95_latency", "direction": "desc"}],
                 "config": {"row_limit": ROW_LIMIT}, **self._window(days)}
        return self._panel(query, {"stages": []}, lambda rows: {"stages": [
            {"stage": r.get("name"), "p50_ms": self._num(r, "p50_latency"), "p95_ms": self._num(r, "p95_latency"), "p99_ms": self._num(r, "p99_latency")} for r in rows]})

    def time_to_first_token(self, days: int | None = None) -> Panel:
        query = {"view": "observations", "metrics": [{"measure": "timeToFirstToken", "aggregation": a} for a in ("p50", "p95")] + [{"measure": "count", "aggregation": "count"}],
                 "dimensions": [], "filters": [{"column": "type", "operator": "=", "value": "GENERATION", "type": "string"}],
                 "config": {"row_limit": 10}, **self._window(days)}
        return self._panel(query, {"p50_ms": None, "p95_ms": None, "generations": 0}, lambda rows: {
            "p50_ms": self._num(rows[0], "p50_timeToFirstToken"), "p95_ms": self._num(rows[0], "p95_timeToFirstToken"), "generations": int(self._num(rows[0], "count_count"))})

    def token_usage(self, days: int | None = None) -> Panel:
        query = {"view": "observations", "metrics": [{"measure": m, "aggregation": "sum"} for m in ("inputTokens", "outputTokens", "totalTokens")],
                 "dimensions": [{"field": "providedModelName"}], "filters": [], "orderBy": [{"field": "sum_totalTokens", "direction": "desc"}],
                 "config": {"row_limit": 50}, **self._window(days)}

        def build(rows: list) -> dict:
            models = [{"model": r.get("providedModelName"), "input_tokens": int(self._num(r, "sum_inputTokens")), "output_tokens": int(self._num(r, "sum_outputTokens")),
                       "total_tokens": int(self._num(r, "sum_totalTokens"))} for r in rows if r.get("providedModelName")]
            return {"models": models, "total_tokens": sum(m["total_tokens"] for m in models)}

        return self._panel(query, {"models": [], "total_tokens": 0}, build)

    def cost(self, days: int | None = None) -> Panel:
        query = {"view": "observations", "metrics": [{"measure": "totalCost", "aggregation": "sum"}], "dimensions": [{"field": "providedModelName"}],
                 "filters": [], "orderBy": [{"field": "sum_totalCost", "direction": "desc"}], "config": {"row_limit": 50}, **self._window(days)}

        def build(rows: list) -> dict:
            models = [{"model": r.get("providedModelName"), "cost_usd": self._num(r, "sum_totalCost")} for r in rows if r.get("providedModelName")]
            return {"models": models, "total_cost_usd": sum(m["cost_usd"] for m in models), "note": "only generations with ingested or inferred cost are counted"}

        return self._panel(query, {"models": [], "total_cost_usd": 0.0, "note": ""}, build)

    def error_and_degraded_rates(self, days: int | None = None) -> Panel:
        """Root observations by level: ERROR = a failed request, WARNING = degraded (the service set these levels; see ``_close_trace``).
        The Metrics API has no ``statusMessage`` dimension, so the level is the only way to tell them apart."""
        query = {"view": "observations", "metrics": [{"measure": "count", "aggregation": "count"}], "dimensions": [{"field": "level"}], "filters": [self.ROOT],
                 "config": {"row_limit": 20}, **self._window(days)}

        def build(rows: list) -> dict:
            by_level: dict[str, int] = {}
            for r in rows:
                level = str(r.get("level", "")).upper()
                by_level[level] = by_level.get(level, 0) + int(self._num(r, "count_count"))
            total, errors, degraded = sum(by_level.values()), by_level.get("ERROR", 0), by_level.get("WARNING", 0)
            return {"requests": total, "errors": errors, "degraded": degraded, "error_rate": errors / total if total else None,
                    "degraded_rate": degraded / total if total else None}

        return self._panel(query, {"requests": 0, "errors": 0, "degraded": 0, "error_rate": None, "degraded_rate": None}, build)

    def cache_hit_rate(self, days: int | None = None) -> Panel:
        query = {"view": "observations", "metrics": [{"measure": "count", "aggregation": "count"}], "dimensions": [{"field": "name"}],
                 "filters": [{"column": "name", "operator": "any of", "value": ["cache.hit", "cache.miss"], "type": "stringOptions"}],
                 "config": {"row_limit": 10}, **self._window(days)}

        def build(rows: list) -> dict:
            counts = {r.get("name"): int(self._num(r, "count_count")) for r in rows}
            hits, misses = counts.get("cache.hit", 0), counts.get("cache.miss", 0)
            return {"hits": hits, "misses": misses, "lookups": hits + misses, "hit_rate": hits / (hits + misses) if hits + misses else None}

        return self._panel(query, {"hits": 0, "misses": 0, "lookups": 0, "hit_rate": None}, build)

    def feedback_summary(self, days: int | None = None) -> Panel:
        query = {"view": "scores-boolean", "metrics": [{"measure": "count", "aggregation": "count"}, {"measure": "value", "aggregation": "avg"}],
                 "dimensions": [{"field": "name"}], "filters": [{"column": "name", "operator": "=", "value": "user_feedback", "type": "string"}],
                 "config": {"row_limit": 10}, **self._window(days)}

        def build(rows: list) -> dict:
            n, rate = int(self._num(rows[0], "count_count")), self._num(rows[0], "avg_value")
            return {"count": n, "positive_rate": rate, "positive": round(n * rate), "negative": n - round(n * rate)}

        return self._panel(query, {"count": 0, "positive_rate": None, "positive": 0, "negative": 0}, build)

    def prompt_version_breakdown(self, days: int | None = None) -> Panel:
        query = {"view": "observations", "metrics": [{"measure": "count", "aggregation": "count"}, {"measure": "latency", "aggregation": "p95"}],
                 "dimensions": [{"field": "promptName"}, {"field": "promptVersion"}],
                 "filters": [{"column": "type", "operator": "=", "value": "GENERATION", "type": "string"}],
                 "orderBy": [{"field": "count_count", "direction": "desc"}], "config": {"row_limit": 50}, **self._window(days)}
        return self._panel(query, {"prompts": []}, lambda rows: {"prompts": [
            {"name": r.get("promptName") or "(local template: not linked)", "version": r.get("promptVersion"), "generations": int(self._num(r, "count_count")),
             "p95_latency_ms": self._num(r, "p95_latency")} for r in rows]})

    def snapshot(self, days: int | None = None) -> dict[str, Panel]:
        """Every panel (each uses its own cached query)."""
        names = ("request_volume_over_time", "latency_percentiles_by_stage", "time_to_first_token", "token_usage", "cost", "error_and_degraded_rates",
                 "cache_hit_rate", "feedback_summary", "prompt_version_breakdown")
        return {n: getattr(self, n)(days) for n in names}


def sample_data() -> dict[str, Panel]:
    """The same panel structures with FAKE, clearly labelled values (``sample=True``) for UI development. Not measurements."""
    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    points = [{"time": (now - timedelta(days=6 - i)).isoformat(), "requests": r} for i, r in enumerate([3, 7, 5, 9, 4, 8, 6])]
    stamp = now.isoformat()
    note = "SAMPLE DATA for UI development; not real measurements"
    mk = lambda data: Panel("ok", data, stamp, note, sample=True)  # noqa: E731
    return {
        "request_volume_over_time": mk({"granularity": "day", "points": points}),
        "latency_percentiles_by_stage": mk({"stages": [{"stage": n, "p50_ms": a, "p95_ms": b, "p99_ms": c} for n, a, b, c in (
            ("rag.request", 1000.0, 2000.0, 3000.0), ("retrieve", 100.0, 200.0, 300.0), ("generation", 800.0, 1700.0, 2600.0), ("context", 5.0, 9.0, 12.0), ("route", 1.0, 2.0, 3.0))]}),
        "time_to_first_token": mk({"p50_ms": 400.0, "p95_ms": 900.0, "generations": 40}),
        "token_usage": mk({"models": [{"model": "sample-model", "input_tokens": 12000, "output_tokens": 3000, "total_tokens": 15000}], "total_tokens": 15000}),
        "cost": mk({"models": [{"model": "sample-model", "cost_usd": 0.0}], "total_cost_usd": 0.0, "note": "sample"}),
        "error_and_degraded_rates": mk({"requests": 40, "errors": 2, "degraded": 1, "error_rate": 0.05, "degraded_rate": 0.025}),
        "cache_hit_rate": mk({"hits": 10, "misses": 30, "lookups": 40, "hit_rate": 0.25}),
        "feedback_summary": mk({"count": 12, "positive_rate": 0.75, "positive": 9, "negative": 3}),
        "prompt_version_breakdown": mk({"prompts": [{"name": "rag-answer-long", "version": 1, "generations": 30, "p95_latency_ms": 1800.0}, {"name": "(local template: not linked)", "version": None, "generations": 5, "p95_latency_ms": 1700.0}]}),
    }
