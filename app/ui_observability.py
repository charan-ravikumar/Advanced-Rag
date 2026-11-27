"""The Observability tab: the Langfuse Metrics API v2 through ``rag_core.observability_reader``, ONE snapshot shared by all visitors.

The free plan allows about 100 Metrics API requests a day, so the snapshot sits behind ``st.cache_data`` with a TTL from ``config.yaml`` (``app.metrics_ttl_seconds``, default one hour;
the reader has its own cache and a local daily query budget behind it). Every panel shows its STATUS (``ok`` / ``no_data`` / ``unavailable`` / ``no_keys``), never an exception.
"Sample data" is a clearly labelled toggle (fake values, for the layout; on by default only when no Langfuse keys exist). Traces are masked and hold no passage text, so there is no
faithfulness trend here; the cost panel says so when no prices are configured.
"""

from __future__ import annotations

import functools
from typing import Any

import pandas as pd
import streamlit as st

from rag_core.observability_reader import sample_data

PANELS = (
    ("request_volume_over_time", "Requests over time"),
    ("latency_percentiles_by_stage", "Latency by stage (p50 / p95 / p99)"),
    ("time_to_first_token", "Time to first token"),
    ("token_usage", "Token usage"),
    ("cost", "Cost"),
    ("error_and_degraded_rates", "Errors and degraded answers"),
    ("cache_hit_rate", "Answer cache hit rate"),
    ("feedback_summary", "User feedback (thumbs)"),
    ("prompt_version_breakdown", "Prompt versions"),
)
STATUS_TEXT = {"ok": "ok", "no_data": "no data in the window", "unavailable": "unavailable", "no_keys": "no Langfuse keys"}


def _snapshot(_reader, days: int | None) -> dict[str, dict]:
    """Every panel as a plain dict (what ``st.cache_data`` stores). The reader is excluded from the cache key (leading underscore)."""
    return {name: panel.as_dict() for name, panel in _reader.snapshot(days).items()}


@functools.lru_cache(maxsize=None)
def _cached(ttl_seconds: int):
    return st.cache_data(ttl=ttl_seconds, show_spinner="Reading the Langfuse metrics…")(_snapshot)


def get_snapshot(reader, ttl_seconds: int, days: int | None = None) -> dict[str, dict]:
    """The shared snapshot: one Metrics API read per TTL for ALL sessions."""
    try:
        return _cached(int(ttl_seconds))(reader, days)
    except Exception:  # noqa: BLE001: the reader never raises; this guards the cache layer itself
        return {name: {"status": "unavailable", "data": None, "as_of": None, "message": "the metrics snapshot could not be read", "sample": False} for name, _ in PANELS}


def _pct(value) -> str:
    return "—" if value is None else f"{value:.1%}"


def _table(rows: list[dict], columns: dict[str, str]) -> None:
    if rows:
        st.dataframe(pd.DataFrame(rows)[list(columns)].rename(columns=columns), hide_index=True, width="stretch")


def _body(name: str, data: Any, prices_configured: bool) -> None:
    if name == "request_volume_over_time":
        points = data.get("points") or []
        if points:
            df = pd.DataFrame(points)
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            st.bar_chart(df, x="time", y="requests", width="stretch")
        else:
            st.caption("No requests in the window.")
    elif name == "latency_percentiles_by_stage":
        stages = data.get("stages") or []
        _table(stages, {"stage": "Stage", "p50_ms": "p50 (ms)", "p95_ms": "p95 (ms)", "p99_ms": "p99 (ms)"})
        st.caption("`rag.request` is the whole request; `retrieve`, `context` and `generation` are its parts. Latencies are measured by Langfuse from the app's own spans.")
    elif name == "time_to_first_token":
        c1, c2, c3 = st.columns(3)
        c1.metric("p50 (ms)", "—" if data.get("p50_ms") is None else f"{data['p50_ms']:.0f}")
        c2.metric("p95 (ms)", "—" if data.get("p95_ms") is None else f"{data['p95_ms']:.0f}")
        c3.metric("Generations", data.get("generations", 0))
    elif name == "token_usage":
        _table(data.get("models") or [], {"model": "Model", "input_tokens": "Input tokens", "output_tokens": "Output tokens", "total_tokens": "Total tokens"})
        st.metric("Total tokens in the window", data.get("total_tokens", 0))
    elif name == "cost":
        if not prices_configured:
            st.info("Prices not configured (`generation.prices` is empty), so no cost is shown: a cost figure would be an estimate.")
        else:
            _table(data.get("models") or [], {"model": "Model", "cost_usd": "Cost (USD)"})
            st.metric("Total cost in the window (USD)", f"{data.get('total_cost_usd', 0.0):.4f}")
            st.caption(data.get("note") or "")
    elif name == "error_and_degraded_rates":
        c1, c2, c3 = st.columns(3)
        c1.metric("Requests", data.get("requests", 0))
        c2.metric("Errors", f"{data.get('errors', 0)} ({_pct(data.get('error_rate'))})")
        c3.metric("Degraded (passages only)", f"{data.get('degraded', 0)} ({_pct(data.get('degraded_rate'))})")
        st.caption("Errors are root observations at level ERROR; degraded requests are at level WARNING.")
    elif name == "cache_hit_rate":
        c1, c2, c3 = st.columns(3)
        c1.metric("Hits", data.get("hits", 0))
        c2.metric("Misses", data.get("misses", 0))
        c3.metric("Hit rate", _pct(data.get("hit_rate")))
    elif name == "feedback_summary":
        c1, c2, c3 = st.columns(3)
        c1.metric("Ratings", data.get("count", 0))
        c2.metric("Thumbs up", data.get("positive", 0))
        c3.metric("Thumbs down", data.get("negative", 0))
        st.caption(f"Share of thumbs up: {_pct(data.get('positive_rate'))}")
    elif name == "prompt_version_breakdown":
        _table(data.get("prompts") or [], {"name": "Prompt", "version": "Version", "generations": "Generations", "p95_latency_ms": "p95 latency (ms)"})


def render_panel(name: str, title: str, panel: dict, prices_configured: bool) -> None:
    status = panel.get("status", "unavailable")
    with st.container(border=True):
        st.markdown(f"**{title}**" + ("  ·  :orange[SAMPLE DATA]" if panel.get("sample") else ""))
        st.caption(f"Status: {STATUS_TEXT.get(status, status)}" + (f" · as of {panel['as_of']}" if panel.get("as_of") else ""))
        if status == "ok":
            _body(name, panel["data"], prices_configured)
        elif status == "no_data":
            st.info(panel.get("message") or "No data in the selected window.")
        elif status == "no_keys":
            st.info("Langfuse keys are not configured, so there is nothing to read. Turn on the sample data to see the layout.")
        else:
            st.warning(panel.get("message") or "The metrics could not be read right now.")
        if panel.get("sample") and panel.get("message"):
            st.caption(panel["message"])


def render_observability(res) -> None:
    st.subheader("Observability")
    st.caption("Metrics of the answers this app has served, read from Langfuse. Traces are masked: they hold no passage text, so there is no faithfulness trend here "
               "(faithfulness is measured offline; see the Evaluation tab).")
    use_sample = st.toggle("Show sample data (fake values, for the layout only)", value=not res.langfuse_keys_present, key="obs_sample")
    if use_sample:
        st.warning("SAMPLE DATA: the values below are fake and exist only to show the layout. They are not measurements of this app.")
        panels = {n: p.as_dict() for n, p in sample_data().items()}
    else:
        panels = get_snapshot(res.reader, res.settings.metrics_ttl_seconds)
        window = getattr(res.reader, "window_days", None)
        st.caption(f"Live data from Langfuse{f' (last {window} days)' if window else ''}. One snapshot is cached for {res.settings.metrics_ttl_seconds // 60} minutes and shared by all visitors "
                   "(the free plan allows very few Metrics API requests per day).")
    cols = st.columns(2)
    for i, (name, title) in enumerate(PANELS):
        with cols[i % 2]:
            render_panel(name, title, panels.get(name) or {"status": "unavailable", "message": "no data for this panel"}, res.prices_configured)
