"""Shared fakes for the app tests: a ready ``Resources`` around a fake retriever and scripted provider, a fake observability reader, artifact helpers and an AppTest runner.
No network, no keys, no real index."""

from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app.artifacts import Artifacts, load_artifacts
from app.bootstrap import OVERRIDE_KEY, Check, Resources, SelfCheck
from app.guards import AppConfig, DailyBudget
from rag_core.config import RagConfig, TracingConfig
from rag_core.observability_reader import Panel
from rag_core.scores import record_feedback
from rag_core.tracing import FakeTracer
from tests.unit.rag_core_fakes import answer_json, FakeLangfuse, make_hit, ok
from tests.unit.test_rag_core_service import make

ROOT = Path(__file__).resolve().parents[2]
ENTRY = str(ROOT / "app" / "streamlit_app.py")
PAPER_ID = "1234.00001"
PAPERS = [{"id": PAPER_ID, "title": "Attention Is All You Need (test fixture)"}, {"id": "2222.00002", "title": "A Second Paper (test fixture)"}]
EXAMPLES = [{"question_id": f"{PAPER_ID}:q{i}", "paper_id": PAPER_ID, "paper_title": PAPERS[0]["title"], "question": f"Example question number {i}?"} for i in range(1, 5)]


def real_artifacts() -> Artifacts:
    """The committed artifacts (app_data/): what the app really shows."""
    return load_artifacts("app_data/results.json", "app_data/papers.json", ROOT)


def fixture_artifacts(results: dict | None = None) -> Artifacts:
    return Artifacts(results=results, papers=list(PAPERS), examples=list(EXAMPLES), source={"license": "CC BY 4.0 (fixture)", "link": "https://example.org/qasper"})


class FakeReader:
    """Stands in for ``ObservabilityReader``: counts snapshot calls; ``statuses`` maps a panel to a status (default ``ok`` with plausible data)."""

    window_days = 7

    def __init__(self, status: str = "ok"):
        self.status, self.calls = status, 0

    def snapshot(self, days=None):
        self.calls += 1
        from rag_core.observability_reader import ObservabilityReader

        if self.status == "ok":
            def http(url, params, auth, timeout_s):
                import json as _json

                from tests.unit.test_rag_core_observability import ok_rows

                return 200, {"data": ok_rows(_json.loads(params["query"]))}

            return ObservabilityReader(http_get=http, env={"LANGFUSE_PUBLIC_KEY": "a", "LANGFUSE_SECRET_KEY": "b"}).snapshot(days)
        names = ("request_volume_over_time", "latency_percentiles_by_stage", "time_to_first_token", "token_usage", "cost", "error_and_degraded_rates", "cache_hit_rate",
                 "feedback_summary", "prompt_version_breakdown")
        return {n: Panel(self.status, {}, "2026-11-23T10:00:00+00:00", None if self.status == "no_keys" else f"fake {self.status}") for n in names}


def make_resources(*, primary=None, secondary=None, hits=None, artifacts=None, retriever=None, keys=True, daily_cap=80, session_cap=20, rate=100, reader=None, feedback=None,
                   rerank_available=True, selfcheck=None, tracing=True, prices=False, index_version="idx-test") -> Resources:
    tracer = FakeTracer() if tracing else None
    svc, retr, a, b, cache, gw = make(primary or [ok(answer_json(("Attention weighs the tokens.", ["S1"])))], secondary, hits=hits, retriever=retriever, tracer=tracer,
                                      tracing=TracingConfig(enabled=tracing), index_version=index_version, cfg=RagConfig(prices={"model-a": {"input": 1.0, "output": 1.0}} if prices else {}))
    check = selfcheck or SelfCheck([Check("Index directory", True, "set and found"), Check("Gemini API key present", keys, "present" if keys else "missing", fatal=False)])
    res = Resources(settings=AppConfig(daily_request_cap=daily_cap, session_request_cap=session_cap, rate_limit_requests=rate, metrics_ttl_seconds=3600), selfcheck=check,
                    service=svc, daily=DailyBudget(daily_cap), reader=reader or FakeReader(), artifacts=artifacts if artifacts is not None else fixture_artifacts(),
                    record_feedback=feedback or (lambda request_id, trace_id, value, comment=None: record_feedback(None, request_id, trace_id, value, comment)),
                    gemini_key_present=keys, langfuse_keys_present=keys, tracing_on=tracing, prices_configured=prices,
                    reranker_available=rerank_available, index_dir="fake-index",
                    version_info={"index_dir": "fake-index", "chunk_count": 8, "index_version": index_version, "manifest": "recursive_fake", "manifest_sha256": "ab" * 32,
                                  "embedder": {"name": "fake-embedder", "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"},
                                  "reranker": {"name": "fake-reranker", "revision": "233902d25c440f23af6f7d6e94d2946bac0bee0a", "cached": rerank_available},
                                  "prompt_version": "test-prompt-version", "prompt_mode": "local", "generator": {"model": "model-a", "reasoning_effort": "low"},
                                  "retrieval": {"mode": "hybrid", "fusion": "rrf", "rrf_k": 60, "vector_weight": 1.0, "lexical_weight": 0.7, "top_k": 5}})
    res._fake = {"retriever": retr, "primary": a, "tracer": tracer, "cache": cache}  # for assertions
    return res


def run_app(resources: Resources | None = None, *, timeout: float = 60) -> AppTest:
    at = AppTest.from_file(ENTRY, default_timeout=timeout)
    if resources is not None:
        at.session_state[OVERRIDE_KEY] = resources
    return at.run()


def ask(at: AppTest, question: str) -> AppTest:
    at.chat_input[0].set_value(question)
    return at.run()


def all_text(at: AppTest) -> str:
    """Every piece of text the page shows (markdown, captions, titles, alerts, widget labels, dataframe cells)."""
    parts = []
    for kind in ("markdown", "caption", "title", "header", "subheader", "warning", "error", "info", "success", "text"):
        parts += [str(e.value) for e in getattr(at, kind)]
    for e in at.dataframe:
        parts.append(e.value.to_csv(index=False))
    for kind in ("button", "text_input", "toggle", "radio", "selectbox", "expander"):
        parts += [str(getattr(e, "label", "")) for e in getattr(at, kind)]
    return "\n".join(parts)
