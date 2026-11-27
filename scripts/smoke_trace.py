"""LIVE tracing smoke test (optional; for eyeballing: prints no numbers meant for docs). Needs GEMINI_API_KEY and both Langfuse keys.

    $env:CHROMA_LOCAL_PATH = "$env:LOCALAPPDATA\\advrag\\chroma_v2"
    python scripts/smoke_trace.py

Three requests, three traces (session id "smoke-trace"):
  1. a normal answerable question                 -> rag.request > route, cache.miss, retrieve, context, generation (linked to the Langfuse prompt)
  2. the SAME question again                      -> a cache hit: rag.request > route, cache.hit (no retrieve, no generation)
  3. a question with Gemini forced to fail        -> degraded: ..., generation at level ERROR, root at level ERROR

Then it flushes, reads each trace back through the SDK's observations API (this can take a few seconds to appear), runs the dashboard reader
once and prints each panel's STATUS (to verify the Metrics API field names), and prints the trace URLs. Only public Qasper questions and
passages go to Gemini and Langfuse (capture mode is the config default: masked, so passage text is not sent to Langfuse at all).
Keys come from the environment or .env (allowlisted names; values never printed).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

QUESTION = "In the paper on detecting smart spammers with a topic model, which topic model is used to build the classification features?"
FAILING_QUESTION = "Which argument components does the argumentation mining paper annotate in user-generated web discourse?"
SESSION = "smoke-trace"


def _consume(svc, question: str):
    result = None
    for event in svc.answer_stream(question, session_id=SESSION):
        if event["type"] == "done":
            result = event["result"]
    return result


def _read_back(client, trace_id: str, wait_s: float = 60.0):
    """Observations of one trace. Ingestion is asynchronous and observations arrive piecemeal, so poll until a root is present and the count
    has stopped growing between two polls."""
    deadline = time.monotonic() + wait_s
    rows, previous = [], -1
    while time.monotonic() < deadline:
        try:
            rows = list(client.api.observations.get_many(trace_id=trace_id, fields="core,basic,model,usage,prompt,metrics", limit=100).data)
        except Exception as exc:  # noqa: BLE001
            return rows, f"{type(exc).__name__}"
        has_root = any(not getattr(r, "parent_observation_id", None) for r in rows)
        if rows and has_root and len(rows) == previous:
            return rows, None
        previous = len(rows)
        time.sleep(4)
    return rows, None if rows else "not visible yet (ingestion can lag; open the URL later)"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from rag_core.env import load_env, present

    load_env()
    have = present("GEMINI_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
    if not all(have.values()):
        print(f"SKIPPED: missing {[k for k, v in have.items() if not v]} (environment or .env); nothing was sent anywhere.")
        return 0
    if not os.environ.get("CHROMA_LOCAL_PATH"):
        print("REFUSED: set CHROMA_LOCAL_PATH to the current store (%LOCALAPPDATA%\\advrag\\chroma_v2).", file=sys.stderr)
        return 2

    from config import cfg
    from eval.run_retrieval_eval import build_real_components
    from rag_core.cache import InMemoryTTLCache
    from rag_core.config import PromptsConfig, RagConfig, TracingConfig
    from rag_core.gateway import build_gateway
    from rag_core.langfuse_client import get_langfuse_client
    from rag_core.observability_reader import ObservabilityReader
    from rag_core.prompt_store import build_prompt_store
    from rag_core.service import RagService, index_version_for
    from rag_core.tracing import build_tracer

    components, _ = build_real_components({"recursive"})
    retriever = components["recursive"]["retriever"]
    rag_cfg, tracing_cfg, prompts_cfg = RagConfig.from_config(cfg), TracingConfig.from_config(cfg), PromptsConfig.from_config(cfg)
    tracer = build_tracer(tracing_cfg)
    prompts = build_prompt_store(prompts_cfg)
    version = index_version_for(retriever.store)
    print(f"tracer: {type(tracer).__name__}   prompts mode: {prompts_cfg.mode}   capture: {tracing_cfg.capture}")

    def service(c: RagConfig, cache) -> RagService:
        return RagService(retriever, build_gateway(c), cache, c, index_version=version, prompts=prompts, tracer=tracer, tracing=tracing_cfg)

    cache = InMemoryTTLCache()
    good = service(rag_cfg, cache)
    broken_cfg = RagConfig.from_config(cfg)
    broken_cfg.gemini_model, broken_cfg.max_attempts = "this-model-id-does-not-exist", 1
    broken = service(broken_cfg, None)

    runs = [("1 normal", _consume(good, QUESTION)), ("2 cache hit (same question)", _consume(good, QUESTION)), ("3 Gemini forced to fail", _consume(broken, FAILING_QUESTION))]
    good.flush_traces(30)
    broken.flush_traces(30)

    client = get_langfuse_client()
    print()
    for label, result in runs:
        print(f"=== {label}")
        print(f"  answer head : {result.answer[:110]!r}")
        print(f"  provider    : {result.provider}  cached={result.cached}  degraded={result.degraded}  prompt={result.prompt.source if result.prompt else None}")
        print(f"  trace id    : {result.trace_id}")
        if not result.trace_id:
            print("  (no trace was recorded)")
            continue
        print(f"  trace url   : {client.get_trace_url(trace_id=result.trace_id)}")
        rows, problem = _read_back(client, result.trace_id)
        print(f"  span names  : {[r.name for r in sorted(rows, key=lambda r: str(getattr(r, 'start_time', '')))] if rows else problem}")
        for r in rows:
            if getattr(r, "type", None) == "GENERATION":
                print(f"  generation  : model={getattr(r, 'model', None)} level={getattr(r, 'level', None)} prompt={getattr(r, 'prompt_name', None)} "
                      f"v{getattr(r, 'prompt_version', None)} usage_keys={sorted(getattr(r, 'usage_details', None) or {})}")
        if rows:
            roots = [r for r in rows if not getattr(r, "parent_observation_id", None)]
            print(f"  root        : {[(r.name, getattr(r, 'level', None)) for r in roots]}")

    print("\nDASHBOARD READER (statuses only; this verifies the Metrics API field names; data can lag a few minutes)")
    reader = ObservabilityReader(window_days=1)
    time.sleep(8)
    for name, panel in reader.snapshot(days=1).items():
        print(f"  {name:<30} {panel.status:<12} {panel.message or ''}")
    print("\nTO SCREENSHOT: (1) the trace tree of trace 1 in the Langfuse UI; (2) Prompts > rag-answer-long > Linked generations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
