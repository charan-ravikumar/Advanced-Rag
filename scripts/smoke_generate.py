"""LIVE smoke test of the generation path (optional; for eyeballing only: prints no timing or token numbers, copy nothing into a doc).

    $env:CHROMA_LOCAL_PATH = "$env:LOCALAPPDATA\\advrag\\chroma_v2"
    python scripts/smoke_generate.py

Runs only when GEMINI_API_KEY is available (from the environment or from .env, loaded by rag_core.env.load_env, which loads an allowlist
of names and never prints a value); otherwise it says so and exits 0. Generation is Gemini only (no failover). What is sent to Gemini:
hand-written PUBLIC questions and passages of public Qasper papers, nothing else.

  A, B. two questions answerable from the corpus                      -> expect a cited answer, valid [S#] markers
  C.    a question the corpus cannot answer                           -> expect abstained (INSUFFICIENT_CONTEXT)
  D.    a question with Gemini forced to fail (an invalid model id)   -> expect a DEGRADED result with the retrieved passages, no crash
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

QUESTIONS = {
    "A (answerable)": "In the paper on detecting smart spammers with a topic model, which topic model is used to build the classification features?",
    "B (answerable)": "What is a Winograd schema and why is it proposed as a challenge for machine translation?",
    "C (not answerable from the corpus)": "What is the boiling point of liquid nitrogen in degrees Celsius?",
    "D (Gemini forced to fail: expect degraded)": "Which argument components does the argumentation mining paper annotate in user-generated web discourse?",
}


def show(label: str, result) -> None:
    print(f"\n=== {label}")
    print(f"answer      : {result.answer}")
    print(f"citations   : {result.citations}   invalid: {result.invalid_citations}")
    print(f"provider    : {result.provider} ({result.model})   abstained: {result.abstained}   degraded: {result.degraded}")
    print(f"attempts    : {[(a.provider, a.outcome, a.error_kind) for a in result.attempts]}")
    print(f"errors      : {[e.code for e in result.errors]}")
    print(f"sources used: {[(s.label, s.doc_id, s.section) for s in result.used_sources]}")
    print(f"retrieved   : {[(s.doc_id, s.section) for s in result.retrieved_sources]}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # model output can contain characters the Windows console codepage lacks
    from rag_core.env import load_env, present

    load_env()  # allowlisted names only; values never printed
    if not present("GEMINI_API_KEY")["GEMINI_API_KEY"]:
        print("SKIPPED: GEMINI_API_KEY is not available (environment or .env); nothing was sent anywhere.")
        return 0
    if not os.environ.get("CHROMA_LOCAL_PATH"):
        print("REFUSED: set CHROMA_LOCAL_PATH to the current store (%LOCALAPPDATA%\\advrag\\chroma_v2).", file=sys.stderr)
        return 2

    from config import cfg
    from eval.run_retrieval_eval import build_real_components
    from rag_core.config import RagConfig
    from rag_core.gateway import build_gateway
    from rag_core.service import RagService, index_version_for

    components, embedder = build_real_components({"recursive"})
    retriever = components["recursive"]["retriever"]
    base = RagConfig.from_config(cfg)
    version = index_version_for(retriever.store)

    def service(rag_cfg: RagConfig) -> RagService:
        return RagService(retriever, build_gateway(rag_cfg), None, rag_cfg, index_version=version)  # no cache: every call is live

    labels = list(QUESTIONS)
    results = {}
    for label in labels[:3]:
        results[label] = _long(service(base), QUESTIONS[label])
        show(label, results[label])
    broken = RagConfig.from_config(cfg)
    broken.gemini_model = "this-model-id-does-not-exist"
    broken.max_attempts = 1  # an invalid model id is a 404: not retried anyway
    results[labels[3]] = _long(service(broken), QUESTIONS[labels[3]])
    show(labels[3], results[labels[3]])

    a, b, c, d = (results[k] for k in labels)
    checks = {
        "A has valid [S#] citations": bool(a.citations) and not a.invalid_citations and not a.degraded,
        "B has valid [S#] citations": bool(b.citations) and not b.invalid_citations and not b.degraded,
        "C abstained": c.abstained and not c.degraded,
        "D is degraded with retrieved passages, no crash": d.degraded and bool(d.retrieved_sources) and any(e.code == "all_providers_failed" for e in d.errors),
    }
    print("\nSMOKE CHECKS")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return 0 if all(checks.values()) else 1


def _long(svc, question: str):
    streamed, result = [], None
    for event in svc.answer_stream(question):
        if event["type"] == "token":
            streamed.append(event["text"])
        elif event["type"] == "done":
            result = event["result"]
    return result


if __name__ == "__main__":
    sys.exit(main())
