"""QUOTA DRY RUN for the generation evaluation (step 3 of Phase 4). Makes a SMALL number of real calls to measure what the plan would cost;
its answers are not analysed and are not reused for any result.

    $env:CHROMA_LOCAL_PATH = "$env:LOCALAPPDATA\\advrag\\chroma_v2"
    python scripts/quota_dry_run.py [--n 5]

What it does (public Qasper questions and public paper text only; keys from the environment or .env, never printed):
1. Reads the dev split and counts what the plan would run (answerable, unanswerable, questions with an oracle context, usable references).
2. Estimates full-paper prompt sizes (arm F) with the conservative token estimator: no network.
3. Generator (Gemini): runs ``--n`` seeded dev questions through arms R, O, T in short mode and R, O in long mode and records the REAL usage per call.
4. Gemini judge: a short burst of tiny calls to see the per-minute limit (stops at the first 429 and records its public quota text, digits masked).
5. Projects the whole plan under the planning limits in ``GROQ_LIMITS`` (the name is historical: these are now UNVERIFIED conservative Gemini values)
   and writes ``runs/gen_dryrun/<UTC>_quota_dry_run/dry_run.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GROQ_LIMITS = {"rpm": 10, "rpd": 1000, "tpm": 100_000, "tpd": 1_000_000,
               "source": "UNVERIFIED conservative planning values, NOT documented limits of gemini-3.5-flash-lite: check the quota of your key in Google AI Studio"}
PREREG = ROOT / "eval" / "PREREGISTRATION_gen.md"
PREREG_HASH = ROOT / "eval" / "PREREGISTRATION_gen.sha256"


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return float(s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))])


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def sample_ids(ids: list[str], n: int, seed: int) -> list[str]:
    return sorted(ids, key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())[:n]


def gemini_rpm_probe(max_calls: int = 16) -> dict:
    """Tiny calls in quick succession until the first 429. Records the status of each call and the (digit-masked) public quota text."""
    from google import genai
    from google.genai import types

    from config import cfg
    from rag_core.config import RagConfig
    from rag_core.tracing import mask_text

    judge = RagConfig.from_config(cfg).judge
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=types.HttpOptions(
        api_version="v1beta", timeout=30000, retry_options=types.HttpRetryOptions(attempts=1)))
    out = {"model": judge["model"], "calls_made": 0, "first_429_after_calls": None, "quota_text": None, "other_errors": []}
    started = time.monotonic()
    for i in range(max_calls):
        try:
            client.models.generate_content(model=judge["model"], contents="Reply with OK", config=types.GenerateContentConfig(
                temperature=0.0, max_output_tokens=16, thinking_config=types.ThinkingConfig(thinking_budget=0)))
            out["calls_made"] += 1
        except Exception as exc:  # noqa: BLE001
            if getattr(exc, "code", None) == 429:
                out["first_429_after_calls"] = out["calls_made"]
                out["quota_text"] = mask_text(str(exc), 700)
                break
            out["other_errors"].append(f"{type(exc).__name__}:{getattr(exc, 'code', None)}")
            break
    out["seconds"] = round(time.monotonic() - started, 1)
    return out


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=5, help="questions per arm and mode for the real Gemini calls")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-gemini", action="store_true")
    args = ap.parse_args(argv)

    from rag_core.env import load_env, present

    load_env()
    have = present("GEMINI_API_KEY")
    print("keys available (booleans):", have)
    if not have["GEMINI_API_KEY"]:
        print("SKIPPED: GEMINI_API_KEY is not available; nothing was sent anywhere.")
        return 0
    if not os.environ.get("CHROMA_LOCAL_PATH"):
        print("REFUSED: set CHROMA_LOCAL_PATH to %LOCALAPPDATA%\\advrag\\chroma_v2.", file=sys.stderr)
        return 2

    from config import cfg
    from eval.gen_arms import (
        FixedContextRetriever,
        oracle_context,
        title_abstract_context,
        usable_references,
    )
    from eval.gold_spans import paragraph_index
    from eval.quota_plan import Limits, Pacer, plan_wall_time, scale_to_budget
    from eval.run_retrieval_eval import build_real_components, preregistration_info
    from rag.tokens import estimate_token_count
    from rag_core.cache import InMemoryTTLCache  # noqa: F401  (documented: no cache is used here)
    from rag_core.config import RagConfig
    from rag_core.gateway import build_gateway
    from rag_core.service import RagService, index_version_for
    from utils.fingerprint import code_fingerprint

    prereg = preregistration_info(PREREG, PREREG_HASH)  # refuses if the pre-registered text changed
    data = ROOT / "eval" / "datasets" / "qasper_dev"
    golden = json.loads((data / "golden_set_v2.json").read_text(encoding="utf-8"))
    split = json.loads((data / "split.json").read_text(encoding="utf-8"))
    corpus = json.loads((data / "corpus.json").read_text(encoding="utf-8"))
    by_id = {q["id"]: q for q in golden["questions"]}
    dev = [by_id[i] for i in sorted(split["dev"])]
    answerable = [q for q in dev if q["answerable"]]
    unanswerable = [q for q in dev if not q["answerable"]]
    paragraphs = {pid: paragraph_index(pid, corpus[pid]) for pid in {q["paper_id"] for q in dev}}
    oracle = {q["id"]: oracle_context(q["paper_id"], q["references"], paragraphs[q["paper_id"]]) for q in answerable}
    o_defined = [q for q in answerable if oracle[q["id"]].hits]
    usable = [q for q in answerable if usable_references(q["references"])]

    counts = {"dev_questions": len(dev), "answerable": len(answerable), "unanswerable": len(unanswerable),
              "answerable_with_a_usable_reference": len(usable), "answerable_with_an_oracle_context": len(o_defined),
              "answerable_oracle_truncated_to_5_paragraphs": sum(1 for q in answerable if oracle[q["id"]].n_dropped_over_cap),
              "oracle_evidence_texts_not_found_in_paper": sum(oracle[q["id"]].n_not_found for q in answerable)}
    full_paper_tokens = []
    for pid in sorted({q["paper_id"] for q in dev}):
        text = " ".join(p.text for p in paragraphs[pid])
        full_paper_tokens.append(estimate_token_count(corpus[pid]["title"] + "\n" + text))
    full_paper = {"papers": len(full_paper_tokens), "estimated_tokens_p50": pct(full_paper_tokens, 0.5), "estimated_tokens_p95": pct(full_paper_tokens, 0.95),
                  "estimated_tokens_max": max(full_paper_tokens),
                  "estimator": "rag.tokens.estimate_token_count (conservative, over-counts for the generator's BPE tokenizer)"}
    print("plan counts:", counts)
    print("full-paper prompt estimate (arm F):", full_paper)

    # ---- real Gemini calls ------------------------------------------------------------------------------------------------
    components, _ = build_real_components({"recursive"})
    retriever = components["recursive"]["retriever"]
    rag_cfg = RagConfig.from_config(cfg)
    rag_cfg.max_attempts = 5
    rag_cfg.timeout_s = 60.0
    gateway = build_gateway(rag_cfg)
    version = index_version_for(retriever.store)
    fixed_o, fixed_t = FixedContextRetriever("oracle"), FixedContextRetriever("title_abstract")
    services = {"R": RagService(retriever, gateway, None, rag_cfg, index_version=version),
                "O": RagService(fixed_o, gateway, None, rag_cfg, index_version=version),
                "T": RagService(fixed_t, gateway, None, rag_cfg, index_version=version)}
    ids = sample_ids([q["id"] for q in o_defined], args.n, args.seed)
    pacer = Pacer(rpm=GROQ_LIMITS["rpm"], tpm=GROQ_LIMITS["tpm"])
    measured: dict[str, list[dict]] = {}
    for arm, mode in (("R", "short"), ("O", "short"), ("T", "short"), ("R", "long"), ("O", "long")):
        label = f"{arm}_{mode}"
        measured[label] = []
        for qid in ids:
            q = by_id[qid]
            if arm == "O":
                fixed_o.set(oracle[qid].hits)
            elif arm == "T":
                fixed_t.set(title_abstract_context(q["paper_id"], corpus[q["paper_id"]]["title"], corpus[q["paper_id"]]["abstract"]))
            est = int(mean([r["total_tokens"] for rows in measured.values() for r in rows]) or 2500)
            pacer.wait(est)
            svc = services[arm]
            res = (svc.answer_short if mode == "short" else svc.answer_long)(q["question"], doc_ids=[q["paper_id"]])
            tokens = (res.usage.prompt_tokens or 0) + (res.usage.completion_tokens or 0)
            pacer.record(tokens)
            measured[label].append({"question_id": qid, "prompt_tokens": res.usage.prompt_tokens, "completion_tokens": res.usage.completion_tokens,
                                    "reasoning_tokens": res.usage.reasoning_tokens, "total_tokens": tokens, "finish_reason": res.finish_reason,
                                    "abstained": res.abstained, "degraded": res.degraded, "n_sources": len(res.used_sources),
                                    "latency_total_ms": res.latency_ms.total})
        rows = measured[label]
        print(f"{label}: n={len(rows)} mean_total_tokens={mean([r['total_tokens'] for r in rows]):.0f} finish={sorted({str(r['finish_reason']) for r in rows})} "
              f"degraded={sum(r['degraded'] for r in rows)}")

    means = {k: mean([r["total_tokens"] for r in v]) for k, v in measured.items()}
    n_all, n_ans, n_o = len(dev), len(answerable), len(o_defined)
    items = {"R_short": (n_all, means["R_short"]), "T_short": (n_all, means["T_short"]), "O_short": (n_o, means["O_short"]),
             "R_long": (n_ans, means["R_long"]), "O_long": (n_o, means["O_long"])}
    limits = Limits(rpm=GROQ_LIMITS["rpm"], rpd=GROQ_LIMITS["rpd"], tpm=GROQ_LIMITS["tpm"], tpd=GROQ_LIMITS["tpd"])
    projection = {"items_calls_and_mean_tokens": {k: list(v) for k, v in items.items()}, "limits": GROQ_LIMITS, "full_plan": plan_wall_time(items, limits),
                  "fraction_of_the_plan_that_fits_in_24h": scale_to_budget(items, limits, 24.0)}
    pre = {k: v for k, v in projection["full_plan"].items()}
    print("PROJECTION (Gemini, full dev plan):", json.dumps(pre))
    print("fraction of the plan that fits in 24 h:", round(projection["fraction_of_the_plan_that_fits_in_24h"], 3))

    gemini = None
    if have["GEMINI_API_KEY"] and not args.no_gemini:
        gemini = gemini_rpm_probe()
        print("Gemini RPM probe:", json.dumps(gemini))

    started = datetime.now(timezone.utc)
    out_dir = ROOT / "runs" / "gen_dryrun" / f"{started.strftime('%Y%m%dT%H%M%SZ')}_quota_dry_run"
    out_dir.mkdir(parents=True)
    record = {"created_utc": started.isoformat(), "code_fingerprint": code_fingerprint(), "preregistration": prereg,
              "generator": {"model": rag_cfg.gemini_model, "reasoning_effort": rag_cfg.reasoning_effort, "max_output_tokens": rag_cfg.max_output_tokens,
                            "short_max_output_tokens": rag_cfg.short_max_output_tokens, "top_k": rag_cfg.top_k},
              "sample": {"question_ids": ids, "seed": args.seed, "n_per_arm_and_mode": len(ids),
                         "note": "real calls to measure cost only; the answers are not analysed or reused"},
              "counts": counts, "full_paper_arm_F": full_paper, "measured_per_call": measured, "mean_total_tokens": means,
              "projection": projection, "gemini_rpm_probe": gemini}
    (out_dir / "dry_run.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
