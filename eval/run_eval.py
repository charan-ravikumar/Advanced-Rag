"""
eval/run_eval.py — RAG evaluation harness with full latency profiling.

Reads the golden dataset specified in config.yaml (eval.dataset_path),
runs the retrieval pipeline against each question, and reports:
  - Recall@5, MRR
  - P50 / P95 / P99 latency (ms)

Results are saved to eval/results/{timestamp}_results.json and
a clean summary table is printed to stdout.

Usage (from project root):
    python -m eval.run_eval
    python -m eval.run_eval --max-questions 20
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ── project root on sys.path ─────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import cfg
from rag.retrieval import retrieve_and_build_context


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _token_f1(a: str, b: str) -> float:
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta or not tb:
        return 0.0
    inter = ta & tb
    p = len(inter) / len(ta)
    r = len(inter) / len(tb)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


MATCH_THRESHOLD = 0.25


def _is_relevant(gold_evidence: list, chunk: str) -> bool:
    return any(_token_f1(ev, chunk) >= MATCH_THRESHOLD for ev in gold_evidence)


def recall_at_k(gold_evidence: list, chunks: list, k: int) -> float:
    return 1.0 if any(_is_relevant(gold_evidence, c) for c in chunks[:k]) else 0.0


def mrr_score(gold_evidence: list, chunks: list) -> float:
    for rank, chunk in enumerate(chunks, start=1):
        if _is_relevant(gold_evidence, chunk):
            return 1.0 / rank
    return 0.0


def _print_table(rows: list[tuple]) -> None:
    """Print a two-column summary table."""
    col_w = max(len(r[0]) for r in rows) + 2
    sep = "─" * (col_w + 14)
    print(f"\n{'Metric':<{col_w}}{'Value':>12}")
    print(sep)
    for label, value in rows:
        print(f"{label:<{col_w}}{value:>12}")
    print()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def run_eval(max_questions: int | None = None) -> dict:
    # ── Load dataset ─────────────────────────────────────────
    dataset_path = ROOT / cfg.eval.dataset_path
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    questions = data["questions"]
    if max_questions:
        questions = questions[:max_questions]

    n = len(questions)
    print(f"\nRunning eval on {n} questions "
          f"[chunking={cfg.pipeline.chunking}  "
          f"retrieval={cfg.pipeline.retrieval}  "
          f"fusion={cfg.pipeline.fusion}  "
          f"reranker={cfg.pipeline.reranker_enabled}]\n")

    r5_scores:  list[float] = []
    mrr_scores: list[float] = []
    latencies:  list[float] = []   # ms per query

    for i, qa in enumerate(questions, 1):
        question      = qa["question"]
        gold_evidence = qa.get("gold_evidence", [])

        t0 = time.perf_counter()
        try:
            result = retrieve_and_build_context(query=question)
        except Exception as exc:
            print(f"  [{i}/{n}] ERROR: {exc}")
            latencies.append(0.0)
            r5_scores.append(0.0)
            mrr_scores.append(0.0)
            continue
        latency_ms = (time.perf_counter() - t0) * 1000

        chunks = [s.content for s in result["sources"]]
        r5  = recall_at_k(gold_evidence, chunks, 5)
        mrr = mrr_score(gold_evidence, chunks)

        r5_scores.append(r5)
        mrr_scores.append(mrr)
        latencies.append(latency_ms)

        lat_breakdown = result.get("latency_ms", {})
        print(
            f"  [{i:>3}/{n}] R@5={r5:.0f}  MRR={mrr:.3f}  "
            f"total={latency_ms:.0f}ms  "
            f"(vec={lat_breakdown.get('vector_search_ms', 0):.0f}ms  "
            f"bm25={lat_breakdown.get('bm25_search_ms', 0):.0f}ms  "
            f"rerank={lat_breakdown.get('reranker_ms', 0):.0f}ms)"
        )

    # ── Compute metrics ───────────────────────────────────────
    lat_arr = np.array(latencies)
    results = {
        "recall@5":          round(float(np.mean(r5_scores)),  4),
        "mrr":               round(float(np.mean(mrr_scores)), 4),
        "latency_p50_ms":    round(float(np.percentile(lat_arr, 50)),  1),
        "latency_p95_ms":    round(float(np.percentile(lat_arr, 95)),  1),
        "latency_p99_ms":    round(float(np.percentile(lat_arr, 99)),  1),
        "latency_mean_ms":   round(float(np.mean(lat_arr)),    1),
        "n_queries":         n,
        "pipeline_config": {
            "chunking":          cfg.pipeline.chunking,
            "retrieval":         cfg.pipeline.retrieval,
            "fusion":            cfg.pipeline.fusion,
            "reranker_enabled":  cfg.pipeline.reranker_enabled,
            "reranker_model":    cfg.pipeline.reranker_model,
            "top_k":             cfg.pipeline.top_k,
            "rrf_k":             cfg.pipeline.rrf_k,
        },
    }

    # ── Print summary ─────────────────────────────────────────
    _print_table([
        ("Recall@5",      f"{results['recall@5']:.4f}"),
        ("MRR",           f"{results['mrr']:.4f}"),
        ("P50 Latency",   f"{results['latency_p50_ms']:.1f}ms"),
        ("P95 Latency",   f"{results['latency_p95_ms']:.1f}ms"),
        ("P99 Latency",   f"{results['latency_p99_ms']:.1f}ms"),
        ("Mean Latency",  f"{results['latency_mean_ms']:.1f}ms"),
        ("N Queries",     str(n)),
    ])

    # ── Save JSON ─────────────────────────────────────────────
    results_dir = ROOT / cfg.eval.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = results_dir / f"{ts}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_path}\n")

    return results


# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-questions", type=int, default=None,
        help="Limit number of questions (default: all)"
    )
    args = parser.parse_args()
    run_eval(max_questions=args.max_questions)
