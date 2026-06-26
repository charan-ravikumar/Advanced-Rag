"""
eval/run_ablation.py — Pipeline ablation runner.

Runs eval/run_eval.py across 4 configs by temporarily patching cfg in-process,
then prints a comparison table showing the contribution of each added feature.

Configs:
  A  recursive + vector-only             (naive baseline)
  B  semantic  + vector-only             (+ better chunking)
  C  semantic  + hybrid + rrf            (+ BM25 & RRF  — RECOMMENDED)
  D  semantic  + hybrid + rrf + reranker (+ CrossEncoder)

Usage (from project root):
    python -m eval.run_ablation
    python -m eval.run_ablation --max-questions 20
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as _config_module
from eval.run_eval import run_eval


# ─────────────────────────────────────────────────────────────
# ABLATION CONFIGS
# ─────────────────────────────────────────────────────────────

ABLATION_CONFIGS = [
    {
        "label":            "A (baseline)",
        "description":      "recursive + vector-only",
        "chunking":         "recursive",
        "retrieval":        "vector",
        "fusion":           "none",
        "reranker_enabled": False,
    },
    {
        "label":            "B",
        "description":      "semantic + vector-only",
        "chunking":         "semantic",
        "retrieval":        "vector",
        "fusion":           "none",
        "reranker_enabled": False,
    },
    {
        "label":            "C  ← recommended",
        "description":      "semantic + hybrid + RRF",
        "chunking":         "semantic",
        "retrieval":        "hybrid",
        "fusion":           "rrf",
        "reranker_enabled": False,
    },
    {
        "label":            "D",
        "description":      "semantic + hybrid + RRF + reranker",
        "chunking":         "semantic",
        "retrieval":        "hybrid",
        "fusion":           "rrf",
        "reranker_enabled": True,
    },
]


# ─────────────────────────────────────────────────────────────
# PATCH / RESTORE HELPERS
# ─────────────────────────────────────────────────────────────

def _patch_cfg(overrides: dict) -> None:
    """Temporarily override cfg.pipeline fields in-process."""
    for key, value in overrides.items():
        setattr(_config_module.cfg.pipeline, key, value)


def _restore_cfg(saved: dict) -> None:
    for key, value in saved.items():
        setattr(_config_module.cfg.pipeline, key, value)


def _save_pipeline_cfg() -> dict:
    p = _config_module.cfg.pipeline
    return {
        "chunking":         p.chunking,
        "retrieval":        p.retrieval,
        "fusion":           p.fusion,
        "reranker_enabled": p.reranker_enabled,
    }


# ─────────────────────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────────────────────

def _print_comparison(rows: list[dict], baseline_r5: float) -> None:
    col_label = 20
    col_desc  = 34
    col_val   = 8
    col_lat   = 14
    col_delta = 16

    header = (
        f"{'Config':<{col_label}}"
        f"{'Description':<{col_desc}}"
        f"{'R@5':>{col_val}}"
        f"{'MRR':>{col_val}}"
        f"{'P50 Latency':>{col_lat}}"
        f"{'vs Baseline':>{col_delta}}"
    )
    sep = "─" * len(header)
    print(f"\n{header}")
    print(sep)

    for row in rows:
        r5    = row["recall@5"]
        mrr   = row["mrr"]
        p50   = row["latency_p50_ms"]
        label = row["label"]
        desc  = row["description"]

        if baseline_r5 > 0:
            delta_pct = (r5 - baseline_r5) / baseline_r5 * 100
            delta_str = f"{delta_pct:+.1f}% R@5"
        else:
            delta_str = "—"

        print(
            f"{label:<{col_label}}"
            f"{desc:<{col_desc}}"
            f"{r5:>{col_val}.4f}"
            f"{mrr:>{col_val}.4f}"
            f"{p50/1000:>{col_lat}.2f}s"
            f"{delta_str:>{col_delta}}"
        )
    print()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def run_ablation(max_questions: int | None = None) -> None:
    saved = _save_pipeline_cfg()
    all_rows = []

    # Note: Config D loads the reranker if reranker_enabled=True.
    # If the reranker model isn't loaded yet (it's lazily skipped when
    # reranker_enabled=False at startup), config D will still use
    # rrf+hybrid results because the reranker object stays None.
    # To get reranker results, start the process with reranker_enabled=true
    # in config.yaml and run only config D.

    for acfg in ABLATION_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Running config {acfg['label']} — {acfg['description']}")
        print(f"{'='*60}")

        overrides = {
            "chunking":         acfg["chunking"],
            "retrieval":        acfg["retrieval"],
            "fusion":           acfg["fusion"],
            "reranker_enabled": acfg["reranker_enabled"],
        }
        _patch_cfg(overrides)

        try:
            result = run_eval(max_questions=max_questions)
        except Exception as exc:
            print(f"Config {acfg['label']} FAILED: {exc}")
            result = {"recall@5": 0.0, "mrr": 0.0,
                      "latency_p50_ms": 0.0, "latency_p95_ms": 0.0,
                      "latency_p99_ms": 0.0}
        finally:
            _restore_cfg(saved)

        all_rows.append({
            **result,
            "label":       acfg["label"],
            "description": acfg["description"],
        })

    baseline_r5 = all_rows[0]["recall@5"] if all_rows else 0.0

    print("\n" + "=" * 60)
    print("ABLATION COMPARISON")
    print("=" * 60)
    _print_comparison(all_rows, baseline_r5)


# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-questions", type=int, default=None,
        help="Limit questions per config (default: all from config.yaml)"
    )
    args = parser.parse_args()
    run_ablation(max_questions=args.max_questions)
