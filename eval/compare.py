"""Paired comparison of two arms of one evaluation run.

    python eval/compare.py --run runs/eval/<run folder> --pair A:C --pair A:D [--metrics HitRate@5,MRR,nDCG@10]

Arms are scored on the same questions, so the comparison is PAIRED: per question the difference
(first arm minus second arm) is computed and bootstrapped, which is more powerful than comparing two means.

Questions about the same paper are correlated (same text, same evidence region), so resampling questions as if
independent understates uncertainty. The PRIMARY interval therefore resamples PAPERS (all of a paper's questions
move together); the per-question bootstrap is kept as a secondary column. Both use 5,000 resamples and a fixed seed.
For every comparison the output shows: number of questions and papers, both means, the mean difference, the
clustered 95% CI (primary), the per-question 95% CI (secondary), wins/losses/ties per question, the clustered
bootstrap p-value and a Holm-adjusted significance flag across ALL comparisons made in this invocation (the count
is printed). A bare difference is never printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_METRICS = ("HitRate@5", "MRR", "nDCG@10")
RESAMPLES = 5000
SEED = 0
ALPHA = 0.05
TIE_TOL = 1e-12


def _interval_and_p(means: np.ndarray, alpha: float) -> tuple[float, float, float]:
    n = len(means)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    le = (np.sum(means <= 0) + 1) / (n + 1)
    ge = (np.sum(means >= 0) + 1) / (n + 1)
    return float(lo), float(hi), float(min(1.0, 2 * min(le, ge)))


def paired_bootstrap(diffs, n_resamples: int = RESAMPLES, seed: int = SEED, alpha: float = ALPHA) -> dict:
    """Per-question bootstrap: mean difference, percentile CI of the resampled means, two-sided bootstrap p-value.

    p = 2 * min(P(mean* <= 0), P(mean* >= 0)) with a +1 correction (resolution 1 / (n_resamples + 1)),
    capped at 1. Deterministic for a fixed seed.
    """
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    if n == 0:
        raise ValueError("no paired observations")
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, n, size=(n_resamples, n))].mean(axis=1)
    lo, hi, p = _interval_and_p(means, alpha)
    return {"mean_diff": float(d.mean()), "ci_low": lo, "ci_high": hi, "p_value": p,
            "wins": int(np.sum(d > TIE_TOL)), "losses": int(np.sum(d < -TIE_TOL)), "ties": int(np.sum(np.abs(d) <= TIE_TOL)), "n": n}


def clustered_bootstrap(diffs, clusters, n_resamples: int = RESAMPLES, seed: int = SEED, alpha: float = ALPHA) -> dict:
    """Cluster bootstrap: resample CLUSTERS (papers) with replacement, keeping all of a cluster's questions together.

    The statistic is the mean difference over all questions in the resampled papers (sum of differences divided by
    the number of questions), so the point estimate equals the plain mean. Clusters are ordered by name so the
    result is deterministic for a fixed seed.
    """
    d = np.asarray(diffs, dtype=float)
    if len(d) == 0:
        raise ValueError("no paired observations")
    if len(clusters) != len(d):
        raise ValueError("one cluster label per observation is required")
    names = sorted(set(clusters))
    index = {c: i for i, c in enumerate(names)}
    sums, counts = np.zeros(len(names)), np.zeros(len(names))
    for diff, c in zip(d, clusters):
        sums[index[c]] += diff
        counts[index[c]] += 1
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(names), size=(n_resamples, len(names)))
    means = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    lo, hi, p = _interval_and_p(means, alpha)
    return {"mean_diff": float(d.mean()), "ci_low": lo, "ci_high": hi, "p_value": p, "n_clusters": len(names), "n": len(d)}


def holm_significant(p_values: list[float], alpha: float = ALPHA) -> list[bool]:
    """Holm step-down: sort ascending, compare the i-th smallest p to alpha / (m - i); stop at the first failure."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    flags = [False] * m
    for rank, i in enumerate(order):
        if p_values[i] <= alpha / (m - rank):
            flags[i] = True
        else:
            break
    return flags


def load_rows(run_dir: Path) -> list[dict]:
    with open(Path(run_dir) / "per_question.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def metric_by_question(rows: list[dict], arm: str, metric: str) -> dict[str, float]:
    """Per-question values of ``metric`` for ``arm``. Questions where the metric is undefined (None, e.g. Evidence-F1 when
    every annotator's evidence is a table or figure) are left out; the same questions are undefined for every arm."""
    return {r["question_id"]: float(r["metrics"][metric]) for r in rows if r["arm"] == arm and r["metrics"][metric] is not None}


def paper_by_question(rows: list[dict]) -> dict[str, str]:
    """question id -> paper id. Rows without a paper_id are their own cluster (equivalent to no clustering)."""
    return {r["question_id"]: r.get("paper_id", r["question_id"]) for r in rows}


def compare(rows: list[dict], pairs: list[tuple[str, str]], metrics=DEFAULT_METRICS,
            n_resamples: int = RESAMPLES, seed: int = SEED) -> dict:
    results = []
    papers = paper_by_question(rows)
    for x, y in pairs:
        for metric in metrics:
            mx, my = metric_by_question(rows, x, metric), metric_by_question(rows, y, metric)
            if not mx or not my:
                raise ValueError(f"arm {x if not mx else y!r} has no rows")
            common = sorted(set(mx) & set(my))
            if len(common) != len(mx) or len(common) != len(my):
                raise ValueError(f"arms {x} and {y} were not scored on the same questions "
                                 f"({len(mx)} vs {len(my)}, {len(common)} in common)")
            diffs = [mx[q] - my[q] for q in common]
            per_q = paired_bootstrap(diffs, n_resamples, seed)
            clu = clustered_bootstrap(diffs, [papers[q] for q in common], n_resamples, seed)
            results.append({
                "arm_x": x, "arm_y": y, "metric": metric, "n": per_q["n"], "n_papers": clu["n_clusters"],
                "mean_x": float(np.mean([mx[q] for q in common])), "mean_y": float(np.mean([my[q] for q in common])),
                "mean_diff": clu["mean_diff"],
                # PRIMARY: papers are resampled, a paper's questions stay together
                "ci_low": clu["ci_low"], "ci_high": clu["ci_high"], "p_value": clu["p_value"],
                # secondary: questions treated as independent (understates uncertainty)
                "qci_low": per_q["ci_low"], "qci_high": per_q["ci_high"], "q_p_value": per_q["p_value"],
                "wins": per_q["wins"], "losses": per_q["losses"], "ties": per_q["ties"],
            })
    flags = holm_significant([r["p_value"] for r in results])
    for r, f in zip(results, flags):
        r["holm_significant"] = f
    return {"comparisons_made": len(results), "alpha": ALPHA, "resamples": n_resamples, "seed": seed, "results": results}


def format_report(report: dict) -> str:
    setting = report.get("setting")
    lines = ([f"SETTING: {setting}"] if setting else []) + [
             f"comparisons made in this invocation: {report['comparisons_made']}  "
             f"(Holm-adjusted at alpha={report['alpha']} on the clustered p-values; bootstrap resamples={report['resamples']}, seed={report['seed']})",
             "difference = first arm minus second arm. PRIMARY CI = 95% paper-clustered paired bootstrap (papers resampled, "
             "their questions kept together); per-q CI = 95% per-question bootstrap, secondary (understates uncertainty). "
             "W/L/T = questions where the first arm is better/worse/equal.", ""]
    header = (f"{'pair':8} {'metric':10} {'n_q':>4} {'n_pap':>5} {'mean_x':>7} {'mean_y':>7} {'diff':>8} {'PRIMARY CI (clustered)':>24} "
              f"{'per-q CI':>19} {'W/L/T':>11} {'p (clust.)':>10} {'Holm sig':>8}")
    lines += [header, "-" * len(header)]
    for r in report["results"]:
        ci = f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
        qci = f"[{r['qci_low']:+.3f}, {r['qci_high']:+.3f}]"
        lines.append(f"{r['arm_x'] + ' vs ' + r['arm_y']:8} {r['metric']:10} {r['n']:>4} {r['n_papers']:>5} {r['mean_x']:>7.3f} {r['mean_y']:>7.3f} "
                     f"{r['mean_diff']:>+8.3f} {ci:>24} {qci:>19} {r['wins']:>3}/{r['losses']:>3}/{r['ties']:>3} "
                     f"{r['p_value']:>10.4f} {'YES' if r['holm_significant'] else 'no':>8}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="run folder containing per_question.jsonl")
    ap.add_argument("--pair", action="append", required=True, help="X:Y, repeatable")
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--resamples", type=int, default=RESAMPLES)
    args = ap.parse_args(argv)
    pairs = [tuple(p.split(":", 1)) for p in args.pair]
    run_dir = Path(args.run)
    report = compare(load_rows(run_dir), pairs, tuple(args.metrics.split(",")), args.resamples, args.seed)
    config_path = run_dir / "config.json"
    if config_path.exists():
        cfg_ = json.loads(config_path.read_text(encoding="utf-8"))
        report["setting"] = cfg_.get("setting")
        report["eval_mode"] = cfg_.get("eval_mode")
    print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
