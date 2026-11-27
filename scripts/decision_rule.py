"""Apply the pre-registered decision rule (eval/PREREGISTRATION.md, section 3) to one paper-mode run. Reports only.

    python scripts/decision_rule.py --run runs/eval/<paper run>

BEST = highest mean nDCG@10 (ties: lower complexity, then name). Every other arm X is compared with BEST (paper-clustered 95% CI,
the 14 comparisons are one Holm family). X is indistinguishable from BEST if its CI for the difference includes 0. The recommended
configuration is the least complex candidate (ties: higher mean nDCG@10, then name). Nothing is frozen by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from eval.compare import compare, format_report, load_rows  # noqa: E402
from eval.run_retrieval_eval import ARMS, Arm  # noqa: E402

METRIC = "nDCG@10"  # default; the corpus-wide setting of Stage 2b uses HitRate@5 (--metric)


def complexity(a: Arm) -> int:
    """One point each: vector leg, lexical leg, fusion step, semantic chunker, stopword-filtered query, non-default RRF
    parameters (anything but k=60 and lexical weight 0.7); two for a reranker; (Stage 2b) two for an external-API embedder and
    two for two-stage routing. Mirrors the tables in the pre-registrations."""
    pts = 0
    pts += a.mode in ("vector", "hybrid")
    pts += a.mode in ("lexical", "hybrid")
    pts += a.mode == "hybrid"
    pts += a.strategy == "semantic"
    pts += a.lexical == "fts_stopword"
    pts += a.fusion == "rrf" and (a.rrf_k != 60 or a.lexical_weight != 0.7)
    pts += 2 * bool(a.rerank)
    pts += 2 * (a.backend != "minilm")
    pts += 2 * bool(a.route_m)
    return int(pts)


def apply_rule(rows: list[dict], arms: dict[str, Arm] = ARMS, n_resamples: int = 5000, seed: int = 0, metric: str = METRIC) -> dict:
    names = sorted({r["arm"] for r in rows})
    means = {n: float(np.mean([r["metrics"][metric] for r in rows if r["arm"] == n])) for n in names}
    cx = {n: complexity(arms[n]) for n in names}
    best = min(names, key=lambda n: (-round(means[n], 12), cx[n], n))
    others = [n for n in names if n != best]
    report = compare(rows, [(best, x) for x in others], (metric,), n_resamples, seed)
    by_x = {r["arm_y"]: r for r in report["results"]}
    indist = {x for x, r in by_x.items() if r["ci_low"] <= 0.0 <= r["ci_high"]}
    holm_indist = {x for x, r in by_x.items() if not r["holm_significant"]}

    def pick(extra: set[str]) -> str:
        return min({best} | extra, key=lambda n: (cx[n], -means[n], n))

    return {"metric": metric, "best": best, "complexity": cx, "mean": means, "comparisons_best_vs_x": by_x,
            "n_comparisons": report["comparisons_made"], "indistinguishable_unadjusted_CI_includes_0": sorted(indist),
            "recommended_primary_rule": pick(indist), "indistinguishable_holm_not_significant": sorted(holm_indist),
            "recommended_holm_sensitivity": pick(holm_indist), "report": report}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--metric", default=METRIC, help="primary metric of the setting (nDCG@10 paper mode; HitRate@5 corpus mode, Stage 2b)")
    ap.add_argument("--out", default=str(ROOT / "runs" / "measurements"))
    args = ap.parse_args(argv)
    from utils.fingerprint import code_fingerprint

    run = Path(args.run)
    cfg_ = json.loads((run / "config.json").read_text(encoding="utf-8"))
    res = apply_rule(load_rows(run), metric=args.metric)
    res["report"]["setting"] = cfg_.get("setting")
    print(format_report(res["report"]))
    print(f"\nBEST on {args.metric}: {res['best']}  (mean {res['mean'][res['best']]:.4f}, complexity {res['complexity'][res['best']]})")
    print(f"arm: mean {args.metric} / complexity / CI-includes-0 vs BEST")
    for n in sorted(res["mean"], key=lambda n: -res["mean"][n]):
        r = res["comparisons_best_vs_x"].get(n)
        print(f"  {n:7} {res['mean'][n]:.4f}  c={res['complexity'][n]}  " + ("(best)" if r is None else
              f"diff(best-X)={r['mean_diff']:+.4f} CI[{r['ci_low']:+.4f},{r['ci_high']:+.4f}] {'includes 0' if n in res['indistinguishable_unadjusted_CI_includes_0'] else 'excludes 0'}; Holm sig: {r['holm_significant']}"))
    print(f"\nRULE OUTCOME (primary, unadjusted CIs): {res['recommended_primary_rule']}")
    print(f"Holm sensitivity outcome:               {res['recommended_holm_sensitivity']}")
    print("Nothing is frozen by this script.")
    started = datetime.now(timezone.utc)
    out = Path(args.out) / f"{started.strftime('%Y%m%dT%H%M%SZ')}_decision_rule"
    out.mkdir(parents=True)
    payload = {"created_utc": started.isoformat(), "code_fingerprint": code_fingerprint(), "source_run": str(run),
               "source_run_config_sha256": hashlib.sha256((run / "config.json").read_bytes()).hexdigest(),
               "preregistration": cfg_.get("preregistration"), "setting": cfg_.get("setting"),
               "scope": f"dev split, {cfg_.get('eval_mode')} mode; development only, nothing frozen", **res}
    (out / "results.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(out / "results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
