"""Check that two evaluation runs agree on everything except timing.

    python eval/determinism.py runs/eval/<run 1> runs/eval/<run 2>

Retrieval is deterministic, so per-question rows must be identical once ``retrieval_latency_ms`` is removed,
summary means must be identical, and the only config differences allowed are the creation time and the run
location. Exit code 0 when identical, 1 otherwise; every difference found is listed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ALLOWED_CONFIG_DIFFS = {"created_utc"}


def _rows(run: Path) -> dict:
    out = {}
    with open(run / "per_question.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            r.pop("retrieval_latency_ms", None)
            out[(r["arm"], r["question_id"])] = r
    return out


def _strip_latency(summary: dict) -> dict:
    s = json.loads(json.dumps(summary))
    for arm in s["arms"].values():
        arm.pop("retrieval_latency_ms", None)
    return s


def compare_runs(run_a: Path, run_b: Path) -> dict:
    run_a, run_b = Path(run_a), Path(run_b)
    ra, rb = _rows(run_a), _rows(run_b)
    differences = []
    metric_diff_rows, retrieved_diff_rows, other_diff_rows = [], [], []
    if set(ra) != set(rb):
        differences.append(f"different (arm, question) sets: {len(set(ra) ^ set(rb))} differ")
    for key in sorted(set(ra) & set(rb)):
        if ra[key] != rb[key]:
            fields = sorted(k for k in set(ra[key]) | set(rb[key]) if ra[key].get(k) != rb[key].get(k))
            differences.append(f"{key}: differing fields {fields}")
            if "metrics" in fields:
                metric_diff_rows.append(key)
            if "retrieved" in fields:
                retrieved_diff_rows.append(key)
            if set(fields) - {"metrics", "retrieved", "n_retrieved"}:
                other_diff_rows.append(key)
    sa = _strip_latency(json.loads((run_a / "summary.json").read_text(encoding="utf-8")))
    sb = _strip_latency(json.loads((run_b / "summary.json").read_text(encoding="utf-8")))
    if sa != sb:
        differences.append("summary.json differs outside the latency fields")
    ca = json.loads((run_a / "config.json").read_text(encoding="utf-8"))
    cb = json.loads((run_b / "config.json").read_text(encoding="utf-8"))
    cfg_diff = sorted(k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k) and k not in ALLOWED_CONFIG_DIFFS)
    if cfg_diff:
        differences.append(f"config.json differs in {cfg_diff}")
    timing_differs = any(
        json.loads(l_a).get("retrieval_latency_ms") != json.loads(l_b).get("retrieval_latency_ms")
        for l_a, l_b in zip(open(run_a / "per_question.jsonl", encoding="utf-8"), open(run_b / "per_question.jsonl", encoding="utf-8"))
    )
    return {"identical_except_timing": not differences, "rows_compared": len(set(ra) & set(rb)),
            "metrics_identical": not metric_diff_rows and set(ra) == set(rb),
            "rows_with_different_metrics": len(metric_diff_rows),
            "rows_with_different_retrieved_lists": len(retrieved_diff_rows),
            "rows_with_other_differences": len(other_diff_rows),
            "differences": differences,
            "latency_fields_differ": timing_differs}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__)
        return 2
    res = compare_runs(Path(argv[0]), Path(argv[1]))
    print(json.dumps(res, indent=2))
    return 0 if res["identical_except_timing"] else 1


if __name__ == "__main__":
    sys.exit(main())
