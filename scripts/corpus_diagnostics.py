"""Corpus-mode diagnostics (read-only; DEV split; no retrieval): does corpus mode fail at FINDING the paper, at RANKING
inside it, or both?

    python scripts/corpus_diagnostics.py --corpus-run runs/eval/<corpus run> --paper-run runs/eval/<paper run>
                                         [--depth-run runs/eval/<corpus run with top_k=20>] [--arms A,B,C,D,E,F]

For each arm: DocHit@k (any retrieved chunk belongs to the gold paper) for k = 1, 5, 10 from the corpus run, and k = 20 from the
depth run (the corpus runs stop at 10); HitRate@5 over all questions and conditional on DocHit@5 = 1 (with n); the paper-mode
HitRate@5 of the SAME arm on the same questions. Saved under runs/measurements/<UTC>_corpus_diagnostics/ with the fingerprint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def load(run: Path, arm: str) -> dict[str, dict]:
    return {r["question_id"]: r for r in map(json.loads, (run / "per_question.jsonl").read_text(encoding="utf-8").splitlines()) if r["arm"] == arm}


def doc_hit_at(row: dict, k: int) -> bool:
    return any(h["doc_id"] == row["paper_id"] for h in row["retrieved"][:k])


def mean(xs):
    xs = list(xs)
    return None if not xs else sum(xs) / len(xs)


def diagnose(corpus: dict, paper: dict, depth: dict | None) -> dict:
    ids = sorted(corpus)
    assert set(ids) == set(paper), "the corpus and paper runs must score the same questions"
    out = {"n_questions": len(ids), "DocHit": {}}
    for k in (1, 5, 10):
        out["DocHit"][f"@{k}"] = mean(doc_hit_at(corpus[i], k) for i in ids)
    out["DocHit"]["@20"] = None if depth is None else mean(doc_hit_at(depth[i], 20) for i in ids)
    out["DocHit"]["@20_source"] = None if depth is None else "separate top_k=20 corpus run (retrieved lists can differ from the top-10 run by HNSW noise)"
    out["DocHit"]["@10_in_depth_run"] = None if depth is None else mean(doc_hit_at(depth[i], 10) for i in ids)
    out["HitRate@5_corpus_all"] = mean(corpus[i]["metrics"]["HitRate@5"] for i in ids)
    out["HitRate@5_paper_all"] = mean(paper[i]["metrics"]["HitRate@5"] for i in ids)
    for label, flag in (("doc_found_at_5", True), ("doc_not_found_at_5", False)):
        sel = [i for i in ids if doc_hit_at(corpus[i], 5) is flag]
        out[label] = {"n": len(sel), "HitRate@5_corpus": mean(corpus[i]["metrics"]["HitRate@5"] for i in sel),
                      "HitRate@5_paper_same_questions": mean(paper[i]["metrics"]["HitRate@5"] for i in sel)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus-run", required=True)
    ap.add_argument("--paper-run", required=True)
    ap.add_argument("--depth-run")
    ap.add_argument("--arms", default="A,B,C,D,E,F")
    ap.add_argument("--out", default=str(ROOT / "runs" / "measurements"))
    args = ap.parse_args(argv)
    from utils.fingerprint import code_fingerprint

    runs = {"corpus": Path(args.corpus_run), "paper": Path(args.paper_run), "depth": Path(args.depth_run) if args.depth_run else None}
    results = {}
    for arm in args.arms.split(","):
        depth = load(runs["depth"], arm) if runs["depth"] else None
        results[arm] = diagnose(load(runs["corpus"], arm), load(runs["paper"], arm), depth)
    started = datetime.now(timezone.utc)
    out = Path(args.out) / f"{started.strftime('%Y%m%dT%H%M%SZ')}_corpus_diagnostics"
    out.mkdir(parents=True)
    payload = {"created_utc": started.isoformat(), "code_fingerprint": code_fingerprint(),
               "scope": "dev split, answerable questions; development only, not publishable",
               "runs": {k: (str(v) if v else None) for k, v in runs.items()},
               "run_config_sha256": {k: hashlib.sha256((v / "config.json").read_bytes()).hexdigest() for k, v in runs.items() if v},
               "arms": results}
    (out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out / "results.json")

    def f(v):
        return "  n/a" if v is None else f"{v:5.3f}"

    print("arm  n   DocHit@1 @5 @10 @20 | HR@5 corpus(all) paper(all) | doc found@5: n  corpus  paper | doc NOT found@5: n  corpus  paper")
    for arm, r in results.items():
        d, y, z = r["DocHit"], r["doc_found_at_5"], r["doc_not_found_at_5"]
        print(f"{arm}  {r['n_questions']}  {f(d['@1'])} {f(d['@5'])} {f(d['@10'])} {f(d['@20'])} | {f(r['HitRate@5_corpus_all'])} {f(r['HitRate@5_paper_all'])} | "
              f"{y['n']:3d} {f(y['HitRate@5_corpus'])} {f(y['HitRate@5_paper_same_questions'])} | {z['n']:3d} {f(z['HitRate@5_corpus'])} {f(z['HitRate@5_paper_same_questions'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
