"""Error analysis of retrieval misses (read-only; DEV split; no new retrieval).

    python scripts/error_analysis.py --run runs/eval/<paper run> --arm D [--seed 0 --sample 25]

Lists every answerable question whose HitRate@10 is 0 for one arm and saves a deterministic sample with the evidence, the
top-3 retrieved chunks and simple diagnostics. The failure categories are PROPOSED heuristics for the owner to review: a
question can fall in several (or none); they are descriptions, not causes, and are NOT used to change any arm.
Output: runs/analysis/<UTC>_error_analysis/ (config.json with the code fingerprint, misses_all.json, sample.json, sample.md,
categories.json, SHA256SUMS).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from rag.lexical_query import STOPWORDS  # noqa: E402

DATASET_DIR = ROOT / "eval" / "datasets" / "qasper_dev"
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
USUAL_SECTION = re.compile(r"intro|related|background|method|approach|model|experiment|setup|setting|result|evaluat|discuss|conclu|"
                           r"data|corpus|implement|analysis|architecture|training|baseline|abstract", re.I)
LOW_OVERLAP = 0.25  # share of the question's content words that occur in the evidence


def content_words(text: str) -> set[str]:
    return {w for w in (m.lower() for m in _WORD.findall(text)) if w not in STOPWORDS}


def overlap_share(question: str, text: str) -> float | None:
    q = content_words(question)
    return None if not q else len(q & content_words(text)) / len(q)


def near_miss(retrieved: list[dict], gold_spans: list[dict]) -> bool:
    """A retrieved chunk touches a gold span (same doc and section, at least 1 character) but was not judged relevant."""
    for h in retrieved:
        if h["relevant"]:
            continue
        for s in gold_spans:
            if (h["doc_id"] == s["doc_id"] and h["section"] == s["section"]
                    and min(h["char_end"], s["char_end"]) - max(h["char_start"], s["char_start"]) > 0):
                return True
    return False


def categories_of(q: dict, row: dict, evidence_overlap: float | None) -> list[str]:
    cats = []
    if evidence_overlap is not None and evidence_overlap < LOW_OVERLAP:
        cats.append("vocabulary_mismatch")
    if len(q["gold_evidence"]) >= 2:
        cats.append("multi_paragraph_evidence")
    if any(not USUAL_SECTION.search(s["section"] or "") for s in q["gold_spans"]):
        cats.append("evidence_in_unusual_section")
    if any(r.get("n_evidence_not_retrievable", 0) > 0 for r in q.get("references", [])):
        cats.append("table_or_figure_context_also_cited")
    if near_miss(row["retrieved"], q["gold_spans"]):
        cats.append("near_miss_partial_overlap")
    if not row["metrics"]["reachable"]:
        cats.append("gold_not_reachable_by_any_chunk")
    return cats or ["none_of_the_proposed"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--arm", default="D")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample", type=int, default=25)
    ap.add_argument("--out", default=str(ROOT / "runs" / "analysis"))
    ap.add_argument("--strategy", default="recursive")
    ap.add_argument("--golden", default=str(DATASET_DIR / "golden_set_v2.json"))
    args = ap.parse_args(argv)

    run = Path(args.run)
    rows = {r["question_id"]: r for r in map(json.loads, (run / "per_question.jsonl").read_text(encoding="utf-8").splitlines())
            if r["arm"] == args.arm}
    golden = {q["id"]: q for q in json.loads(Path(args.golden).read_text(encoding="utf-8"))["questions"]}
    miss_ids = sorted(i for i, r in rows.items() if r["metrics"]["HitRate@10"] == 0)

    from config import cfg
    from db.store import build_store
    from utils.fingerprint import code_fingerprint

    store = build_store(cfg, args.strategy)

    def detail(qid: str, with_chunks: bool) -> dict:
        q, row = golden[qid], rows[qid]
        evidence_text = " ".join(q["gold_evidence"])
        ov = overlap_share(q["question"], evidence_text)
        out = {
            "question_id": qid, "paper_id": q["paper_id"], "question": q["question"],
            "answer_types": sorted({r["type"] for r in q.get("references", [])}),
            "n_annotators": len(q.get("references", [])),
            "n_evidence_paragraphs": len(q["gold_evidence"]),
            "gold_sections": sorted({s["section"] for s in q["gold_spans"]}),
            "question_words_found_in_evidence": None if ov is None else round(ov, 3),
            "categories_PROPOSED": categories_of(q, row, ov),
            "reachable": bool(row["metrics"]["reachable"]),
            "metrics": {k: row["metrics"][k] for k in ("HitRate@10", "DocHit@10", "EvidenceRecall@10", "EvidenceF1@5", "nDCG@10")},
        }
        if with_chunks:
            out["gold_evidence_first_300_chars"] = [e[:300] for e in q["gold_evidence"]]
            top = row["retrieved"][:3]
            texts = {c.chunk_id: c.text for c in store.get_chunks([h["chunk_id"] for h in top])}
            out["top3_retrieved"] = [{"rank": h["rank"], "section": h["section"], "legs": h["legs"], "text_first_200_chars": texts[h["chunk_id"]][:200]}
                                     for h in top]
            out["question_words_found_in_top3"] = [None if (c := overlap_share(q["question"], texts[h["chunk_id"]])) is None else round(c, 3)
                                                   for h in top]
        return out

    allm = [detail(i, False) for i in miss_ids]
    sample_ids = sorted(random.Random(args.seed).sample(miss_ids, min(args.sample, len(miss_ids))))
    sample = [detail(i, True) for i in sample_ids]

    cat_counts = Counter(c for m in allm for c in m["categories_PROPOSED"])
    type_all = Counter(t for q in (golden[i] for i in rows) for t in {r["type"] for r in q.get("references", [])})
    type_miss = Counter(t for m in allm for t in m["answer_types"])
    categories = {
        "status": "PROPOSED by the assistant; awaiting the owner's review. Multi-label and heuristic; not used to change any arm.",
        "arm": args.arm, "questions_scored": len(rows), "questions_with_HitRate@10_eq_0": len(miss_ids),
        "definitions": {
            "vocabulary_mismatch": f"fewer than {LOW_OVERLAP:.0%} of the question's content words occur in the evidence text",
            "multi_paragraph_evidence": "the question has 2 or more distinct evidence paragraphs",
            "evidence_in_unusual_section": "a gold section name does not match the usual section-name pattern",
            "table_or_figure_context_also_cited": "an annotator also cited a table or figure as evidence (not retrievable as text)",
            "near_miss_partial_overlap": "a retrieved chunk overlaps a gold span by >= 1 character but fails the 0.5 length-neutral rule",
            "gold_not_reachable_by_any_chunk": "no stored chunk can satisfy the relevance rule for this question",
            "none_of_the_proposed": "no heuristic fired"},
        "counts_over_all_misses": dict(sorted(cat_counts.items(), key=lambda kv: -kv[1])),
        "counts_over_sample": dict(sorted(Counter(c for m in sample for c in m["categories_PROPOSED"]).items(), key=lambda kv: -kv[1])),
        "answer_types_all_scored_questions_with_at_least_one_such_answer": dict(type_all),
        "answer_types_among_misses_with_at_least_one_such_answer": dict(type_miss),
    }

    started = datetime.now(timezone.utc)
    out = Path(args.out) / f"{started.strftime('%Y%m%dT%H%M%SZ')}_error_analysis"
    out.mkdir(parents=True)
    config = {"created_utc": started.isoformat(), "source_run": str(run), "arm": args.arm, "seed": args.seed, "sample_size": len(sample),
              "sampling": f"random.Random({args.seed}).sample(sorted miss ids, {args.sample})", "code_fingerprint": code_fingerprint(),
              "golden_set_sha256": hashlib.sha256(Path(args.golden).read_bytes()).hexdigest(),
              "source_run_config_sha256": hashlib.sha256((run / "config.json").read_bytes()).hexdigest(), "store": store.name}
    files = {"config.json": config, "misses_all.json": allm, "sample.json": sample, "categories.json": categories}
    for name, obj in files.items():
        (out / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    md = [f"# Error analysis sample (arm {args.arm}, {len(sample)} of {len(miss_ids)} misses; categories are PROPOSED)\n"]
    for s in sample:
        md += [f"## {s['question_id']}", f"**Q:** {s['question']}", f"- answer types: {s['answer_types']}; evidence paragraphs: {s['n_evidence_paragraphs']}; "
               f"gold sections: {s['gold_sections']}; question words found in evidence: {s['question_words_found_in_evidence']}",
               f"- categories (PROPOSED): {s['categories_PROPOSED']}"]
        md += [f"- **gold evidence:** {e}" for e in s["gold_evidence_first_300_chars"]]
        md += [f"- top-{t['rank']} [{t['section']}] {t['text_first_200_chars']!r}" for t in s["top3_retrieved"]] + [""]
    (out / "sample.md").write_text("\n".join(md), encoding="utf-8")
    (out / "SHA256SUMS").write_text("".join(f"{hashlib.sha256((out / n).read_bytes()).hexdigest()}  {n}\n"
                                            for n in ("config.json", "misses_all.json", "sample.json", "categories.json", "sample.md")), encoding="utf-8")
    print(out)
    print(json.dumps({k: categories[k] for k in ("questions_scored", "questions_with_HitRate@10_eq_0", "counts_over_all_misses",
                                                  "counts_over_sample")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
