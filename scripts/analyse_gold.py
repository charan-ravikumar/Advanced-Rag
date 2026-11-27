"""Read-only analysis of the Qasper dev questions for the corpus papers (Evaluation v2, Phase 1).

    python scripts/analyse_gold.py

Writes runs/measurements/qasper_gold_<UTC>/results.json (with the code fingerprint) and prints a summary.
Builds nothing: it does not touch golden_set.json, corpus.json, any store or any index.
"""

import collections
import json
import os
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"

from eval.gold_spans import (  # noqa: E402
    ANSWERABLE,
    UNANSWERABLE,
    classify_question,
    gold_spans_for_question,
)
from rag.ingestion import file_sha256  # noqa: E402
from utils.fingerprint import code_fingerprint  # noqa: E402

RAW = ROOT / ".cache" / "qasper" / "qasper-dev-v0.3.json"
CORPUS = ROOT / "eval" / "datasets" / "qasper_dev" / "corpus.json"
GOLDEN = ROOT / "eval" / "datasets" / "qasper_dev" / "golden_set.json"


def _locate_all(items, corpus):
    """items: list of (question_id, paper_id, evidence list). Returns summary + excluded ids."""
    paragraph_conf = collections.Counter()
    lost_questions, partial_questions, spans_per_q = [], [], []
    for qid, pid, evidence in items:
        res = gold_spans_for_question(pid, corpus[pid], evidence)
        paragraph_conf.update(res["per_evidence"])
        spans_per_q.append(len(res["spans"]))
        if res["all_lost"]:
            lost_questions.append({"id": qid, "reason": "every evidence paragraph not_found"})
        elif "not_found" in res["per_evidence"]:
            partial_questions.append({"id": qid, "not_found": res["per_evidence"].count("not_found"), "of": len(evidence)})
    return {
        "questions": len(items),
        "evidence_paragraphs": sum(paragraph_conf.values()),
        "evidence_paragraph_location": {k: paragraph_conf.get(k, 0) for k in ("exact", "normalised", "ambiguous", "not_found")},
        "questions_losing_all_evidence": len(lost_questions),
        "questions_losing_some_evidence": len(partial_questions),
        "excluded_question_ids": lost_questions,
        "partial_question_ids": partial_questions,
        "spans_per_question": {"min": min(spans_per_q), "median": st.median(spans_per_q), "max": max(spans_per_q)} if spans_per_q else None,
    }


def main() -> int:
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))

    status = collections.Counter()
    reasons = collections.Counter()
    candidates = []  # kept answerable: (id, paper_id, evidence)
    per_paper_ans = collections.Counter()
    unans_ids = []
    questions_total = 0
    for pid in sorted(corpus):
        for qa in raw[pid].get("qas", []):
            questions_total += 1
            st_, why, rec = classify_question(pid, raw[pid], qa)
            status[st_] += 1
            reasons[why] += 1
            if st_ == ANSWERABLE:
                candidates.append((rec["id"], pid, rec["gold_evidence"]))
                per_paper_ans[pid] += 1
            elif st_ == UNANSWERABLE:
                unans_ids.append(rec["id"])

    def capped(cap):
        return sum(min(n, cap) for n in per_paper_ans.values())

    golden_answerable = [(q["id"], q["paper_id"], q["gold_evidence"]) for q in golden["questions"] if q["answerable"]]
    dup_title_papers = 0
    from rag.chunk_adapter import qasper_to_document
    for pid, paper in corpus.items():
        titles = [s.section_title for s in qasper_to_document(pid, paper).sections]
        dup_title_papers += len(titles) != len(set(titles))

    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_fingerprint": code_fingerprint(),
        "inputs": {"raw_dev": str(RAW.relative_to(ROOT)), "raw_dev_sha256": file_sha256(RAW),
                   "corpus_sha256": file_sha256(CORPUS), "golden_set_sha256": file_sha256(GOLDEN)},
        "raw_dev": {"papers_in_raw_dev": len(raw), "corpus_papers": len(corpus), "corpus_papers_found_in_raw_dev": len(set(corpus) & set(raw)),
                    "questions_in_corpus_papers": questions_total},
        "by_status": dict(status),
        "by_reason": dict(reasons),
        "answerable_usable": {
            "total": len(candidates),
            "papers_with_at_least_one": len(per_paper_ans),
            "per_paper_questions": {"min": min(per_paper_ans.values()), "median": st.median(per_paper_ans.values()), "max": max(per_paper_ans.values())},
            "if_capped_per_paper": {"cap_2": capped(2), "cap_3": capped(3), "cap_5": capped(5), "uncapped": len(candidates)},
        },
        "unanswerable_usable": {"total": len(unans_ids)},
        "current_golden_set": {"questions": len(golden["questions"]),
                               "answerable": len(golden_answerable),
                               "unanswerable": sum(not q["answerable"] for q in golden["questions"]),
                               "papers_with_questions": len({q["paper_id"] for q in golden["questions"]})},
        "papers_with_duplicate_section_titles": dup_title_papers,
        "gold_span_location_current_golden_answerable": _locate_all(golden_answerable, corpus),
        "gold_span_location_all_usable_answerable": _locate_all(candidates, corpus),
    }
    d = ROOT / "runs" / "measurements" / ("qasper_gold_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(d / "results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
