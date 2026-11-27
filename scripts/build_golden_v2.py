"""Build golden_set_v2.json (answerable questions with located gold spans) and split.json.

    python scripts/build_golden_v2.py [--cap 3]

Inputs (read-only): .cache/qasper/qasper-dev-v0.3.json (already cached, no download) and
eval/datasets/qasper_dev/corpus.json. Outputs: golden_set_v2.json and split.json next to the corpus.
The v1 golden_set.json is not touched.

Selection (deterministic, no randomness): for every corpus paper, keep at most ``cap`` answerable
questions, ordered by sha256(question id); keep ALL unanswerable questions. The same builder filters
as v1 apply (annotator agreement on answerability, no table/figure-only evidence); excluded questions
and their reasons are listed in the output, nothing is dropped silently.
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"

from eval.gold_spans import (  # noqa: E402
    ANSWER_TYPES, ANSWERABLE, UNANSWERABLE, classify_question, gold_spans_for_question, reference_records,
)
from eval.split import DEV_PERCENT, split_of, split_questions  # noqa: E402
from rag.ingestion import file_sha256  # noqa: E402

RAW = ROOT / ".cache" / "qasper" / "qasper-dev-v0.3.json"
DATASET_DIR = ROOT / "eval" / "datasets" / "qasper_dev"
CORPUS = DATASET_DIR / "corpus.json"


def _order_key(question_id: str) -> str:
    return hashlib.sha256(question_id.encode("utf-8")).hexdigest()


def build(cap: int) -> tuple[dict, dict]:
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))

    answerable_by_paper: dict[str, list[dict]] = {}
    unanswerable: list[dict] = []
    excluded: list[dict] = []
    qa_by_id: dict[str, tuple[str, dict]] = {}
    for pid in sorted(corpus):
        for qa in raw[pid].get("qas", []):
            status, reason, rec = classify_question(pid, raw[pid], qa)
            qid = f"{pid}:{qa.get('question_id', '')}"
            qa_by_id[qid] = (pid, qa)
            if status == ANSWERABLE:
                answerable_by_paper.setdefault(pid, []).append(rec)
            elif status == UNANSWERABLE:
                unanswerable.append(rec)
            else:
                excluded.append({"id": qid, "reason": reason})

    selected: list[dict] = []
    not_selected_cap: list[str] = []
    for pid, recs in sorted(answerable_by_paper.items()):
        recs = sorted(recs, key=lambda r: _order_key(r["id"]))
        selected.extend(recs[:cap])
        not_selected_cap.extend(r["id"] for r in recs[cap:])

    location = Counter()
    questions: list[dict] = []
    for rec in selected:
        res = gold_spans_for_question(rec["paper_id"], corpus[rec["paper_id"]], rec["gold_evidence"])
        location.update(res["per_evidence"])
        refs = reference_records(raw[rec["paper_id"]], qa_by_id[rec["id"]][1])
        questions.append({**rec, "references": refs, "gold_spans": [s.to_dict() for s in res["spans"]], "evidence_location": res["per_evidence"]})
    for rec in unanswerable:
        refs = reference_records(raw[rec["paper_id"]], qa_by_id[rec["id"]][1])
        questions.append({**rec, "references": refs, "gold_spans": [], "evidence_location": []})
    questions.sort(key=lambda q: q["id"])

    type_counts = Counter(r["type"] for q in questions for r in q["references"])
    golden = {
        "dataset": "qasper-v0.3",
        "split": "dev (all questions of the 200 corpus papers; dev/test is defined in split.json)",
        "version": 2,
        "selection": {"cap_answerable_per_paper": cap, "order": "sha256(question id) ascending", "unanswerable": "all kept"},
        "references_note": "references = one record per annotator: answer, type (extractive|abstractive|yes_no|unanswerable), "
                           "evidence (the annotator's verbatim evidence PARAGRAPHS present in the paper text), "
                           "n_evidence_not_retrievable (table/figure or non-verbatim evidence, not listed). "
                           "gold_answers/gold_evidence/gold_spans are the unions kept from v1 for compatibility.",
        "inputs": {"raw_dev_sha256": file_sha256(RAW), "corpus_sha256": file_sha256(CORPUS)},
        "n_questions": len(questions),
        "n_answerable": sum(q["answerable"] for q in questions),
        "n_unanswerable": sum(not q["answerable"] for q in questions),
        "annotator_answers_by_type": {t: type_counts.get(t, 0) for t in ANSWER_TYPES + ("none",)},
        "annotator_answers_with_mixed_fields": sum(r["mixed_types"] for q in questions for r in q["references"]),
        "evidence_paragraph_location": {k: location.get(k, 0) for k in ("exact", "normalised", "ambiguous", "not_found")},
        "excluded_by_builder_filters": sorted(excluded, key=lambda e: e["id"]),
        "answerable_not_selected_because_of_cap": sorted(not_selected_cap),
        "questions": questions,
    }
    return golden, {"questions": questions}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cap", type=int, default=3)
    args = ap.parse_args()

    golden, _ = build(args.cap)
    golden_path = DATASET_DIR / "golden_set_v2.json"
    golden_path.write_text(json.dumps(golden, indent=2, ensure_ascii=False), encoding="utf-8")
    golden_sha = file_sha256(golden_path)

    questions = golden["questions"]
    ids = split_questions(questions)
    by_id = {q["id"]: q for q in questions}
    counts = {
        s: {
            "questions": len(ids[s]),
            "answerable": sum(by_id[i]["answerable"] for i in ids[s]),
            "unanswerable": sum(not by_id[i]["answerable"] for i in ids[s]),
            "papers": len({by_id[i]["paper_id"] for i in ids[s]}),
            "annotator_answers_by_type": dict(Counter(r["type"] for i in ids[s] for r in by_id[i]["references"])),
            "questions_with_at_least_one_answer_of_type": {
                t: sum(any(r["type"] == t for r in by_id[i]["references"]) for i in ids[s]) for t in ANSWER_TYPES},
        }
        for s in ("dev", "test")
    }
    paper_overlap = {by_id[i]["paper_id"] for i in ids["dev"]} & {by_id[i]["paper_id"] for i in ids["test"]}
    assert not paper_overlap, f"papers in both splits: {sorted(paper_overlap)[:3]}"
    assert all(split_of(by_id[i]["paper_id"]) == s for s in ids for i in ids[s])

    split = {
        "rule": f"sha256(paper_id) as an integer, mod 100 < {DEV_PERCENT} -> dev, otherwise test",
        "golden_set": "golden_set_v2.json",
        "golden_set_sha256": golden_sha,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "counts": counts,
        "note": "Unanswerable questions are NOT scored by retrieval metrics; they are reserved for the generation evaluation.",
        "dev": ids["dev"],
        "test": ids["test"],
    }
    split_path = DATASET_DIR / "split.json"
    split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(golden_path)
    print(split_path)
    print(json.dumps({"golden_set_sha256": golden_sha, "n_questions": golden["n_questions"], "n_answerable": golden["n_answerable"],
                      "n_unanswerable": golden["n_unanswerable"], "evidence": golden["evidence_paragraph_location"],
                      "counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
