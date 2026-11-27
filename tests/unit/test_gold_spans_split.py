"""Gold-span location, question classification, and the dev/test split + golden_set_v2 consistency."""

import json
from collections import Counter
from pathlib import Path

import pytest

from eval.gold_spans import (
    ANSWERABLE,
    DROPPED,
    UNANSWERABLE,
    _builder,
    classify_question,
    gold_spans_for_question,
    locate_evidence,
)
from eval.split import DEV_PERCENT, split_of, split_questions
from rag.chunk_adapter import qasper_to_document

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / ".cache" / "qasper" / "qasper-dev-v0.3.json"
DATASET = ROOT / "eval" / "datasets" / "qasper_dev"


# ---- locating evidence -------------------------------------------------------------------------------

PAPER = {
    "title": "T",
    "abstract": "An abstract sentence.",
    "full_text": [
        {"section_name": "Intro", "paragraphs": ["First paragraph here.", "A repeated paragraph.", "Third one."]},
        {"section_name": "Methods", "paragraphs": ["A repeated paragraph.", "Spaced   out\n text here."]},
    ],
}


def test_exact_evidence_gets_section_relative_offsets():
    doc = qasper_to_document("p", PAPER)
    spans, conf = locate_evidence(doc, "p", "First paragraph here.")
    assert conf == "exact" and len(spans) == 1
    s = spans[0]
    assert (s.doc_id, s.section) == ("p", "Intro")
    assert doc.sections[s.section_idx].content[s.char_start:s.char_end] == "First paragraph here."


def test_a_paragraph_repeated_in_two_sections_is_ambiguous_and_keeps_every_occurrence():
    doc = qasper_to_document("p", PAPER)
    spans, conf = locate_evidence(doc, "p", "A repeated paragraph.")
    assert conf == "ambiguous" and sorted(s.section for s in spans) == ["Intro", "Methods"]
    assert all(doc.sections[s.section_idx].content[s.char_start:s.char_end] == "A repeated paragraph." for s in spans)


def test_whitespace_differences_are_located_as_normalised():
    doc = qasper_to_document("p", PAPER)
    spans, conf = locate_evidence(doc, "p", "Spaced out text here.")
    assert conf == "normalised" and len(spans) == 1 and spans[0].section == "Methods"


def test_missing_evidence_is_not_found_and_never_raises():
    doc = qasper_to_document("p", PAPER)
    assert locate_evidence(doc, "p", "This text is not in the paper.") == ([], "not_found")
    assert locate_evidence(doc, "p", "   ") == ([], "not_found")


def test_question_that_loses_all_evidence_is_reported_not_dropped():
    res = gold_spans_for_question("p", PAPER, ["nowhere", "also nowhere"])
    assert res["all_lost"] is True and res["spans"] == [] and res["per_evidence"] == ["not_found", "not_found"]
    mixed = gold_spans_for_question("p", PAPER, ["nowhere", "Third one."])
    assert mixed["all_lost"] is False and mixed["per_evidence"] == ["not_found", "exact"]


def test_abstract_is_a_section_too():
    doc = qasper_to_document("p", PAPER)
    spans, conf = locate_evidence(doc, "p", "An abstract sentence.")
    assert conf == "exact" and spans[0].section == "Abstract"


# ---- classification mirrors the builder --------------------------------------------------------------

def _qa(answers, question="Q?"):
    return {"question": question, "question_id": "x", "answers": [{"answer": a} for a in answers]}


def test_classification_reasons_on_constructed_questions():
    paper = {"abstract": "abs", "full_text": [{"section_name": "S", "paragraphs": ["para one", "para two"]}]}
    ok = {"unanswerable": False, "free_form_answer": "a", "evidence": ["para one"]}
    assert classify_question("p", paper, _qa([ok]))[:2] == (ANSWERABLE, "kept")
    assert classify_question("p", paper, _qa([{"unanswerable": True}]))[:2] == (UNANSWERABLE, "all_annotators_unanswerable")
    assert classify_question("p", paper, _qa([ok, {"unanswerable": True}]))[:2] == (DROPPED, "annotators_disagree_on_answerability")
    assert classify_question("p", paper, _qa([]))[:2] == (DROPPED, "no_question_text_or_no_annotations")
    fig = {"unanswerable": False, "free_form_answer": "a", "evidence": ["FLOAT SELECTED: Table 1"]}
    assert classify_question("p", paper, _qa([fig]))[:2] == (DROPPED, "evidence_is_table_or_figure_only")
    none = {"unanswerable": False, "free_form_answer": "a", "evidence": []}
    assert classify_question("p", paper, _qa([none]))[:2] == (DROPPED, "no_evidence_given")
    nov = {"unanswerable": False, "free_form_answer": "a", "evidence": ["a sentence, not a paragraph"]}
    assert classify_question("p", paper, _qa([nov]))[:2] == (DROPPED, "evidence_not_a_verbatim_paragraph")
    noans = {"unanswerable": False, "evidence": ["para one"]}
    assert classify_question("p", paper, _qa([noans]))[:2] == (DROPPED, "no_answer_text")


@pytest.mark.skipif(not RAW.exists(), reason="raw Qasper dev split is not cached")
def test_classifier_agrees_with_the_builders_parse_question_on_the_whole_raw_split():
    b = _builder()
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    counts = Counter()
    for pid in sorted(raw)[:120]:
        for qa in raw[pid].get("qas", []):
            status, reason, rec = classify_question(pid, raw[pid], qa)
            expected = b.parse_question(pid, raw[pid], qa)
            assert (rec is None) == (expected is None) or status == DROPPED
            if expected is None:
                assert status == DROPPED and reason != "other"
            else:
                assert rec == expected and status == (ANSWERABLE if expected["answerable"] else UNANSWERABLE)
            counts[status] += 1
    assert counts[ANSWERABLE] > 0 and counts[DROPPED] > 0


# ---- split ---------------------------------------------------------------------------------------------

def test_split_rule_is_the_documented_one():
    import hashlib

    for pid in ["1234.56789", "1601.02403", "abc", ""]:
        bucket = int(hashlib.sha256(pid.encode()).hexdigest(), 16) % 100
        assert split_of(pid) == ("dev" if bucket < DEV_PERCENT else "test")


def test_split_is_deterministic_and_roughly_40_60():
    ids = [f"{y}.{n:05d}" for y in (1601, 1707, 1810, 1910, 2001) for n in range(400)]
    assert [split_of(i) for i in ids] == [split_of(i) for i in ids]
    dev_share = sum(split_of(i) == "dev" for i in ids) / len(ids)
    assert 0.33 < dev_share < 0.47


def test_split_questions_keeps_a_paper_in_one_split():
    qs = [{"id": f"{p}:{i}", "paper_id": p} for p in ["a", "b", "c", "d", "e", "f"] for i in range(3)]
    out = split_questions(qs)
    papers = {s: {i.split(":")[0] for i in ids} for s, ids in out.items()}
    assert not papers["dev"] & papers["test"] and len(out["dev"]) + len(out["test"]) == 18


@pytest.mark.skipif(not (DATASET / "golden_set_v2.json").exists(), reason="golden_set_v2.json not built")
def test_golden_set_v2_and_split_json_are_consistent():
    import hashlib

    golden_bytes = (DATASET / "golden_set_v2.json").read_bytes()
    golden = json.loads(golden_bytes)
    split = json.loads((DATASET / "split.json").read_text(encoding="utf-8"))
    assert split["golden_set_sha256"] == hashlib.sha256(golden_bytes).hexdigest()
    by_id = {q["id"]: q for q in golden["questions"]}
    assert len(by_id) == len(golden["questions"]) == golden["n_questions"]
    dev, test = set(split["dev"]), set(split["test"])
    assert not dev & test and dev | test == set(by_id)
    assert not {by_id[i]["paper_id"] for i in dev} & {by_id[i]["paper_id"] for i in test}
    assert all(split_of(by_id[i]["paper_id"]) == "dev" for i in dev) and all(split_of(by_id[i]["paper_id"]) == "test" for i in test)
    per_paper = Counter(q["paper_id"] for q in golden["questions"] if q["answerable"])
    assert max(per_paper.values()) <= golden["selection"]["cap_answerable_per_paper"]
    for q in golden["questions"]:
        if q["answerable"]:
            assert q["gold_spans"] and q["evidence_location"] and "not_found" not in q["evidence_location"]
            assert {s["doc_id"] for s in q["gold_spans"]} == {q["paper_id"]}
        else:
            assert q["gold_spans"] == [] and q["gold_evidence"] == []
    counts = split["counts"]
    for s, ids in (("dev", dev), ("test", test)):
        assert counts[s]["questions"] == len(ids)
        assert counts[s]["answerable"] == sum(by_id[i]["answerable"] for i in ids)
        assert counts[s]["answerable"] + counts[s]["unanswerable"] == len(ids)
    assert golden["evidence_paragraph_location"]["not_found"] == 0
