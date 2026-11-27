"""Paragraph-level Evidence-F1 (the official Qasper evidence metric applied to retrieved chunks), with hand-built cases,
plus the annotator-reference records and the primary/extension labelling."""

import json
from pathlib import Path

import pytest

from eval.gold_spans import ANSWER_TYPES, answer_type, paragraph_index, reference_records
from rag.chunk_adapter import qasper_to_document
from rag.eval_metrics import (
    Paragraph,
    evidence_f1,
    evidence_f1_at_k,
    paragraphs_of_chunk,
    question_metrics,
    retrieved_paragraph_texts,
)
from rag.schemas_ingest import Chunk

ROOT = Path(__file__).resolve().parents[2]


def chunk(start, end, *, section="S", doc="p"):
    return Chunk.create(doc_id=doc, section=section, chunk_idx=0, char_start=start, char_end=end, strategy="recursive",
                        text="x", doc_version="v")


# three paragraphs of one section, as qasper_to_document lays them out (joined by a blank line, offsets +2)
P0 = Paragraph("S", 0, 0, 0, 100, "paragraph zero")
P1 = Paragraph("S", 0, 1, 102, 250, "paragraph one")
P2 = Paragraph("S", 0, 2, 252, 400, "paragraph two")
OTHER = Paragraph("T", 1, 0, 0, 100, "paragraph in another section")
PARAS = [P0, P1, P2, OTHER]


# ---- chunk -> paragraphs ----------------------------------------------------------------------------------------

def test_a_chunk_maps_to_every_paragraph_it_overlaps():
    assert paragraphs_of_chunk(chunk(50, 120), PARAS) == [P0, P1]  # spans the boundary between 0 and 1
    assert paragraphs_of_chunk(chunk(110, 240), PARAS) == [P1]  # inside one paragraph
    assert paragraphs_of_chunk(chunk(0, 400), PARAS) == [P0, P1, P2]  # a long chunk covers three


def test_touching_a_paragraph_without_sharing_a_character_does_not_count():
    assert paragraphs_of_chunk(chunk(100, 102), PARAS) == []  # the blank-line gap between P0 and P1
    assert paragraphs_of_chunk(chunk(0, 100), PARAS) == [P0]


def test_other_section_and_unlocated_chunks_map_to_nothing():
    assert paragraphs_of_chunk(chunk(10, 50, section="U"), PARAS) == []
    assert paragraphs_of_chunk(chunk(-1, -1), PARAS) == []
    assert paragraphs_of_chunk(chunk(10, 50, section="T"), PARAS) == [OTHER]


def test_minimum_overlap_is_a_parameter():
    assert paragraphs_of_chunk(chunk(95, 130), PARAS, min_overlap_chars=10) == [P1]  # 5 chars of P0 < 10, 28 of P1
    assert retrieved_paragraph_texts([chunk(95, 130)], PARAS, 10) == {"paragraph one"}


def test_paragraph_identity_is_the_text_so_duplicates_collapse():
    dup = Paragraph("S", 0, 3, 402, 500, "paragraph zero")  # same text as P0, elsewhere
    assert retrieved_paragraph_texts([chunk(0, 90), chunk(410, 490)], PARAS + [dup]) == {"paragraph zero"}


# ---- F1 on hand-built sets ----------------------------------------------------------------------------------------

def test_perfect_and_empty_and_disjoint():
    assert evidence_f1({"a", "b"}, [{"a", "b"}]) == 1.0
    assert evidence_f1(set(), [{"a"}]) == 0.0
    assert evidence_f1({"x"}, [{"a"}]) == 0.0


def test_precision_recall_f1_values():
    # predicted {a,b,c}, gold {a,d}: tp=1, precision 1/3, recall 1/2, F1 = 2*(1/3)(1/2)/(1/3+1/2) = 0.4
    assert evidence_f1({"a", "b", "c"}, [{"a", "d"}]) == pytest.approx(0.4)
    # a superset prediction: precision 1/2, recall 1 -> 2/3
    assert evidence_f1({"a", "b"}, [{"a"}]) == pytest.approx(2 / 3)
    # a subset prediction: precision 1, recall 1/2 -> 2/3
    assert evidence_f1({"a"}, [{"a", "b"}]) == pytest.approx(2 / 3)


def test_max_over_annotators_is_taken():
    gold = [{"a", "b", "c", "d"}, {"a"}]  # the second annotator needs only "a"
    assert evidence_f1({"a"}, gold) == 1.0
    assert evidence_f1({"a"}, [gold[0]]) == pytest.approx(2 * 1 * 0.25 / 1.25)  # the first alone gives 0.4


def test_annotators_without_evidence_are_skipped_not_counted_as_zero():
    assert evidence_f1({"a"}, [set(), {"a"}]) == 1.0
    assert evidence_f1({"a"}, [set(), set()]) is None
    assert evidence_f1({"a"}, []) is None


def test_evidence_f1_at_k_uses_only_the_top_k_chunks():
    retrieved = [chunk(300, 380), chunk(110, 240), chunk(0, 90)]  # P2 first, then P1, then P0
    gold = [{"paragraph one"}]
    assert evidence_f1_at_k(retrieved, PARAS, gold, 1) == 0.0
    assert evidence_f1_at_k(retrieved, PARAS, gold, 2) == pytest.approx(2 / 3)  # {two, one} vs {one}
    assert evidence_f1_at_k(retrieved, PARAS, gold, 3) == pytest.approx(0.5)  # {two, one, zero} vs {one}: P=1/3, R=1


def test_a_long_chunk_that_swallows_three_paragraphs_is_penalised_on_precision():
    assert evidence_f1_at_k([chunk(0, 400)], PARAS, [{"paragraph one"}], 5) == pytest.approx(0.5)


def test_question_metrics_includes_evidence_f1_only_when_given_its_inputs():
    from eval.gold_spans import GoldSpan

    spans = [GoldSpan("p", "S", 0, 102, 250, 0, "exact")]
    with_f1 = question_metrics([chunk(110, 240)], spans, "p", 1, paragraphs=PARAS, reference_evidence=[{"paragraph one"}])
    assert (with_f1["EvidenceF1@1"], with_f1["EvidenceF1@5"], with_f1["EvidenceF1@10"]) == (1.0, 1.0, 1.0)
    without = question_metrics([chunk(110, 240)], spans, "p", 1)
    assert without["EvidenceF1@5"] is None


# ---- the paragraph index matches the document the chunkers see ----------------------------------------------------------

PAPER = {
    "title": "T", "abstract": "Abstract text.",
    "full_text": [{"section_name": "Intro", "paragraphs": ["First paragraph.", "  Second paragraph.  ", ""]},
                  {"section_name": "Empty", "paragraphs": ["", "  "]},
                  {"section_name": "Methods", "paragraphs": ["Third."]}],
}


def test_paragraph_index_lines_up_with_the_documents_sections():
    doc = qasper_to_document("p", PAPER)
    index = paragraph_index("p", PAPER)
    assert [p.text for p in index] == ["Abstract text.", "First paragraph.", "Second paragraph.", "Third."]
    for p in index:
        sec = doc.sections[p.section_idx]
        assert sec.section_title == p.section and sec.content[p.char_start:p.char_end] == p.text


@pytest.mark.skipif(not (ROOT / "eval/datasets/qasper_dev/corpus.json").exists(), reason="corpus not present")
def test_paragraph_index_lines_up_on_every_real_paper():
    corpus = json.loads((ROOT / "eval/datasets/qasper_dev/corpus.json").read_text(encoding="utf-8"))
    for pid, paper in corpus.items():
        doc = qasper_to_document(pid, paper)
        for p in paragraph_index(pid, paper):
            assert doc.sections[p.section_idx].content[p.char_start:p.char_end] == p.text


# ---- annotator references -----------------------------------------------------------------------------------------------

def test_answer_types():
    assert answer_type({"unanswerable": True}) == ("unanswerable", False)
    assert answer_type({"unanswerable": False, "extractive_spans": ["x"], "yes_no": None, "free_form_answer": ""}) == ("extractive", False)
    assert answer_type({"unanswerable": False, "extractive_spans": [], "yes_no": False, "free_form_answer": ""}) == ("yes_no", False)  # "no" is a yes/no answer
    assert answer_type({"unanswerable": False, "extractive_spans": [], "yes_no": None, "free_form_answer": "text"}) == ("abstractive", False)
    assert answer_type({"unanswerable": False, "extractive_spans": ["x"], "free_form_answer": "y"}) == ("extractive", True)  # mixed: reported
    assert answer_type({}) == ("none", False)
    assert set(ANSWER_TYPES) == {"extractive", "abstractive", "yes_no", "unanswerable"}


def test_reference_records_keep_every_annotator_with_type_and_own_evidence():
    paper = {"abstract": "abs", "full_text": [{"section_name": "S", "paragraphs": ["para one", "para two"]}]}
    qa = {"question": "Q?", "question_id": "q", "answers": [
        {"answer": {"unanswerable": False, "extractive_spans": ["span"], "yes_no": None, "free_form_answer": "", "evidence": ["para one"]}},
        {"answer": {"unanswerable": False, "extractive_spans": [], "yes_no": True, "free_form_answer": "", "evidence": ["para two", "FLOAT SELECTED: Table 1", "not a paragraph"]}},
        {"answer": {"unanswerable": False, "extractive_spans": [], "yes_no": None, "free_form_answer": "free", "evidence": []}}]}
    refs = reference_records(paper, qa)
    assert [r["type"] for r in refs] == ["extractive", "yes_no", "abstractive"]
    assert [r["answer"] for r in refs] == ["span", "yes", "free"]
    assert [r["evidence"] for r in refs] == [["para one"], ["para two"], []]
    assert [r["n_evidence_not_retrievable"] for r in refs] == [0, 2, 0]  # the table and the non-verbatim string are counted, not listed


@pytest.mark.skipif(not (ROOT / "eval/datasets/qasper_dev/golden_set_v2.json").exists(), reason="golden_set_v2.json not built")
def test_golden_set_v2_keeps_all_annotators_and_their_types():
    g = json.loads((ROOT / "eval/datasets/qasper_dev/golden_set_v2.json").read_text(encoding="utf-8"))
    split = json.loads((ROOT / "eval/datasets/qasper_dev/split.json").read_text(encoding="utf-8"))
    for q in g["questions"]:
        refs = q["references"]
        assert refs and all(r["type"] in ANSWER_TYPES for r in refs)
        if q["answerable"]:
            assert all(r["type"] != "unanswerable" for r in refs)
            assert set(e for r in refs for e in r["evidence"]) == set(q["gold_evidence"])
            assert any(r["evidence"] for r in refs)
        else:
            assert all(r["type"] == "unanswerable" and r["answer"] is None for r in refs)
    by_id = {q["id"]: q for q in g["questions"]}
    for s in ("dev", "test"):
        counted = {}
        for i in split[s]:
            for r in by_id[i]["references"]:
                counted[r["type"]] = counted.get(r["type"], 0) + 1
        assert counted == split["counts"][s]["annotator_answers_by_type"]
    assert g["annotator_answers_with_mixed_fields"] == 0
