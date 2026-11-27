"""Qasper question classification and gold-span location (pure: no network, no model, no store).

``classify_question`` applies EXACTLY the builder's filters (scripts/build_qasper_eval_set.parse_question)
but says why a question was dropped. ``locate_evidence`` finds an evidence paragraph in the sections the
chunkers see (``qasper_to_document``), so a gold span uses the same coordinates as a chunk:
(doc_id, section title, section-relative char_start, char_end).

Location confidence per evidence paragraph:
  exact       found verbatim, exactly once in the document
  normalised  found only after whitespace normalisation, exactly once
  ambiguous   found more than once (every occurrence is kept as a span: any of them is acceptable evidence)
  not_found   not found
Nothing is dropped silently: ``not_found`` evidence is reported with the question id.
"""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass
from pathlib import Path

from rag.chunk_adapter import _normalise_with_map, _occurrences, qasper_to_document

ROOT = Path(__file__).resolve().parents[1]
_BUILDER = None


def _builder():
    """Load scripts/build_qasper_eval_set.py by path (scripts/ is not a package). Import has no side effects."""
    global _BUILDER
    if _BUILDER is None:
        spec = importlib.util.spec_from_file_location("build_qasper_eval_set", ROOT / "scripts" / "build_qasper_eval_set.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _BUILDER = module
    return _BUILDER


@dataclass(frozen=True)
class GoldSpan:
    doc_id: str
    section: str
    section_idx: int  # index into Document.sections (disambiguates duplicate titles)
    char_start: int
    char_end: int
    evidence_idx: int  # which evidence paragraph of the question this span belongs to
    confidence: str

    def to_dict(self) -> dict:
        return asdict(self)


# Question-level statuses
UNANSWERABLE = "unanswerable"
ANSWERABLE = "answerable"
DROPPED = "dropped"


def classify_question(paper_id: str, paper: dict, qa: dict) -> tuple[str, str, dict | None]:
    """Return ``(status, reason, record)``; ``record`` is the golden-set record (as the builder writes it)
    for kept questions. Mirrors ``parse_question`` branch by branch."""
    b = _builder()
    question = (qa.get("question") or "").strip()
    annotations = [a.get("answer", {}) for a in qa.get("answers", [])]
    if not question or not annotations:
        return DROPPED, "no_question_text_or_no_annotations", None

    record = b.parse_question(paper_id, paper, qa)
    unanswerable_flags = [a.get("unanswerable", False) for a in annotations]
    if all(unanswerable_flags):
        return UNANSWERABLE, "all_annotators_unanswerable", record
    if any(unanswerable_flags):
        return DROPPED, "annotators_disagree_on_answerability", None
    if record is not None:
        return ANSWERABLE, "kept", record

    # dropped inside the answerable branch: find out why
    paragraphs = b._paper_paragraphs(paper)
    answers = [t for t in (b._answer_text(a) for a in annotations) if t]
    evidence_raw = [e.strip() for a in annotations for e in (a.get("evidence") or []) if e and e.strip()]
    if not answers:
        return DROPPED, "no_answer_text", None
    if not evidence_raw:
        return DROPPED, "no_evidence_given", None
    non_float = [e for e in evidence_raw if not e.startswith("FLOAT SELECTED")]
    if not non_float:
        return DROPPED, "evidence_is_table_or_figure_only", None
    if not any(e in paragraphs for e in non_float):
        return DROPPED, "evidence_not_a_verbatim_paragraph", None
    return DROPPED, "other", None  # cannot happen if parse_question and this mirror agree


def locate_evidence(doc, doc_id: str, evidence: str, evidence_idx: int = 0) -> tuple[list[GoldSpan], str]:
    """Locate one evidence paragraph in ``doc`` (a Document from ``qasper_to_document``)."""
    if not evidence or not evidence.strip():
        return [], "not_found"
    exact: list[tuple[int, int]] = []
    for si, sec in enumerate(doc.sections):
        for pos in _occurrences(sec.content, evidence, 0):
            exact.append((si, pos))
    if exact:
        conf = "exact" if len(exact) == 1 else "ambiguous"
        spans = [GoldSpan(doc_id, doc.sections[si].section_title, si, pos, pos + len(evidence), evidence_idx, conf) for si, pos in exact]
        return spans, conf

    norm_ev, _ = _normalise_with_map(evidence)
    if not norm_ev:
        return [], "not_found"
    found: list[tuple[int, int, int]] = []
    for si, sec in enumerate(doc.sections):
        norm_sec, smap = _normalise_with_map(sec.content)
        for pos in _occurrences(norm_sec, norm_ev, 0):
            found.append((si, smap[pos], smap[pos + len(norm_ev) - 1] + 1))
    if not found:
        return [], "not_found"
    conf = "normalised" if len(found) == 1 else "ambiguous"
    spans = [GoldSpan(doc_id, doc.sections[si].section_title, si, s, e, evidence_idx, conf) for si, s, e in found]
    return spans, conf


def gold_spans_for_question(paper_id: str, paper: dict, evidence: list[str]) -> dict:
    """Locate every evidence paragraph of one question.

    Returns ``{"spans": [GoldSpan...], "per_evidence": [confidence...], "all_lost": bool}``.
    """
    doc = qasper_to_document(paper_id, paper)
    spans: list[GoldSpan] = []
    per_evidence: list[str] = []
    for i, ev in enumerate(evidence):
        s, conf = locate_evidence(doc, paper_id, ev, i)
        spans.extend(s)
        per_evidence.append(conf)
    return {"spans": spans, "per_evidence": per_evidence, "all_lost": bool(per_evidence) and all(c == "not_found" for c in per_evidence)}


# ---- annotator references: answers with their type and evidence -------------------------------------------------------

ANSWER_TYPES = ("extractive", "abstractive", "yes_no", "unanswerable")


def answer_type(answer: dict) -> tuple[str, bool]:
    """``(type, mixed)`` for one annotator's answer. Qasper gives each answer exactly one type; ``mixed`` is True
    if more than one answer field is populated (reported, never silently resolved). Priority when mixed:
    unanswerable, extractive, yes_no, abstractive."""
    flags = {
        "unanswerable": bool(answer.get("unanswerable")),
        "extractive": bool(answer.get("extractive_spans")),
        "yes_no": answer.get("yes_no") is not None,
        "abstractive": bool(answer.get("free_form_answer")),
    }
    present = [t for t in ("unanswerable", "extractive", "yes_no", "abstractive") if flags[t]]
    return (present[0] if present else "none"), len(present) > 1


def reference_records(paper: dict, qa: dict) -> list[dict]:
    """One record per annotator: ``{answer, type, mixed_types, evidence, n_evidence_not_retrievable}``.

    ``evidence`` holds the annotator's evidence PARAGRAPHS that exist verbatim in the paper's text (what a
    retriever over abstract + body can return). Table/figure evidence ("FLOAT SELECTED") and anything that is not a
    verbatim paragraph is counted in ``n_evidence_not_retrievable`` but not listed.
    """
    b = _builder()
    paragraphs = b._paper_paragraphs(paper)
    out = []
    for a in (x.get("answer", {}) for x in qa.get("answers", [])):
        typ, mixed = answer_type(a)
        evidence, lost = [], 0
        for ev in a.get("evidence") or []:
            ev = ev.strip()
            if not ev:
                continue
            if not ev.startswith("FLOAT SELECTED") and ev in paragraphs:
                if ev not in evidence:
                    evidence.append(ev)
            else:
                lost += 1
        out.append({"answer": None if typ == "unanswerable" else b._answer_text(a), "type": typ, "mixed_types": mixed,
                    "evidence": evidence, "n_evidence_not_retrievable": lost})
    return out


def paragraph_index(paper_id: str, paper: dict) -> list:
    """Every paragraph of the document as ``Paragraph`` records in the coordinates chunks use.

    Mirrors ``qasper_to_document``: the abstract is a section of one paragraph; every other section is its
    non-empty paragraphs (stripped) joined by a blank line, so the paragraphs of a section start at 0 and each next
    one starts 2 characters after the previous one ends.
    """
    from rag.eval_metrics import Paragraph

    out: list = []
    doc = qasper_to_document(paper_id, paper)
    abstract = (paper.get("abstract") or "").strip()
    sources: list[list[str]] = []
    if abstract:
        sources.append([abstract])
    for section in paper.get("full_text", []):
        paras = [p.strip() for p in section.get("paragraphs", []) if p.strip()]
        if paras:
            sources.append(paras)
    assert len(sources) == len(doc.sections), "paragraph index and document sections disagree"
    for si, (paras, sec) in enumerate(zip(sources, doc.sections)):
        pos = 0
        for pi, text in enumerate(paras):
            out.append(Paragraph(sec.section_title, si, pi, pos, pos + len(text), text))
            pos += len(text) + 2
        assert pos - 2 == len(sec.content), "section content is not the joined paragraphs"
    return out
