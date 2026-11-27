"""Adapter: current chunker output -> list[Chunk] with offsets and an honest confidence label.

Offset confidence (side channel, NOT stored in the vector store metadata):
  exact      chunk text found verbatim, exactly one candidate position
  normalised found only after whitespace normalisation, one candidate position
  ambiguous  more than one candidate position; the first one is used
  not_found  could not be located; offsets are -1

Every chunker splits within a single section (``section_pos`` in the raw chunk
metadata), so chunks are located inside that section's text and ``spans_sections``
is always False (a permanent test invariant). Absolute offsets are kept in
``doc_char_start`` / ``doc_char_end``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from schemas import Document, Section

from config import cfg
from rag.chunkers import CHUNKER_MAP
from rag.schemas_ingest import Chunk, sha256_text

STRATEGIES = ("recursive", "layout", "semantic")
CONFIDENCES = ("exact", "normalised", "ambiguous", "not_found")

_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class OffsetInfo:
    confidence: str
    spans_sections: bool
    doc_char_start: int
    doc_char_end: int
    by_search: bool = False  # True only when the chunker gave no usable offsets and the text was searched for


@dataclass
class AdaptedDocument:
    doc_id: str
    strategy: str
    doc_version: str
    chunks: list[Chunk]
    offset_info: dict[str, OffsetInfo]  # keyed by chunk_id


def qasper_to_document(paper_id: str, paper: dict) -> Document:
    """Corpus entry -> Document. Mirrors scripts/ingest_qasper.paper_to_document,
    but doc_id is the paper_id and the title is only a label."""
    title = paper.get("title", "Unknown")
    sections: list[Section] = []
    abstract = (paper.get("abstract") or "").strip()
    if abstract:
        sections.append(
            Section(
                content=abstract,
                section_title="Abstract",
                metadata={"section_index": 0, "section_name": "Abstract"},
            )
        )
    for idx, section in enumerate(paper.get("full_text", []), start=1):
        name = section.get("section_name", "") or f"Section {idx}"
        text = "\n\n".join(p.strip() for p in section.get("paragraphs", []) if p.strip())
        if not text:
            continue
        sections.append(
            Section(content=text, section_title=name, metadata={"section_index": idx, "section_name": name})
        )
    return Document(
        content="\n\n".join(s.content for s in sections),
        metadata={"source": title, "paper_id": paper_id, "title": title, "paper_title": title, "file_type": "qasper"},
        sections=sections,
    )


def _normalise_with_map(s: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs to one space and strip; map each output char to its source index."""
    out: list[str] = []
    idx: list[int] = []
    prev_space = True
    for i, ch in enumerate(s):
        if ch.isspace():
            if not prev_space:
                out.append(" ")
                idx.append(i)
                prev_space = True
        else:
            out.append(ch)
            idx.append(i)
            prev_space = False
    if out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


def _occurrences(haystack: str, needle: str, start: int) -> list[int]:
    found, pos = [], haystack.find(needle, start)
    while pos != -1:
        found.append(pos)
        pos = haystack.find(needle, pos + 1)
    return found


def locate(text: str, region: str, hint: int = 0) -> tuple[int, int, str]:
    """Find ``text`` in ``region`` at or after ``hint``. Returns (start, end, confidence)."""
    if not text:
        return -1, -1, "not_found"
    hits = _occurrences(region, text, hint)
    if hits:
        return hits[0], hits[0] + len(text), "exact" if len(hits) == 1 else "ambiguous"
    norm_region, rmap = _normalise_with_map(region)
    norm_text, _ = _normalise_with_map(text)
    if not norm_text:
        return -1, -1, "not_found"
    # translate hint into normalised coordinates
    nhint = next((k for k, src in enumerate(rmap) if src >= hint), len(rmap))
    hits = _occurrences(norm_region, norm_text, nhint)
    if not hits:
        return -1, -1, "not_found"
    s = rmap[hits[0]]
    e = rmap[hits[0] + len(norm_text) - 1] + 1
    return s, e, "normalised" if len(hits) == 1 else "ambiguous"


def _section_ranges(document: Document) -> list[tuple[int, int, str]]:
    """(start, end, title) of each section inside document.content.

    ``qasper_to_document`` joins sections with a blank line, so positions are computed cumulatively; a search
    is only a fallback for Documents built another way (and is used only when the computed slice disagrees).
    """
    ranges, pos = [], 0
    for sec in document.sections:
        if document.content[pos:pos + len(sec.content)] != sec.content:
            found = document.content.find(sec.content, pos)
            if found == -1:
                return [(0, len(document.content), "")]
            pos = found
        ranges.append((pos, pos + len(sec.content), sec.section_title))
        pos += len(sec.content) + 2  # "\n\n" between sections
    return ranges or [(0, len(document.content), "")]


def adapt_document(document: Document, doc_id: str, strategy: str, embedder=None) -> AdaptedDocument:
    """Run the existing chunker for ``strategy`` and convert its output to list[Chunk].

    Offsets come from the chunker (``section_char_start`` / ``section_char_end``, computed while splitting) and
    are only VERIFIED here against the section text. Searching for the chunk text is a fallback for chunkers
    that report no offsets; ``OffsetInfo.by_search`` says when it was used.
    """
    if strategy not in CHUNKER_MAP:
        raise ValueError(f"unknown strategy: {strategy!r}")
    chunker = CHUNKER_MAP[strategy]
    raw = chunker(document, embedder=embedder)

    doc_version = sha256_text(document.content)
    ranges = _section_ranges(document)
    chunks: list[Chunk] = []
    info: dict[str, OffsetInfo] = {}
    max_overlap = cfg.chunking.recursive_chunk_overlap if strategy == "recursive" else 0
    cur_sec, prev_s, prev_e = None, -1, 0

    for raw_chunk in raw:
        text = raw_chunk.content
        if not text.strip():
            continue
        sec_no = raw_chunk.metadata.get("section_pos", 0)
        sec_start, sec_end, title = ranges[sec_no] if sec_no < len(ranges) else ranges[0]
        region = document.content[sec_start:sec_end]
        if sec_no != cur_sec:
            cur_sec, prev_s, prev_e = sec_no, -1, 0
        ks, ke = raw_chunk.metadata.get("section_char_start"), raw_chunk.metadata.get("section_char_end")
        by_search = False
        s = e = -1
        conf = "not_found"
        if ks is not None and ke is not None and 0 <= ks < ke <= len(region):
            seg = region[ks:ke]
            if seg == text:
                s, e, conf = ks, ke, "exact"
            elif _WS.sub(" ", seg).strip() == _WS.sub(" ", text).strip():
                s, e, conf = ks, ke, "normalised"  # e.g. semantic chunks join paragraphs with a blank line
        if s < 0:
            by_search = True
            cursor = max(prev_s + 1, prev_e - max_overlap)
            s, e, conf = locate(text, region, cursor)
            if s < 0 and cursor > prev_s + 1:
                s, e, conf = locate(text, region, prev_s + 1)
                if s >= 0:
                    conf = "ambiguous"
        if s >= 0:
            prev_s, prev_e = s, e
            doc_s, doc_e = sec_start + s, sec_start + e
            char_start, char_end = s, e
        else:
            doc_s = doc_e = -1
            char_start = char_end = -1
        spans = False

        chunk = Chunk.create(
            doc_id=doc_id,
            section=title,
            chunk_idx=len(chunks),
            char_start=char_start,
            char_end=char_end,
            strategy=strategy,
            text=text,
            doc_version=doc_version,
        )
        chunks.append(chunk)
        info[chunk.chunk_id] = OffsetInfo(conf, spans, doc_s, doc_e, by_search)

    return AdaptedDocument(doc_id, strategy, doc_version, chunks, info)
