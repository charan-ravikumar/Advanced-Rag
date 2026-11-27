"""Chunk contract + adapter tests (HashEmbedder, no network)."""

import pytest

from rag.chunk_adapter import STRATEGIES, adapt_document, locate, qasper_to_document
from rag.embedders import HashEmbedder
from rag.schemas_ingest import Chunk, make_chunk_id, sha256_text

EMB = HashEmbedder()


def _paras(topic: str, n: int) -> list[str]:
    return [f"{topic} paragraph {i} discusses {topic} models and their evaluation in detail. " * 4 for i in range(n)]


def _paper(mutate: dict | None = None) -> dict:
    secs = {
        "Introduction": _paras("alpha", 6),
        "Methods": _paras("beta", 8),
        "Results": _paras("gamma", 6),
    }
    if mutate:
        for name, (i, old, new) in mutate.items():
            secs[name][i] = secs[name][i].replace(old, new, 1)
    return {
        "title": "A Title",
        "abstract": "Short abstract about alpha beta gamma.",
        "full_text": [{"section_name": k, "paragraphs": v} for k, v in secs.items()],
    }


def _adapt(paper, strategy, doc_id="1234.5678"):
    doc = qasper_to_document(doc_id, paper)
    return doc, adapt_document(doc, doc_id, strategy, embedder=EMB)


def _norm(s):
    return " ".join(s.split())


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_deterministic(strategy):
    _, a = _adapt(_paper(), strategy)
    _, b = _adapt(_paper(), strategy)
    assert [c.chunk_id for c in a.chunks] == [c.chunk_id for c in b.chunks]
    assert a.doc_version == b.doc_version
    assert a.chunks


def test_different_strategy_different_ids():
    ids = {s: {c.chunk_id for c in _adapt(_paper(), s)[1].chunks} for s in STRATEGIES}
    assert not (ids["recursive"] & ids["layout"])
    assert not (ids["recursive"] & ids["semantic"])
    assert not (ids["layout"] & ids["semantic"])
    assert make_chunk_id("d", "recursive", 0, "h") != make_chunk_id("d", "layout", 0, "h")


def test_chunk_id_formula():
    c = Chunk.create(doc_id="d", section="s", chunk_idx=3, char_start=0, char_end=1,
                     strategy="layout", text="x", doc_version="v")
    assert c.chunk_id == make_chunk_id("d", "layout", 3, sha256_text("x"))
    assert len(c.chunk_id) == 16


def test_one_char_change_layout_changes_only_that_section():
    _, base = _adapt(_paper(), "layout")
    _, mod = _adapt(_paper({"Methods": (2, "beta", "bata")}), "layout")
    assert base.doc_version != mod.doc_version
    base_ids, mod_ids = {}, {}
    for c in base.chunks:
        base_ids.setdefault(c.section, set()).add(c.chunk_id)
    for c in mod.chunks:
        mod_ids.setdefault(c.section, set()).add(c.chunk_id)
    changed = {s for s in base_ids if base_ids[s] != mod_ids[s]}
    assert changed == {"Methods"}


def test_one_char_change_recursive_is_local_and_changes_doc_version():
    _, base = _adapt(_paper(), "recursive")
    _, mod = _adapt(_paper({"Methods": (2, "beta", "bata")}), "recursive")
    assert base.doc_version != mod.doc_version
    b = {c.chunk_idx: c for c in base.chunks}
    m = {c.chunk_idx: c for c in mod.chunks}
    changed = [i for i in b if i in m and b[i].chunk_id != m[i].chunk_id]
    assert changed, "the edited text must change at least one chunk"
    assert len(changed) < len(b) // 2
    assert all(b[i].section in {"Methods", "Introduction", "Results"} for i in changed)
    # chunks that start in other, non-adjacent sections are untouched
    assert all(b[i].section == "Methods" for i in changed) or len(changed) <= 3


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_offsets_round_trip_modulo_whitespace(strategy):
    doc, res = _adapt(_paper(), strategy)
    sections = {s.section_title: s.content for s in doc.sections}
    checked = 0
    for c in res.chunks:
        oi = res.offset_info[c.chunk_id]
        if oi.confidence in ("exact", "normalised") and not oi.spans_sections:
            assert _norm(sections[c.section][c.char_start:c.char_end]) == _norm(c.text)
            checked += 1
    assert checked > 0


def test_locate_whitespace_tolerant_and_labels():
    region = "alpha  beta\n\ngamma delta"
    s, e, conf = locate("beta gamma", region)
    assert conf == "normalised" and _norm(region[s:e]) == "beta gamma"
    assert locate("alpha", "alpha x alpha")[2] == "ambiguous"
    assert locate("zeta", region)[2] == "not_found"
    assert locate("alpha", "alpha x")[2] == "exact"


def test_metadata_flat_no_none_and_round_trip():
    _, res = _adapt(_paper(), "layout")
    for c in res.chunks:
        md = c.to_metadata()
        assert "text" not in md
        assert all(v is not None and isinstance(v, (str, int, float, bool)) for v in md.values())
        assert Chunk.from_metadata_and_text(md, c.text) == c


def test_doc_id_is_paper_id_not_title():
    _, res = _adapt(_paper(), "layout", doc_id="1601.02403")
    assert {c.doc_id for c in res.chunks} == {"1601.02403"}
