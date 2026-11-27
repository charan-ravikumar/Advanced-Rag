"""Property tests for the chunkers (HashEmbedder, no network).

Properties, for every strategy:
  coverage      every non-whitespace character of every section is inside some chunk span
  non-empty     no empty / whitespace-only chunks
  deterministic two runs give identical chunks
  char cap      no chunk longer than chunking.max_chunk_chars
  token cap     no chunk longer than min(chunking.max_tokens, embedder max_seq_length)

Coverage is evaluated on absolute document offsets (side channel of the adapter),
so chunks that cross a section boundary still count for both sections.
"""

import re

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from config import cfg
from rag.chunk_adapter import STRATEGIES, _section_ranges, adapt_document, qasper_to_document
from rag.embedders import HashEmbedder

EMB = HashEmbedder()
MAX_CHARS = cfg.chunking.max_chunk_chars
MAX_TOKENS = min(cfg.chunking.max_tokens, EMB.max_seq_length)

_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def count_tokens(text: str) -> int:
    """Simple injected token counter: words and punctuation marks."""
    return len(_TOKEN.findall(text))


WORDS = {
    "a": "alpha model training dataset corpus annotation label".split(),
    "b": "network latency server cache memory thread queue".split(),
    "c": "protein cell enzyme sequence gene expression assay".split(),
}
UNICODE = [
    "Ünïcödé résumé naïve café — “quoted” text ✓ über straße.",
    "数据集 包含 许多 样本 。 模型 在 其上 训练 。",
    "Привет мир. Это тест токенизации и сегментации предложений.",
    "emoji 🙂 mix 🚀 of symbols ∑ ∫ ≈ and words appear here.",
]


def _sentences(topic: str, n: int) -> str:
    ws = WORDS[topic]
    return " ".join(
        " ".join(ws[(i + j) % len(ws)] for j in range(8)).capitalize() + "." for i in range(n)
    )


SHORT = st.sampled_from(["Hi.", "Yes.", "See Table 1.", "ok", "Fig. 2 shows it."])
NORMAL = st.builds(_sentences, st.sampled_from("abc"), st.integers(2, 12))
LONG = st.builds(_sentences, st.sampled_from("abc"), st.integers(60, 140))
UNI = st.builds(lambda s, k: " ".join([s] * k), st.sampled_from(UNICODE), st.integers(1, 40))
REPEATED = st.sampled_from(
    ["This exact paragraph is repeated many times in the document.", _sentences("a", 5)]
)
PARAGRAPH = st.one_of(SHORT, NORMAL, NORMAL, LONG, UNI, REPEATED)
SECTION = st.lists(PARAGRAPH, min_size=1, max_size=8)
PAPER = st.lists(SECTION, min_size=1, max_size=4)


def build(paper_sections) -> tuple:
    paper = {
        "title": "T",
        "abstract": "",
        "full_text": [
            {"section_name": f"S{i}", "paragraphs": paras} for i, paras in enumerate(paper_sections)
        ],
    }
    return qasper_to_document("0000.00000", paper)


# Example count, derandomization and the other Hypothesis settings come from the profile that
# tests/conftest.py loads ("ci" by default, "explore" via ADVRAG_HYPOTHESIS_PROFILE); see CLAUDE.md.
SETTINGS = settings()


def _run(sections, strategy):
    doc = build(sections)
    return doc, adapt_document(doc, "0000.00000", strategy, embedder=EMB)


def _uncovered(doc, res) -> list[tuple[str, str]]:
    """Return (section, snippet) for runs of uncovered non-whitespace characters."""
    covered = [False] * len(doc.content)
    for c in res.chunks:
        oi = res.offset_info[c.chunk_id]
        if oi.doc_char_start >= 0:
            for i in range(oi.doc_char_start, oi.doc_char_end):
                covered[i] = True
    missing = []
    for start, end, title in _section_ranges(doc):
        i = start
        while i < end:
            if not covered[i] and not doc.content[i].isspace():
                j = i
                while j < end and not covered[j]:
                    j += 1
                missing.append((title, doc.content[i:j][:60]))
                i = j
            else:
                i += 1
    return missing


@pytest.mark.parametrize("strategy", STRATEGIES)
@SETTINGS
# Permanent example: this input (a 114-sentence paragraph then a 2-sentence one, repeating vocabulary) was found by
# this property and used to lose the last paragraph in the recursive chunker (positions were searched for with
# str.find). Generation depends on which modules are imported (Hypothesis reads literals from sys.modules), so every
# bug the property finds must be pinned here instead of trusting generation to find it again.
@example([[_sentences("a", 114), _sentences("a", 2)]])
@given(PAPER)
def test_coverage_no_content_loss(strategy, sections):
    doc, res = _run(sections, strategy)
    assert not [c for c in res.chunks if res.offset_info[c.chunk_id].confidence == "not_found"]
    assert _uncovered(doc, res) == []


@pytest.mark.parametrize("strategy", STRATEGIES)
@SETTINGS
@example([[_sentences("a", 114), _sentences("a", 2)]])
@given(PAPER)
def test_offsets_are_computed_and_slice_back_to_the_chunk_text(strategy, sections):
    """Every chunk's recorded (char_start, char_end) slices its section to the chunk text (modulo whitespace) and no
    chunk was located by searching."""
    doc, res = _run(sections, strategy)
    by_title = {}
    for s in doc.sections:
        by_title.setdefault(s.section_title, []).append(s.content)
    norm = lambda t: re.sub(r"\s+", " ", t).strip()  # noqa: E731
    for c in res.chunks:
        assert res.offset_info[c.chunk_id].by_search is False
        assert c.char_start >= 0
        assert any(norm(sec[c.char_start:c.char_end]) == norm(c.text) for sec in by_title[c.section])


@pytest.mark.parametrize("strategy", STRATEGIES)
@SETTINGS
@given(PAPER)
def test_no_empty_chunks(strategy, sections):
    _, res = _run(sections, strategy)
    assert all(c.text.strip() for c in res.chunks)


@pytest.mark.parametrize("strategy", STRATEGIES)
@SETTINGS
@given(PAPER)
def test_deterministic(strategy, sections):
    _, a = _run(sections, strategy)
    _, b = _run(sections, strategy)
    assert [(c.chunk_id, c.text) for c in a.chunks] == [(c.chunk_id, c.text) for c in b.chunks]


@pytest.mark.parametrize("strategy", STRATEGIES)
@SETTINGS
@given(PAPER)
def test_char_cap(strategy, sections):
    _, res = _run(sections, strategy)
    assert max((len(c.text) for c in res.chunks), default=0) <= MAX_CHARS


@pytest.mark.parametrize("strategy", STRATEGIES)
@SETTINGS
@given(PAPER)
def test_token_cap(strategy, sections):
    _, res = _run(sections, strategy)
    assert max((count_tokens(c.text) for c in res.chunks), default=0) <= MAX_TOKENS


@SETTINGS
@given(PAPER)
def test_recursive_effective_size_cap(sections):
    """Recursive chunks may exceed recursive_chunk_size ONLY through the within-section merge of a
    short tail, so the effective cap is chunk_size + min_chunk_chars - 1 (not the 2000 hard cap)."""
    _, res = _run(sections, "recursive")
    cap = cfg.chunking.recursive_chunk_size + cfg.chunking.min_chunk_chars - 1
    assert max((len(c.text) for c in res.chunks), default=0) <= cap


@pytest.mark.parametrize("strategy", STRATEGIES)
@SETTINGS
@given(PAPER)
def test_never_spans_sections(strategy, sections):
    """Permanent invariant: no chunk of any strategy crosses a section boundary."""
    doc, res = _run(sections, strategy)
    ranges = _section_ranges(doc)
    for c in res.chunks:
        oi = res.offset_info[c.chunk_id]
        assert oi.spans_sections is False
        if oi.doc_char_start >= 0:
            assert any(s <= oi.doc_char_start and oi.doc_char_end <= e for s, e, _ in ranges)


@pytest.mark.parametrize("strategy", STRATEGIES)
@SETTINGS
@given(PAPER)
def test_undersized_chunks_not_merged_across_sections(strategy, sections):
    """Chunk count per section depends only on that section's own content."""
    doc, res = _run(sections, strategy)
    for i, section_paras in enumerate(sections):
        solo_doc, solo = _run([section_paras], strategy)
        assert [c.text for c in res.chunks if c.section == f"S{i}"] == [c.text for c in solo.chunks]


# ---- minimal, hand-written reproductions of the two expected bugs ----------

def test_minimal_semantic_drops_small_leading_group():
    """A tiny paragraph followed by a ~1990-char paragraph: they cannot merge
    (over the char cap), and the small leftover group is discarded."""
    big = ("beta " * 400)[:1995].strip()
    doc, res = _run([["Tiny intro.", big]], "semantic")
    assert _uncovered(doc, res) == []


def test_regression_recursive_chunker_covers_periodic_text():
    """Regression for the str.find bug (fixed by offset-native splitting): a long periodic paragraph followed by a short
    one used to lose the short paragraph. A normal test now; it was a strict xfail while the bug was open."""
    doc, res = _run([[_sentences("a", 114), _sentences("a", 2)]], "recursive")
    assert _uncovered(doc, res) == []
    assert not any(res.offset_info[c.chunk_id].by_search for c in res.chunks)


def test_minimal_layout_chunk_exceeds_embedding_window():
    """One section of 300 short sentences is a single chunk far over max_tokens."""
    doc, res = _run([[_sentences("a", 60)]], "layout")
    assert max(count_tokens(c.text) for c in res.chunks) <= MAX_TOKENS


@pytest.mark.model
@pytest.mark.slow
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_fallback_counter_is_safe_under_real_tokenizer(strategy):
    """Chunks built with the NO-tokenizer fallback (HashEmbedder) must still fit the real window
    on real scientific text (8 Qasper papers, including inline math).

    The plain regex counter failed this: its chunks reached 357 real tokens. The fallback is a
    heuristic and is NOT safe for heavily accented or CJK text (it undercounts it); that is
    why production passes the real embedder and ingestion re-checks every chunk.
    """
    import json
    from pathlib import Path

    model = _real_model()
    if model is None:
        pytest.skip("embedding model not in local HuggingFace cache")
    corpus = json.loads((Path(__file__).resolve().parents[2] / "eval/datasets/qasper_dev/corpus.json").read_text(encoding="utf-8"))
    papers = list(corpus.items())[:8]
    docs = [qasper_to_document(pid, p) for pid, p in papers]
    limit = min(cfg.chunking.max_tokens, model.max_seq_length)
    for doc in docs:
        for c in adapt_document(doc, "x", strategy, embedder=EMB).chunks:
            n = model.count_tokens(c.text)
            assert n <= limit, f"{n} real tokens > {limit}: {c.text[:60]!r}"


# ---- real tokenizer, only when the model is cached --------------------------

def _real_model():
    """The real embedder, or None when the pinned revision is not cached."""
    from rag.embedders import SentenceTransformerEmbedder

    emb = SentenceTransformerEmbedder.from_config(cfg)
    try:
        emb.model  # loads from the local cache only
    except RuntimeError:
        return None
    return emb


@pytest.mark.model
@pytest.mark.slow
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_real_tokenizer_token_cap(strategy):
    model = _real_model()
    if model is None:
        pytest.skip("embedding model not in local HuggingFace cache")
    doc = build([[_sentences("a", 80), _sentences("b", 90)], [" ".join(UNICODE * 30)]])
    res = adapt_document(doc, "0000.00000", strategy, embedder=model)
    limit = min(cfg.chunking.max_tokens, model.max_seq_length)
    for c in res.chunks:
        # measured with the raw tokenizer, independently of the chunker's counter
        n = len(model.tokenizer(c.text, add_special_tokens=True)["input_ids"])
        assert n <= limit, f"{n} tokens > {limit}"
