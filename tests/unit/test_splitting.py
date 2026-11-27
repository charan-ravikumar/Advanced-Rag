"""The offset-native splitters against frozen reference implementations (the old code), and the guarantee that
no chunker searches for a chunk's position.

Reference oracles in this file:
  * langchain's RecursiveCharacterTextSplitter (the splitter the old recursive_chunk used),
  * the previous string-returning ``split_to_fit``,
  * the previous paragraph pipeline of the semantic chunker (collapse 3+ newlines, split on blank lines, strip).
"""

import ast
import json
import re
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.chunkers import _paragraph_spans
from rag.splitting import recursive_split_spans
from rag.tokens import (
    estimate_token_count,
    split_to_fit,
    split_to_fit_spans,
    split_to_fit_with_offsets,
)

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "eval" / "datasets" / "qasper_dev" / "corpus.json"

# ---- text strategies ------------------------------------------------------------------------------------

ALPHABET = st.sampled_from(list("abcde  .\n\n\t,é"))
RANDOM_TEXT = st.text(alphabet=ALPHABET, max_size=1500)
PERIODIC_TEXT = st.builds(lambda unit, k: unit * k, st.text(alphabet=ALPHABET, min_size=1, max_size=40), st.integers(1, 60))
PROSE = st.builds(
    lambda words, seps: "".join(w + s for w, s in zip(words, seps)),
    st.lists(st.sampled_from(["alpha", "beta", "gamma", "delta", "model", "data."]), min_size=1, max_size=300),
    st.lists(st.sampled_from([" ", " ", " ", "\n", "\n\n", "\n\n\n", "  "]), min_size=300, max_size=300),
)
ANY_TEXT = st.one_of(RANDOM_TEXT, PERIODIC_TEXT, PROSE)
SIZES = st.tuples(st.integers(1, 200), st.integers(0, 200)).filter(lambda t: t[1] <= t[0])


# ---- recursive splitter vs LangChain -----------------------------------------------------------------------

@given(ANY_TEXT, SIZES)
def test_recursive_spans_equal_langchain_chunks_on_arbitrary_text(text, sizes):
    size, overlap = sizes
    reference = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap).split_text(text)
    spans = recursive_split_spans(text, size, overlap)
    assert [text[a:b] for a, b in spans] == reference
    assert all(0 <= a < b <= len(text) for a, b in spans)


@given(PERIODIC_TEXT, st.sampled_from([(100, 20), (1000, 200), (50, 0)]))
def test_recursive_spans_on_periodic_text_are_exact(text, sizes):
    """The old locate-by-find failed here: every span must slice back to exactly its chunk text."""
    size, overlap = sizes
    reference = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap).split_text(text)
    assert [text[a:b] for a, b in recursive_split_spans(text, size, overlap)] == reference


def test_recursive_spans_on_the_text_that_broke_str_find():
    """Regression: the input hypothesis found (a long periodic paragraph, then a short one)."""
    from tests.unit.test_chunker_properties import _sentences

    text = _sentences("a", 114) + "\n\n" + _sentences("a", 2)
    reference = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(text)
    spans = recursive_split_spans(text, 1000, 200)
    assert [text[a:b] for a, b in spans] == reference
    assert spans[-1][1] == len(text)  # the final paragraph is covered


def test_recursive_spans_validate_arguments():
    for bad in [(0, 0), (10, 11), (10, -1)]:
        with pytest.raises(ValueError):
            recursive_split_spans("abc", *bad)
    assert recursive_split_spans("", 10, 2) == [] and recursive_split_spans(" \n ", 10, 2) == []


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus not present")
def test_recursive_spans_equal_langchain_on_every_real_section():
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    ref_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    checked = 0
    for paper in corpus.values():
        texts = [paper.get("abstract", "")] + ["\n\n".join(p.strip() for p in s["paragraphs"] if p.strip()) for s in paper["full_text"]]
        for text in (t for t in texts if t.strip()):
            assert [text[a:b] for a, b in recursive_split_spans(text, 1000, 200)] == ref_splitter.split_text(text)
            checked += 1
    assert checked > 2000


# ---- split_to_fit vs the previous implementation ----------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WORD_GAP = re.compile(r"\s+")


def _ref_spans(text, separator):
    spans, pos = [], 0
    for m in separator.finditer(text):
        if m.start() > pos:
            spans.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    return spans


def reference_split_to_fit(text, max_chars, max_tokens, count):
    """Frozen copy of the string-returning implementation that preceded the offset-native one."""

    def fits(s):
        return len(s) <= max_chars and count(s) <= max_tokens

    def hard_split(s):
        out = []
        while s:
            n = min(len(s), max_chars)
            while n > 1 and count(s[:n]) > max_tokens:
                n = max(1, n // 2) if n > 8 else n - 1
            out.append(s[:n])
            s = s[n:].lstrip()
        return out

    def rec(s, level):
        s = s.strip()
        if not s:
            return []
        if fits(s):
            return [s]
        if level > 1:
            return hard_split(s)
        spans = _ref_spans(s, _SENTENCE_END if level == 0 else _WORD_GAP)
        pieces, cur = [], None

        def flush():
            piece = s[cur[0]:cur[1]]
            pieces.extend([piece] if fits(piece) else rec(piece, level + 1))

        for a, b in spans:
            if cur is None:
                cur = (a, b)
            elif fits(s[cur[0]:b]):
                cur = (cur[0], b)
            else:
                flush()
                cur = (a, b)
        if cur is not None:
            flush()
        return pieces

    return rec(text, 0)


@given(ANY_TEXT, st.integers(5, 300), st.integers(3, 80))
def test_split_to_fit_matches_the_previous_implementation_and_offsets_are_exact(text, max_chars, max_tokens):
    expected = reference_split_to_fit(text, max_chars, max_tokens, estimate_token_count)
    spans = split_to_fit_spans(text, max_chars, max_tokens, estimate_token_count)
    assert [text[a:b] for a, b in spans] == expected
    assert split_to_fit(text, max_chars, max_tokens, estimate_token_count) == expected
    assert split_to_fit_with_offsets(text, max_chars, max_tokens, estimate_token_count) == [(a, text[a:b]) for a, b in spans]
    assert all(a < b for a, b in spans) and [a for a, _ in spans] == sorted(a for a, _ in spans)


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus not present")
def test_split_to_fit_matches_the_previous_implementation_on_real_sections():
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    n = 0
    for paper in list(corpus.values())[:60]:
        for s in paper["full_text"]:
            text = "\n\n".join(p.strip() for p in s["paragraphs"] if p.strip())
            if text:
                assert [text[a:b] for a, b in split_to_fit_spans(text, 2000, 256, estimate_token_count)] == \
                    reference_split_to_fit(text, 2000, 256, estimate_token_count)
                n += 1
    assert n > 100


# ---- semantic paragraphs ---------------------------------------------------------------------------------------

def reference_paragraphs(content):
    text = re.sub(r"\n{3,}", "\n\n", content).strip()
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


@given(ANY_TEXT)
def test_paragraph_spans_equal_the_previous_paragraph_pipeline(content):
    spans = _paragraph_spans(content)
    assert [content[a:b] for a, b in spans] == reference_paragraphs(content)


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus not present")
def test_paragraph_spans_equal_the_previous_pipeline_on_real_sections():
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    for paper in corpus.values():
        for s in paper["full_text"]:
            text = "\n\n".join(p.strip() for p in s["paragraphs"] if p.strip())
            assert [text[a:b] for a, b in _paragraph_spans(text)] == reference_paragraphs(text)


# ---- nothing searches for a chunk's position -----------------------------------------------------------------------

@pytest.mark.parametrize("module", ["rag/chunkers.py", "rag/tokens.py", "rag/splitting.py"])
def test_chunking_modules_never_search_for_a_position(module):
    """Offsets are computed, never searched for: no str.find/rfind/index/rindex call in the chunking modules."""
    tree = ast.parse((ROOT / module).read_text(encoding="utf-8"))
    searches = [(n.lineno, n.func.attr) for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in {"find", "rfind", "index", "rindex"}]
    assert searches == []
