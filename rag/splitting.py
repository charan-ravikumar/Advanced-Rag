"""Offset-native recursive character splitting.

``recursive_split_spans`` reproduces ``langchain_text_splitters.RecursiveCharacterTextSplitter``
(separators ``["\\n\\n", "\\n", " ", ""]``, ``keep_separator=True`` i.e. "start", ``strip_whitespace=True``,
``length_function=len``) but returns ``(start, end)`` positions in the input text instead of strings.
The positions are COMPUTED while splitting, never searched for: the old code located each returned chunk
with ``str.find``, which can land one period early in periodic or repeated text.

Why exact positions are possible: with ``keep_separator="start"`` every split is a contiguous slice of the
text, a merged chunk is the slice from its first split's start to its last split's end, and stripping only
moves the two ends inward. tests/unit/test_splitting.py proves equality with the LangChain output on random
text and on the real corpus.
"""

from __future__ import annotations

import re

DEFAULT_SEPARATORS = ("\n\n", "\n", " ", "")


def _pieces(text: str, lo: int, hi: int, separator: str) -> list[tuple[int, int]]:
    """Non-empty contiguous pieces of text[lo:hi]; each separator is kept at the START of the piece after it."""
    if not separator:
        return [(i, i + 1) for i in range(lo, hi)]
    starts = [lo + m.start() for m in re.finditer(re.escape(separator), text[lo:hi])]
    bounds = [lo, *starts, hi]
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]


def _strip(text: str, start: int, end: int) -> tuple[int, int] | None:
    seg = text[start:end]
    left = len(seg) - len(seg.lstrip())
    stripped = seg.strip()
    if not stripped:
        return None
    return start + left, start + left + len(stripped)


def _merge(text: str, splits: list[tuple[int, int]], chunk_size: int, chunk_overlap: int) -> list[tuple[int, int]]:
    """LangChain ``_merge_splits`` with an empty separator, on spans."""
    out: list[tuple[int, int]] = []
    current: list[tuple[int, int]] = []
    total = 0

    def emit() -> None:
        if not current:
            return
        span = _strip(text, current[0][0], current[-1][1])
        if span is not None:
            out.append(span)

    for s, e in splits:
        length = e - s
        if total + length > chunk_size:
            if current:
                emit()
                while total > chunk_overlap or (total + length > chunk_size and total > 0):
                    total -= current[0][1] - current[0][0]
                    current = current[1:]
        current.append((s, e))
        total += length
    emit()
    return out


def _split(text: str, lo: int, hi: int, separators: tuple[str, ...], chunk_size: int, chunk_overlap: int) -> list[tuple[int, int]]:
    region = text[lo:hi]
    separator, rest = separators[-1], ()
    for i, candidate in enumerate(separators):
        if not candidate:
            separator = candidate
            break
        if candidate in region:
            separator, rest = candidate, separators[i + 1:]
            break

    final: list[tuple[int, int]] = []
    good: list[tuple[int, int]] = []
    for s, e in _pieces(text, lo, hi, separator):
        if e - s < chunk_size:
            good.append((s, e))
            continue
        if good:
            final.extend(_merge(text, good, chunk_size, chunk_overlap))
            good = []
        if not rest:
            final.append((s, e))  # LangChain appends an oversized last-resort piece unstripped
        else:
            final.extend(_split(text, s, e, rest, chunk_size, chunk_overlap))
    if good:
        final.extend(_merge(text, good, chunk_size, chunk_overlap))
    return final


def recursive_split_spans(text: str, chunk_size: int, chunk_overlap: int,
                          separators: tuple[str, ...] = DEFAULT_SEPARATORS) -> list[tuple[int, int]]:
    """``(start, end)`` of every chunk, in order, such that ``text[start:end]`` equals the LangChain chunk."""
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap > chunk_size:
        raise ValueError("need chunk_size > 0 and 0 <= chunk_overlap <= chunk_size")
    return _split(text, 0, len(text), tuple(separators), chunk_size, chunk_overlap)
