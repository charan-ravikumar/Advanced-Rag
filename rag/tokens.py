"""Token counting and size-bounded splitting shared by all chunkers.

Pieces returned by ``split_to_fit`` are exact substrings of the input, so
character offsets can still be recovered by the adapter.
"""

import re
from typing import Callable

from config import cfg

_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WORD_GAP = re.compile(r"\s+")
SPECIAL_TOKENS = 2  # [CLS] and [SEP] count against the model window


def regex_token_count(text: str) -> int:
    """Plain words-and-punctuation count. Undercounts WordPiece tokens: do not use as a cap."""
    return len(_TOKEN.findall(text))


_PIECE = re.compile(r"[^\W_]+|[^\s]", re.UNICODE)


def estimate_token_count(text: str) -> int:
    """Conservative WordPiece estimate for when no real tokenizer is available (tests).

    One token per punctuation mark, underscore or CJK character, one per 3 characters of any
    other word, plus the special tokens. It deliberately overestimates ordinary English (WordPiece
    needs about 1.2 tokens per word) so that math, numbers and code do not slip past the window.
    It is still a heuristic: production chunking uses the embedder's own tokenizer, and ingestion
    verifies the real count of every stored chunk.
    """
    total = SPECIAL_TOKENS
    for tok in _PIECE.findall(text):
        if tok[0].isalnum():
            cjk = len(_CJK.findall(tok))
            total += cjk + (-(-(len(tok) - cjk) // 3) if len(tok) > cjk else 0)
        else:
            total += 1
    return total


def token_counter_for(embedder=None) -> Callable[[str], int]:
    """Count with the embedder's own tokenizer when it has one, otherwise the estimate."""
    if embedder is not None and hasattr(embedder, "count_tokens"):
        return embedder.count_tokens
    tokenizer = getattr(embedder, "tokenizer", None)
    if tokenizer is not None:
        return lambda t: len(tokenizer.tokenize(t)) + SPECIAL_TOKENS
    return estimate_token_count


def max_tokens_for(embedder=None) -> int:
    """Chunk token cap: the configured cap, never above the embedder's window."""
    limit = cfg.chunking.max_tokens
    window = getattr(embedder, "max_seq_length", None)
    return min(limit, window) if window else limit


def max_chars() -> int:
    return cfg.chunking.max_chunk_chars


def _spans(text: str, separator: re.Pattern) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for m in separator.finditer(text):
        if m.start() > pos:
            spans.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    return spans


def split_to_fit_spans(text: str, max_chars: int, max_tokens: int, count: Callable[[str], int]) -> list[tuple[int, int]]:
    """``(start, end)`` of pieces of ``text`` within both caps: sentences, then words, then characters.

    Positions are computed while splitting (never searched for), so ``text[start:end]`` is exactly the piece.
    """

    def fits(a: int, b: int) -> bool:
        return (b - a) <= max_chars and count(text[a:b]) <= max_tokens

    def strip(a: int, b: int) -> tuple[int, int] | None:
        seg = text[a:b]
        stripped = seg.strip()
        if not stripped:
            return None
        left = len(seg) - len(seg.lstrip())
        return a + left, a + left + len(stripped)

    def hard_split(a: int, b: int) -> list[tuple[int, int]]:
        out = []
        while a < b:
            n = min(b - a, max_chars)
            while n > 1 and count(text[a:a + n]) > max_tokens:
                n = max(1, n // 2) if n > 8 else n - 1
            out.append((a, a + n))
            a += n
            while a < b and text[a].isspace():  # == s[n:].lstrip()
                a += 1
        return out

    def rec(a: int, b: int, level: int) -> list[tuple[int, int]]:
        st = strip(a, b)
        if st is None:
            return []
        a, b = st
        if fits(a, b):
            return [(a, b)]
        if level > 1:
            return hard_split(a, b)
        spans = [(a + x, a + y) for x, y in _spans(text[a:b], _SENTENCE_END if level == 0 else _WORD_GAP)]
        pieces: list[tuple[int, int]] = []
        cur = None
        for s, e in spans:
            if cur is None:
                cur = (s, e)
            elif fits(cur[0], e):
                cur = (cur[0], e)
            else:
                pieces.extend([cur] if fits(*cur) else rec(cur[0], cur[1], level + 1))
                cur = (s, e)
        if cur is not None:
            pieces.extend([cur] if fits(*cur) else rec(cur[0], cur[1], level + 1))
        return pieces

    return rec(0, len(text), 0)


def split_to_fit(text: str, max_chars: int, max_tokens: int, count: Callable[[str], int]) -> list[str]:
    """Pieces of ``text`` within both caps (see ``split_to_fit_spans``)."""
    return [text[a:b] for a, b in split_to_fit_spans(text, max_chars, max_tokens, count)]


def split_to_fit_with_offsets(text: str, max_chars: int, max_tokens: int, count) -> list[tuple[int, str]]:
    """``(start offset in text, piece)`` for every piece; offsets are computed, not searched for."""
    return [(a, text[a:b]) for a, b in split_to_fit_spans(text, max_chars, max_tokens, count)]
