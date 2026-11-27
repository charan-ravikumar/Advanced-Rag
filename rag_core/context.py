"""Context builder: which retrieved chunks go into the prompt, in relevance order, within a token budget.

Budget accounting. The generator (Gemini) uses a tokenizer we do not load here, so the budget is measured with
``rag.tokens.estimate_token_count``: a deliberately conservative WordPiece estimate (about one token per three characters of a
word plus one per punctuation mark), which overestimates ordinary English for BPE tokenizers (roughly four characters per token).
Over-counting only means fewer chunks fit; it never lets the prompt exceed the context window. Each source also pays a fixed
overhead for its <source ...> wrapper. Pass ``count_tokens`` to use a real tokenizer instead.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from rag.tokens import estimate_token_count
from rag_core.schemas import Source

# DESIGN DEFAULT, NOT A TUNED VALUE. Five chunks is a conventional starting point. The evaluation so far (dev split, retrieval
# only) shows that Evidence-F1 falls as k grows while recall rises, so there is no k that is best for everything; the right
# number needs an end-to-end answer-quality evaluation, which has not been done. See PROJECT_CONTEXT.
DEFAULT_MAX_CHUNKS = 5
DEFAULT_MAX_TOKENS = 3000
PER_SOURCE_OVERHEAD_TOKENS = 40  # the <source id=... paper=... section=...> wrapper and its newlines, rounded up
MIN_TRUNCATED_TOKENS = 16  # a truncated first chunk keeps at least this many tokens of text
TRUNCATION_MARK = " [...]"


@dataclass
class ContextResult:
    sources: list[Source] = field(default_factory=list)  # labelled S1..Sn, in relevance order; text may be truncated
    used_chunk_ids: list[str] = field(default_factory=list)
    tokens: int = 0  # estimated tokens of the texts plus wrapper overhead
    truncated_first: bool = False  # the first chunk alone exceeded the budget and was cut
    dropped: int = 0  # retrieved chunks that did not fit (or exceeded max_chunks) and were left out


def truncate_to_tokens(text: str, budget: int, count_tokens: Callable[[str], int] = estimate_token_count) -> str:
    """The longest prefix of ``text`` (cut at a word boundary where possible) that fits ``budget`` tokens, plus a marker.
    Never empty for a non-empty ``text``."""
    budget = max(budget, 1)
    if count_tokens(text) <= budget:
        return text
    lo, hi = 1, len(text)
    while lo < hi:  # largest prefix length whose prefix + marker fits
        mid = (lo + hi + 1) // 2
        if count_tokens(text[:mid] + TRUNCATION_MARK) <= budget:
            lo = mid
        else:
            hi = mid - 1
    cut = text[:lo]
    boundary = max(cut.rfind(" "), cut.rfind("\n"))
    if boundary > len(cut) // 2:
        cut = cut[:boundary]
    return cut.rstrip() + TRUNCATION_MARK


def build_context(hits: Sequence, max_tokens: int = DEFAULT_MAX_TOKENS, *, max_chunks: int = DEFAULT_MAX_CHUNKS,
                  count_tokens: Callable[[str], int] = estimate_token_count) -> ContextResult:
    """Keep ``hits`` (objects with chunk_id, doc_id, section, text, rank, score) in the order given, label them S1..Sn and stop
    at ``max_chunks`` or when the next chunk would exceed ``max_tokens``. A first chunk that alone exceeds the budget is
    truncated rather than dropped, so a non-empty ``hits`` never yields an empty context."""
    out = ContextResult()
    seen: set[str] = set()
    candidates = [h for h in hits if h.text and h.chunk_id not in seen and not seen.add(h.chunk_id)]  # type: ignore[func-returns-value]  # add() returns None: keeps first occurrences
    for position, hit in enumerate(candidates):
        if len(out.sources) >= max_chunks:
            out.dropped = len(candidates) - position
            break
        text = hit.text
        cost = count_tokens(text) + PER_SOURCE_OVERHEAD_TOKENS
        if out.tokens + cost > max_tokens:
            if not out.sources:  # never an empty context: cut the first chunk to fit instead
                text = truncate_to_tokens(text, max(max_tokens - PER_SOURCE_OVERHEAD_TOKENS, MIN_TRUNCATED_TOKENS), count_tokens)
                cost = count_tokens(text) + PER_SOURCE_OVERHEAD_TOKENS
                out.truncated_first = True
            else:
                out.dropped = len(candidates) - position
                break
        out.sources.append(Source(chunk_id=hit.chunk_id, doc_id=hit.doc_id, section=getattr(hit, "section", "") or "", text=text,
                                  rank=int(hit.rank), score=float(hit.score), label=f"S{len(out.sources) + 1}"))
        out.used_chunk_ids.append(hit.chunk_id)
        out.tokens += cost
    return out
