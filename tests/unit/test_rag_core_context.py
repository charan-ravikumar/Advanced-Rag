"""rag_core.context: relevance order, budget, S-labels, never-empty context."""

from types import SimpleNamespace

import pytest

from rag.tokens import estimate_token_count
from rag_core.context import (DEFAULT_MAX_CHUNKS, DEFAULT_MAX_TOKENS, MIN_TRUNCATED_TOKENS, PER_SOURCE_OVERHEAD_TOKENS, TRUNCATION_MARK,
                              build_context, truncate_to_tokens)
from tests.unit.rag_core_fakes import make_hit


def words(n: int, tag: str = "w") -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


def test_default_is_five_chunks_and_documented_as_a_design_default():
    assert DEFAULT_MAX_CHUNKS == 5
    source = open("rag_core/context.py", encoding="utf-8").read()
    assert "DESIGN DEFAULT, NOT A TUNED VALUE" in source


def test_relevance_order_is_kept_and_sources_are_labelled_s1_to_sn():
    hits = [make_hit(i) for i in range(1, 4)]
    ctx = build_context(hits, 10_000)
    assert [s.label for s in ctx.sources] == ["S1", "S2", "S3"]
    assert ctx.used_chunk_ids == ["c1", "c2", "c3"] == [s.chunk_id for s in ctx.sources]
    assert [s.rank for s in ctx.sources] == [1, 2, 3] and ctx.sources[0].score == 1.0 and ctx.sources[0].section == "Methods"
    assert ctx.sources[0].text == hits[0].text and not ctx.truncated_first and ctx.dropped == 0


def test_at_most_max_chunks_are_used_and_the_rest_counted_as_dropped():
    ctx = build_context([make_hit(i) for i in range(1, 9)], 100_000)
    assert len(ctx.sources) == DEFAULT_MAX_CHUNKS and ctx.dropped == 3
    assert len(build_context([make_hit(i) for i in range(1, 9)], 100_000, max_chunks=2).sources) == 2


def test_the_budget_stops_at_the_first_chunk_that_does_not_fit_and_never_skips_ahead():
    big, small = words(300), words(3)
    hits = [make_hit(1, text=words(20)), make_hit(2, text=big), make_hit(3, text=small)]
    budget = estimate_token_count(hits[0].text) + PER_SOURCE_OVERHEAD_TOKENS + 50
    ctx = build_context(hits, budget)
    assert ctx.used_chunk_ids == ["c1"] and ctx.dropped == 2  # c3 is small enough but ranks below c2: relevance order is not broken
    assert ctx.tokens <= budget


def test_the_estimated_total_never_exceeds_the_budget():
    hits = [make_hit(i, text=words(40, f"t{i}_")) for i in range(1, 9)]
    for budget in (150, 400, 900, 3000):
        ctx = build_context(hits, budget)
        assert ctx.tokens <= budget or ctx.truncated_first


def test_an_oversize_first_chunk_is_truncated_not_dropped_so_the_context_is_never_empty():
    hits = [make_hit(1, text=words(2000)), make_hit(2)]
    ctx = build_context(hits, 300)
    assert len(ctx.sources) == 1 and ctx.truncated_first and ctx.dropped == 1
    text = ctx.sources[0].text
    assert text.endswith(TRUNCATION_MARK) and 0 < len(text) < len(hits[0].text) and hits[0].text.startswith(text[: -len(TRUNCATION_MARK)])
    assert estimate_token_count(text) + PER_SOURCE_OVERHEAD_TOKENS <= 300


def test_a_budget_smaller_than_the_wrapper_overhead_still_yields_a_non_empty_truncated_source():
    ctx = build_context([make_hit(1, text=words(500))], 5)
    assert len(ctx.sources) == 1 and ctx.truncated_first and len(ctx.sources[0].text) > len(TRUNCATION_MARK)
    assert estimate_token_count(ctx.sources[0].text) >= 1


def test_a_later_oversize_chunk_is_dropped_not_truncated():
    ctx = build_context([make_hit(1, text=words(10)), make_hit(2, text=words(2000))], 400)
    assert ctx.used_chunk_ids == ["c1"] and not ctx.truncated_first and ctx.dropped == 1


def test_empty_input_gives_an_empty_context_and_empty_or_duplicate_chunks_are_skipped():
    assert build_context([], 1000).sources == []
    ctx = build_context([make_hit(1), make_hit(1), make_hit(2, text=""), make_hit(3)], 10_000)
    assert ctx.used_chunk_ids == ["c1", "c3"] and [s.label for s in ctx.sources] == ["S1", "S2"]


def test_a_custom_token_counter_is_used_instead_of_the_estimate():
    hits = [make_hit(i, text="word " * 10) for i in range(1, 5)]
    one_token_each = build_context(hits, 4 * (1 + PER_SOURCE_OVERHEAD_TOKENS), count_tokens=lambda t: 1)
    assert len(one_token_each.sources) == 4
    heavy = build_context(hits, 4 * (1 + PER_SOURCE_OVERHEAD_TOKENS), count_tokens=lambda t: 1000)
    assert len(heavy.sources) == 1 and heavy.truncated_first  # even one chunk exceeds that counter's budget: truncated, not empty


def test_truncate_to_tokens_cuts_at_a_word_boundary_and_is_never_empty():
    text = words(500)
    cut = truncate_to_tokens(text, 40)
    assert cut.endswith(TRUNCATION_MARK) and estimate_token_count(cut) <= 40
    body = cut[: -len(TRUNCATION_MARK)]
    assert body and text.startswith(body) and text[len(body)] == " "  # ended on a word boundary
    assert truncate_to_tokens("short text", 100) == "short text"
    assert truncate_to_tokens("x" * 5000, MIN_TRUNCATED_TOKENS).endswith(TRUNCATION_MARK)  # no whitespace to cut at: still non-empty


def test_hit_objects_only_need_the_documented_attributes():
    hit = SimpleNamespace(chunk_id="a", doc_id="d", text="hello", rank=3, score=0.5)  # no section attribute
    ctx = build_context([hit], DEFAULT_MAX_TOKENS)
    assert ctx.sources[0].section == "" and ctx.sources[0].rank == 3
