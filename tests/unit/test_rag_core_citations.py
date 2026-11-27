"""rag_core.citations: marker parsing, validation against the provided labels, abstention sentinel."""

import pytest

from rag_core.citations import could_become_abstention, is_abstention, parse_citations, strip_code

FIVE = [f"S{i}" for i in range(1, 6)]


def cited(text, valid=FIVE):
    p = parse_citations(text, valid)
    return p.cited, p.invalid


def test_simple_markers_in_order_of_first_appearance_without_duplicates():
    assert cited("Attention helps [S2]. It scales [S1] and [S2] again [S3].") == (["S2", "S1", "S3"], [])


def test_adjacent_markers_are_two_citations():
    assert cited("It works [S1][S2].") == (["S1", "S2"], [])
    assert cited("It works [S1] [S2][S3]") == (["S1", "S2", "S3"], [])


def test_comma_separated_ids_in_one_bracket():
    assert cited("It works [S1, S3].") == (["S1", "S3"], [])
    assert cited("It works [S1,S2 , S4]") == (["S1", "S2", "S4"], [])


def test_a_marker_for_a_source_that_was_not_provided_is_invalid():
    assert cited("Claim [S9].") == ([], ["S9"])
    assert cited("Claim [S1][S9][S0].") == (["S1"], ["S9", "S0"])
    assert cited("Claim [S1, S7]") == (["S1"], ["S7"])


def test_s9_with_only_five_sources_is_invalid_but_s5_is_valid():
    assert cited("a [S5] b [S9]") == (["S5"], ["S9"])
    assert cited("a [S2]", ["S1", "S2"]) == (["S2"], [])
    assert cited("a [S3]", ["S1", "S2"]) == ([], ["S3"])


@pytest.mark.parametrize("text", ["No markers here.", "[S]", "[s1]", "[S 1]", "S1", "(S1)", "[S1", "S1]", "[S-1]", "[SS1]", "[1]", "[S1a]", "[]", ""])
def test_malformed_markers_are_not_markers(text):
    assert cited(text) == ([], [])


def test_leading_zero_marker_is_reported_as_invalid_not_silently_mapped():
    assert cited("x [S01]") == ([], ["S01"])


def test_a_half_open_marker_does_not_swallow_a_later_good_one():
    assert cited("see [S1 and then [S2] works") == (["S2"], [])


def test_markers_inside_inline_or_fenced_code_are_not_citations():
    assert cited("Use `arr[S1]` to index.") == ([], [])
    assert cited("```python\nx = table[S2]\n```\nThe result [S3].") == (["S3"], [])
    assert cited("Open fence never closes ```x[S1] and [S2]") == ([], [])
    assert cited("`a[S1]` then real [S2]") == (["S2"], [])


def test_a_marker_glued_to_a_word_still_counts():
    assert cited("uses attention[S1].") == (["S1"], [])


def test_marker_count_includes_repeats():
    assert parse_citations("[S1] [S1] [S2, S2]", FIVE).markers == 4


@pytest.mark.parametrize("text,expected", [("LDA is used\u3010S1\u3011.", (["S1"], [])), ("a\u3010S1\u3011\u3010S2\u3011", (["S1", "S2"], [])), ("a\u3010S1, S2\u3011", (["S1", "S2"], [])),
                                           ("a\uff3bS3\uff3d", (["S3"], [])), ("x [S1] y \u3010S2\u3011", (["S1", "S2"], [])), ("a\u3010S9\u3011", ([], ["S9"])),
                                           ("a \u3010 S1 \u3011", (["S1"], []))])
def test_lenticular_and_full_width_brackets_are_markers_too(text, expected):
    """Observed live: at low reasoning effort the generator cites as \u3010S1\u3011. The ASCII-only parser read that as 'no citations'."""
    assert cited(text) == expected


@pytest.mark.parametrize("text", ["\u3010S\u3011", "\u3010s1\u3011", "\u3010S 1\u3011", "\u3010S1", "S1\u3011", "\u3010S-1\u3011", "\u3010\u3011", "\u3014S1\u3015"])
def test_malformed_lenticular_markers_are_not_markers(text):
    assert cited(text) == ([], [])


def test_lenticular_markers_inside_code_are_still_ignored():
    assert cited("Use `arr\u3010S1\u3011` here, then real \u3010S2\u3011.") == (["S2"], [])


def test_the_service_reads_a_lenticular_citation_from_the_generator_as_a_valid_citation():
    from tests.unit.rag_core_fakes import ok
    from tests.unit.test_rag_core_service import done_of, events, make

    svc, *_ = make([ok("Attention weighs the tokens\u3010S1\u3011\u3010S9\u3011.")])
    r = done_of(events(svc))
    assert r.citations == ["S1"] and r.invalid_citations == ["S9"]


def test_strip_code_removes_fences_and_inline_code():
    assert "S1" not in strip_code("a `b[S1]` c ```d[S1]``` e")


@pytest.mark.parametrize("text", ["INSUFFICIENT_CONTEXT", "  INSUFFICIENT_CONTEXT\n", "INSUFFICIENT_CONTEXT.", '"INSUFFICIENT_CONTEXT"', "`INSUFFICIENT_CONTEXT`",
                                  "**INSUFFICIENT_CONTEXT**", "INSUFFICIENT_CONTEXT The sources discuss training but not this."])
def test_the_sentinel_means_abstained(text):
    assert is_abstention(text)


@pytest.mark.parametrize("text", ["The answer is INSUFFICIENT_CONTEXT because", "", "insufficient_context", "INSUFFICIENT", "Yes.", "Based on [S1], X."])
def test_everything_else_is_not_an_abstention(text):
    assert not is_abstention(text)


@pytest.mark.parametrize("prefix,expected", [("", True), (" ", True), ("I", True), ("INS", True), ("INSUFFICIENT_CONTEX", True), ('"INSUF', True),
                                             ("INSUFFICIENT_CONTEXT", False), ("The", False), ("Yes", False), ("IN a", False), ("INSUFFICIENT_CONTEXT.", False)])
def test_hold_back_applies_only_while_the_text_could_still_become_the_sentinel(prefix, expected):
    assert could_become_abstention(prefix) is expected
