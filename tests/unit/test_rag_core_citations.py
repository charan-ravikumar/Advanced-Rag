"""rag_core.citations: the SHORT answer's abstention sentinel (the long answer is structured output: tests/unit/test_rag_core_claims.py)."""

import pytest

import rag_core.citations as citations
from rag_core.citations import could_become_abstention, is_abstention


def test_the_regex_citation_parser_is_gone():
    assert not hasattr(citations, "parse_citations") and not hasattr(citations, "strip_code")


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
