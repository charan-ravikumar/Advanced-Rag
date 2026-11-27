"""Stopword-filtered lexical query (arm B2): the query builder and the wrapper. Real SQLite FTS5 in a tmp dir; no network."""

import pytest

from rag.lexical import SqliteFtsIndex
from rag.lexical_query import (
    STOPWORDS,
    StopwordFilteredLexical,
    filter_stopwords,
    stopword_match_query,
)
from rag.schemas_ingest import Chunk


def chunk(doc_id: str, idx: int, text: str) -> Chunk:
    return Chunk.create(doc_id=doc_id, section="S", chunk_idx=idx, char_start=0, char_end=len(text),
                        strategy="recursive", text=text, doc_version="v1")


def test_stopwords_are_removed_and_order_and_case_are_kept():
    assert filter_stopwords("What is the Learning Rate of the Optimizer?") == "learning rate optimizer"


def test_a_question_made_only_of_stopwords_falls_back_to_the_unfiltered_query():
    assert filter_stopwords("What is it?") == ""
    assert stopword_match_query("What is it?") == '"what" OR "is" OR "it"'


def test_empty_and_punctuation_only_queries_have_no_expression():
    for q in ("", "   ", "?!.,;:", "---", "()"):
        assert filter_stopwords(q) == ""
        assert stopword_match_query(q) is None


def test_punctuation_is_stripped_and_hyphenated_words_are_split():
    assert filter_stopwords("state-of-the-art, (BERT)!") == "state art bert"


@pytest.mark.parametrize("q", ['"unbalanced', "a OR b AND NOT c", "col:value NEAR(x y)", "x* ^y {z}", "it's \"quoted\" -dash", "naïve café"])
def test_fts5_syntax_characters_never_reach_the_expression_as_operators(q):
    expr = stopword_match_query(q)
    assert expr is not None
    for part in expr.split(" OR "):
        assert part.startswith('"') and part.endswith('"') and part.count('"') == 2  # every term is one quoted literal


def test_the_boolean_words_are_stopwords_only_when_they_are_in_the_list():
    assert {"and", "or", "not"} <= STOPWORDS and "near" not in STOPWORDS
    assert filter_stopwords("alpha AND beta OR gamma NOT delta NEAR epsilon") == "alpha beta gamma delta near epsilon"


def test_the_wrapper_searches_with_the_filtered_query_and_keeps_the_filter_and_k(tmp_path):
    ix = SqliteFtsIndex(tmp_path / "lex.sqlite3")
    try:
        ix.upsert([
            chunk("d1", 0, "the of the of what what is is the of"),  # only function words
            chunk("d1", 1, "attention improves translation quality"),
            chunk("d2", 0, "attention in another document"),
        ])
        plain = [cid for cid, _ in ix.search("what is the attention", 5)]
        wrapped = StopwordFilteredLexical(ix)
        filtered = wrapped.search("what is the attention", 5)
        assert len(plain) == 3 and len(filtered) == 2  # the function-word chunk no longer matches
        assert all(cid in plain for cid, _ in filtered)
        only_d2 = wrapped.search("what is the attention", 5, doc_ids=["d2"])
        assert len(only_d2) == 1 and only_d2[0][0] in ix.doc_chunk_ids("d2")
        assert len(wrapped.search("what is the attention", 1)) == 1
        assert wrapped.search("what is it", 5)  # fallback: all stopwords -> unfiltered query still answers
        assert wrapped.search("", 5) == [] and wrapped.count() == 3  # delegation
    finally:
        ix.close()
