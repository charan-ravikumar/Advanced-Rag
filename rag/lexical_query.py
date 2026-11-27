"""Stopword-filtered lexical query (arm B2).

The base lexical leg ORs every word of the question. Questions are mostly function words ("what is the ... of the ..."),
which match almost every chunk and add noise to BM25. ``StopwordFilteredLexical`` wraps any LexicalIndex, removes the
stopwords from the query text and delegates to the wrapped index, which still escapes the remaining words into a safe
FTS5 expression (``rag.lexical.escape_query``). If every word is a stopword the unfiltered query is used, so a
question is never answered with nothing only because of the filter.

Import-safe: no I/O at import. The stopword list is fixed here (a standard English list) and is part of the code fingerprint.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rag.lexical import escape_query

_WORD = re.compile(r"[^\W_]+", re.UNICODE)

STOPWORDS: frozenset[str] = frozenset("""
i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its itself
they them their theirs themselves what which who whom this that these those am is are was were be been being have has had having
do does did doing a an the and but if or because as until while of at by for with about against between into through during
before after above below to from up down in out on off over under again further then once here there when where why how all any
both each few more most other some such no nor not only own same so than too very s t can will just don should now
""".split())


def filter_stopwords(query: str) -> str:
    """The words of ``query`` (lower-cased, in order) without stopwords, joined by spaces; may be empty."""
    return " ".join(w for w in (m.lower() for m in _WORD.findall(query)) if w not in STOPWORDS)


def stopword_match_query(query: str) -> str | None:
    """The FTS5 MATCH expression for the filtered query, falling back to the unfiltered one when nothing is left.
    ``None`` only when the query has no word at all."""
    return escape_query(filter_stopwords(query)) or escape_query(query)


class StopwordFilteredLexical:
    """LexicalIndex that drops stopwords from the query before searching ``inner``."""

    def __init__(self, inner):
        self.inner = inner

    def search(self, query: str, k: int, doc_ids: Sequence[str] | None = None):
        return self.inner.search(filter_stopwords(query) or query, k, doc_ids=doc_ids)

    def __getattr__(self, name):  # upsert, delete_by_doc, count, ... are delegated untouched
        return getattr(self.inner, name)
