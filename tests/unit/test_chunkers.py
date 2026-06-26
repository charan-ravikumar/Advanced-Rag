"""Unit tests for the chunking strategies."""

import pytest
from unittest.mock import MagicMock

from schemas import Chunk, Document, Section


def _make_document(content: str, source: str = "test_paper") -> MagicMock:
    """Build a minimal mock Document compatible with the chunkers."""
    doc = MagicMock()
    doc.content = content
    doc.metadata = {"source": source}
    doc.sections = []
    return doc


def _make_document_with_sections(sections: list[dict]) -> MagicMock:
    """Build a mock Document with pre-parsed sections."""
    doc = MagicMock()
    doc.content = " ".join(s["content"] for s in sections)
    doc.metadata = {"source": "test_paper"}
    mocked_sections = []
    for s in sections:
        sec = MagicMock()
        sec.content = s["content"]
        sec.section_title = s.get("title", "Untitled")
        sec.metadata = {}
        mocked_sections.append(sec)
    doc.sections = mocked_sections
    return doc


LONG_TEXT = "\n\n".join(
    [
        "This section introduces the methodology used in the experiment.",
        "The dataset consists of 500 annotated examples collected from multiple sources.",
        "We apply a transformer-based model fine-tuned on the training split.",
        "Results show a significant improvement over the baseline approach.",
        "The evaluation metric used is F1 score on the held-out test set.",
        "Future work will explore larger models and multilingual settings.",
    ]
    * 3  # repeat to ensure multiple chunks
)


class TestRecursiveChunk:
    def test_returns_list(self):
        from rag.chunkers import recursive_chunk
        doc = _make_document("Hello world. " * 200)
        result = recursive_chunk(doc)
        assert isinstance(result, list)

    def test_chunks_are_chunk_instances(self):
        from rag.chunkers import recursive_chunk
        doc = _make_document("Some text. " * 300)
        result = recursive_chunk(doc)
        assert all(isinstance(c, Chunk) for c in result)

    def test_chunking_strategy_metadata(self):
        from rag.chunkers import recursive_chunk
        doc = _make_document("Word " * 500)
        result = recursive_chunk(doc)
        for chunk in result:
            assert chunk.metadata["chunking_strategy"] == "recursive"

    def test_source_preserved(self):
        from rag.chunkers import recursive_chunk
        doc = _make_document("Word " * 200, source="my_paper")
        result = recursive_chunk(doc)
        for chunk in result:
            assert chunk.metadata["source"] == "my_paper"

    def test_empty_content(self):
        from rag.chunkers import recursive_chunk
        doc = _make_document("")
        result = recursive_chunk(doc)
        assert result == []


class TestLayoutChunk:
    def test_one_chunk_per_section(self):
        from rag.chunkers import layout_chunk
        sections = [
            {"title": "Introduction", "content": "Intro text here."},
            {"title": "Methods", "content": "Methods description."},
            {"title": "Results", "content": "Results summary."},
        ]
        doc = _make_document_with_sections(sections)
        result = layout_chunk(doc)
        assert len(result) == len(sections)

    def test_section_title_in_metadata(self):
        from rag.chunkers import layout_chunk
        sections = [{"title": "Abstract", "content": "Abstract text."}]
        doc = _make_document_with_sections(sections)
        result = layout_chunk(doc)
        assert result[0].metadata["section_title"] == "Abstract"

    def test_chunking_strategy_is_layout(self):
        from rag.chunkers import layout_chunk
        sections = [{"title": "Intro", "content": "Text."}]
        doc = _make_document_with_sections(sections)
        result = layout_chunk(doc)
        assert result[0].metadata["chunking_strategy"] == "layout"

    def test_empty_sections(self):
        from rag.chunkers import layout_chunk
        doc = _make_document_with_sections([])
        result = layout_chunk(doc)
        assert result == []


class TestSemanticChunk:
    def test_returns_list(self):
        from rag.chunkers import semantic_chunk
        doc = _make_document(LONG_TEXT)
        result = semantic_chunk(doc)
        assert isinstance(result, list)

    def test_chunks_are_chunk_instances(self):
        from rag.chunkers import semantic_chunk
        doc = _make_document(LONG_TEXT)
        result = semantic_chunk(doc)
        assert all(isinstance(c, Chunk) for c in result)

    def test_chunking_strategy_metadata(self):
        from rag.chunkers import semantic_chunk
        doc = _make_document(LONG_TEXT)
        result = semantic_chunk(doc)
        for chunk in result:
            assert chunk.metadata["chunking_strategy"] == "semantic"

    def test_empty_content_returns_empty(self):
        from rag.chunkers import semantic_chunk
        doc = _make_document("")
        result = semantic_chunk(doc)
        assert result == []

    def test_min_chunk_size_respected(self):
        from rag.chunkers import semantic_chunk
        doc = _make_document(LONG_TEXT)
        result = semantic_chunk(doc, min_chunk_chars=10)
        for chunk in result:
            assert len(chunk.content) >= 10 or len(result) == 1
