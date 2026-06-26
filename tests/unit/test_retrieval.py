"""Unit tests for query routing and retrieve_and_build_context (mocked)."""

import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# classify_query
# ---------------------------------------------------------------------------

class TestClassifyQuery:
    """Test the fast regex path of classify_query (no LLM calls)."""

    @pytest.mark.parametrize(
        "query",
        [
            "hi",
            "hello",
            "how are you?",
            "thanks",
            "thank you",
            "bye",
            "good morning",
        ],
    )
    def test_conversational_queries_route_direct(self, query):
        from rag.retrieval import classify_query
        result = classify_query(query)
        assert result == "direct", f"Expected 'direct' for query: {query!r}"

    @pytest.mark.parametrize(
        "query",
        [
            "What dataset do they use?",
            "How does the model handle out-of-domain data?",
            "Which language has six dialects in the paper?",
            "Describe the evaluation methodology used.",
        ],
    )
    def test_research_queries_route_rag(self, query):
        """Research queries should route to 'rag' without an LLM call
        (they don't match conversational patterns)."""
        from rag.retrieval import classify_query
        # Patch the LLM call to always return 'rag' so we test deterministically
        with patch("rag.retrieval.groq_client") as mock_client:
            mock_choice = MagicMock()
            mock_choice.choices[0].message.content = "rag"
            mock_client.chat.completions.create.return_value = mock_choice
            result = classify_query(query)
        # Either the regex already says 'rag', or the LLM (mocked) says 'rag'
        assert result == "rag", f"Expected 'rag' for query: {query!r}"


# ---------------------------------------------------------------------------
# retrieve_and_build_context
# ---------------------------------------------------------------------------

class TestRetrieveAndBuildContext:
    """Smoke-test retrieve_and_build_context with all external I/O mocked."""

    def _make_mock_collection(self, n_results=3):
        mock_col = MagicMock()
        mock_col.query.return_value = {
            "ids": [[f"id{i}" for i in range(n_results)]],
            "documents": [[f"doc content {i}" for i in range(n_results)]],
            "distances": [[0.1 * i for i in range(n_results)]],
            "metadatas": [
                [
                    {"source": f"paper_{i}", "chunking_strategy": "semantic"}
                    for i in range(n_results)
                ]
            ],
        }
        mock_col.count.return_value = n_results
        return mock_col

    def test_returns_required_keys(self):
        from rag.retrieval import retrieve_and_build_context

        mock_col = self._make_mock_collection()

        with (
            patch("rag.retrieval.collection", mock_col),
            patch("rag.retrieval.encode_query", return_value=[0.1] * 384),
        ):
            result = retrieve_and_build_context("What dataset do they use?")

        assert "context" in result
        assert "sources" in result
        assert "latency_ms" in result

    def test_latency_is_non_negative(self):
        from rag.retrieval import retrieve_and_build_context

        mock_col = self._make_mock_collection()

        with (
            patch("rag.retrieval.collection", mock_col),
            patch("rag.retrieval.encode_query", return_value=[0.1] * 384),
        ):
            result = retrieve_and_build_context("How does BM25 work?")

        assert result["latency_ms"] >= 0

    def test_empty_collection_returns_context_string(self):
        from rag.retrieval import retrieve_and_build_context

        mock_col = self._make_mock_collection(n_results=0)
        mock_col.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "distances": [[]],
            "metadatas": [[]],
        }

        with (
            patch("rag.retrieval.collection", mock_col),
            patch("rag.retrieval.encode_query", return_value=[0.1] * 384),
        ):
            result = retrieve_and_build_context("irrelevant query")

        assert isinstance(result["context"], str)
        assert isinstance(result["sources"], list)
