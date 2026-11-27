"""Unit tests for query routing (rag/router.py)."""

import pytest

from rag.router import classify_by_rules, route_query

# ---------------------------------------------------------------------------
# router (rules only, no LLM, no clients)
# ---------------------------------------------------------------------------

class TestClassifyQuery:
    """The rule-based router: pure, deterministic, no LLM."""

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
        assert classify_by_rules(query) == "direct", f"Expected 'direct' for query: {query!r}"
        assert route_query(query) == "direct"

    @pytest.mark.parametrize(
        "query",
        [
            "What dataset do they use?",
            "How does the model handle out-of-domain data?",
            "Which language has six dialects in the paper?",
            "Describe the evaluation methodology used.",
        ],
    )
    def test_research_queries_are_not_decided_by_rules_and_route_rag(self, query):
        """The old version of this test mocked the LLM to answer 'rag', so it passed without ever
        exercising the router. Now it asserts the rules abstain (None) and the LLM-free route is rag."""
        assert classify_by_rules(query) is None
        assert route_query(query) == "rag"

    @pytest.mark.parametrize("query", ["hey there", "see you later", "what's up", "Thanks!", "Good evening.", "Thank you very much", "ok"])
    def test_more_small_talk_phrases(self, query):
        assert route_query(query) == "direct"

    def test_short_query_with_question_mark_is_not_small_talk(self):
        assert classify_by_rules("BM25?") is None
        assert route_query("BM25?") == "rag"

    @pytest.mark.parametrize("query", ["attention mechanism", "BERT", "transformer architecture", "word2vec", "dropout rate",
                                       "sentiment analysis", "F1 score"])
    def test_short_keyword_queries_are_research_queries(self, query):
        """v1 treated any 1-3 word query without a question mark as small talk, so 'attention mechanism' never reached retrieval."""
        assert classify_by_rules(query) is None
        assert route_query(query) == "rag"

    @pytest.mark.parametrize("query", ["Hello, how does BERT work?", "hi what dataset do they use", "thanks, but what is the baseline?",
                                       "help me understand the loss function", "hello world benchmark", "no free lunch theorem",
                                       "yes/no questions in QA"])
    def test_a_message_that_merely_starts_with_a_greeting_is_still_a_research_query(self, query):
        assert route_query(query) == "rag"

    def test_the_allowlist_is_whole_message_not_prefix_and_ignores_case_punctuation_and_spacing(self):
        assert route_query("  Hello!!  ") == "direct" and route_query("THANK   YOU.") == "direct" and route_query("what\u2019s up?") == "direct"
        assert route_query("") == "rag" and route_query("???") == "rag"

    def test_whitespace_and_case_insensitive(self):
        assert route_query("   HELLO   ") == "direct"

    def test_router_module_has_no_side_effect_imports(self):
        import rag.router as router

        assert not hasattr(router, "groq_client") and not hasattr(router, "logger")
