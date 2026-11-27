# Test baseline (before any refactoring)

> **Historical.** This is the first baseline, taken before the import-time side effects were removed. The current baseline is `docs/TEST_BASELINE.md`.

- Command: `pytest tests -q -rA` with the venv at `%LOCALAPPDATA%\advrag\venv`, `HF_HUB_OFFLINE=1` (stops the model download). No fake credentials set, nothing fixed.
- Environment: `docs/env_baseline.txt`. The embedding model `all-MiniLM-L6-v2` is **not** in the local HuggingFace cache.
- Result: **16 passed, 37 failed** (53 tests).
- Caveat: all 37 failures are import-time failures. The test bodies never ran, so any wrong test or genuine bug behind them is **unknown** until the import side effects are removed. Re-baseline after the next step.

## Category counts

| Category | Count |
|---|---|
| missing dependency | 0 |
| needs live network or credentials at import | 37 |
| wrong test (assertion is incorrect) | 0 (unknown behind the 37) |
| genuine bug | 0 (unknown behind the 37) |
| unknown | 0 |
| passed | 16 |

## Import-time side effects (to remove next step)

| Module line | What it does |
|---|---|
| `rag/chunkers.py:69` `embedding_model = SentenceTransformer(cfg.embedding.model)` | Loads / downloads the HF model on import. Needs the HF cache or network. |
| `rag/embeddings.py:19` `embedding_model = SentenceTransformer(cfg.embedding.model)` | Same, reached via `rag/retrieval.py:30`. |
| `db/vectordb.py:25` `client = chromadb.CloudClient(...)` (after `load_dotenv()` at line 14) | Connects to Chroma Cloud with credentials from `.env` on import. |
| `db/vectordb.py:42` `collection = client.get_or_create_collection(...)` | Network call to Chroma Cloud on import (can create the collection). |
| `rag/retrieval.py:190` `all_data = collection.get(` | Reads the whole cloud collection on import. |
| `rag/retrieval.py:211` `bm25 = BM25Okapi(tokenized_corpus)` | Builds the BM25 index from that cloud data on import. |
| `rag/retrieval.py:104` `load_dotenv()`, `:131` `groq_client = OpenAI(`, `:156` `gemini_client = genai.Client(` | Read API keys from `.env` and build clients at import (no request made at construction). |
| `rag/context_builder.py:10` `tokenizer = tiktoken.get_encoding(` | Downloads the tiktoken vocabulary on first use (public file). |
| `cache/cache.py:28` `load_dotenv()`, `:55` `redis.ConnectionPool(` | Reads Redis config at import; pool is lazy. Not imported by the current tests. |

## Per-test results

Failure reasons: **A** = `rag/chunkers.py:69` cannot load the model (HF cache empty, offline). **B** = `rag/retrieval.py` import fails at `rag/embeddings.py:19` (same model), after `db/vectordb.py` has already run its Chroma Cloud lines on import.

| Test id | Result | Category | Reason |
|---|---|---|---|
| test_chunkers.py::TestRecursiveChunk::test_returns_list | FAIL | needs network/credentials at import | A |
| test_chunkers.py::TestRecursiveChunk::test_chunks_are_chunk_instances | FAIL | needs network/credentials at import | A |
| test_chunkers.py::TestRecursiveChunk::test_chunking_strategy_metadata | FAIL | needs network/credentials at import | A |
| test_chunkers.py::TestRecursiveChunk::test_source_preserved | FAIL | needs network/credentials at import | A |
| test_chunkers.py::TestRecursiveChunk::test_empty_content | FAIL | needs network/credentials at import | A |
| test_chunkers.py::TestLayoutChunk::test_one_chunk_per_section | FAIL | needs network/credentials at import | A |
| test_chunkers.py::TestLayoutChunk::test_section_title_in_metadata | FAIL | needs network/credentials at import | A |
| test_chunkers.py::TestLayoutChunk::test_chunking_strategy_is_layout | FAIL | needs network/credentials at import | A |
| test_chunkers.py::TestLayoutChunk::test_empty_sections | FAIL | needs network/credentials at import | A |
| test_chunkers.py::TestSemanticChunk::test_returns_list | FAIL | needs network/credentials at import | A |
| test_chunkers.py::TestSemanticChunk::test_chunks_are_chunk_instances | FAIL | needs network/credentials at import | A |
| test_chunkers.py::TestSemanticChunk::test_chunking_strategy_metadata | FAIL | needs network/credentials at import | A |
| test_chunkers.py::TestSemanticChunk::test_empty_content_returns_empty | FAIL | needs network/credentials at import | A |
| test_chunkers.py::TestSemanticChunk::test_min_chunk_size_respected | FAIL | needs network/credentials at import | A |
| test_eval_metrics.py::TestRetrievalMetrics::test_first_relevant_rank | PASS | | |
| test_eval_metrics.py::TestRetrievalMetrics::test_no_relevant_chunk | PASS | | |
| test_eval_metrics.py::TestRetrievalMetrics::test_doc_hit_rank | PASS | | |
| test_eval_metrics.py::TestAnswerMetrics::test_exact_match_normalises | PASS | | |
| test_eval_metrics.py::TestAnswerMetrics::test_f1_partial_overlap | PASS | | |
| test_eval_metrics.py::TestAnswerMetrics::test_f1_no_overlap | PASS | | |
| test_eval_metrics.py::TestAnswerMetrics::test_best_over_references | PASS | | |
| test_eval_metrics.py::TestAnswerMetrics::test_refusal_detection | PASS | | |
| test_eval_metrics.py::TestGoldenSet::test_schema_and_counts | PASS | | |
| test_fingerprint.py (7 tests) | PASS | | |
| test_fusion.py::TestWeightedRRF::test_returns_dict | FAIL | needs network/credentials at import | B |
| test_fusion.py::TestWeightedRRF::test_all_doc_ids_present | FAIL | needs network/credentials at import | B |
| test_fusion.py::TestWeightedRRF::test_higher_weight_boosts_rank | FAIL | needs network/credentials at import | B |
| test_fusion.py::TestWeightedRRF::test_scores_are_positive | FAIL | needs network/credentials at import | B |
| test_fusion.py::TestWeightedRRF::test_empty_lists | FAIL | needs network/credentials at import | B |
| test_fusion.py::TestWeightedRRF::test_single_list_populated | FAIL | needs network/credentials at import | B |
| test_fusion.py::TestWeightedRRF::test_different_k_values[10] | FAIL | needs network/credentials at import | B |
| test_fusion.py::TestWeightedRRF::test_different_k_values[60] | FAIL | needs network/credentials at import | B |
| test_fusion.py::TestWeightedRRF::test_different_k_values[100] | FAIL | needs network/credentials at import | B |
| test_retrieval.py::TestClassifyQuery::test_conversational_queries_route_direct[hi] | FAIL | needs network/credentials at import | B |
| ...[hello] | FAIL | needs network/credentials at import | B |
| ...[how are you?] | FAIL | needs network/credentials at import | B |
| ...[thanks] | FAIL | needs network/credentials at import | B |
| ...[thank you] | FAIL | needs network/credentials at import | B |
| ...[bye] | FAIL | needs network/credentials at import | B |
| ...[good morning] | FAIL | needs network/credentials at import | B |
| test_retrieval.py::TestClassifyQuery::test_research_queries_route_rag[What dataset do they use?] | FAIL | needs network/credentials at import | B |
| ...[How does the model handle out-of-domain data?] | FAIL | needs network/credentials at import | B |
| ...[Which language has six dialects in the paper?] | FAIL | needs network/credentials at import | B |
| ...[Describe the evaluation methodology used.] | FAIL | needs network/credentials at import | B |
| test_retrieval.py::TestRetrieveAndBuildContext::test_returns_required_keys | FAIL | needs network/credentials at import | B |
| test_retrieval.py::TestRetrieveAndBuildContext::test_latency_is_non_negative | FAIL | needs network/credentials at import | B |
| test_retrieval.py::TestRetrieveAndBuildContext::test_empty_collection_returns_context_string | FAIL | needs network/credentials at import | B |

(Paths are under `tests/unit/`. `...[x]` rows belong to the same test function as the row above.)
