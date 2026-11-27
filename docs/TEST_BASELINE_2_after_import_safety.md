# Test baseline (current)

> **Historical.** Second baseline, taken after the import-time side effects were removed but before the Retriever existed (118 passed, 23 skipped). The current baseline is `docs/TEST_BASELINE.md`.

The first baseline (16 passed, 37 failed, every failure an import-time side effect) is kept as
[TEST_BASELINE_initial_red.md](TEST_BASELINE_initial_red.md). This file replaces it after the
import-time side effects were removed from the chunkers, embeddings, `db/vectordb.py` and
`rag/context_builder.py`, and a hermetic guard was added (`tests/conftest.py`).

- Full run: `pytest -p no:cacheprovider -q -rs` (no `--ignore`), venv at `%LOCALAPPDATA%\advrag\venv`.
- Quick run: `pytest --fast -p no:cacheprovider -q -rs`.
- Environment and model revision: `docs/env_baseline.txt`. The model `all-MiniLM-L6-v2` is cached at the pinned revision, so the `model` tests run.
- Collected: **141** tests.

## Counts

| Category | Full run | `--fast` run |
|---|---|---|
| passed | **118** | 100 |
| skipped: needs Retriever refactor | **23** | 23 |
| skipped: `--fast` profile (slow tests) | 0 | 18 |
| real failures | **0** | 0 |
| missing dependency / wrong test / genuine bug / unknown | 0 | 0 |
| wall time | 289 s (pytest) / 296 s (shell) | 35.7 s (pytest) / 43.5 s (shell) |

Compared with the first baseline (53 tests: 16 passed, 37 failed): the 37 import-time failures
are gone. Of those 37, **14** (`test_chunkers.py`) now pass; **23** (`test_fusion.py` 9,
`test_retrieval.py` 14) are skipped by an explicit module-level `pytest.mark.skip` because
`rag/retrieval.py` still has import-time side effects (TODO block at the top of that file).
The other 88 tests are new since the first baseline.

## Skipped: needs Retriever refactor (23)

Reason shown in the report: `needs Retriever refactor, see TODO at the top of rag/retrieval.py`.
Their bodies have never run against the current code, so whether they pass is **unknown**.

| File | Tests |
|---|---|
| `tests/unit/test_fusion.py` (9) | `TestWeightedRRF::test_returns_dict`, `test_all_doc_ids_present`, `test_higher_weight_boosts_rank`, `test_scores_are_positive`, `test_empty_lists`, `test_single_list_populated`, `test_different_k_values[10]`, `[60]`, `[100]` |
| `tests/unit/test_retrieval.py` (14) | `TestClassifyQuery::test_conversational_queries_route_direct[hi]`, `[hello]`, `[how are you?]`, `[thanks]`, `[thank you]`, `[bye]`, `[good morning]`; `TestClassifyQuery::test_research_queries_route_rag[What dataset do they use?]`, `[How does the model handle out-of-domain data?]`, `[Which language has six dialects in the paper?]`, `[Describe the evaluation methodology used.]`; `TestRetrieveAndBuildContext::test_returns_required_keys`, `test_latency_is_non_negative`, `test_empty_collection_returns_context_string` |

## Skipped by `--fast` only (18, run in the full suite)

Marker `slow`: they import torch or run the real model.
`test_import_guard.py` module-import probe (11) and `test_missing_model_cache_raises_clear_error_not_download` (1),
and the 6 `model`-marked tests in `test_chunker_properties.py` (`test_real_tokenizer_token_cap[x3]`,
`test_fallback_counter_is_safe_under_real_tokenizer[x3]`).
`--fast` also lowers hypothesis from 200 to 15 examples per property.

## Passed, by file (full run)

| File | Tests | Result |
|---|---|---|
| `test_chunk_contract.py` | 13 | all pass |
| `test_chunker_properties.py` | 30 | all pass (6 need the cached model) |
| `test_chunkers.py` | 14 | all pass |
| `test_eval_metrics.py` | 9 | all pass |
| `test_fingerprint.py` | 7 | all pass |
| `test_import_guard.py` | 17 | all pass |
| `test_ingestion.py` | 14 | all pass |
| `test_store.py` | 14 | all pass |
| `test_fusion.py`, `test_retrieval.py` | 23 | skipped (above) |

## Open issues

- `rag/retrieval.py` is not import-safe; fixing it (a `Retriever` class with an injected store, embedder and LLM clients) turns the 23 skipped tests into a real result.
- The retrieval tests were never run against current code, so nothing is known about whether they are correct.
