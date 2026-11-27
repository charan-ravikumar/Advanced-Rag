# Test baseline (current)

> **Historical.** Third baseline, taken after the Retriever existed but before Evaluation v2 (276 passed, 3 skipped, 0 failed). The current baseline is `docs/TEST_BASELINE.md`.

History: [TEST_BASELINE_initial_red.md](TEST_BASELINE_initial_red.md) (16 passed, 37 failed, all import-time failures) and
[TEST_BASELINE_2_after_import_safety.md](TEST_BASELINE_2_after_import_safety.md) (118 passed, 23 skipped). This file replaces
them after the pure fusion and router modules, the SQLite FTS5 lexical index and the `Retriever` were added.

- Full run: `pytest -p no:cacheprovider -q -rs` (no `--ignore`), venv at `%LOCALAPPDATA%\advrag\venv`.
- Quick run: `pytest --fast -p no:cacheprovider -q -rs`.
- Environment and model revision: `docs/env_baseline.txt`.
- Collected: **279** tests.

## Counts

| Category | Full run | `--fast` run |
|---|---|---|
| passed | **276** | 253 |
| skipped: needs Retriever refactor | **3** | 3 |
| skipped: `--fast` profile (slow tests) | 0 | 23 |
| real failures | **0** | 0 |
| missing dependency / wrong test / genuine bug / unknown | 0 | 0 |
| time | 267 s (pytest) / 274 s (shell) | 37 to 38 s (pytest) / 42 to 43 s (shell), two runs |

Compared with the previous baseline (141 collected: 118 passed, 23 skipped): the 23 previously skipped tests were resolved as
follows. **9** (`test_fusion.py`) now run against `rag/fusion.py` and pass. **11** (`TestClassifyQuery`) now run against the pure
`rag/router.py` and pass. **3** (`TestRetrieveAndBuildContext`) remain skipped and are wrong by construction (below).
138 tests are new: fusion (14 added), router (7 added), lexical index (46), dual-store ingestion (13), Retriever (53),
and 5 more import-safety probes for the new modules.

## Still skipped: needs Retriever refactor (3)

`tests/unit/test_retrieval.py::TestRetrieveAndBuildContext`, skipped at class level with the reason
*needs Retriever refactor: these tests target `rag.retrieval.retrieve_and_build_context`, which still imports LLM clients and builds a
BM25 index at import (see TODO in `rag/retrieval.py`); they are also wrong by construction; the new `rag.retriever` tests replace them.*

| Test | Why it is wrong by construction |
|---|---|
| `test_returns_required_keys` | Patches `rag.retrieval.collection` only; the module-level `bm25` / `all_documents` built at import from the real collection are not patched, so the hybrid path mixes mocked and real data. Checks only that three keys exist. |
| `test_latency_is_non_negative` | `latency_ms` is a difference of two `time.time()` readings, so `>= 0` holds for any implementation (even if latency were never measured). It cannot fail. |
| `test_empty_collection_returns_context_string` | Empties only the mocked vector leg; the real import-time BM25 index still returns hits, so it never tests an empty collection. The mock ignores `where`, so it cannot test filtering either. |

Also corrected: `TestClassifyQuery::test_research_queries_route_rag` mocked the LLM to answer "rag", so it passed without exercising the
router. It now asserts the rules abstain (`classify_by_rules` is `None`) and the LLM-free route is `rag`.

## Skipped by `--fast` only (23, run in the full suite)

Marker `slow`: they import torch or run the real model. The module-import probe (14 modules + 2 state/cloud checks),
`test_missing_model_cache_raises_clear_error_not_download`, and the 6 `model`-marked tests in `test_chunker_properties.py`.
`--fast` also lowers hypothesis from 200 to 8 examples per property.

## Passed, by file (full run)

| File | Tests | Result |
|---|---|---|
| `test_chunk_contract.py` | 13 | all pass |
| `test_chunker_properties.py` | 30 | all pass (6 need the cached model) |
| `test_chunkers.py` | 14 | all pass |
| `test_eval_metrics.py` | 9 | all pass |
| `test_fingerprint.py` | 7 | all pass |
| `test_fusion.py` | 23 | all pass |
| `test_import_guard.py` | 22 | all pass |
| `test_ingestion.py` | 14 | all pass |
| `test_ingestion_lexical.py` | 13 | all pass |
| `test_lexical.py` | 46 | all pass |
| `test_retrieval.py` | 21 | 18 pass, 3 skipped (above) |
| `test_retriever.py` | 53 | all pass |
| `test_store.py` | 14 | all pass |

## Open issues

- `rag/retrieval.py` is still not import-safe (TODO block at the top); the API and `scripts/run_evaluation.py` still use their own copies of the retrieval logic.
- v1 router rule kept unchanged: a query of three words or fewer without `?` is treated as small talk (pinned by `test_short_keyword_query_without_question_mark_is_direct_by_the_old_rule`).
