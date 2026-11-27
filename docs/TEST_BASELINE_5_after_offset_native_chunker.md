# Test baseline (current)

History: [initial red](TEST_BASELINE_initial_red.md) (16 passed, 37 failed), [after import safety](TEST_BASELINE_2_after_import_safety.md)
(118 passed, 23 skipped), [after the Retriever](TEST_BASELINE_3_after_retriever.md) (276 passed, 3 skipped, 0 failed),
[after Evaluation v2](TEST_BASELINE_4_after_eval_v2.md) (343 passed, **1 failed**, 1 xfail, 3 skipped).
This file replaces them after the offset-native chunker fix, Evidence-F1 and the paper-primary labelling.

- Full run: `pytest -p no:cacheprovider -q -rs` (no `--ignore`), venv at `%LOCALAPPDATA%\advrag\venv`.
- Quick run: `pytest --fast -p no:cacheprovider -q -rs`.
- Collected: **397** tests. Hypothesis profile: `ci` (derandomized) by default; see `CLAUDE.md` rule 8.

## Counts

| Category | Full run | `--fast` run |
|---|---|---|
| passed | **394** | 369 |
| failed (real) | **0** | 0 |
| expected failure (xfail) | 0 | 0 |
| skipped: needs Retriever refactor | 3 | 3 |
| skipped: `--fast` profile (slow tests) | 0 | 25 |
| time | 328.5 s (pytest) / 336 s (shell) | 54.8 s (pytest) / 60.7 s (shell) |

`--fast` is above the old ~30 s target (it is about 55 s): the suite grew from 348 to 397 tests and most of the time is per-test setup
(Chroma and FTS fixtures, imports), not Hypothesis. Two subprocess tests (about 13 s) were moved to `slow` in this step.

## What changed since the previous baseline

- The one real failure (`test_coverage_no_content_loss[recursive]`) is fixed: `recursive_chunk` now computes offsets natively
  (`rag/splitting.py`) instead of searching with `str.find`. The strict xfail became the normal test
  `test_regression_recursive_chunker_covers_periodic_text`, and the falsifying input is a permanent `@example` on the coverage test.
- New: `test_splitting.py` (12), `test_hypothesis_profiles.py` (4), `test_evidence_f1.py` (17); `test_compare.py` grew 10 to 17
  (clustered bootstrap); `test_eval_runner.py` 24 to 28 (Evidence-F1, setting labels, default mode); `test_chunker_properties.py` 31 to 34;
  `test_ingestion.py` +1 (signature covers every chunker dependency); `test_lexical.py` +1 (naming for non-default store dirs).

## Still skipped: needs Retriever refactor (3)

`tests/unit/test_retrieval.py::TestRetrieveAndBuildContext` (`test_returns_required_keys`, `test_latency_is_non_negative`,
`test_empty_collection_returns_context_string`): wrong by construction and targeting the old `rag.retrieval` module (it builds LLM clients
and a BM25 index at import); replaced by `tests/unit/test_retriever.py`. See
[TEST_BASELINE_2_after_import_safety.md](TEST_BASELINE_2_after_import_safety.md) for the per-test reasons.

## Skipped by `--fast` only (25, run in the full suite)

Marker `slow`: the module-import probe (14 modules + 2 checks), `test_missing_model_cache_raises_clear_error_not_download`, the 6
`model`-marked tests, and the 2 subprocess tests in `test_hypothesis_profiles.py`. `--fast` also lowers Hypothesis from 200 to 8 examples
per property.

## By file (full run)

| File | Tests | Result |
|---|---|---|
| `test_chunk_contract.py` | 13 | all pass |
| `test_chunker_properties.py` | 34 | all pass |
| `test_chunkers.py` | 14 | all pass |
| `test_compare.py` | 17 | all pass |
| `test_eval_metrics.py` (old metrics) | 9 | all pass |
| `test_eval_runner.py` | 28 | all pass |
| `test_evidence_f1.py` | 17 | all pass |
| `test_fingerprint.py` | 7 | all pass |
| `test_fusion.py` | 23 | all pass |
| `test_gold_spans_split.py` | 12 | all pass |
| `test_hypothesis_profiles.py` | 4 | all pass |
| `test_import_guard.py` | 22 | all pass |
| `test_ingestion.py` | 15 | all pass |
| `test_ingestion_lexical.py` | 13 | all pass |
| `test_lexical.py` | 47 | all pass |
| `test_relevance.py` | 17 | all pass |
| `test_retrieval.py` | 21 | 18 pass, 3 skipped |
| `test_retriever.py` | 58 | all pass |
| `test_splitting.py` | 12 | all pass |
| `test_store.py` | 14 | all pass |

## Open issues

- Vector search (Chroma HNSW) is not bit-reproducible across processes: retrieved lists can differ at deep ranks between runs; in the latest dev runs one corpus-mode row (of 1,044) also changed `EvidenceF1@10`. Paper mode was identical (see PROJECT_CONTEXT).
- `rag/retrieval.py` is still not import-safe; the API is not wired to the new Retriever.
- `--fast` is about 55 s, not 30 s.
