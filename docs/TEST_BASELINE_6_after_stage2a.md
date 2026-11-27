# Test baseline (current)

History: [initial red](TEST_BASELINE_initial_red.md) (16 passed, 37 failed), [after import safety](TEST_BASELINE_2_after_import_safety.md)
(118 passed, 23 skipped), [after the Retriever](TEST_BASELINE_3_after_retriever.md) (276 passed, 3 skipped, 0 failed),
[after Evaluation v2](TEST_BASELINE_4_after_eval_v2.md) (343 passed, 1 failed, 1 xfail, 3 skipped),
[after the offset-native chunker](TEST_BASELINE_5_after_offset_native_chunker.md) (394 passed, 3 skipped, 0 failed).
This file replaces them after Stage 2a (pre-registration hash, 15-arm grid, reranker, stopword lexical query, `corpus_titled` mode,
decision-rule and analysis scripts).

- Full run: `pytest -p no:cacheprovider -q -rs` (no `--ignore`), venv at `%LOCALAPPDATA%\advrag\venv`.
- Quick run: `pytest --fast -p no:cacheprovider -q -rs`.
- Collected: **428** tests. Hypothesis profile: `ci` (derandomized) by default; see `CLAUDE.md` rule 8.

## Counts

| Category | Full run | `--fast` run |
|---|---|---|
| passed | **425** | 398 |
| failed (real) | **0** | 0 |
| skipped: needs Retriever refactor | 3 | 3 |
| skipped: `--fast` profile (slow tests) | 0 | 27 |
| time | 385.7 s (pytest) / 395 s (shell) | 64.6 s (pytest) / 71 s (shell) |

`--fast` is above the old ~30 s target (about 65 s): the suite grew from 397 to 428 tests this stage and most of the time is per-test
setup (Chroma and FTS fixtures, imports), not Hypothesis.

## What changed since the previous baseline

- New: `test_lexical_query.py` (12: stopword filter, fallback, punctuation, FTS5 syntax characters, wrapper), `test_rerankers.py` (4: lazy,
  local-only, missing cache raises, real model), `test_stage2a_analysis.py` (7: decision rule incl. the complexity table in the
  pre-registration, error-analysis helpers, corpus diagnostics).
- `test_eval_runner.py` 28 to 35: `corpus_titled` queries, the new arms' settings, the 15-arm registry equals the pre-registered list, rerank
  arm without a reranker is refused, variant retrievers share store and index, end-to-end 15 arms with a fake reranker, pre-registration hash
  recorded / appending deviations allowed / tampering refused. The default test arm set is now `core` (A-F) so hermetic tests never load the real reranker.
- `test_compare.py` 17 to 18: a metric undefined for a question (Evidence-F1 `None`) is left out for both arms.

## Still skipped: needs Retriever refactor (3)

`tests/unit/test_retrieval.py::TestRetrieveAndBuildContext` (`test_returns_required_keys`, `test_latency_is_non_negative`,
`test_empty_collection_returns_context_string`): wrong by construction and targeting the old `rag.retrieval` module (it builds LLM clients
and a BM25 index at import); replaced by `tests/unit/test_retriever.py`. See
[TEST_BASELINE_2_after_import_safety.md](TEST_BASELINE_2_after_import_safety.md) for the per-test reasons.

## Skipped by `--fast` only (27, run in the full suite)

Marker `slow`: the module-import probe (14 modules + 2 checks), `test_missing_model_cache_raises_clear_error_not_download`, the 6
`model`-marked tests (including the real reranker test), the 2 subprocess tests in `test_hypothesis_profiles.py`, and the reranker
missing-cache test. `--fast` also lowers Hypothesis from 200 to 8 examples per property.

## By file (full run)

| File | Tests | Result |
|---|---|---|
| `test_chunk_contract.py` | 13 | all pass |
| `test_chunker_properties.py` | 34 | all pass |
| `test_chunkers.py` | 14 | all pass |
| `test_compare.py` | 18 | all pass |
| `test_eval_metrics.py` (old metrics) | 9 | all pass |
| `test_eval_runner.py` | 35 | all pass |
| `test_evidence_f1.py` | 17 | all pass |
| `test_fingerprint.py` | 7 | all pass |
| `test_fusion.py` | 23 | all pass |
| `test_gold_spans_split.py` | 12 | all pass |
| `test_hypothesis_profiles.py` | 4 | all pass |
| `test_import_guard.py` | 22 | all pass |
| `test_ingestion.py` | 15 | all pass |
| `test_ingestion_lexical.py` | 13 | all pass |
| `test_lexical.py` | 47 | all pass |
| `test_lexical_query.py` | 12 | all pass |
| `test_relevance.py` | 17 | all pass |
| `test_rerankers.py` | 4 | all pass |
| `test_retrieval.py` | 21 | 18 pass, 3 skipped |
| `test_retriever.py` | 58 | all pass |
| `test_splitting.py` | 12 | all pass |
| `test_stage2a_analysis.py` | 7 | all pass |
| `test_store.py` | 14 | all pass |

## Open issues

- Vector search (Chroma HNSW) is not bit-reproducible across processes in general; in the Stage 2a repeats (paper mode, 15 arms, and
  corpus_titled) all rows were identical apart from timing.
- `rag/retrieval.py` is still not import-safe; the API is not wired to the new Retriever.
- `--fast` is about 65 s, not 30 s.
