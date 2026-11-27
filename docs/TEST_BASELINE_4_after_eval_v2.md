# Test baseline (current)

History: [initial red](TEST_BASELINE_initial_red.md) (16 passed, 37 failed), [after import safety](TEST_BASELINE_2_after_import_safety.md)
(118 passed, 23 skipped), [after the Retriever](TEST_BASELINE_3_after_retriever.md) (276 passed, 3 skipped, 0 failed).
This file replaces them after Evaluation v2 (span metrics, paired bootstrap, runner with test-split guard, gold spans, split).

- Full run: `pytest -p no:cacheprovider -q -rs` (no `--ignore`), venv at `%LOCALAPPDATA%\advrag\venv`.
- Quick run: `pytest --fast -p no:cacheprovider -q -rs`.
- Collected: **348** tests.

## Counts

| Category | Full run | `--fast` run |
|---|---|---|
| passed | **343** | 321 |
| **failed (real, known bug, see below)** | **1** | 0 |
| expected failure (strict xfail, known bug) | 1 | 1 |
| skipped: needs Retriever refactor | 3 | 3 |
| skipped: `--fast` profile (slow tests) | 0 | 23 |
| time | 303 s (pytest) / 312 s (shell) | 50 s (pytest) / 56 s (shell), two runs |

`--fast` is now above the old ~30 s target: the suite grew by 69 tests this step. The slowest groups are the
property tests and the Chroma-backed ingestion, lexical and retriever fixtures.

## The one failure: a real, known bug (needs the owner's decision)

`tests/unit/test_chunker_properties.py::test_coverage_no_content_loss[recursive]` **fails in the full run**, with the falsifying input
`sections=[[_sentences('a', 114), _sentences('a', 2)]]` (a 114-sentence paragraph then a 2-sentence one, repeating vocabulary).
It passes when the file runs alone: importing `tests/unit/test_gold_spans_split.py` is enough to change the outcome (most likely because
Hypothesis seeds generation with literals from imported modules; not proven; an empty example database does not change it). It is
deterministic given the set of imported modules, not random.

Cause (from reading the code and one reproduction): `recursive_chunk` locates each splitter chunk with `str.find` starting from the previous
end minus the overlap; in periodic or repeated text that can land one period early, so a merged tail is sliced from the wrong place and the
last paragraph is lost. Fixing it changes `rag/chunkers.py`, which is in the ingest signature, so both ingested collections would have to be
rebuilt. Not fixed. Pinned by `test_known_bug_recursive_chunker_mislocates_periodic_text` (strict xfail). Real data is not affected:
`runs/measurements/store_audit_20261121T105815Z/results.json` (0 of 7,255 recursive and 0 of 6,442 semantic stored chunks differ from the
section text at their offsets; 0 uncovered source characters).

I left the property test unchanged and red rather than marking it expected-to-fail, because that would also hide any different
recursive coverage failure.

## Still skipped: needs Retriever refactor (3)

`tests/unit/test_retrieval.py::TestRetrieveAndBuildContext` (`test_returns_required_keys`, `test_latency_is_non_negative`,
`test_empty_collection_returns_context_string`): wrong by construction and targeting the old `rag.retrieval` module; replaced by
`tests/unit/test_retriever.py`. See [TEST_BASELINE_2_after_import_safety.md](TEST_BASELINE_2_after_import_safety.md) for the per-test reasons.

## Skipped by `--fast` only (23, run in the full suite)

Marker `slow`: the module-import probe (14 modules + 2 checks), `test_missing_model_cache_raises_clear_error_not_download`, and the 6
`model`-marked tests. `--fast` also lowers hypothesis from 200 to 8 examples per property.

## By file (full run)

| File | Tests | Result |
|---|---|---|
| `test_chunk_contract.py` | 13 | all pass |
| `test_chunker_properties.py` | 31 | 29 pass, 1 fails (known bug above), 1 xfail (known bug) |
| `test_chunkers.py` | 14 | all pass |
| `test_compare.py` | 10 | all pass |
| `test_eval_metrics.py` (old metrics) | 9 | all pass |
| `test_eval_runner.py` | 24 | all pass |
| `test_fingerprint.py` | 7 | all pass |
| `test_fusion.py` | 23 | all pass |
| `test_gold_spans_split.py` | 12 | all pass |
| `test_import_guard.py` | 22 | all pass |
| `test_ingestion.py` | 14 | all pass |
| `test_ingestion_lexical.py` | 13 | all pass |
| `test_lexical.py` | 46 | all pass |
| `test_relevance.py` | 17 | all pass |
| `test_retrieval.py` | 21 | 18 pass, 3 skipped |
| `test_retriever.py` | 58 | all pass |
| `test_store.py` | 14 | all pass |

## Open issues

- The recursive-chunker bug above (decision: fix and rebuild both collections, or mark the property test expected-to-fail).
- Vector search (Chroma HNSW) is not bit-reproducible across processes: retrieved lists can differ at deep ranks between runs while the metrics stayed identical in the dev runs (see PROJECT_CONTEXT).
- `rag/retrieval.py` is still not import-safe; the API is not wired to the new Retriever.
