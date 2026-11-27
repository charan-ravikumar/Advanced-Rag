# Test baseline (current)

History: [initial red](TEST_BASELINE_initial_red.md) (16 passed, 37 failed), [after import safety](TEST_BASELINE_2_after_import_safety.md)
(118 passed, 23 skipped), [after the Retriever](TEST_BASELINE_3_after_retriever.md) (276 passed, 3 skipped, 0 failed),
[after Evaluation v2](TEST_BASELINE_4_after_eval_v2.md) (343 passed, 1 failed, 1 xfail, 3 skipped),
[after the offset-native chunker](TEST_BASELINE_5_after_offset_native_chunker.md) (394 passed, 3 skipped, 0 failed),
[after Stage 2a](TEST_BASELINE_6_after_stage2a.md) (425 passed, 3 skipped, 0 failed).
This file replaces them after Phase 2 (the generation path, `rag_core/`) and the Stage 2b code that was written but not executed.

- Full run: `pytest -p no:cacheprovider -q -rs` (no `--ignore`), venv at `%LOCALAPPDATA%\advrag\venv`.
- Quick run: `pytest --fast -p no:cacheprovider -q -rs`.
- Collected: **761** tests. Hypothesis profile: `ci` (derandomized) by default; see `CLAUDE.md` rule 8.

## Counts

| Category | Full run | `--fast` run |
|---|---|---|
| passed | **758** | 721 |
| failed (real) | **0** | 0 |
| skipped: needs Retriever refactor (old `rag.retrieval` tests) | 3 | 3 |
| skipped: `--fast` profile (slow tests) | 0 | 37 |
| time | 437 s (pytest) / 446 s (shell) | 183 s (pytest) / 189 s (shell) |

**Timing caveat.** `--fast` was about 55 s before this phase and measured 167 to 189 s here. The slowest tests in the duration report are the same
pre-existing ones (Chroma-backed ingestion and retriever fixtures), each about twice as slow as before, and the OneDrive sync process was
using a large share of the CPU while the suite ran (this project folder is inside OneDrive and many new files were just created). The new
`rag_core` tests take a few seconds in total. Treat the timings of this run as inflated by machine load, not as a regression of the suite; re-measure
when OneDrive is idle (or move the project outside OneDrive).

## What changed since the previous baseline

- **Phase 2, `rag_core/` (new, 8 test files, 260 tests):** `test_rag_core_context.py` (12), `test_rag_core_prompts.py` (10: delimiters, injection text stays
  inside its `<source>`, escaping), `test_rag_core_citations.py` (49: `[S1][S2]`, `[S9]` with five sources, malformed markers, code spans, abstention
  sentinel and its streaming hold-back), `test_rag_core_gateway.py` (66: per-call timeout, retry with backoff and jitter, circuit breaker with half-open,
  failover only before the first token, `MidStreamError`, attempt records, the Groq and Gemini wrappers against fake clients and the REAL SDK exception
  classes, key scrubbing, the old Gemini `config={"timeout": 30}` bug), `test_rag_core_cache.py` (30), `test_rag_core_service.py` (58: event order, used
  sources equal the prompt sources, degraded mode, mid-stream, failover, cache hit/miss per key ingredient, input guard, short mode, latency fields),
  `test_rag_core_integration.py` (5: real Retriever, tmp Chroma and FTS5, hash embedder), `test_rag_core_imports.py` (30: no forbidden imports, no
  `print` or logging, keys only in the gateway, nothing new imports the deprecated `rag.retrieval`, config defaults equal `config.yaml`).
  All of them use fake providers: no network and no real keys.
- **Router fix:** `rag/router.py` now treats a message as small talk only if the WHOLE message is an allowlisted phrase. `test_retrieval.py` grew from 21 to
  39 tests (the old router tests were rewritten to assert the new behaviour in both directions: "attention mechanism", "BERT" and "Hello, how does BERT work?"
  are research queries; "hello", "Thanks!" and "good morning" are small talk).
- **Import guard:** the package and its nine modules (ten entries) were added to the poisoned-subprocess import probe (`test_import_guard.py`, 22 to 32 tests; those probes are `slow`).
- **Stage 2b code (written, NOT executed, dormant):** `test_gemini_embedder.py` (20) and `test_routing.py` (18) keep testing code that no evaluation uses;
  `test_eval_runner.py` grew to 41 tests (Stage 2b arms and prereg 2b hash handling). See `eval/PREREGISTRATION_2b.md`, "NOT EXECUTED".
- **Mutation check (not part of the suite):** six deliberate bugs were injected into `rag_core` (no escaping, no sentinel hold-back, oversize first chunk dropped,
  degraded results cached, retry after a mid-stream failure, cache key blind to the prompt version); 19 tests failed as intended; all were reverted.

## Still skipped: needs Retriever refactor (3)

`tests/unit/test_retrieval.py::TestRetrieveAndBuildContext` (`test_returns_required_keys`, `test_latency_is_non_negative`,
`test_empty_collection_returns_context_string`): wrong by construction and targeting the old `rag.retrieval` module (it builds LLM clients
and a BM25 index at import; it now carries a DEPRECATED banner); replaced by `tests/unit/test_retriever.py` and the `rag_core` tests. See
[TEST_BASELINE_2_after_import_safety.md](TEST_BASELINE_2_after_import_safety.md) for the per-test reasons.

## Skipped by `--fast` only (37, run in the full suite)

Marker `slow`: the module-import probe (now 24 modules + 2 checks), `test_missing_model_cache_raises_clear_error_not_download`, the `model`-marked tests (including
the real reranker test), the 2 subprocess tests in `test_hypothesis_profiles.py`, and the reranker missing-cache test. `--fast` also lowers Hypothesis from
200 to 8 examples per property.

## By file (full run)

| File | Tests | Result |
|---|---|---|
| `test_chunk_contract.py` | 13 | all pass |
| `test_chunker_properties.py` | 34 | all pass |
| `test_chunkers.py` | 14 | all pass |
| `test_compare.py` | 18 | all pass |
| `test_eval_metrics.py` (old metrics) | 9 | all pass |
| `test_eval_runner.py` | 41 | all pass |
| `test_evidence_f1.py` | 17 | all pass |
| `test_fingerprint.py` | 7 | all pass |
| `test_fusion.py` | 23 | all pass |
| `test_gemini_embedder.py` (Stage 2b, dormant) | 20 | all pass |
| `test_gold_spans_split.py` | 12 | all pass |
| `test_hypothesis_profiles.py` | 4 | all pass |
| `test_import_guard.py` | 32 | all pass |
| `test_ingestion.py` | 15 | all pass |
| `test_ingestion_lexical.py` | 13 | all pass |
| `test_lexical.py` | 47 | all pass |
| `test_lexical_query.py` | 12 | all pass |
| `test_rag_core_cache.py` | 30 | all pass |
| `test_rag_core_citations.py` | 49 | all pass |
| `test_rag_core_context.py` | 12 | all pass |
| `test_rag_core_gateway.py` | 66 | all pass |
| `test_rag_core_imports.py` | 30 | all pass |
| `test_rag_core_integration.py` | 5 | all pass |
| `test_rag_core_prompts.py` | 10 | all pass |
| `test_rag_core_service.py` | 58 | all pass |
| `test_relevance.py` | 17 | all pass |
| `test_rerankers.py` | 4 | all pass |
| `test_retrieval.py` | 39 | 36 pass, 3 skipped |
| `test_retriever.py` | 58 | all pass |
| `test_routing.py` (Stage 2b, dormant) | 18 | all pass |
| `test_splitting.py` | 12 | all pass |
| `test_stage2a_analysis.py` | 8 | all pass |
| `test_store.py` | 14 | all pass |

## Open issues

- No live provider has been exercised: `scripts/smoke_generate.py` is written and skips unless both `GROQ_API_KEY` and `GEMINI_API_KEY` are set (they were not).
  The model ids in `config.yaml` (`generation`) are settings, not verified against the live catalogues.
- Vector search (Chroma HNSW) is not bit-reproducible across processes in general; in the Stage 2a repeats all rows were identical apart from timing.
- `rag/retrieval.py` is deprecated and still imported by four legacy entry points (`app/main.py`, `eval/run_eval.py`, `rag/pipeline.py`, `streamlit_app.py`);
  nothing new may import it. The old API is not wired to the new Retriever or to `rag_core`.
- Suite timing is inflated by OneDrive sync (see the caveat above).
