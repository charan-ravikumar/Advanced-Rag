# Test baseline (current)

History: [initial red](TEST_BASELINE_initial_red.md) (16 passed, 37 failed), [after import safety](TEST_BASELINE_2_after_import_safety.md)
(118 passed, 23 skipped), [after the Retriever](TEST_BASELINE_3_after_retriever.md) (276 passed, 3 skipped, 0 failed),
[after Evaluation v2](TEST_BASELINE_4_after_eval_v2.md) (343 passed, 1 failed, 1 xfail, 3 skipped),
[after the offset-native chunker](TEST_BASELINE_5_after_offset_native_chunker.md) (394 passed, 3 skipped, 0 failed),
[after Stage 2a](TEST_BASELINE_6_after_stage2a.md) (425 passed, 3 skipped, 0 failed),
[after Phase 2, the generation path](TEST_BASELINE_7_after_phase2.md) (758 passed, 3 skipped, 0 failed).
This file replaces them after Phase 3 (Groq only, Langfuse tracing, Langfuse Prompt Management, the read side for the Observability tab).

- Full run: `pytest -p no:cacheprovider -q -rs` (no `--ignore`), venv at `%LOCALAPPDATA%\advrag\venv`.
- Quick run: `pytest --fast -p no:cacheprovider -q -rs`.
- Collected: **1000** tests. Hypothesis profile: `ci` (derandomized) by default; see `CLAUDE.md` rule 8.

## Counts

| Category | Full run | `--fast` run |
|---|---|---|
| passed | **997** | 950 |
| failed (real) | **0** | 0 |
| skipped: needs Retriever refactor (old `rag.retrieval` tests) | 3 | 3 |
| skipped: `--fast` profile (slow tests) | 0 | 47 |
| time | 398 s (pytest) / 404 s (shell) | 112 s (pytest) / 117 s (shell) |

**Timing caveat.** The project folder is inside OneDrive and its sync process competes for the CPU: the previous baseline measured the same suite
(then 761 tests) at 183 s for `--fast` and this run at 112 s for 1000 tests, an earlier run in this phase took 146 s. The `rag_core` tests themselves take a few
seconds in total; the time is in older Chroma-backed fixtures. Treat timings as load-dependent, not as a property of the code.

## What changed since the previous baseline (Phase 3)

All new tests are hermetic: a fake provider, a fake tracer, a fake prompt client, a fake Langfuse client, scripted HTTP and a fake clock. No network, no keys.

- **`test_rag_core_prompt_store.py` (54):** a valid Langfuse prompt is used with its version, label and hash; the `get_prompt` arguments (chat type, `production`
  label, TTL, timeout, no retries); pins by version; every fetch error falls back to the local template; every INVALID prompt shape is rejected (14 shapes: missing,
  extra and misplaced variables, missing sentinel or citation instruction, bad role, wrong number of system messages, placeholders, over the length cap, empty); variables
  stay inert after compile for adversarial text; two tests use the REAL SDK prompt class offline (no client, no network) and check that our compile equals the SDK's;
  two slow subprocess tests check that `mode=local` and no-keys never import the SDK.
- **`test_rag_core_tracing.py` (41):** masking (every key shape, URLs, e-mail addresses, long numbers, unicode, truncation after redaction), the three capture modes,
  cost only from configured prices, the fail-open wrapper (a tracer that raises on every call, on enter, on exit, on update), sampling, the daily cap with a fake clock,
  `LangfuseTracer` against a fake client (parenting, `end()`, usage mapping, first-token time, prompt link, bounded flush, the root level), one slow subprocess test (no SDK import).
- **`test_rag_core_service_tracing.py` (31):** the span tree and attributes for a normal answer, cache hit, all providers failing (root `WARNING`), mid-stream failure (root `ERROR`),
  retrieval failure, internal error, invalid question, small talk, abstention, prompt fallback events (fetch error `WARNING`, invalid prompt `ERROR`), a linked Langfuse
  prompt; a raising tracer leaves the events and the result identical to an untraced run (4 scenarios); no key-like string reaches the tracer in any capture mode;
  `answer_short` is untraced by default; an aborted consumer still ends the root; at most 8 observations.
- **`test_rag_core_observability.py` (53):** score schema and validation, feedback with a deterministic score id, masked comment, fail-open; the reader: every status, HTTP 429/401/403/500,
  exceptions and malformed bodies, TTL and error TTL, the daily budget and its UTC reset, sample shapes equal real shapes, and a pin of every dimension and filter column to the list the live API accepts.
- **`test_rag_core_prompt_sync.py` (9):** compare and seed with a fake server: dry run changes nothing, only missing prompts are created, idempotent, a differing prompt is never overwritten without the explicit flag, old versions are untouched.
- **`test_rag_core_groq_only.py` (10):** the default gateway has exactly one provider; with a fake OpenAI-style client and the real SDK exception classes: success, 500, 404, 401 and a missing key degrade with the retrieved passages and never leak the key, 429 is retried, a broken stream is a mid-stream error.
- **Extended:** `test_rag_core_imports.py` (30 to 62: no SDK or HTTP client at module level in any module, the new modules are in the documented set), `test_import_guard.py` (32 to 39: the six new modules in the poisoned-subprocess probe), `test_rag_core_cache.py` (30 to 32: the key carries the prompt id), `test_rag_core_service.py` (58, adapted).
- **Mutation checks (not part of the suite, both reverted):** the prompt validation switched off made 16 tests fail; the service using the raw tracer without the fail-open wrapper made 7 fail.
- **Bugs the tests found before any live run:** the reader's cache key contained the moving time window (a repeated read never hit the cache); one reader panel was not guarded against an odd response shape.
- **Bugs the live run found that fakes could not:** the configured Groq model id does not exist on the account (fixed, `scripts/check_model_ids.py`); the Metrics API has no `statusMessage` column, so the degraded-rate query failed silently (the root level is now the signal and a test pins the valid columns).

## Still skipped: needs Retriever refactor (3)

`tests/unit/test_retrieval.py::TestRetrieveAndBuildContext` (`test_returns_required_keys`, `test_latency_is_non_negative`,
`test_empty_collection_returns_context_string`): wrong by construction and targeting the old `rag.retrieval` module (it builds LLM clients
and a BM25 index at import; it carries a DEPRECATED banner); replaced by `tests/unit/test_retriever.py` and the `rag_core` tests. See
[TEST_BASELINE_2_after_import_safety.md](TEST_BASELINE_2_after_import_safety.md) for the per-test reasons.

## Skipped by `--fast` only (47, run in the full suite)

Marker `slow`: the module-import probe (now 31 modules plus 3 checks), `test_missing_model_cache_raises_clear_error_not_download`, the `model`-marked tests (including
the real reranker test), the 2 subprocess tests in `test_hypothesis_profiles.py`, the reranker missing-cache test, and three subprocess tests in the new Phase 3 files
(`mode=local` and no-keys never import the SDK; building a tracer without keys never imports it). `--fast` also lowers Hypothesis from 200 to 8 examples per property.

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
| `test_import_guard.py` | 39 | all pass |
| `test_ingestion.py` | 15 | all pass |
| `test_ingestion_lexical.py` | 13 | all pass |
| `test_lexical.py` | 47 | all pass |
| `test_lexical_query.py` | 12 | all pass |
| `test_rag_core_cache.py` | 32 | all pass |
| `test_rag_core_citations.py` | 49 | all pass |
| `test_rag_core_context.py` | 12 | all pass |
| `test_rag_core_gateway.py` | 66 | all pass |
| `test_rag_core_groq_only.py` | 10 | all pass |
| `test_rag_core_imports.py` | 62 | all pass |
| `test_rag_core_integration.py` | 5 | all pass |
| `test_rag_core_observability.py` | 53 | all pass |
| `test_rag_core_prompt_store.py` | 54 | all pass |
| `test_rag_core_prompt_sync.py` | 9 | all pass |
| `test_rag_core_prompts.py` | 10 | all pass |
| `test_rag_core_service.py` | 58 | all pass |
| `test_rag_core_service_tracing.py` | 31 | all pass |
| `test_rag_core_tracing.py` | 41 | all pass |
| `test_relevance.py` | 17 | all pass |
| `test_rerankers.py` | 4 | all pass |
| `test_retrieval.py` | 39 | 36 pass, 3 skipped |
| `test_retriever.py` | 58 | all pass |
| `test_routing.py` (Stage 2b, dormant) | 18 | all pass |
| `test_splitting.py` | 12 | all pass |
| `test_stage2a_analysis.py` | 8 | all pass |
| `test_store.py` | 14 | all pass |

## Open issues

- Live checks done by hand (not in the suite): `scripts/check_model_ids.py`, `scripts/smoke_generate.py` (Groq only), `scripts/seed_prompts.py`, `scripts/check_prompt_drift.py` and `scripts/smoke_trace.py`. They need keys and the network and are run only when the owner asks.
- `LangfuseTracer` and the reader are tested hermetically against fakes; only the live smoke exercised them against the real SDK and API. The real-SDK prompt classes are tested offline.
- The Observability panels `feedback_summary` (no feedback written yet) and `cost` (no prices configured, so zero) are not meaningful yet.
- The Gemini judge model id exists, but the judge itself is a later phase; a thinking model needs a thinking budget or a larger output limit there.
- Vector search (Chroma HNSW) is not bit-reproducible across processes in general; in the Stage 2a repeats all rows were identical apart from timing.
- `rag/retrieval.py` is deprecated and still imported by four legacy entry points (`app/main.py`, `eval/run_eval.py`, `rag/pipeline.py`, `streamlit_app.py`);
  nothing new may import it. The old API is not wired to the new Retriever or to `rag_core`.
- Suite timing is load-dependent (OneDrive sync; see the caveat above).
