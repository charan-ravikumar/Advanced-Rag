# Test baseline (current)

History: [initial red](TEST_BASELINE_initial_red.md), [after import safety](TEST_BASELINE_2_after_import_safety.md), [after the Retriever](TEST_BASELINE_3_after_retriever.md),
[after Evaluation v2](TEST_BASELINE_4_after_eval_v2.md), [after the offset-native chunker](TEST_BASELINE_5_after_offset_native_chunker.md),
[after Stage 2a](TEST_BASELINE_6_after_stage2a.md), [after Phase 2](TEST_BASELINE_7_after_phase2.md) (758 passed), [after Phase 3](TEST_BASELINE_8_after_phase3.md) (997 passed, 3 skipped).
This file replaces them after Phase 4 (generation evaluation, minimal scope).

- Full run: `pytest -p no:cacheprovider -q -rs` (main venv `%LOCALAPPDATA%\advrag\venv`). Quick run: `pytest --fast -p no:cacheprovider -q -rs`.
- Evaluation venv (`venv_eval`, RAGAS): `tests/unit/test_ragas_adapter.py` runs only there (it skips in the main venv): `& "$env:LOCALAPPDATA\advrag\venv_eval\Scripts\python.exe" -m pytest tests/unit/test_ragas_adapter.py -p no:cacheprovider -q -o addopts=""`.
- Collected in the main venv: **1169** tests.

## Counts

| Category | Full run (main venv) | `--fast` run (main venv) | `test_ragas_adapter.py` in `venv_eval` |
|---|---|---|---|
| passed | **1166** | 1110 | 5 |
| failed | **0** | 0 | 0 |
| skipped: needs Retriever refactor (old `rag.retrieval` tests) | 3 | 3 | - |
| skipped: ragas is not installed in the main venv (`test_ragas_adapter.py`) | 1 | 1 | - |
| skipped: `--fast` profile (slow tests) | 0 | 56 | - |
| time | 253 s (pytest) / 258 s (shell) | 110 s / 115 s | 9 s |

Timings depend on machine load (the folder is inside OneDrive); treat them as indicative.

## What changed since the previous baseline (Phase 4)

All new tests are hermetic: a fake generator, a fake judge, a fake clock, tmp directories. No network, no keys, no real store.

- **`test_answer_metrics.py` (32):** SQuAD normalisation edge cases (articles inside words, Unicode quotes and dashes, a narrow no-break space, markdown), hand-computed token F1 values,
  max over references (not the mean), ties, table-only and unanswerable references skipped, the comma join equals the space join, yes/no accuracy, abstention as a miss, citation validity, paper-clustered intervals.
- **`test_run_generation_eval.py` (35):** the seeded sample, every cache-key ingredient, the disk cache, the token ledger and governor, a complete session, rows readable by `compare.py`, the same context in both modes,
  the retrieval arguments equal the service's, the budget stopping BEFORE the call that would exceed it and repeated sessions finishing with every (question, mode) generated exactly once, a 429 caching nothing,
  a server error retried next run, a changed reasoning effort or context being a miss, the test split and a changed pre-registration refused.
- **`test_judge_and_ragas_runner.py` (41):** the judge cache key (every ingredient), valid-only caching, per-minute and per-day 429 handling, the session call budget, usage per metric, error classification,
  the scoring run folder labelled EXPLORATORY with the judge recorded, abstentions skipped, a quota stop then a resume paying for each distinct judge call once, a modified generation run refused, the analytics
  opt-out set before any third-party import, and one subprocess test that the evaluation modules import no ragas or provider SDK.
- **`test_gen_arms_and_quota.py` (20):** the oracle and title contexts (code that exists but is not wired into a run), the plan arithmetic, the sliding-window pacer, `RagService.answer_long`.
- **`test_ragas_adapter.py` (5, `venv_eval` only):** the adapter against the REAL ragas collections metrics with a fake judge: faithfulness from verdicts, relevancy from embeddings, three judge calls per answer, all cached.
- **Extended:** `test_rag_core_groq_only.py` (reasoning effort sent, recorded, validated, keyed; reasoning tokens kept separate), `test_rag_core_citations.py` (the lenticular `【S1】` and full-width brackets,
  lists, malformed forms, code, a service-level case), `test_import_guard.py` (eight new modules in the poisoned-subprocess probe), `test_rag_core_imports.py` (eval runners listed as entry points).
- **Mutation checks (not part of the suite, both reverted):** the mean instead of the max in `answer_f1` made 2 tests fail; a judge cache key that ignores the metric name made 3 tests fail.
- **Bugs found by the live runs, not by the fakes:** the model cites as `【S1】` at low reasoning effort and the ASCII-only parser read it as no citations (fixed, tested); the free-tier daily limits are the real constraint
  (the quota dry run and the resumable design follow from it).

## Still skipped

- `tests/unit/test_retrieval.py::TestRetrieveAndBuildContext` (3): wrong by construction and targeting the deprecated `rag.retrieval` (see [TEST_BASELINE_2_after_import_safety.md](TEST_BASELINE_2_after_import_safety.md)).
- `tests/unit/test_ragas_adapter.py` (module): skipped in the main venv by design (ragas is installed only in `venv_eval`); it passes there.
- Skipped by `--fast` only (56): marker `slow` (module-import probes, the real-model tests, subprocess tests); `--fast` also lowers Hypothesis from 200 to 8 examples.

## Open issues

- Live checks done by hand and not in the suite: `scripts/check_model_ids.py`, `scripts/quota_dry_run.py`, `eval/run_generation_eval.py`, `eval/run_ragas.py`. They need keys and the network.
- The RAGAS scoring of the first generation run is incomplete (2 of 34 answers scored: the judge's daily quota ended); it resumes from its cache.
- `scripts/smoke_*` and the live runners were exercised once; the real judge adapter (`build_gemini_raw`) was exercised only by that partial run.
- `rag/retrieval.py` is deprecated and still imported by four legacy entry points; nothing new may import it.
