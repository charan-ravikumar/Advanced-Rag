# Test baseline (current)

History: [initial red](TEST_BASELINE_initial_red.md), [after import safety](TEST_BASELINE_2_after_import_safety.md), [after the Retriever](TEST_BASELINE_3_after_retriever.md),
[after Evaluation v2](TEST_BASELINE_4_after_eval_v2.md), [after the offset-native chunker](TEST_BASELINE_5_after_offset_native_chunker.md),
[after Stage 2a](TEST_BASELINE_6_after_stage2a.md), [after Phase 2](TEST_BASELINE_7_after_phase2.md) (758 passed), [after Phase 3](TEST_BASELINE_8_after_phase3.md) (997 passed, 3 skipped),
[after Phase 4](TEST_BASELINE_9_after_phase4.md) (1166 passed, 4 skipped).
This file replaces them after Phase 5 (the Streamlit app, local only).

- Full run: `pytest -p no:cacheprovider -q -rs` (main venv `%LOCALAPPDATA%\advrag\venv`). Quick run: `pytest --fast -p no:cacheprovider -q -rs`.
- Evaluation venv (`venv_eval`, RAGAS): `tests/unit/test_ragas_adapter.py` runs only there (it skips in the main venv): `& "$env:LOCALAPPDATA\advrag\venv_eval\Scripts\python.exe" -m pytest tests/unit/test_ragas_adapter.py -p no:cacheprovider -q -o addopts=""`.
- Collected in the main venv: **1332** tests.

## Counts

| Category | Full run (main venv) | `--fast` run (main venv) |
|---|---|---|
| passed | **1328** | 1263 |
| failed | **0** | 0 |
| skipped: needs Retriever refactor (old `rag.retrieval` tests) | 3 | 3 |
| skipped: ragas is not installed in the main venv (`test_ragas_adapter.py`) | 1 | 1 |
| skipped: `--fast` profile (slow tests) | 0 | 65 |
| time | 459 s (pytest) / 466 s (shell) | 255 s / 262 s |

Timings depend on machine load (the folder is inside OneDrive); the AppTest files are the slowest per test (each run starts a script); treat the numbers as indicative.
`test_ragas_adapter.py` in `venv_eval` was not part of this phase and was not re-run (5 passed at the end of Phase 4).

## What changed since the previous baseline (Phase 5)

All new tests are hermetic: a fake service, fake providers, a fake tracer and reader, fake clocks, tmp directories. No network, no keys, no real store.
(The real app was also started by hand against the real store; see PROJECT_CONTEXT.)

- **`test_app_guards.py` (16):** the access gate (constant-time compare on bytes, the lockout, the secret never in a repr or message), the sliding-window rate limit and the session cap on a fake clock,
  the process-wide daily budget (thread safety, resets on the next UTC day, release), and `check_question` being the service's own validator.
- **`test_rag_core_passages_only.py` (5):** `RagService.passages_only`: retrieval with no model call, no cache and no trace, a degraded result with the passages and a typed error.
- **`test_app_support.py` (61):** the marker normaliser and styler (`【S1】`, `［S1］`, partial markers while streaming, `$` escaping, a marker touching bold text), `classify`, the artifact loader (never raises, drops sections
  without a scope), the artifact builder (fails on a missing run folder, scope or fingerprint; the claims-log cross-check; freshness), the bootstrap (every fatal check failing in turn, key warnings, versions, the
  manifest lookup, a locked retriever across threads, no raise on bad artifacts).
- **`test_app_ui.py` (72, Streamlit AppTest):** every tab in every state (no data, no keys, unavailable), the chat happy path, every state (abstained, degraded, mid-stream error, invalid question, daily budget reached,
  cached), scope and filters, examples, feedback (right trace id, masked comment, disabled without a trace id) and the rerun surviving a thumb, the access-code gate (wrong blocks, right opens, the lockout, the code from
  `st.secrets`, never rendered), rate limit and daily cap shared by sessions, the self-check error screen and the retry, the Evaluation numbers equal the artifact, "What the data does NOT show", app modules importing no SDK,
  and only the bootstrap touching the environment.
- **Extended:** `test_import_guard.py` (eight new modules in the poisoned-subprocess probe: `eval.app_results` and seven `app.*` modules), `test_rag_core_imports.py` (`app/bootstrap.py` listed as the app's one `load_env` caller).
- **Mutation check (not part of the suite, reverted):** removing the daily-cap branch (`if not res.daily.try_acquire():`) in `app/ui_chat.py::submit` made 4 tests fail.
- **Bugs found while testing, not by the first version of the tests:** the lockout on the fifth wrong code left the input enabled until the next rerun (fixed: redraw at once); a form value is committed only with the submit
  click, which several first tests got wrong (the tests were wrong, not the app); a marker touching bold text (`**ROUGE-L**[S1]`) rendered as literal asterisks in a real browser, which AppTest cannot see (fixed, and a test now
  renders the Markdown with a parser); the old test of `load_env` callers did not know the app (updated).

## Still skipped

- `tests/unit/test_retrieval.py::TestRetrieveAndBuildContext` (3): wrong by construction and targeting the deprecated `rag.retrieval` (see [TEST_BASELINE_2_after_import_safety.md](TEST_BASELINE_2_after_import_safety.md)).
- `tests/unit/test_ragas_adapter.py` (module): skipped in the main venv by design (ragas is installed only in `venv_eval`); it passes there.
- Skipped by `--fast` only (65): marker `slow` (module-import probes, the real-model tests, subprocess tests, one AppTest file-level probe); `--fast` also lowers Hypothesis from 200 to 8 examples.

## Open issues

- Live checks done by hand and not in the suite: `scripts/check_model_ids.py`, `scripts/quota_dry_run.py`, `eval/run_generation_eval.py`, `eval/run_ragas.py`, and the real app in a browser (one example question, cached on the second ask).
- No automated browser test: AppTest does not render Markdown or run a real frontend.
- The RAGAS scoring of the first generation run is incomplete (2 of 34 answers scored: the judge's daily quota ended); it resumes from its cache. The Evaluation tab therefore shows no RAGAS numbers.
- `app_data/results.json` must be rebuilt (`scripts/build_app_artifacts.py`) after any new run; `--check` and a test detect a stale file.
- `rag/retrieval.py` is deprecated and still imported by four legacy entry points; nothing new may import it.
