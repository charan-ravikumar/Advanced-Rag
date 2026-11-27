# CLAUDE.md — standing rules for this repository

1. **No destructive actions without asking.** Never run any script that calls
   `clear_collection()` or otherwise deletes data without asking the owner first.
   Never touch Chroma Cloud; all dev and eval use a local
   `chromadb.PersistentClient` under `%LOCALAPPDATA%\advrag\chroma` (outside
   OneDrive; opened by `db.store.build_store(cfg, strategy)`, one collection per
   chunking strategy). The store has no "clear collection" method on purpose.
2. **Packages.** Public PyPI is allowed. Install into a virtualenv, never the
   system Python, and use the pinned versions in the lock files. Do not change
   global pip configuration. Ask the owner before adding or upgrading a
   dependency. If a package is unavailable, stop and tell the owner.
3. **No data leaves the machine.** Do not send repo code or any non-public data
   to external services. Public dataset downloads are fine.
4. **Small steps, verified.** Work in small steps. After each step run the
   tests and show the output. Never claim something works without running it.
5. **Evidence discipline.** No number may appear in README or docs unless it
   comes from a saved run artifact. Every published number goes into
   `docs/CLAIMS_LOG.md` with: metric, value, run folder, code fingerprint.
6. **Code fingerprint** = sha256 over all `.py` and `.yaml` files in the repo
   (sorted by path), computed by the helper in `utils/fingerprint.py`
   (`python -m utils.fingerprint`). It is used in place of a git SHA.
7. **Decision log.** After every task, append an entry to `PROJECT_CONTEXT.md`
   using the template at the top of that file.
8. **Python environment.** Always use the virtualenv at
   `%LOCALAPPDATA%\advrag\venv` (outside OneDrive), e.g.
   `& "$env:LOCALAPPDATA\advrag\venv\Scripts\python.exe" -m pytest`. Exact
   versions are recorded in `docs/env_baseline.txt` and as a lock in
   `requirements.lock` (core + serve + dev; regenerate with pip-tools, see its header). The test harness sets
   `HF_HUB_OFFLINE=1` itself (`tests/conftest.py`).
   **Test commands** (run from the repo root, no `--ignore` needed):
   - everyday: `pytest --fast -p no:cacheprovider -q -rs`. About 310 s of pytest
     time (the old 30 s target is NOT met: the suite is now ~1334 tests + 62 skipped, and most
     of the time is per-test setup, Streamlit AppTest start-up and model-free
     fixtures, not Hypothesis). Uses 8 hypothesis
     examples per property and skips tests marked `slow` (torch-importing probes,
     the real-model tests and the subprocess profile tests).
   - full: `pytest -p no:cacheprovider -q -rs`. About 460 s (NOT re-measured in Week 1). 200 hypothesis
     examples per property plus the `slow` and `model` tests. Run it before
     finishing a task and before quoting any chunking result.
   **Hypothesis profiles** (`tests/conftest.py`): `ci` is the default for normal
   and `--fast` runs (`derandomize=True`, no example database, so a run is
   reproducible for a fixed set of collected modules). `explore` uses random
   seeds and prints the failing blob: select it with
   `$env:ADVRAG_HYPOTHESIS_PROFILE="explore"; pytest ...` when hunting new bugs.
   Every failure found by `explore` MUST be pinned with `@example` (see
   `test_chunker_properties.py`), otherwise CI will not keep finding it.
   (The skipped `TestRetrieveAndBuildContext` class was deleted with
   `rag/retrieval.py`; `tests/unit/test_retrieval.py` now holds only the router tests.)
9. **`scripts/run_evaluation.py` was deleted (Week 1, Step 1).** It used to
   overwrite README.md and `eval/results/benchmark_summary.md`; the summaries it
   wrote are UNVERIFIED history.
10. **Import hazards.** `.env` holds real Chroma Cloud credentials. The only code
    that could reach Chroma Cloud (`db/vectordb.py`) and the import-unsafe legacy
    module (`rag/retrieval.py`) were deleted in Week 1, Step 1: nothing in the
    repo can open a cloud client any more, and `tests/conftest.py` still blocks
    `chromadb.CloudClient` and non-loopback network access. The chunkers and
    every module listed in `tests/unit/test_import_guard.py` are import-safe
    (lazy).
11. **Embedding model.** `sentence-transformers/all-MiniLM-L6-v2` is pinned to
    revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (`config.yaml`,
    `embedding.revision`) and is loaded from the local HuggingFace cache only
    (`local_files_only=True`). Never download or change it without asking.
    Tests run hermetically: `tests/conftest.py` blocks non-loopback network
    access and `chromadb.CloudClient`.
12. *(Retired in Week 1, Step 1: `rag/context_builder.py` was deleted and nothing
    imports tiktoken.)*
13. **`scripts/ingest_qasper.py` was deleted.** Use `scripts/ingest.py`,
    which is incremental, never clears a collection, and verifies the store.
    `CHROMA_LOCAL_PATH` overrides the local store directory (OneDrive paths are
    refused).
14. **Lexical index.** The SQLite FTS5 index lives next to the Chroma dir
    (`%LOCALAPPDATA%\advrag\lexical_<strategy>.sqlite3` for the default `chroma`
    dir; `lexical_<dirname>_<strategy>.sqlite3` for any other dir, e.g.
    `lexical_chroma_v2_recursive.sqlite3`; `lexical_index_path`).
    Build or repair it with `scripts/build_lexical.py --strategy <s>` (reads the
    store, never embeds, never deletes from the vector store). Ingestion writes
    both stores and verifies count and chunk-id parity. Do not re-ingest a built
    strategy just to add the lexical index.
15. **`scripts/smoke_retrieve.py` is for eyeballing only.** It prints no
    metrics; never copy its output into a doc.
16. **Evaluation v2 (retrieval only).** `eval/run_retrieval_eval.py` calls only
    `Retriever.retrieve`: no LLM, no network. Question set: `golden_set_v2.json`;
    dev/test split by paper in `split.json`. **Dev numbers are for development
    only: never put them in README or `docs/CLAIMS_LOG.md`.** The TEST split is
    run only when the owner asks: it needs `--i-understand-this-is-final`, the
    frozen config `eval/test_freeze.json` (created 2026-11-22 at the owner's request: D; never edit or recreate it)
    and appends to `eval/test_ledger.jsonl`. THE TEST SPLIT HAS BEEN RUN ONCE (ledger: 1 line, run
    `runs/eval/20261122T144233Z_3afc204d_test_paper`); do not run it again without the owner's explicit instruction. Never tune on test. Compare
    arms only with `eval/compare.py` (paired bootstrap, Holm across all
    comparisons); check run-to-run agreement with `eval/determinism.py`.
17. **Stores.** `%LOCALAPPDATA%\advrag\chroma` holds the OLD stores (built with
    the pre-fix chunker; kept, do not delete). `%LOCALAPPDATA%\advrag\chroma_v2`
    holds the current ones (recursive and semantic, built with the offset-native
    chunker; select with `$env:CHROMA_LOCAL_PATH`). Evaluation runs that must
    describe the current code use `chroma_v2`. Changing any chunking module
    changes the ingest signature, and ingestion then refuses to write to a store
    built with another one: ask the owner before re-ingesting.
18. **Evaluation labelling.** PAPER mode (`--mode paper`, the default) is the
    PRIMARY setting (Qasper's canonical single-document task); corpus mode is
    labelled "EXTENSION: corpus-wide" and `--mode corpus_titled` "EXTENSION:
    corpus-wide, title-qualified" in `config.json`, `summary.json` and the
    `compare.py` output. `compare.py`'s PRIMARY interval is the paper-clustered
    bootstrap; the per-question interval is secondary.
19. **Pre-registration (Stage 2a).** `eval/PREREGISTRATION.md` fixes the arms,
    metrics, comparisons and the decision rule; its sha256 is in
    `eval/PREREGISTRATION.sha256` and in every run's `config.json`. NEVER edit
    the original text (the runner refuses to run if it changed); deviations go
    in an appended, dated `## Deviations` section. Arms are defined in
    `eval/run_retrieval_eval.py::ARMS` (`--arm core` = A-F, `--arm all` = 15).
    `scripts/decision_rule.py` applies the rule (reports only; nothing is
    frozen). Add or change an arm only through a NEW pre-registration.
20. **Reranker.** `cross-encoder/ms-marco-MiniLM-L-6-v2` is pinned to revision
    `233902d25c440f23af6f7d6e94d2946bac0bee0a` (`config.yaml`,
    `pipeline.reranker_revision`), downloaded once with the owner's approval and
    loaded lazily from the local cache only (`rag/rerankers.py`). Never
    download or change it without asking. It is off by default: the reranker is
    chosen per call (`rerank=True`: the app sidebar and evaluation arms G/H), and
    the `pipeline.reranker_enabled` key in `config.yaml` is documentation only,
    nothing reads it.
    Reranker latency is on this CPU (about 1.4 s per query on dev).
21. **Generation path (`rag_core/`).** Import-safe (no client, model or network
    at import); provider SDKs are imported lazily inside the providers. API keys
    come ONLY from the environment (`GROQ_API_KEY`, `GEMINI_API_KEY`), are read at
    the first request, and must never be printed, logged or written (the package
    has no `print`/logging; `tests/unit/test_rag_core_imports.py` enforces it).
    All unit tests use fake providers: no network, no real keys. `rag.retrieval`
    was deleted; `tests/unit/test_rag_core_imports.py` fails if any file imports it. The Stage 2b
    modules (`rag/router_docs.py`, `rag/gemini_embedder.py`) are dormant and must
    not be imported by `rag_core` (Stage 2b is recorded as NOT EXECUTED in
    `eval/PREREGISTRATION_2b.md`). No performance number goes into a doc without a
    saved run. `scripts/smoke_generate.py` is live and optional: it needs
    `GROQ_API_KEY` and `CHROMA_LOCAL_PATH`, prints no timings or token counts, and
    sends only public questions and public paper text. Generation is Groq ONLY
    (`generation.providers: ["groq"]`); Gemini is only the offline evaluation judge
    (`generation.judge`) and is not in the request path.
22. **Secrets and `.env`.** NEVER open, cat, print, echo, grep the values of, or
    otherwise display `.env` or any API key. Check presence by name only, printing
    booleans (`rag_core.env.present`). `.env` is read ONLY by `rag_core/env.py::
    load_env()`, which loads an allowlist of five names (GROQ_API_KEY,
    GEMINI_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL),
    never overrides an existing variable, and is called only from scripts and app
    entry points, never at import. `.env` also holds old Chroma Cloud and Redis
    variables: the allowlist deliberately does not load them. `.env.example` lists
    names only and must never contain a value. Only public Qasper questions and
    paper text may go to Groq, Gemini or Langfuse.
23. **Tracing (Langfuse).** Optional and fail-open: `RagService` always wraps the
    tracer in `SafeTracer`; a tracer bug or outage must never change or break an
    answer. Default capture is `masked` (no passage text, no compiled prompt, key
    patterns redacted in every mode). Do not add observations per chunk or per
    stream delta (free-plan unit budget; the span tree is bounded at 8 and a test
    enforces it). Evaluation runs (`answer_short`) are untraced unless
    `tracing.trace_evaluation` is true. The Metrics API v2 allows very few
    requests per day on the free plan: never loop over reader panels, never lower
    the reader's cache TTL, and keep every dimension and filter column within the
    list pinned in `tests/unit/test_rag_core_observability.py`. Live scripts that
    contact Langfuse (`scripts/smoke_trace.py`, `seed_prompts.py`,
    `check_prompt_drift.py`) are run only when the owner asks.
24. **Prompts.** The templates in `rag_core/prompts.py` are the source of truth
    and the fallback; Langfuse holds the served copies (label `production`). A
    prompt fetched from Langfuse is validated before use (`validate_template`);
    change the validator and the templates together. Any wording change bumps
    `PROMPT_VERSION`. `scripts/seed_prompts.py` creates only what is missing
    (`--apply`) and never edits or deletes a version; `--update-differing` creates a
    NEW version and needs the owner's approval. Evaluation runs use the local
    templates or a pinned version (`prompts.pin`), never the moving `production`
    label.
25. **Evaluation environment (RAGAS).** RAGAS 0.4.4 lives in its OWN venv,
    `%LOCALAPPDATA%\advrag\venv_eval` (pinned in `requirements-eval.lock`, never in
    the main lock), because installing it into the main venv would
    downgrade pinned packages (openai, rich, tenacity). Run only
    `eval/run_ragas.py` there; generation and all tests run in the main venv. Do
    NOT install or upgrade anything in the main venv for RAGAS. **Every eval entry
    point must set `RAGAS_DO_NOT_TRACK=true` BEFORE importing ragas** (otherwise
    ragas posts usage events to a third party). Details: `docs/ragas_notes.md`.
26. **Generation evaluation (Phase 4, minimal scope).** Pre-registration:
    `eval/PREREGISTRATION_gen.md` (+ `.sha256`; the runners refuse to run if its original
    text changed; deviations are appended under `## Deviations`, never rewritten). Scope is
    arm R only, one seeded sample (seed and ids in `config.json`), dev split only.
    `eval/run_generation_eval.py` (main venv) writes answers to `runs/gen/`; `eval/run_ragas.py`
    (venv_eval) scores a generation run folder. **The generation test run is REFUSED** unless
    `--i-understand-this-is-final` AND `eval/gen_freeze.json` exist; that file is created only
    when the owner asks. Every output of these runners is labelled EXPLORATORY (n=40, an
    uncalibrated judge): no number from them goes into the README; a `docs/CLAIMS_LOG.md` row
    is allowed only if it says EXPLORATORY and gives the run folder and the code fingerprint.
    **Token governor:** `--daily-token-budget` (default 180,000, below Groq's documented
    200,000 tokens/day for `openai/gpt-oss-120b`) counts the tokens of the responses in
    `runs/gen/token_ledger.jsonl` over a rolling 24 hours; pass tokens spent elsewhere in the
    window with `--already-used-tokens`. A quota stop exits with code 3; running the same command
    resumes (generations and judge calls are cached on disk under `runs/gen/cache` and
    `runs/gen/judge_cache`). **Never delete those caches or the ledger without asking** (they hold
    paid work); a cache-format change bumps the schema instead. Do not edit `.py`/`.yaml` files while
    a run is in progress (the run records a code fingerprint at the end). Only public Qasper text
    goes to Groq and Gemini. Reasoning effort (`generation.reasoning_effort`, default `low`) is
    recorded on every answer and is part of the cache key.
27. **Streamlit app (Phase 5, local only).** Entry `app/streamlit_app.py`; run it with
    `$env:CHROMA_LOCAL_PATH="$env:LOCALAPPDATA\advrag\chroma_v2"; & "$env:LOCALAPPDATA\advrag\venv\Scripts\python.exe" -m streamlit run app\streamlit_app.py`
    (`ADVRAG_INDEX_DIR` takes precedence over `CHROMA_LOCAL_PATH`). Streamlit is pinned in
    `requirements.lock`; the API facts and AppTest gotchas are in `docs/streamlit_notes.md`.
    Only `app/streamlit_app.py` and `app/bootstrap.py` read the environment: `bootstrap.prepare_environment`
    is the ONLY caller of `load_env` and copies `st.secrets` (GROQ_API_KEY, LANGFUSE_*, APP_ACCESS_CODE) into
    `os.environ`; no `app/ui_*` module may call `load_env` or touch `.env` (`rag_core/gateway.py` reads the key from the environment at the first request, as before). Never open `.env`; never render a key or
    the access code (booleans only; tests assert this). App modules import no SDK except streamlit and the
    project (an AST test enforces it) and never import `rag.retrieval`. All limits are in `config.yaml` under
    `app.*` (daily cap, session cap, rate limit, access-code lockout, metrics TTL); the daily cap is a request
    count shared by all sessions, beyond which no LLM call is made (`RagService.passages_only`). Thumbs are
    saved FIRST to a local SQLite file (`app.feedback_path`, default `%LOCALAPPDATA%\advrag\feedback.sqlite3`,
    keyed by request id, comment masked) and forwarded to Langfuse only when the answer has a trace id
    (`rag_core/scores.py::record_feedback`). The
    Evaluation and About tabs read ONLY `app_data/results.json` and `app_data/papers.json`, built by
    `scripts/build_app_artifacts.py` (it fails if a number has no run folder, a matching fingerprint or a scope;
    `--check` reports a stale artifact; rebuild after any new run). The app never runs an evaluation, never reads
    the test split, and the Observability tab makes at most one Metrics API request per TTL for all visitors
    (rule 23). AppTest does not render Markdown or a browser: a change to how text is shown needs a real-browser
    look as well as the tests. Tests use a fake service, tracer and reader (`tests/unit/app_fakes.py`): no network.
