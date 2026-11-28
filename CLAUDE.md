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
    come ONLY from the environment (`GEMINI_API_KEY`; `GROQ_API_KEY` is still
    allowlisted but nothing reads it), are read at the first request, and must never be printed, logged or written (the package
    has no `print`/logging; `tests/unit/test_rag_core_imports.py` enforces it).
    All unit tests use fake providers: no network, no real keys. `rag.retrieval`
    was deleted; `tests/unit/test_rag_core_imports.py` fails if any file imports it. The Stage 2b
    modules (`rag/router_docs.py`, `rag/gemini_embedder.py`) are dormant and must
    not be imported by `rag_core` (Stage 2b is recorded as NOT EXECUTED in
    `eval/PREREGISTRATION_2b.md`). No performance number goes into a doc without a
    saved run. `scripts/smoke_generate.py` is live and optional: it needs
    `GEMINI_API_KEY` and `CHROMA_LOCAL_PATH`, prints no timings or token counts, and
    sends only public questions and public paper text. **Generation is Gemini ONLY**
    (`generation.providers: ["gemini"]`, model `generation.gemini_model` =
    `gemini-3.5-flash-lite`). `GroqProvider` was removed from `rag_core/gateway.py`;
    `build_gateway` knows only `gemini` and rejects `groq`
    (`tests/unit/test_rag_core_gemini_only.py`). The gateway still supports a
    provider list (failover), but only one provider is configured, so there is no
    failover. The offline evaluation judge (`generation.judge.model` =
    `gemini-3.8-flash`) is a DIFFERENT model from the generator but the SAME family
    (Gemini), and it is not in the request path. Groq `openai/gpt-oss-120b` is
    history: it produced the earlier exploratory generation run and caches.
    **Long answers are structured output, not parsed prose.** The long-mode
    request carries a per-request JSON response schema (`rag_core/claims.py::answer_schema`,
    `SCHEMA_VERSION` `claims-1`, Gemini `response_json_schema`): `status`
    (`answered` | `insufficient_context`) and `claims`, each with `text` and
    `citations` of `{source, quote}`. `source` is an enum of exactly the labels that
    were sent. The code checks every label against the sources that were sent and
    looks every quote up word for word in its source (`locate_quote`: HTML-unescaped,
    whitespace-collapsed, case-sensitive, at least `MIN_QUOTE_WORDS` = 3 words). It
    records `start`/`end` offsets and `verified` on each `SpanCitation`
    (`AnswerResult.claims`). `AnswerResult.answer` is RENDERED by the code from
    the claims ("claim [S1] claim [S2][S3]"). A reply that is not a structured
    answer is `invalid_output` (degraded, passages shown). JSON cut off by the
    output limit keeps the claims that arrived and is `partial` with error
    `truncated`. While streaming, only the claim text is shown, never the JSON
    and never an abstention (`StreamRenderer`). The regex `[S#]` parser was
    removed; `rag_core/citations.py` keeps only the SHORT answer's
    `INSUFFICIENT_CONTEXT` sentinel helpers. Short answers stay plain text
    with no schema. A verified quote proves the words are in the passage, not
    that the passage supports the claim.
22. **Secrets and `.env`.** NEVER open, cat, print, echo, grep the values of, or
    otherwise display `.env` or any API key. Check presence by name only, printing
    booleans (`rag_core.env.present`). `.env` is read ONLY by `rag_core/env.py::
    load_env()`, which loads an allowlist of five names (GROQ_API_KEY,
    GEMINI_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL),
    never overrides an existing variable, and is called only from entry points
    (scripts, the API startup in `api/main.py`, and
    `app/bootstrap.py::prepare_environment`), never at import. `.env` also holds old Chroma Cloud and Redis
    variables: the allowlist deliberately does not load them. `.env.example` lists
    names only and must never contain a value. Only public Qasper questions and
    paper text may go to Gemini or Langfuse.
23. **Tracing (OpenTelemetry, viewed in Langfuse).** Optional and fail-open: `RagService` always wraps the
    tracer in `SafeTracer`; a tracer bug or outage must never change or break an
    answer. The tracer is `rag_core/otel_tracing.py::OtelTracer`. It writes
    OpenTelemetry spans through a PRIVATE `TracerProvider` (never the global one),
    with explicit parent contexts (never the ambient current span). Every
    generation span carries the GenAI semantic conventions (`gen_ai.*`), and every
    span carries Langfuse's OTel attributes (`langfuse.observation.*`,
    `langfuse.trace.name`, `session.id`, copied from the Langfuse SDK). A test
    compares them with the installed SDK's own encoding. Exporters
    (`tracing.exporters`): `langfuse` exports over OTLP/HTTP to
    `<LANGFUSE_BASE_URL>/api/public/otel/v1/traces` with Basic auth
    (`rag_core/langfuse_client.py::langfuse_span_exporter`, the only place that
    reads the keys). `otlp` is a plain `OTLPSpanExporter()` that the
    OpenTelemetry SDK configures from the standard `OTEL_EXPORTER_OTLP_*`
    variables; it is off by default. Both receive the same masked attributes.
    The Langfuse SDK is still used for prompts, feedback scores and the Metrics
    API, but no longer for traces. Do not add OpenTelemetry auto-instrumentation
    packages (FastAPI/httpx) without asking: they are not in the lock, and they
    would add spans beyond the bounded tree. Default capture is `masked` (no passage text, no compiled prompt, key
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
    `PROMPT_VERSION`. The long prompt now asks for structured claims
    (`PROMPT_VERSION` `2026-11-27.1`), and its validator requires `insufficient_context`
    and `"quote"`. A Langfuse copy of the OLD free-text long prompt is therefore
    REFUSED (local fallback, `prompt.fallback` event) until the owner approves
    `seed_prompts.py --update-differing`. `scripts/seed_prompts.py` creates only what is missing
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
    **Generator and judge:** the generator is Gemini `gemini-3.5-flash-lite` and the judge is
    Gemini `gemini-3.8-flash` (rule 21). They are different models of the same family, so
    family-level self-preference is still possible; `run_ragas.py::family_note` writes this
    into every scoring run. The model id was confirmed live on 2026-11-27, but the judge's
    settings check on that model failed with HTTP 503/504 and is NOT yet verified. The switches
    (generator Groq to Gemini, judge 2.5-flash to 3.8-flash, long answers to structured claims)
    are recorded in the dated 2026-11-27 Deviation of `eval/PREREGISTRATION_gen.md`. A run with
    them is a NEW exploratory run, not comparable with the earlier one. Long-mode rows carry
    `claims` and the EXPLORATORY `QuoteVerified` metric (`eval/answer_metrics.py`), and the
    generation cache schema is 3.
    **Token governor:** `--daily-token-budget` (default 180,000) is the project's own budget.
    It is NOT a documented Gemini limit: the 180,000 was chosen to sit below Groq's old 200,000
    tokens/day. The pacer (`GEMINI_RPM`/`GEMINI_TPM` in `run_generation_eval.py`) uses
    conservative, UNVERIFIED placeholders, not the key's real quota. The governor counts the tokens of the responses in
    `runs/gen/token_ledger.jsonl` over a rolling 24 hours; pass tokens spent elsewhere in the
    window with `--already-used-tokens`. A quota stop exits with code 3; running the same command
    resumes (generations and judge calls are cached on disk under `runs/gen/cache` and
    `runs/gen/judge_cache`). **Never delete those caches or the ledger without asking** (they hold
    paid work); a cache-format change bumps the schema instead. Do not edit `.py`/`.yaml` files while
    a run is in progress (the run records a code fingerprint at the end). Only public Qasper text
    goes to Gemini. Reasoning effort is still recorded on every answer and is part of the cache
    key, but it is `None` (Gemini has no such parameter; `config.yaml` no longer sets
    `generation.reasoning_effort`). Gemini's thinking is controlled by
    `generation.gemini_thinking_budget` (answers; unset = not sent) and
    `generation.judge.thinking_budget` (judge). The model id is in the cache key, so old Groq
    cache entries are never served for Gemini.
27. **API server and Streamlit client (Phase 5, local only).** Two processes. The answer
    pipeline runs ONLY in the FastAPI server (`api/`); the Streamlit app is a CLIENT of it
    and does not run `rag_core` in-process.
    - **API server** (`api/main.py`, routes in `api/routes.py`; tests `tests/unit/test_api.py`).
      Start it first:
      `$env:CHROMA_LOCAL_PATH="$env:LOCALAPPDATA\advrag\chroma_v2"; & "$env:LOCALAPPDATA\advrag\venv\Scripts\python.exe" -m uvicorn api.main:app --host 127.0.0.1 --port 8000`
      (`ADVRAG_INDEX_DIR` takes precedence over `CHROMA_LOCAL_PATH`; the index dir and
      `GEMINI_API_KEY` are needed by THIS process). At startup it calls `load_env` and builds
      the index, embedder, gateway, tracer and feedback store once
      (`app.bootstrap.build_resources`). If the self-check fails it still starts, answers
      `503 not_ready` and lists the failed checks on `GET /v1/ready`. When `ADVRAG_API_TOKEN`
      is set, every endpoint except `/v1/health` needs `Authorization: Bearer <token>`. Bind to
      127.0.0.1 unless a token is set. Endpoints: `/v1/health`, `/v1/ready`, `/v1/meta`,
      `/v1/ask` (SSE), `/v1/answer`, `/v1/retrieve`, `/v1/feedback`, `/v1/observability`.
      The server enforces the limits in `config.yaml` under `app.*`. The per-key rate limit
      and the session cap use `slowapi` (`api/limits.py`; the key is `X-Session-Id`, in
      memory, so a restart resets it). The daily cap is a request count shared by all
      clients; beyond it the server makes no LLM call and returns the passages with a
      `daily_budget_reached` error (`RagService.passages_only`). Thumbs are saved by the
      server FIRST to a local SQLite file (`app.feedback_path`, default
      `%LOCALAPPDATA%\advrag\feedback.sqlite3`, keyed by request id, comment masked). They go
      to Langfuse only when the answer has a trace id (`rag_core/scores.py::record_feedback`).
    - **Streamlit client** (entry `app/streamlit_app.py`): `& "$env:LOCALAPPDATA\advrag\venv\Scripts\python.exe" -m streamlit run app\streamlit_app.py`.
      `app/bootstrap.py::build_client_resources` opens no index, loads no model and reads no
      key. It talks to the API at `ADVRAG_API_URL` (default `http://127.0.0.1:8000`), sending
      `ADVRAG_API_TOKEN` if the server has one, through the adapters in `app/remote.py` over
      `api/client.py` (`httpx`, imported lazily inside the client; app modules never import
      it directly). It mirrors the API's self-check. If the API cannot
      be reached, the page shows a self-check error, and a failure mid-answer becomes a typed
      `error` event, never a stack trace. The access-code gate (`APP_ACCESS_CODE`) stays in
      the client. Streamlit is pinned in `requirements.lock`; the Streamlit API facts and
      AppTest gotchas are in `docs/streamlit_notes.md`.
    Only `app/streamlit_app.py` and `app/bootstrap.py` read the environment on the client side.
    `bootstrap.prepare_environment` is the client's only caller of `load_env`, and it copies an
    allowlist of `st.secrets` (`bootstrap.SECRET_NAMES`: GEMINI/GROQ keys, LANGFUSE_*,
    APP_ACCESS_CODE, ADVRAG_API_URL, ADVRAG_API_TOKEN) into `os.environ`. No `app/ui_*` module may
    call `load_env` or touch `.env`. Never open `.env`; never render a key, the API token or the
    access code (booleans only; tests assert this). App modules import no SDK except streamlit
    and the project (an AST test enforces it) and never import `rag.retrieval`. The
    Evaluation and About tabs read ONLY `app_data/results.json` and `app_data/papers.json`, built by
    `scripts/build_app_artifacts.py` (it fails if a number has no run folder, a matching fingerprint or a scope;
    `--check` reports a stale artifact; rebuild after any new run). Those two files are read
    locally by the client. NOTE: `app_data/results.json` was built from the old Groq runs, so it
    describes the Groq generator. The app never runs an evaluation and never reads the test
    split. The Observability tab gets its data through `GET /v1/observability`, and the server
    makes at most one Metrics API request per TTL for all visitors (rule 23). AppTest does not
    render Markdown or a browser: a change to how text is shown needs a real-browser look as
    well as the tests. UI tests inject fake resources (`tests/unit/app_fakes.py`: fake service,
    tracer and reader). API tests (`tests/unit/test_api.py`) run the real FastAPI app in-process
    with FastAPI's `TestClient`, a fake retriever and scripted providers. Neither uses the
    network.
28. **Official Qasper test set (held out).** `eval/heldout.py` is the only code that
    may touch it. It was downloaded once on 2026-11-27 to
    `%LOCALAPPDATA%\advrag\heldout\qasper_test_v0.3` (outside the repo, read-only);
    `eval/heldout_manifest.json` pins its sha256 values (never edit it). NEVER open,
    read, parse, print, copy, ingest or delete those files, and never run the
    evaluator script stored with them. `python -m eval.heldout verify|status` is safe
    (it hashes bytes only). `freeze` is run ONLY when the owner asks: it writes
    `eval/heldout_freeze.json` (data hash, arms, top_k, candidate_k, code
    fingerprint, pre-registration hash), which is never edited or recreated.
    `open_heldout` returns the data path at most ONCE: it needs explicit
    confirmation and a matching freeze, and it records the opening in
    `eval/heldout_ledger.jsonl` and in `OPENED.json` next to the data before
    returning. Never delete either record. Both evaluation runners refuse the data
    (or a byte-identical copy) as an input. No runner consumes it yet: building its
    question set and index is part of the final evaluation and needs its own
    pre-registration. It is NOT the "test" split of `split.json` (rule 16), which is
    a slice of the official DEV papers and has already been run once.
29. **Docker (`Dockerfile`, `compose.yaml`).** `docker compose run --rm --build ingest` once, then
    `docker compose up --build` (API 127.0.0.1:8000, UI 127.0.0.1:8501). Packages: public PyPI at the
    locked versions, torch from PyTorch's public CPU wheel index; no secrets in the build. The image
    holds no keys, models or index; only the variable NAMES listed in `compose.yaml` reach a container.
    NEVER run `docker compose config` without `--quiet` (it prints the values from `.env`); use
    `--env-file /dev/null` for tests that must not contact Gemini or Langfuse. Do not remove the
    `advrag_data` volume (`down -v`) without asking. `tests/unit/test_docker_files.py` pins these rules.
30. **CI (`.github/workflows/ci.yml`).** Jobs: `ruff check .`, `mypy` (rag_core, `[tool.mypy]`), `pytest --fast`,
    Docker build plus `scripts/ci_smoke.sh` (`/v1/ready`), `pip-audit` on `requirements.lock`. Keep all green; tool
    pins are in `requirements-ci.txt`. Never silence a failure with a blanket ignore. A new `--ignore-vuln` id goes
    in `.github/pip-audit-ignore.txt` with its reason and needs the owner's approval. The chromadb ids there hold
    only while no Chroma server or HttpClient is used. `scripts/ci_fetch_model.py` downloads a model and runs ONLY in
    CI. `scripts/ci_smoke.sh` uses its own compose project and removes only that project's volume.
    **`main` is protected** (turned on 2026-11-28 with the owner's approval): the five jobs are required checks,
    admins included, and force pushes and deletion are off. A direct push to `main` is rejected: push a branch,
    open a pull request, wait for the checks, then merge it. Renaming a job means updating the required checks.
    Never relax or remove the protection without asking.
31. **Run registry (`runs/index.jsonl`).** One row per run folder, built by `scripts/runs_index.py` from the files in
    the folders (id, path, timestamp, code fingerprint, config hash, split, metrics, files, content hash). Never edit it
    by hand: after adding or changing a run folder, run `python scripts/runs_index.py` and commit the index with the run.
    CI (a step of the `ruff` job) runs `--check`, which also fails if README.md or `docs/CLAIMS_LOG.md` name a run
    folder that does not exist. Being in the index does not make a number publishable: rules 5, 16 and 26 still apply
    (it holds dev and EXPLORATORY numbers).
