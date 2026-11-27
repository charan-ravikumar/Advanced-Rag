# 03 — Code Review

**Scope:** every live module (`rag/retriever.py`, `rag/lexical.py`, `rag/ingestion.py`, `rag/chunkers.py`, `db/store.py`, `rag_core/*`, `app/*`, `eval/*`) plus the legacy modules that your claims come from. Line numbers are from the audited tree (fingerprint `d76f6665…`).
**Evidence tags:** **[V]** verified by reading the code · **[T]** verified by a test run or a saved artifact · **[M]** verified by an AUDIT MEASUREMENT (`06` Appendix A) · **[R]** reasoned from code or library behaviour, *not* verified. I did not modify any application file.
**Severity:** *Critical* = wrong results, data loss or a security hole on the live path · *High* = significant production, quality or reproducibility risk on the live path · *Medium* · *Low*. **Legacy-path findings are rated "if revived"**: the code is dead but still advertised by `README.md`, `Makefile` and `resume_bullets_v2.md`, so it is also a *presentation* risk.

## 1. Summary

| Severity | Live path | Legacy path (dead, advertised) |
|---|---|---|
| Critical | **0** | 2 (L-01, L-04) |
| High | 4 (N-01 … N-04) | 4 (L-02, L-03, L-06, L-12) |
| Medium | 9 (N-05 … N-13) | 7 (L-05, L-07 … L-11, L-13) |
| Low | 6 (N-14 … N-19) | 1 (L-14) |

There is **no Critical defect on the live path**, and that is a real change from the v1 audit (which had five). The High items are about *serving*, *unmeasured safety behaviour* and *reproducibility*, not about correctness bugs.

**Fix first (P0/P1 in `05`):** N-04 (the documented commands do not run), N-03 (dependencies), N-01 (global lock), N-02 (abstention unmeasured), L-04 (delete the unauthenticated legacy server).

---

## 2. Live-path findings

### High

#### N-01 — One process-wide lock serialises all retrieval; one SQLite connection shared across threads
* **File / symbol:** `app/bootstrap.py:167-178` `LockedRetriever`; `rag/lexical.py:81-89` `SqliteFtsIndex.__init__(check_same_thread=False)` **[V]**
* **Issue:** a single `threading.Lock` wraps the *entire* `Retriever.retrieve()` (query embedding, Chroma, FTS5, and the cross-encoder when `rerank=True`). There is no lock-acquisition timeout. The owner documents the reason ("SQLite and the embedder are not safe for concurrent use"), but it was not measured, and no test exercises two concurrent sessions.
* **Why it matters:** hold time is the retrieval latency. TEST artifact: D p50 59.8 ms / p95 286 ms / **max 3,466 ms**; with rerank (arm G) **p50 1,242 ms / p95 1,943 ms / max 3,133 ms** **[T]**. With rerank on, the ceiling is ≈ 0.8 retrievals/s for *all* users combined, and one slow call (the 3.4 s outliers are real) stalls every other session. Streamlit pins one thread per session for the full LLM stream, so this lock is the main shared choke point.
* **Fix:** SQLite in WAL mode supports concurrent readers, so use a thread-local read-only connection for the lexical leg and drop the lock there. Verify that MiniLM `encode` under `torch.inference_mode()` and Chroma `query` are safe concurrently (write a 2-thread test first); keep a lock only around whatever fails. Put the cross-encoder behind a bounded semaphore. Add `lock.acquire(timeout=…)` and return a "busy" state instead of blocking forever.
* **Expected improvement:** retrieval concurrency from 1 to roughly the core count (4 logical CPUs on the audit machine) for the non-rerank path. **Measure with the Locust plan in `06` §3**; this is an estimate, not a result.

#### N-02 — No retrieval-confidence gate; behaviour on unanswerable questions was never measured
* **File / symbol:** `rag/retriever.py:82` `retrieve` (always returns `top_k`); `rag_core/service.py:293` (`if not ctx.sources:` abstains only on an *empty* result) **[V]**
* **Issue:** the old `vector_threshold` / `bm25_threshold` knobs still sit in `config.yaml` but no live code reads them. Whether the model abstains depends entirely on the prompt. The retrieval eval excludes unanswerable questions (`eval/run_retrieval_eval.py`), and the generation eval reports only `AbstentionRate_on_answerable` (0.200, n=40, EXPLORATORY) **[T]**. `golden_set_v2.json` contains **48 unanswerable dev questions** that no runner scores **[V]**.
* **Why it matters:** the false-answer rate (the model answering when the paper does not say) is *the* RAG safety number, and it is unknown. It is also the first question a hiring manager asks.
* **Fix:** (1) add a `false_answer_rate` / `false_abstain_rate` pair to the generation runner using the 48 unanswerable questions (dev only; pre-register as a deviation or a new pre-registration, per CLAUDE.md rule 19); (2) calibrate a fused-score margin or reranker-score threshold on dev; (3) gate in `RagService` ("no passage above threshold" → abstain without an LLM call; saves tokens too).
* **Expected improvement:** a measured false-answer rate with a CI; fewer wasted LLM calls on off-topic questions.

#### N-03 — Three dependency sources disagree; `requirements.txt` is a polluted freeze
* **File:** `requirements.txt:42,79,111-112,118,133,165,169,191,211`; `pyproject.toml:2-3,13,28`; `docs/env_baseline.txt` **[V]**
* **Issue:** `requirements.txt` (236 lines) contains `nipype`, `nibabel`, `pyxnat`, `kubernetes`, `spacy`, `unstructured`, **`fitz==0.0.1.dev2`** (a different package that collides with PyMuPDF's `fitz`), **`pathlib==1.0.1`** (an obsolete backport that shadows the stdlib module), `openai==3.57.0` while `pyproject.toml` says `openai>=1.0,<2.0`, an unpinned `redis>=5.0.0`; it is also a *different set* from the environment the tests actually ran in (236 lines vs. the 165-line `docs/env_baseline.txt`). The true environment is that baseline (Python 3.13.14), recorded in a comment-headed file that nothing consumes. `pyproject.toml` omits `langfuse`, a hard dependency of the live tracer **[V]**, and has no `[tool.hatch.build]` package selection, so `pip install -e .` (Makefile `install`) very likely fails or ships nothing **[R: not run, to avoid modifying the pinned venv]**.
* **Fix:** generate one lockfile from the real import graph (`uv pip compile` / `pip-tools`) with extras `serve`, `eval`, `dev`; delete the junk freeze; add a CI job that does a clean `pip install` and imports every live package.
* **Expected improvement:** a clone-to-running-tests path that works on a clean machine in one command.

#### N-04 — The documented commands do not run
* **File:** `README.md:7,11,66,72,79`; `Makefile:30,33,36`; `scripts/ingest_qasper.py:1` **[V]**
* **Issue:** `make ingest` and the README Quick Start run `scripts/ingest_qasper.py`, whose first line is `raise SystemExit("Deprecated: wipes the collection…")`. `make run-app` / README run the *root* `streamlit_app.py`, which imports the deprecated `rag.retrieval` (import-time clients, Chroma Cloud, BM25 over the whole corpus; `langchain_huggingface` is not installed). `make run-api` runs `app/main.py`, which cannot import (`openinference` missing; legacy chain). The README CI badge points to `your-org/advanced-rag`; the README says "backed by ChromaDB Cloud".
* **Why it matters:** an interviewer who clones the repo follows the README first. Every documented path fails.
* **Fix:** make the three commands true (`scripts/ingest.py`, `app/streamlit_app.py` via the documented env vars, and the new API), remove the badge until CI exists, and move the legacy files into an `legacy/` folder or delete them (the v1 snapshot exists).

### Medium

| ID | File / symbol | Exact issue | Why it matters | Fix | Expected improvement |
|---|---|---|---|---|---|
| **N-05** | `app/guards.py:50,100,135`; `app/streamlit_app.py:64` **[V]** | `SessionGuard`, `AccessGate` lock-out and the access state live in `st.session_state` (per browser session). Only `DailyBudget` is process-wide | A reload or new tab resets the rate limit, the per-session cap and the **access-code lock-out** (unlimited guesses with parallel sessions). One client can burn the whole 80/day budget and take the demo offline for everyone | Process-wide failed-attempt counter with exponential lock-out; a trusted-proxy client key when deployed; API-key quotas once there is a service | Abuse cost moves from "one browser tab" to "one credential" |
| **N-06** | `rag_core/gateway.py:186-191`; `rag_core/errors.py:10` **[V]**, behaviour **[R]** | A stream with **no visible text** is converted to `ProviderError("other", EmptyResponse)`. `"other"` ∈ `BREAKER_KINDS` and is not retryable. For a reasoning model (`gpt-oss-120b`) that spends the whole `max_tokens` on reasoning, the real signal (`finish_reason="length"`) is discarded | Three such questions in a row open the breaker for 30 s and **every user gets the degraded notice while Groq is healthy**; it is user-triggerable. Rare today (measured completion ≈ 67 to 116 tokens per call vs. cap 1,024) | Pass `Final.finish_reason` into the classification: `length` → typed `"length"` error, **not** breaker-eligible, one retry with a higher cap or `reasoning_effort="low"`; surface "answer too long" | Removes a user-triggered global degradation path |
| **N-07** | `rag/chunkers.py:234,251` `_group_paragraphs` **[V]+[M]** | Boundary detection embeds 7-paragraph sliding windows (`semantic_window_size: 3`) and compares *adjacent, 6/7-overlapping* windows. MiniLM truncates at 256 tokens | **77.5% of windows exceed 256 tokens; the centre paragraph is truncated away entirely in 28.1% of all windows and partly in 19.7%** (median paragraph = 90 tokens, so 3 preceding paragraphs consume the budget). Similarities are high regardless of topic shifts. On TEST, semantic = recursive (+0.006, ns) | Embed single paragraphs (or sentences) and compare to a running centroid or a percentile breakpoint; plot the similarity series; or retire the strategy and present it as a negative result | A defensible "semantic" story, or fewer lines to maintain |
| **N-08** | `rag/lexical.py:60` `escape_query`, `:130-147` `search` (the `doc_id IN (...)` clause is line 144); `rag/lexical_query.py` **[V]**, scaling **[M]** | Every question term, stopwords included, is OR-ed (≤ 64), and `doc_id` is an `UNINDEXED` FTS5 column, so the document filter is applied *after* the MATCH has visited every matching row. The docstring's "restricts the candidate set BEFORE ranking and limiting" is true semantically, not computationally | On synthetic copies of your corpus **[M]**: corpus-wide p50 **23.8 → 170.7 → 1,068 ms** at 200 → 2,000 → 10,000 docs (×45 for ×50); a **single-paper-filtered** query, your primary task, **15.1 → 115 → 589 ms** with the same ~37-chunk answer set; max 31 s. TEST at 7.3K chunks: lexical p50 20 ms, max 1,881 ms **[T]**. Stopword filtering is only ≈ 2× faster (574 ms at 10K) and does not change the slope | (1) Constrain the query by `rowid`s fetched from `chunk_map` (indexed on `doc_id`), or index `doc_id` as a token column and AND it into the MATCH; (2) cap terms by IDF; (3) re-measure at 10K docs. Dev showed no quality cost for stopword filtering (D2 0.4979 vs D 0.5001 nDCG@10, DEV only) | Single-paper lexical latency flat in corpus size; paper-mode retrieval stays near its 60 ms median at 10K docs instead of ≈ 0.6 s. **Medium today, High above ≈ 2K documents** |
| **N-09** | `rag_core/*` (0 logging calls, enforced by a test); `app/*` | No structured logs and no metrics on the live path. Failures are visible only in the UI result or in Langfuse, which is sampled at 150 traces/day (`tracing.daily_request_cap`) and needs keys | When Langfuse is off, sampled out or down, a provider outage, a breaker open or a slow retrieval leaves **no local trace** | JSON logger with `request_id`, stage latencies, error codes (never content); Prometheus RED + per-stage histograms (the `Retriever` already returns `latency_ms` per stage) | Debuggable without a SaaS; the "operational thinking" evidence interviewers look for |
| **N-10** | `rag/ingestion.py:107` `verify_store`; `db/store.py:152-153` `doc_ids` **[V]** | `list(store.iter_chunks())` loads every chunk (text + metadata) into memory; `doc_ids()` iterates all chunks | Fine at 7K chunks. At 363K chunks (10K papers) it is a multi-GB, minutes-long step; at 1M it fails | `col.get(include=[])` for ids, `metadatas` only for counts, sample texts for round-trip checks, paginate | Ingestion verification that scales linearly in time, constant in memory |
| **N-11** | `.env` (747 B, repo root, inside a OneDrive-synced folder); `rag_core/env.py` allowlist **[V]** (file never opened) | Per CLAUDE.md rules 10 and 22 the file holds real credentials, including legacy Chroma Cloud and Redis variables the allowlist deliberately ignores. `.gitignore` protects git, not OneDrive | Secrets are being synced to cloud storage; unused cloud credentials should not exist. (The v1 zip contains no `.env`: 0 of 72 entries) | Move the file to `%LOCALAPPDATA%\advrag\.env` (`load_env(path=…)` already supports a path), delete unused cloud variables, rotate any key that was synced | Smaller blast radius if the folder is shared or the account is compromised |
| **N-12** | `app/bootstrap.py:284`; `rag_core/service.py:77,175` **[V]** | `index_version` (`<chunk count>:<manifest sha>`) is computed once at start-up and baked into every cache key | A `delete_by_doc` or re-ingest while the process runs leaves the key unchanged: stale cached answers for up to the 1 h TTL | Recompute lazily with a short TTL (a stat of the manifest/`store.count()`), or have ingestion write a version file that `RagService` reads | Cache never outlives the data it was built from |
| **N-13** | `rag_core/prompt_store.py:87-110` `LangfusePromptStore._resolve`; `rag_core/service.py` (prompt resolved *before* the cache lookup) **[V]**, outage behaviour **[R]** | A failed fetch is not remembered: every request re-calls `client.get_prompt(..., fetch_timeout_seconds=2, max_retries=0)` | During a Langfuse outage each request, cache hits included, can pay up to 2 s before falling back to the local template. The "cache hit < 50 ms" target cannot hold then | Negative cache / tiny breaker (e.g. 60 s) around the fetch; resolve at start-up and refresh in the background | Cache-hit latency independent of Langfuse |

### Low

| ID | File / symbol | Exact issue | Fix |
|---|---|---|---|
| **N-14** | `rag_core/config.py:79-85` `_overlay`; `config.py`; `config.yaml:5,10` **[V]** | Unknown YAML keys under `generation`/`prompts`/`tracing`/`app` are silently ignored (`if hasattr(section, name)` iterates the dataclass fields), so a typo uses the default. Duplicate knobs: `pipeline.top_k` vs `generation.top_k`, `pipeline.rerank_k: 10 # PLACEHOLDER` vs `RagConfig.rerank_k = 20`. Line 5 still says `reranker_enabled: false # +11.7s on CPU` (measured p50 1.19 s) | pydantic-settings models with `extra="forbid"`; one source per knob; fix the stale comment |
| **N-15** | `rag_core/gateway.py:163,174` `LLMGateway.stream` **[V]** | `deadline_s=90` is checked only *between* attempts; `timeout_s=30` is an httpx read timeout between chunks, so a slow-drip stream can hold a thread indefinitely | Check a monotonic deadline on every delta and end with `MidStreamError(partial)` |
| **N-16** | `rag_core/scores.py:72` `submit_score` **[V]** | `if not trace_id: return False`: thumbs feedback is silently dropped for any request that was not traced (cap, sampling, no keys) | Persist feedback locally (SQLite/JSONL keyed by `request_id`) and forward asynchronously |
| **N-17** | `rag_core/cache.py:77`; `app/guards.py:135` **[V]** | Cache and daily budget are in-process: lost on restart, not shared across workers; no single-flight (two identical simultaneous questions both call the LLM) | Fine for one process. Add a `RedisCache` implementing the existing `CacheBackend` protocol only when a second replica exists |
| **N-18** | `pyproject.toml` `requires-python = ">=3.10"` **[V]** | Every test and run used Python 3.13.14; 3.10–3.12 never exercised | CI matrix, or raise the floor to what is tested |
| **N-19** | `rag/chunkers.py:21`; `pyproject.toml` `scikit-learn` **[V]** | A module-level `RecursiveCharacterTextSplitter` is built at import "for reference only"; `scikit-learn` is imported for a single `cosine_similarity` | Move the reference splitter into the test oracle; `numpy` dot products suffice |

---

## 3. Legacy-path findings (dead code, but the source of your claims)

All of these live in modules with **0% test coverage** that cannot be imported in your venv. They matter because a reviewer who opens the file sees them.

| ID | File:line | Issue | Severity if revived | Status of the live replacement |
|---|---|---|---|---|
| **L-01** | `rag/retrieval.py:392` | Gemini fallback passes `config={"timeout": 30}`; `GenerateContentConfig(timeout=30)` raises `ValidationError` (**reproduced on google-genai 3.6.1**). The fallback can never succeed, so the "automatic failover" bullet was never true | **Critical** | `GeminiProvider` is correct, tested with fakes, not enabled (`providers: ["groq"]`) |
| **L-02** | `rag/retrieval.py:403-519` | After Groq fails *mid-stream*, the handler `yield`s a full Gemini answer after the partial Groq text (splice); `total_tokens += len(delta.split())` | High | `MidStreamError`, never spliced |
| **L-03** | `app/main.py:348-369` | `set_cached_response(request.query, full_response)` runs unconditionally, so the failure text "Generation failed on both Groq and Gemini." is cached for 3,600 s | High | `is_cacheable` (`rag_core/cache.py:64`) |
| **L-04** | `app/main.py:137-226,258` | `POST /cache/flush`, `GET /metrics`, `/cache/stats` have no auth or rate limit; `cache/cache.py:240,284` uses `KEYS rag:cache:*` (O(N), blocks Redis); raw user queries are `print()`-ed (`:258`) | **Critical** | No HTTP surface on the live path |
| **L-05** | `rag/retrieval.py:812,830,896` | `hash(doc)` as chunk identity (process-salted; identical text from different papers collapses) and an O(C×N) loop `for i, doc in enumerate(all_documents)` | Medium | Deterministic `chunk_id`; hydrate by id |
| **L-06** | `rag/retrieval.py:220,241` | Whole corpus pulled into RAM with `collection.get()` and BM25-indexed at import; metadata filter ignored by BM25 | High | Persistent FTS5 index with filter parity |
| **L-07** | `rag/context_builder.py:248` | `enforce_token_budget` `break`s at the first chunk that does not fit, so an oversized first (merged) chunk yields an **empty context**; 13 `print()` calls; `hash(content)` dedupe | Medium | `rag_core/context.py` truncates the first chunk instead (100% tested) |
| **L-08** | `app/main.py:308`; `observability/metrics.py` | "Latency saved" uses a hard-coded `6.0` s baseline; "tokens saved" is `len(text.split())`; no labels; default buckets (max 10 s) for 120 s operations | Medium | None (no live metrics: see N-09) |
| **L-09** | `rag/retrieval.py:255-285` | `classify_query` makes a synchronous 15 s-timeout LLM call for every non-chit-chat query; not called by `app/main.py` at all | Medium | Rules-only `rag/router.py` |
| **L-10** | `rag/retrieval.py` (module scope); `app/main.py:93-120` | Import-time side effects: `load_dotenv`, Groq/Gemini clients from env, model load, OTLP exporter to a hard-coded `127.0.0.1:4317`, `OpenAIInstrumentor().instrument()` (captures prompts/completions into spans) | Medium | `rag_core` is import-safe (a test enforces it) |
| **L-11** | root `streamlit_app.py:82,116,144,168,183,233,263,284` | 8× `unsafe_allow_html=True`; untrusted values can be interpolated into HTML | Medium | New `app/` has none (`escape_md`) |
| **L-12** | `scripts/run_evaluation.py:87,1026,1034` | Benchmarks `BAAI/bge-reranker-base` while docs name MiniLM; `qpm = round(60 / avg_total, 1)`; `"error_rate": 0.0` literal; the script also overwrites `README.md` (CLAUDE.md rule 9) | High (claim integrity) | `eval/run_retrieval_eval.py` + claims log |
| **L-13** | `rag/loaders.py:21,250-252` | `fitz.open` never closed; `load_md` merges front-matter *after* `source`/`file_type` so a document can overwrite them (**citation spoofing**); `source` is an absolute path; no size/type/zip-bomb limits. Not wired to anything and 0% tested | Medium **(High the day an upload endpoint is added)** | Not applicable (no upload path) |
| **L-14** | `rag/embeddings.py`, `rag/pipeline.py`, `rag/ingest.py` | Dead or legacy: per-chunk `encode`, `uuid4` ids, `embedded_chunks.json` dump, `print` logging; `rag/ingest.py` pushes to Chroma Cloud | Low | `rag/ingestion.py` |

---

## 4. Remaining v1 Medium and Low findings: status

| v1 ID | Now |
|---|---|
| R-03 O(C×N) loop · R-06 squared-L2 `1/(1+d)` · R-07 naive tokenizer · R-09 stale handle · R-10 thread-per-request | **Fixed / removed on the live path** (FTS5 porter tokenizer, cosine collections, no module globals, sequential by design) |
| R-08 context builder · I-03 uncapped layout chunks · I-04 random ids and collection wipe · I-05 per-chunk encode | **Fixed** (`rag_core/context.py`; `split_to_fit_spans`; deterministic ids + `delete_by_doc`; `EMBED_BATCH = 32`) |
| L-04 hard-coded models/no backoff · L-05 prompt design · L-06 token counting · L-07 router · L-08 swallowed errors | **Fixed** (gateway, versioned prompts with `[S#]` citations, API `usage`, rules router, typed errors) |
| B-01 sync handlers · B-05 health · B-08 metrics labels · B-09 lifespan | **Moot or open**: no API on the live path; re-appears as N-01/N-09 and in the roadmap |
| B-03 `print` logging · B-04 import-time I/O · B-06 Redis `KEYS`/weak key · B-07 OTel misconfig | **Fixed on the live path**; legacy remains (L-04, L-08, L-10) |
| E-05 judge defaults to 0.5 · E-06 stale docs | **E-05 fixed** (failures become `None` and are counted, `run_ragas.py`); **E-06 open** (docs carry `UNVERIFIED` banners but still describe a FAISS + spaCy system) |
| I-06 loader spoofing · X-01 dead code | **Open** (L-13, L-14) |
| T-02 unused fixture | Not re-checked |

## 5. Evaluation-framework review (agent-assisted, then reconciled with the live path)

I delegated a read-only review of `eval/*.py` and the evaluation scripts and reconciled it with what I verified directly. Findings worth acting on:

| Area | Finding | Severity |
|---|---|---|
| Pre-registration | `eval/run_retrieval_eval.py:296-309` `preregistration_info` only checks the hash **if the hash file exists**; fail-open for non-registered runs. The hash file exists today (`eval/PREREGISTRATION.sha256`) | Low |
| Statistics | `eval/compare.py:62-64` the clustered bootstrap is **question-weighted** (`sums/counts` after resampling papers), so papers with many questions dominate; the p-value is a bootstrap sign-probability, not a null-centred test; the Holm family is "everything in one `compare()` call" rather than the pre-registered set (the TEST invocation did contain exactly the 3 pre-registered comparisons) | Medium (state the estimand) |
| Determinism | `eval/determinism.py:60-63` zips rows without checking counts; determinism is checked by a separate script, not enforced by the runner | Low |
| Generation judge | `eval/run_ragas.py:78-93` judge failures become `None` metrics and scoring continues: the run on disk scored **2 of 34** rows before the Gemini daily quota (HTTP 429) | Medium (n too small to claim anything; correctly *not* claimed) |
| Legacy | `eval/run_ablation.py:155-160` turns a failed configuration into all-zero metrics (a broken arm looks "worst" instead of invalid); `eval/run_eval.py`, `run_ablation.py`, `feature_contrib.py` are dead (0% covered) | High if run |
| Terminology | Legacy "Recall@5" = token-F1 ≥ 0.25 hit rate; the live runner reports HitRate/EvidenceRecall; `old_*` columns remain beside the new metrics | Low |

## 6. Static code-quality metrics **[M]** (AST scan, `06` Appendix A)

| Package | Files | LOC | Return annotated | Arg annotated | `print()` | broad `except Exception` |
|---|---|---|---|---|---|---|
| `rag_core` | 17 | 2,774 | 82% | 92% | **0** | 23 (those inspected are intentional fail-open, `# noqa: BLE001`) |
| `app` | 10 | 1,960 | 84% | 81% | 7 (`app/main.py`) | 12 |
| `rag` | 23 | 4,665 | 65% | 70% | 24 (legacy files) | 11 |
| `db` / `cache` / `observability` | 8 | 758 | 82% / 75% / **0%** | 53% / 100% / 0% | 0 | 5 |
| `eval` | 19 | 3,354 | 80% | 85% | 76 (CLI output) | 6 |
| **All 102 files** | 102 | 18,034 | **78%** | **81%** | 272 (164 in `scripts/`) | 70 |

`ruff` and `mypy` are **not installed** in the pinned venv, so lint and type-check status is **unknown** (`pyproject.toml` configures only `ruff`; there is no mypy/pyright config). Test coverage and the full list of 0% modules are in `06` §1.

## 7. What is good and should not be "fixed"

* `rag_core/gateway.py`: the retry / breaker / failover / mid-stream contract. Keep it; add N-06 and N-15.
* `rag_core/cache.py::make_cache_key` and `is_cacheable`: complete and conservative.
* `rag/ingestion.py` + `db/store.py`: deterministic ids, reverse-order batch upsert, ingest-signature refusal, post-run verification.
* `rag_core/prompts.py`: escaping of *all* untrusted text before templating, delimiter discipline, validation of fetched prompts. "Reduces the risk; does not remove it" is the right claim.
* `app/artifacts` + `eval/app_results.py`: a UI that can only render numbers a builder has traced to a run folder.
* The test harness: hermetic network guard, fakes that use the *real* SDK exception classes, Hypothesis profiles with pinned `@example`s.
