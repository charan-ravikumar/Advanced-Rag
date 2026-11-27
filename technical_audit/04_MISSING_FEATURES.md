# 04 — Missing Features and Capabilities

**Reading guide.** Status: ✅ implemented (with evidence) · 🟡 partial · ❌ missing. "Live" means reachable from the current Streamlit app (`app/`, `rag_core/`, `rag/retriever.py`). Legacy-only code counts as ❌ for the live system and is noted. Coverage figures are from the run in `01` §0. Where something looked absent I searched the whole tree (`Dockerfile*`, compose, `.github/`, `locust`/`k6`, `prometheus*`, `grafana`, `mypy`, `pre-commit` → no matches) before saying so.

## 1. Capability matrix (sections A–H of your brief)

### A. RAG architecture

| Capability | Status | Evidence / gap |
|---|---|---|
| Retrieval quality (measured) | ✅ | TEST (292 q, 124 papers): D nDCG@10 0.456, HitRate@5 0.682; reranked G 0.514. Corpus-wide (EXTENSION, DEV) is weak: D Hit@5 ≈ 0.18, so "search all papers" needs work |
| Hybrid (vector + BM25) | ✅ | `rag/retriever.py`; FTS5 `bm25()` (porter, unicode61) + Chroma cosine. Gain over vector-only not significant after Holm |
| Fusion strategy | ✅ | `fuse_rrf` weighted (k=60, 1.0/0.7), deterministic ties; `naive_interleave` baseline. Missing: score-normalised fusion arm |
| Semantic chunking quality | 🟡 | Implemented and property-tested, but boundary detector is flawed (N-07: 77.5% of windows truncated) and ties recursive on TEST |
| Context relevance | 🟡 | Relevance order kept, 5 chunks, 3,000-token *estimate* (`rag_core/context.py`). No neighbour expansion, no passage-level compression |
| Reranking | 🟡 | Pinned MiniLM cross-encoder works (+0.058 nDCG, significant) but **off by default**; +1.19 s p50 on CPU; no adaptive use |
| Query transformation / multi-query / decomposition | ❌ | None. Every question is independent; no conversation memory (`ui_chat.py` docstring) |
| Context compression | ❌ | Not needed yet: prompts average ≈ 1.1K tokens/call (EXPLORATORY run) |
| Duplicate retrieval handling | ✅ | `build_context` de-duplicates by `chunk_id`; one chunk found by both legs appears once with both ranks. Near-duplicate *text* across papers is not collapsed |
| Long-document handling | 🟡 | Section-aware recursive/semantic splitting with token cap; Qasper papers only. No hierarchical retrieval (section → paragraph) |
| Retrieval scalability | 🟡 | Persistent FTS5 + HNSW (no import-time corpus load). Lexical leg **measured** to 367K synthetic chunks (grows ~linearly; see `02` §7); vector leg not tested beyond 7.3K chunks |
| Metadata filtering | ✅ | `doc_ids` filter applied **before** ranking in both legs (filter parity, tested). Only `doc_id` is filterable |
| Incremental indexing | ✅ | Skip unchanged `doc_version`; replace changed docs via `delete_by_doc`; reverse-order upsert; manifest + verification (`rag/ingestion.py`, 97.2% covered) |
| Document versioning | 🟡 | `doc_version` hash stored; only the latest version is kept, no history or as-of queries |
| Deletion and re-indexing | ✅ | `delete_by_doc` on both stores; parity check afterwards. No public API for it |
| Document upload and parsing | ❌ | Ingestion accepts **Qasper-format JSON only** (`rag/chunk_adapter.py`). `rag/loaders.py` (PDF/DOCX/HTML/MD/XLSX) is legacy, 0% tested, unwired |

### B. LLM engineering

| Capability | Status | Evidence / gap |
|---|---|---|
| Model selection | 🟡 | `openai/gpt-oss-120b` verified against the live model list by `scripts/check_model_ids.py`, reasoning effort "low" recorded per call; not tuned (config says so). No cost/quality comparison across models |
| Fallback architecture | 🟡 | `LLMGateway` + `GeminiProvider` correct and tested with fakes; **disabled** (`providers: ["groq"]`); never exercised live in the answer path |
| Retries / backoff / jitter / `Retry-After` | ✅ | `rag_core/gateway.py`; 48 tests with a fake clock |
| Circuit breaker | ✅ | Per provider, half-open with one trial; N-06 notes one mis-classification |
| Timeout handling | 🟡 | Per-call httpx timeout 30 s, attempt deadline 90 s; no wall-clock cap during streaming (N-15) |
| Token management | 🟡 | Conservative *estimate* for the budget; real `usage` captured per call (prompt, completion, reasoning). No per-key token budgets |
| Prompt management | ✅ | Versioned, sha256-keyed, Langfuse copy validated against required substrings, local fallback, pin for evaluation |
| Structured outputs | ❌ | Free text with `[S#]` markers, parsed by regex. No JSON-schema / constrained decoding; the short-answer mode takes the first line |
| Streaming reliability | ✅ | Typed events; abstention sentinel held back; mid-stream error never spliced; partial answers surfaced |
| Context-window management | 🟡 | Fixed 5 chunks / 3,000-token estimate; truncates the first chunk rather than dropping it |
| Provider abstraction | ✅ | `Provider` protocol, two SDK wrappers, SDKs imported lazily |
| Model observability | 🟡 | Langfuse generation spans with usage, finish reason, attempts. Cost only if `prices` configured (it is `{}`) |
| Cost optimisation | ❌ | Daily request cap is a proxy, not a cost model; no semantic cache, no routing |
| Response validation | 🟡 | Citation labels checked against provided sources; abstention detected. **No support check** (a valid `[S1]` does not mean S1 supports the claim) |
| Malformed responses | ✅ | Empty text → typed failure; `length` truncation reported and never cached; lenticular `【S1】` brackets handled |

### C. Production backend

| Capability | Status | Evidence / gap |
|---|---|---|
| HTTP service (FastAPI) | ❌ | Live path has **no HTTP API** (deliberate: `PROJECT_CONTEXT.md`). Legacy `app/main.py` is dead |
| Async vs sync | ❌ | Fully synchronous; Streamlit thread per session; sync generators |
| Connection pooling / resource mgmt | 🟡 | One Chroma client, one SQLite connection, one lazily-built SDK client, all process-global |
| Concurrency | ❌ | One global retrieval lock (N-01); no concurrent test; capacity unknown |
| API design / validation | 🟡 | `validate_question` (type, printable, 500 chars) in `RagService`; no schema'd API |
| Authentication / authorisation | 🟡 | Optional shared access code (`hmac.compare_digest`); open "demo mode" when unset; no users, keys or roles |
| Rate limiting / abuse | 🟡 | Per-session limits + global daily cap; bypassable by reload (N-05) |
| Error handling | ✅ | Typed errors, degraded and partial results, never a stack trace in the UI |
| Health checks (liveness/readiness) | 🟡 | A start-up **self-check** screen (index, FTS5, parity, manifest, model cache, warm-up). No `/healthz`/`/readyz` |
| Background processing | ❌ | Ingestion is an offline CLI |
| Graceful shutdown | ❌ | Not applicable to Streamlit; needed for a service (drain streams, close SQLite) |
| Configuration | 🟡 | `config.yaml` + env allowlist, import-safe; unknown keys ignored (N-14) |
| **Can it serve concurrent users reliably?** | **No evidence** | A handful of users: probably yes (Groq stream dominates; retrieval p50 60 ms). Beyond that: unmeasured and structurally serialised |

### D. Performance and scalability: see `02` §7 for the measured and modelled analysis

### E. Reliability and resilience

| Capability | Status | Evidence / gap |
|---|---|---|
| Retry policy / exponential backoff | ✅ | Gateway only; retrieval has none (a failure → degraded "search unavailable" result) |
| Circuit breaker | ✅ | LLM only; none for Chroma/SQLite/Langfuse (see N-13) |
| Timeout propagation | 🟡 | Per-provider; none on the retrieval leg or the lock |
| Fallback / partial-failure handling | ✅ | `degraded`, `partial`, `abstained`, passages-only budget mode |
| Redis outage | n/a live | Live path has no Redis. Legacy `cache/cache.py` fails open (good) but is untested |
| Vector-store outage | 🟡 | `retrieval_failed` typed event; no retry, no read-only replica |
| LLM-provider outage | ✅/🟡 | Breaker + degraded notice; single provider so no failover in practice |
| Idempotency | 🟡 | Ingestion idempotent by `doc_version`; no request idempotency keys (not needed without writes) |
| Graceful degradation | ✅ | Daily budget → retrieved passages only; Langfuse down → local prompt + no tracing |

### F. LLM evaluation and observability

| Capability | Status | Evidence / gap |
|---|---|---|
| Retrieval metrics | ✅ | HitRate, MRR, nDCG@10, EvidenceRecall/F1, span relevance; paired clustered bootstrap + Holm |
| Faithfulness | 🟡 | RAGAS adapter + Gemini judge exists; **2 of 34 rows scored** (quota 429), uncalibrated, so no claim |
| Answer relevancy | 🟡 | Same as above |
| Context precision / recall | ❌/🟡 | Not computed for the *used* context. **Free win**: `eval/gold_spans.py` already locates gold evidence, so context precision/recall of `used_sources` can be computed deterministically |
| Hallucination detection | ❌ | No NLI/entailment check; no false-answer rate on unanswerable questions (N-02) |
| Citation correctness | 🟡 | Validity (label exists) 50/50 on n=40; **correctness (supports the claim) not measured** |
| Regression testing / CI gating | ❌ | No CI; a quality regression would not fail anything |
| Online evaluation | ❌ | Traces deliberately not judged (privacy, quota) |
| User feedback | 🟡 | Thumbs → Langfuse score, dropped when untraced (N-16) |
| Trace-level debugging | ✅ | Langfuse span tree (≤ 8 observations), prompt version + sha, attempts, finish reason |
| Token and cost tracking | 🟡 | Tokens yes (ledger, per-call usage); cost no (`prices: {}`) |
| Latency breakdown | ✅ | Per stage (embed/vector/lexical/fuse/hydrate/rerank) in eval and traces; no metrics export |
| Dataset management | ✅ | Versioned golden set (`version`, input SHA256), dev/test split by paper, test ledger, freeze files |

### G. AI safety and security

| Capability | Status | Evidence / gap |
|---|---|---|
| Prompt-injection protection | 🟡 | Escape all untrusted text, `<source>` delimiters, "data not instructions" rule, tests with adversarial values. No classifier, no output filter; "reduces, does not remove" |
| Malicious-document handling | ❌ | No upload path. Legacy loaders have no size/type/zip-bomb limits; front-matter can overwrite `source` (L-13) |
| PII detection / redaction | 🟡 | Masking of keys, emails, URLs, long numbers **only in what is sent to Langfuse**. No PII scan of documents or answers |
| Input validation | ✅ | Question type/length/printable |
| Output validation | 🟡 | Citation + abstention checks only |
| Secret management | 🟡 | Allowlist loader, keys never logged/rendered (tests assert it); `.env` sits in a synced folder (N-11) |
| API authentication / authorisation | 🟡/❌ | See C |
| Tenant isolation | ❌ | None. The filter-parity design makes it cheap to add (see §3 #13) |
| Data leakage | 🟡 | Question text and masked metadata go to Langfuse (a third party); fine for public Qasper text, not for private documents |
| Abuse / rate limiting | 🟡 | N-05 |

### H. DevOps and deployment

| Capability | Status | Evidence / gap |
|---|---|---|
| Dockerfile / compose | ❌ | None |
| CI/CD, GitHub Actions | ❌ | None (README badge is a placeholder) |
| Automated testing | ✅ (locally) | 1,332 tests, 252 s; not in CI |
| Linting | 🟡 | `ruff` configured in `pyproject.toml`; **not installed** in the venv; never run |
| Type checking | ❌ | No mypy/pyright; 78% return / 81% argument annotation |
| Dependency pinning | 🟡 | `docs/env_baseline.txt` is a true freeze; `requirements.txt` is polluted; `pyproject.toml` loose |
| Container health checks / IaC | ❌ | None |
| Prometheus / Grafana / alerting | ❌ | Legacy metric definitions only; no scrape config, dashboards or alert rules |
| Logging infrastructure | ❌ | No structured logs on the live path |
| Environment separation | 🟡 | `prompts.mode` local/langfuse; test vs eval vs app env discipline is strong; no dev/stage/prod config |

## 2. Gap summary

**Missing (❌)**: HTTP service · async · upload/parse · background ingestion · structured output · query transformation · tenant isolation · CI/CD · containers · metrics/alerts · structured logging · type-check gate · regression gate · hallucination and support checks · context precision/recall · load test.

**Partial (🟡) and closest to done**: reranking (works, not shipped) · failover (built, disabled) · feedback (built, drops untraced) · faithfulness (built, 2/34) · cost tracking (tokens yes, prices no) · rate limiting (per session only) · versioning (hash, no history) · health (self-check, no endpoints).

## 3. Advanced features: what each would change, and whether it is worth it

Verdict key: **DO** = high return for this portfolio · **LATER** = valuable only after a prerequisite · **SKIP** = adds complexity without proportionate evidence. Effort is engineering hours for a strong implementation including tests and an evaluation.

| # | Feature | Problem it solves / why it matters | Already addressed? (evidence) | What must change in *this* code | Effort | Verdict |
|---|---|---|---|---|---|---|
| 1 | Adaptive retrieval by query complexity | One fixed pipeline wastes latency on easy questions and under-serves hard ones; cost control | 🟡 per-call knobs (`mode`, `rerank`, `top_k`), a UI toggle; no automatic choice | A `plan_query()` in `RagService` using cheap signals (first-pass RRF score margin, question length, numerals, "compare/why" cues) to enable rerank and widen `candidate_k`; evaluate as a cost-quality curve (rerank = +1.19 s for +0.058 nDCG) | 10–14 | **DO.** Monetises your best measured result under a latency budget |
| 2 | Query rewriting / decomposition | Short, under-specified, follow-up and multi-hop questions retrieve badly; corpus-wide Hit@5 ≈ 0.18 (DEV) | ❌ | LLM rewrite behind a cache and flag; condense follow-ups once chat memory exists; test on the corpus-wide mode | 12–16 | **LATER.** Only if you add chat memory or "search all papers" as a headline mode |
| 3 | Agentic retrieval | Multi-step tool use for multi-hop questions | ❌ | Bounded plan-act loop | 20–30 | **SKIP.** No multi-hop eval data; latency, cost and eval complexity explode; reads as a buzzword |
| 4 | Hybrid search with configurable fusion | Different corpora favour different fusion | ✅ `fusion ∈ {rrf, naive}`, `rrf_k`, weights; arms C, D, R20_*, R60_* on dev | Optionally a convex-combination arm via a *new* pre-registration | 6–8 | **SKIP** (done). Do not re-open |
| 5 | Dynamic reranking | Reranker cost is only worth paying when ranking is uncertain | 🟡 on/off only; `rerank_k` fixed at 20 | Early-exit on score margin; ONNX/int8 cross-encoder; batch size tuning; cap candidates | 10–16 | **DO** (merge with #1) |
| 6 | Context compression | Cost and "lost in the middle" | ❌; prompts ≈ 1.1K tokens/call | Extractive sentence selection | 10–14 | **SKIP.** No evidence of need; adds latency |
| 7 | Semantic caching | Paraphrases miss an exact-match cache | ❌ exact normalised key | Embedding-similarity lookup with a strict threshold; evaluate false-hit rate on paraphrase pairs | 12–16 | **LATER.** Needs traffic; a wrong cache hit is worse than a miss |
| 8 | Multi-level caching | Share hits across replicas; skip embed + search for repeated retrieval | 🟡 single in-process TTL-LRU, rich key | `RedisCache` behind the existing `CacheBackend` protocol; retrieval-result cache keyed by (query, filter, index version) | 8–12 | **LATER.** Add Redis when a 2nd replica exists; the justification is then concrete |
| 9 | Asynchronous document ingestion | Users add documents; embedding is CPU-heavy (measured below) | ❌ CLI only | `POST /v1/documents` → job row in SQLite → worker process → status endpoint; idempotent via `doc_version`; file validation and size limits | 20–28 | **DO (P2).** Fills the "upload → parse → index" lifecycle gap. **No Celery/Kafka needed** |
| 10 | Background task processing | Keep the API responsive during ingestion | ❌ | Same as #9 (a separate worker process, not `BackgroundTasks`) | (in #9) | **DO** (folded into #9) |
| 11 | Incremental indexing | Re-embedding everything on each change is O(N) | ✅ `ingest_corpus`, 97.2% covered | A demo script and a metric: "re-ingest after editing 1 of 200 papers touched n chunks" | 2–3 | **DONE.** Just demonstrate it |
| 12 | Document versioning | Audit trails and as-of answers | 🟡 `doc_version` hash, latest only | Keep old chunks with `valid_to`, filter by version, show version in provenance | 10–14 | **LATER** (only if the target domain needs audit) |
| 13 | Multi-tenancy | Isolation is the first enterprise question | ❌; but `doc_ids` pre-filter parity exists and is tested | `tenant_id` metadata on chunk + FTS5, key→tenant map, **mandatory** server-side filter at the `Retriever` boundary, per-tenant cache key and quota, a property test "no result from another tenant" | 16–24 | **DO (P2).** High interview value, cheap *because* of filter parity |
| 14 | Feedback and evaluation loop | Production signal into regression data | 🟡 thumbs → Langfuse, dropped if untraced | Local feedback table; export thumbs-down questions to a review queue → candidate regression set | 8–12 | **DO (P2)** |
| 15 | A/B testing of retrieval configs | Choose between configs on real traffic | ✅ *offline* pre-registered arms | Request-level deterministic bucketing + per-arm metrics; or **shadow mode** (run arm B silently, compare) | 10–14 | **LATER.** Without traffic it is theatre; shadow mode is the honest version |
| 16 | Online RAG evaluation | Detect drift in production | ❌ | Cheap heuristic metrics (citation-validity rate, abstention rate, empty-retrieval rate, latency SLO, feedback) exported as metrics; no online LLM-judge | 10–14 | **DO the cheap metrics** (in #5 of the top-5); skip online judging |
| 17 | LLM-as-judge evaluation | Scale beyond exact-match metrics | 🟡 Gemini judge + custom RAGAS adapter; n=2 scored; uncalibrated | Hand-label 50–100 answers, report agreement (κ) per metric, finish n ≥ 100 within quota | 14–20 | **DO.** Turns "exploratory" into "claimable" |
| 18 | Hallucination / faithfulness checks | Support, not just citation validity | ❌ runtime; 🟡 offline RAGAS | Offline: NLI/cross-encoder entailment of each cited sentence vs its source; false-answer rate on the 48 unanswerable questions; runtime flag later | 16–24 | **DO offline**; runtime check **LATER** (latency) |
| 19 | Model routing by complexity/cost | Cheaper model for easy questions | ❌ single model; gateway only does failover | Needs #1's complexity signal, configured prices, and measured quality per question type (extractive F1 0.547 vs abstractive 0.166 vs yes/no 0.167, n=25/9/6, EXPLORATORY) | 12–18 | **LATER.** Without a measured quality delta it is a buzzword |
| 20 | Cost-aware retrieval and generation | Make $/request visible and budgeted | 🟡 tokens recorded; `prices: {}` | Configure list prices (source + date), cost per request in logs and metrics, per-key budgets | 4–6 | **DO (cheap).** Label as "shadow cost" while on the free tier |
| 21 | Query-level tracing | Debug one request end to end | ✅ Langfuse span tree, masked by default, 97.5% covered | Add OTLP export option + W3C `traceparent` once an API exists | 8–12 | **DONE;** extend with the API |
| 22 | Load testing and capacity planning | "How many users?" is unanswerable today | ❌ (the legacy "14.3 QPM" is 1/latency) | Locust scenarios (fake LLM to isolate retrieval; real LLM for e2e), open-loop arrivals, SLOs, capacity model | 12–18 | **DO (top 5)** |

## 4. Should you adopt RAGAS, DeepEval or stay custom?

* **Keep your custom pipeline as the backbone.** Its integrity controls (pre-registration, split by paper, ledger, bootstrap, fingerprints) are the differentiator, and no framework provides them.
* **RAGAS: keep, but narrowly.** Faithfulness and answer relevancy are recognised names an interviewer expects. Today's reality is an uncalibrated Gemini judge that scored 2 of 34 answers before the quota ended. Do not expand it. Calibrate it on 50–100 human labels, report agreement, and cap claims at what that supports.
* **DeepEval: do not add.** It would duplicate RAGAS, add a dependency and another judge. Its one real benefit, pytest-style assertions for CI, takes about 50 lines on top of what you have (a frozen mini-set, a fake provider, thresholds).
* **Cheaper and more reliable than another LLM judge:** compute **context precision and recall of the *used* sources deterministically** from `gold_spans` (you already locate gold evidence), **false-answer / false-abstain rate** on the unanswerable set, and **citation support** with a local entailment model. These are free, repeatable and quota-proof.

## 5. What not to build (complexity without proportionate value)

Kubernetes, Kafka, Celery, Elasticsearch/OpenSearch, a second vector database, an agent framework, GraphRAG, fine-tuned embeddings, or multi-LLM debate. Each needs a measured trigger that you do not have: a second replica (Redis), more than a few million chunks or concurrent writers (a dedicated search engine), real traffic (A/B, semantic cache), multi-hop evaluation data (agents). Build the measurement first; the trigger will tell you which one is justified.
