> **UNVERIFIED: written before the rebuild. Do not quote. See docs/CLAIMS_LOG.md.**

# 04 — Missing Features and Engineering Opportunities

Method: for every item I searched the whole repo (`grep` across `*.py`, config, docs) before declaring it absent. Where something exists, I cite it. Ratings: **Complexity** S (≤ 8 h) / M (8–24 h) / L (24 h+). **Worth it** is judged for a ₹35 LPA *portfolio* project, not for a company roadmap.

---

## Part 1 — Capability status by evaluation dimension (A–H)

Legend: ✅ present and sound · ⚠️ partial / defective · ❌ absent.

### A. RAG architecture

| Capability | Status | Evidence |
|---|---|---|
| Hybrid retrieval | ⚠️ | Implemented (`retrieval.py`), but corpus for BM25 questionable (R-02), unfiltered by metadata (R-05), strategies mixed (R-04). |
| Fusion strategy | ✅/⚠️ | Weighted RRF k=60 OK; not truncated (R-01); weights (1.0/0.7) are not tuned or justified by data. |
| Semantic chunking quality | ⚠️ | Boundary logic flawed, silent drop (I-01, I-02); gain over recursive not significant. |
| Reranking | ⚠️ | Config-gated, default off; unbounded input; benchmark model ≠ configured model; no latency budget. |
| Query transformation / multi-query / decomposition | ❌ | `grep -i "rewrite\|decompos\|multi.?query\|hyde"` → nothing. |
| Context compression | ❌ | Only dedupe + neighbour merge + token cut. |
| Duplicate handling | ⚠️ | Exact-text dedupe only; near-duplicates across strategies survive. |
| Long documents | ⚠️ | Per-page / per-section chunks unbounded; embedding window 256 tokens truncates. |
| Metadata filtering | ⚠️ | Vector leg only (Streamlit UI). Not exposed in API. |
| Incremental indexing / versioning / deletion | ❌ | uuid4 IDs, `clear_collection()` on every ingest; Chroma `delete`/`upsert` are never called (only `delete_collection`). BM25 not refreshed until restart. |
| Retrieval scalability | ❌ | In-memory `rank_bm25`, bulk `get()`. See scale table in `02_ARCHITECTURE_REVIEW.md` §4.4. |

### B. LLM engineering

| Capability | Status | Evidence |
|---|---|---|
| Model fallback | ⚠️ | Present but broken (L-01), mid-stream splice (L-02). |
| Retries / backoff / circuit breaker | ❌ | SDK `max_retries=1` only; eval script has Chroma backoff, production doesn't. |
| Timeouts | ⚠️ | Per-call numbers (15/30/60/120 s) but no end-to-end deadline. |
| Token management | ⚠️ | tiktoken for context; `split()` for output; no `max_tokens` on the main generation. |
| Prompt management | ❌ | One inline string, no versioning. |
| Structured outputs / validation | ❌ | Free text only; no citation schema. |
| Provider abstraction | ❌ | `groq_client` / `gemini_client` used directly. |
| Observability of LLM calls | ⚠️ | `OpenAIInstrumentor` (undeclared dependency); no token/cost/TTFT metrics. |
| Cost optimisation | ⚠️ | Exact-match cache only. No model routing, no semantic cache, no cost accounting. |
| Malformed-response handling | ❌ | Chunk parse errors swallowed (`except: continue`). |

### C. Production backend

| Capability | Status |
|---|---|
| FastAPI structure | ⚠️ single file, no routers/DI |
| Async | ❌ sync handlers and clients |
| Connection pooling | ✅ Redis pool (20); ❌ others default |
| Auth / rate limit / request validation | ❌ |
| Error handling | ⚠️ inconsistent (some swallowed, some 500) |
| Health checks | ⚠️ single `/` endpoint, no readiness |
| Background processing | ❌ |
| Graceful shutdown | ❌ |
| Config management | ⚠️ `config.yaml` + env + hard-coded constants; no validation (`SimpleNamespace`) |

**Can it handle concurrent users reliably?** Not today. Up to ~40 sync threads, 10 concurrent Chroma Cloud reads per collection (documented), per-worker RAM copy of BM25, no backpressure, no limits. It handles a handful of concurrent demo users.

### D. Performance and scalability — see the scale table (§4.4 of `02_ARCHITECTURE_REVIEW.md`). Gap: **no load test, no profiling, no capacity model**.

### E. Reliability and resilience — see the failure table (§5 of `02_ARCHITECTURE_REVIEW.md`). Gaps: breaker, backoff, degraded mode, idempotent ingestion, error-not-cached.

### F. LLM evaluation and observability

| Capability | Status | Notes |
|---|---|---|
| Retrieval metrics | ✅ Recall@5/10, MRR | lexical relevance (E-03) |
| Faithfulness / answer relevancy | ⚠️ | Home-grown 8B-judge prompts (noisy); RAGAS code exists but disabled (`ENABLE_RAGAS=False`) and RAGAS rows are "N/A" in results. |
| Context precision / recall | ❌ | |
| Hallucination / citation correctness | ❌ | No citation format to check. |
| Regression testing in CI | ❌ | |
| Online evaluation / user feedback | ❌ | No feedback endpoint. |
| Trace-level debugging | ⚠️ | OTel spans, disconnected (B-07). |
| Token and cost tracking | ❌ | |
| Latency breakdown | ✅ | Per-stage `latency_ms` returned and in Prometheus (retrieval/rerank/generation). |
| Dataset management | ⚠️ | 100-question JSON, seed recorded; no versioning, no hold-out, no adversarial / unanswerable set. |

**RAGAS / DeepEval?** Adds value *only* for answer-level metrics (faithfulness, answer relevancy) and only if you (1) calibrate the judge against ~30 human labels and (2) use a stronger judge than the generator. For retrieval, your own doc-id-based metrics are better than any framework. Recommendation: **custom retrieval eval + a small, calibrated LLM-judge module**; use RAGAS only as a cross-check, not as the headline number. DeepEval's main appeal is pytest-style CI assertions. You can get 80% of that with a 50-line pytest that loads a frozen eval set.

### G. AI safety and security

| Capability | Status |
|---|---|
| Prompt-injection protection | ❌ (S-02) |
| Malicious document handling | ❌ (no size/type sniffing, HTML-injection in UI S-03, front-matter metadata override I-06) |
| PII detection/redaction | ❌ (queries logged raw) |
| Input / output validation | ❌ / ❌ |
| Secret management | ⚠️ `.env` ignored by git ✅; no `.env.example`; keys read at import; no secret scanning. (I grepped for key patterns: **none committed**.) |
| AuthN / AuthZ / tenant isolation / rate limiting | ❌ |

Realistic impacts: cost-abuse via open endpoint (S-01); indirect prompt injection steering answers or exfiltrating other chunks (S-02); stored HTML injection in the UI (S-03); metadata spoofing that makes a malicious document cite as a trusted `source` (I-06); filter bypass leaking out-of-scope chunks (R-05).

### H. DevOps and deployment

Dockerfile ❌ · Compose ❌ · CI ❌ (badge is a placeholder) · tests ⚠️ (not hermetic) · lint ⚠️ (ruff config exists; not enforced) · type checking ❌ · dependency pinning ❌ (polluted freeze) · health checks ❌ · IaC ❌ · Prometheus config/Grafana ❌ · alerting ❌ · env separation ❌.

---

## Part 2 — The 22 candidate differentiators

Ordered as in your list. "**Verdict**" is the recommendation (✅ do · 🟡 do a *light* version · ❌ skip).

### 1. Adaptive retrieval by query complexity
* **Problem:** Cheap queries pay for reranking/expansion they don't need; hard ones don't get it.
* **Why it matters:** Latency/cost vs quality is the core production trade-off.
* **Current state:** Binary router only (`classify_query`: direct vs rag), not wired to the API.
* **Change:** Route to `{no-retrieval, vector-only, hybrid, hybrid+rerank}` using cheap features (length, entities, numbers, quoted terms) and *measured* per-route quality on your eval set.
* **Benefit:** Lower median latency at equal recall; a data-driven story. **Complexity:** M. **Verdict:** 🟡 *Only after* the eval is trustworthy; otherwise it's a buzzword.

### 2. Query rewriting and decomposition
* **Problem:** Context-free questions ("What dataset do they use?") and multi-hop questions retrieve poorly.
* **Current state:** None.
* **Change:** One LLM rewrite step (standalone question using chat history) + optional decomposition for "and/compare" queries; fuse sub-query results with RRF.
* **Benefit:** Real recall gains on multi-turn and compound questions; enables chat history, which Streamlit currently lacks. **Complexity:** M. **Verdict:** ✅ (high interview value if you *measure* it with an ablation).

### 3. Agentic retrieval
* **Problem:** Questions needing iterative lookup.
* **Current state:** None. **Complexity:** L. **Verdict:** ❌ Adds latency/non-determinism that your current eval can't validate. Mention as a deliberate non-goal.

### 4. Configurable fusion strategies
* **Current state:** RRF vs "none" only (`cfg.pipeline.fusion`), weights fixed 1.0/0.7.
* **Change:** Strategy registry: RRF, weighted RRF, normalised-score interpolation (min-max / z-score), vector-only. Tune weights on a *dev split* by grid search and report on a held-out test split.
* **Benefit:** Evidence for the 0.7 weight and for RRF over naive merge (replaces the buggy arm). **Complexity:** S. **Verdict:** ✅ cheap and valuable.

### 5. Dynamic reranking
* **Current state:** Static on/off; unbounded input; default off.
* **Change:** Rerank only top-20; skip when first-stage margin is high or when the latency budget is low; use a small cross-encoder (MiniLM-L-6 or bge-reranker-v2-m3 *only* if GPU). Measure actual latency (batch size, `torch.set_num_threads`, ONNX/int8).
* **Benefit:** Turns the "reranker is too slow on CPU" non-result into an engineering result. **Complexity:** S–M. **Verdict:** ✅ light version.

### 6. Context compression
* **Problem:** The 1,500-token budget wastes tokens on irrelevant sentences.
* **Change:** Sentence-level extractive filtering (score sentences vs query with the bi-encoder, keep top-N + neighbours). Skip LLM-based compression (cost, latency).
* **Benefit:** More relevant evidence per token. **Complexity:** M. **Verdict:** 🟡 optional.

### 7. Semantic caching
* **Problem:** Exact-match cache misses paraphrases.
* **Current state:** Exact normalised-string key only.
* **Change:** Embed the query (already done), look up nearest cached query in Chroma/Redis vector set with a high threshold (≥ 0.95) *and* identical retrieval config/index version; store citation IDs.
* **Risk:** Wrong-answer hits. Needs a measured false-hit rate. **Complexity:** M. **Verdict:** 🟡 only after cache keys include index/config version (B-06); present it with its false-positive analysis.

### 8. Multi-level caching
* **Current state:** Embedding LRU + Redis answer cache (not tiers).
* **Change:** Add a small in-process TTL cache in front of Redis for hot answers; cache retrieval results (query-embedding → candidate IDs) separately from answers so a prompt change doesn't invalidate retrieval.
* **Verdict:** 🟡 Light. Real value is correct invalidation (versioned keys), not more tiers.

### 9. Asynchronous document ingestion
* **Current state:** CLI scripts only; no upload endpoint.
* **Change:** `POST /v1/documents` → 202 + `job_id`; FastAPI `BackgroundTasks` (or a single worker process) runs parse → chunk → embed → upsert; `GET /v1/jobs/{id}`.
* **Benefit:** Completes the "Document Upload → Response" lifecycle you describe. **Complexity:** M. **Verdict:** ✅ **core**.

### 10. Background task processing (Celery/Kafka?)
* **Verdict:** ❌ Celery/Kafka. `BackgroundTasks` + a SQLite jobs table is enough for this scale. Introduce a queue only if you must survive restarts mid-job *and* show real concurrent ingest load. Be ready to explain this threshold.

### 11. Incremental indexing
* **Current state:** None (see I-04). **Change:** Deterministic IDs, `upsert`, per-doc delete, FTS5 incremental updates, `index_version` bump.
* **Benefit:** Removes restart-to-reindex and the wipe-everything ingest. **Complexity:** M. **Verdict:** ✅ **core**.

### 12. Document versioning
* **Change:** `doc_id`, `doc_version`, `content_sha256`, `active` flag; skip re-embedding unchanged content; atomically flip active version.
* **Complexity:** S–M (once 11 exists). **Verdict:** ✅ (cheap on top of 11).

### 13. Multi-tenancy
* **Current state:** None; and the BM25 filter bypass (R-05) would leak across tenants.
* **Change:** `tenant_id` metadata on every chunk and FTS row; API key → tenant; mandatory filter injected server-side; cache key includes tenant; isolation tests.
* **Complexity:** M. **Verdict:** 🟡 Do the *isolation tests* and the mandatory-filter plumbing. It is a strong security/system-design talking point. Skip tenant admin UIs.

### 14. User feedback and evaluation loops
* **Change:** `POST /v1/feedback {request_id, rating, comment}`; store request trace (query, chunk IDs, answer, config) in SQLite; weekly export of thumbs-down into the eval set.
* **Complexity:** S–M. **Verdict:** ✅ cheap and signals product thinking, but be honest that you have no real traffic.

### 15. A/B testing of retrieval configurations
* **Change:** Deterministic bucketing by hash(request_id) → `config_id`; log `config_id` in spans/metrics; offline replay of logged queries against both configs.
* **Verdict:** 🟡 Offline replay is realistic; "online A/B" without traffic is theatre.

### 16. Online RAG evaluation
* **Change:** Sample 1–5% of requests, run cheap reference-free checks (citation validity, answer-vs-context NLI/overlap, not-found rate) async and emit Prometheus gauges.
* **Verdict:** 🟡 Light version tied to citation validation.

### 17. LLM-as-a-judge
* **Current state:** Exists but unreliable (E-05).
* **Change:** Stronger judge than generator, rubric + JSON output, temperature 0, 2 samples, calibrate vs ~30 human labels (report agreement/Cohen's κ), exclude failures instead of defaulting to 0.5.
* **Verdict:** ✅ fix rather than add.

### 18. Hallucination and faithfulness checks
* **Change:** Enforce `[S#]` citations; verify each cited ID exists in the supplied context; sentence-level support check (cheap NLI model or judge) on a sample; "not found" path tested on unanswerable questions (QASPER has them; you filtered them out).
* **Benefit:** Turns "Do not hallucinate" in the prompt into a measurable property. **Verdict:** ✅ **high value**.

### 19. Model routing by complexity/cost
* **Change:** Tiny policy: short factoid + high retrieval margin → 8B; long/multi-hop → larger model; log cost per route.
* **Verdict:** 🟡 Only with cost accounting (see 20). Without data it's speculation.

### 20. Cost-aware retrieval and generation
* **Change:** Count prompt/completion tokens from API `usage`, price table in config, per-request `cost_usd` in span + Prometheus; budget guard (max context tokens by route).
* **Complexity:** S. **Verdict:** ✅ small, clearly real, great dashboard material.

### 21. Query-level tracing
* **Current state:** Partial and disconnected (B-07).
* **Change:** FastAPI instrumentation, thread context propagation, attributes (route, chunk IDs, tokens, cache, degraded). Store `trace_id` in the response header and feedback.
* **Verdict:** ✅ **core** (low effort, high visual payoff).

### 22. Load testing and capacity planning
* **Current state:** None. "QPM" is `60/latency`.
* **Change:** Locust/k6 scenarios (closed-loop concurrency 1→50; cache-hit mix; failure injection), record RPS, p50/p95/p99, error rate, TTFT, CPU/RAM; identify the knee (Chroma 10-read ceiling, thread pool, LLM rate limit).
* **Verdict:** ✅ **core**. This is the missing "performance & capacity testing" for Level 3.

### Summary of verdicts

| Do (core) | Do (light) | Skip |
|---|---|---|
| 9, 11, 12, 17, 18, 20, 21, 22, 2, 4 | 1, 5, 6, 7, 8, 13, 14, 15, 16, 19 | 3 (agentic), 10 (Celery/Kafka) |

---

## Part 3 — Things not on your list that matter more than most of the above

1. **A measured, correct baseline.** Run "recursive + vector-only" and "semantic + vector-only" and report them (right now one is estimated).
2. **A paper-scoped evaluation** (retrieve within `paper_id`) in addition to corpus-wide. It separates retrieval quality from question ambiguity.
3. **An unanswerable-question set** to test "I could not find that information".
4. **Embedding-model sanity check.** MiniLM-L6 is dated. A one-day comparison (e.g. `bge-small-en-v1.5`, `e5-small-v2`, `all-mpnet-base-v2`) is likely to move Recall more than any pipeline feature. It also fixes the 256-token truncation if you choose a longer-context model.
5. **A frozen dev/test split** so tuning (weights, thresholds, chunk size) doesn't overfit the 100 questions you report on.
