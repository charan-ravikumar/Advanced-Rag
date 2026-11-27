> **UNVERIFIED: written before the rebuild. Do not quote. See docs/CLAIMS_LOG.md.**

# 06 — Benchmarking and Testing Strategy

Goal: every number you publish is (a) produced by the **same code path the API runs**, (b) accompanied by uncertainty, and (c) traceable to a committed artefact (config + git SHA + per-question output).

---

## 1. What exists today and why it's not enough

| Asset | Useful for | Limitation |
|---|---|---|
| `evaluation/checkpoints/*.jsonl` | Paired analysis (I used it for the audit) | Only the 06-15 run; per-question rows lack config, SHA, retrieved chunk IDs |
| `scripts/run_evaluation.py` | Resume-able, multi-arm harness | Separate retriever (≠ prod), buggy hybrid arm (E-01), lexical relevance (E-03), noisy judge (E-05), writes README (E-04) |
| `eval/run_eval.py`, `eval/run_ablation.py` | Uses production retrieval | Broken dataset path; `chunking` override is a no-op; reranker may be `None` |
| `tests/unit` (~35) | RRF + regex-router + chunker smoke | Not hermetic; one wrong assertion; no cache/API/context tests |

---

## 2. Unit testing strategy (hermetic, < 30 s, no network, no model downloads)

**Rules:** no import-time I/O (requires the `Retriever`/`LLMGateway` refactor, B-04); fake the embedder with a deterministic hash-based vectoriser; use real `schemas` objects, not `MagicMock`.

| Area | Tests | Bugs they would have caught |
|---|---|---|
| Chunkers | Property: normalised concatenation of chunks contains every input paragraph (Hypothesis); `max_chunk_chars` respected; no empty chunks; deterministic output; chunk token length ≤ embedder window | I-01, I-02, I-03 |
| RRF | Exact scores for a hand-computed example (k=60, weights 1/0.7); ties are deterministic; empty lists; list truncation; monotonicity (adding a doc to a list never lowers others' scores) | regression of E-01 semantics |
| Retriever | Fake vector store + fake lexical index: (a) BM25-only top hit survives fusion; (b) fused list ≤ `rerank_k`; (c) metadata filter excludes docs from **both** legs; (d) same doc from two strategies dedupes | R-01, R-04, R-05 |
| Context builder | Relevance order preserved; oversize first chunk doesn't produce empty context; budget respected using the same tokenizer as prod; `used_chunk_ids` correct | R-08 |
| Router | Table test: "BERT architecture" → rag; "hi" → direct; LLM failure → rag; LLM path stubbed | L-07 |
| LLM gateway | Fake provider scripted with sequences: `429, 200`; `timeout ×3 → breaker open`; `fail before first token → failover`; `fail after first token → typed error, no splice`; `GenerateContentConfig(**cfg)` constructs | L-01, L-02, L-04 |
| Cache | `fakeredis`: hit/miss; TTL; **errors/not-found never cached**; key varies by config/index version/tenant; Redis exception → miss; no `KEYS` | L-03, B-06 |
| API | `TestClient` with fakes: validation (length, empty), auth 401/403, rate-limit 429, SSE framing (`event: token … event: done`), error event on provider failure | S-01, B-02 |
| Loaders | Front-matter cannot override `source`; HTML with nested divs yields text; oversize file rejected | I-06 |

**Targets:** line coverage ≥ 75% on `rag/`, `llm/`, `cache/`, `app/`; mutation spot-check on `weighted_rrf` and the chunker merge.
**CI:** `ruff check`, `mypy` (strict on new packages), `pytest -q --cov --cov-fail-under=75`.

## 3. Integration testing strategy

Run in compose or with `testcontainers` (Redis) and a **local Chroma** (`chromadb.PersistentClient` in a tmp dir; same API as Cloud for query/upsert, so the Cloud-specific limits are covered separately by contract tests).

| Scenario | Assertion |
|---|---|
| Ingest 20 fixture docs → query | Top-1 doc id correct for 10 hand-written questions |
| Re-ingest unchanged doc | 0 new embeddings; chunk count unchanged (idempotency) |
| New version of a doc | Old chunks gone, new present, cache invalidated (`index_version`) |
| Delete doc | No chunk of that doc retrievable by vector **or** lexical leg |
| Tenant A vs B | Zero cross-tenant results for 50 adversarial queries |
| Chroma 5xx injected (toxiproxy or stub) | Retry → FTS-only degraded response with `degraded=true`; no 500 |
| Redis down | Requests succeed; added latency ≤ socket timeout; breaker opens |
| Provider A down | Failover to B; metrics `llm_failover_total` increments |
| Provider fails mid-stream | Client receives `error` event; nothing cached |
| Chroma Cloud contract (nightly, optional) | Pagination at 300, write batches at 300, doc ≤ 16 KB |

## 4. RAG evaluation (offline)

### 4.1 Dataset management
* Keep QASPER, but add a **frozen split**: 50 dev (tuning), 100 test (reporting; never used for tuning), + 30 unanswerable questions (QASPER `unanswerable=True`, currently filtered out) + 20 red-team cases (prompt injection documents, filter-bypass attempts).
* Version it: `evaluation/datasets/<name>@<sha256>.jsonl`; each record has `{question, paper_id, gold_doc_id, gold_spans: [{char_start,char_end}], answer, answerable}`.
* Report two regimes:
  1. **Paper-scoped** (filter by `paper_id`): isolates retrieval/ranking inside a document.
  2. **Corpus-wide**: realistic but needs paper-qualified or rewritten questions; 71/100 current questions are ≤ 8 words and context-dependent.

### 4.2 Metrics
| Layer | Metric | Notes |
|---|---|---|
| Retrieval | Recall@{1,5,10}, MRR, nDCG@10 | Relevance = chunk overlaps a gold span (≥ 50% of the span or ≥ 50% of chunk tokens inside the span). Length-neutral. |
| Context | **Context recall** (gold span inside the text actually sent to the LLM), context precision (fraction of context tokens inside gold spans) | Captures the context-builder loss that Recall@5 hides (R-08). |
| Generation | Answer correctness (token-F1 + judge), **faithfulness** (claims supported by cited context), **citation validity** (cited ids ∈ provided ids), **abstention accuracy** on unanswerables | Judge calibrated on 30 human labels; report agreement. |
| System | Latency per stage (p50/p95/p99), TTFT, tokens, $ / query, cache hit rate | From traces, not from wall-clock `time.time()` around everything. |

### 4.3 Statistics
* Report each metric with a **95% paired-bootstrap CI** (5,000 resamples, seed fixed) and the *win/loss counts* between configs. (For reference, on your current data: Hybrid+RRF vs vector-only is 6/0 with CI [+2, +11] pp; reranker and semantic-vs-recursive are not distinguishable.)
* Call a difference "real" only if the CI excludes 0 **on the test split**.
* For > 3 comparisons, state the number of comparisons (or apply Holm correction).

### 4.4 Ablation matrix (all through the production `Retriever`)
| Arm | Chunking | Retrieval | Fusion | Rerank |
|---|---|---|---|---|
| A | recursive | vector | – | – |
| B | semantic | vector | – | – |
| C | semantic | hybrid | naive interleave | – |
| D | semantic | hybrid | RRF (1.0/0.7) | – |
| E | semantic | hybrid | RRF (tuned on dev) | – |
| F | semantic | hybrid | RRF | MiniLM top-20 |
| G | semantic | hybrid | RRF | bge-reranker-base top-20 (GPU optional) |
Add rows for embedder swaps (P2-3) only after A–G are clean.

### 4.5 Regression gate (CI)
A 30-question frozen "smoke" set with a local Chroma fixture. Fail the PR if Recall@5 drops by more than 3 pp or p50 retrieval latency regresses by > 25% (tolerances calibrated from three repeated runs).

## 5. Load and performance testing

### 5.1 Tools and modes
* **Locust** (Python, fits your stack) or **k6**.
* Two modes: **(a) stubbed LLM** (fixed 300 ms TTFT, 20 tok/s) to isolate retrieval, cache and API; **(b) real LLM** at low concurrency to measure true TTFT/E2E (watch provider rate limits; label results as provider-dependent).

### 5.2 Scenarios
| Scenario | Shape | Question answered |
|---|---|---|
| Baseline | 1 user, 200 requests | Clean per-stage latency distribution (warm models, cache off) |
| Step load | 1→5→10→20→40→80 users, 3 min each, closed loop | Where is the knee? (thread pool = 40, Chroma Cloud 10 concurrent reads/collection, CPU) |
| Cache mix | 70% repeat / 30% unique | Cache hit latency end-to-end (not just the Redis GET), miss/hit ratio effect on p95 |
| Spike | 5→60 users in 10 s | Rate-limit/backpressure behaviour: 429s, not timeouts or 500s |
| Soak | 20 users, 60 min | Memory growth (embedding LRU, Python object churn), connection leaks |
| Fault injection | Kill Redis; add 2 s latency to Chroma; Groq returns 429 | Graceful degradation and recovery time |
| Ingestion under load | Upload 20 docs while 20 users query | Query p95 stays within budget; no stale/partial reads |

### 5.3 Measurements to record per run
RPS (achieved), p50/p95/p99 E2E and **per stage**, TTFT, error rate by class, 429 rate, CPU %, RSS MB, Chroma call count/latency, Redis latency, tokens/s, cost. Always include **hardware and Python/library versions**.

### 5.4 Methodology rules (avoid the current pitfalls)
1. **Units:** state QPS or QPM explicitly; compute from `completed_requests / wall_clock_seconds` at fixed concurrency, never `60 / latency` (which equals throughput only for concurrency = 1).
2. **Warm-up:** discard the first N requests (model load, connection setup).
3. **Percentiles, not means**; also publish the max and investigate outliers (your data has a 16,129 s sample that destroyed a mean).
4. **No sleeps inside timed sections** (the eval harness sleeps 1.5 s between Groq calls; keep that out of reported latency).
5. **Separate network/provider latency from your own code:** report retrieval-only and E2E separately.
6. **Repeat 3×** and report the range; fail the benchmark if run-to-run CV exceeds ~15%.

## 6. Performance metrics (SLOs to define and then verify)

| SLI | Proposed SLO (tune to your measurements) | Source |
|---|---|---|
| Retrieval p95 (hybrid, no rerank) | ≤ 800 ms | `rag_stage_latency_seconds{stage="retrieval"}` |
| Rerank p95 (top-20, MiniLM, CPU) | measure, then set | stage histogram |
| TTFT p95 (cache miss) | ≤ 2.5 s with provider healthy | `llm_ttft_seconds` |
| Cache-hit E2E p95 | ≤ 100 ms (stubbed) | load test |
| Error rate (5xx) | < 1% at ≤ capacity knee | `rag_request_total{outcome}` |
| Availability of degraded mode | Chroma outage → 100% responses with `degraded=true` | chaos test |
| Cost/query | report; guard ≤ configured budget | `llm_cost_usd_total` |

## 7. Acceptance criteria (definition of done for "defensible")

**Correctness**
- [ ] Chunker property tests pass (no content loss; sizes within embedder window).
- [ ] Fused candidate list ≤ `rerank_k`; BM25-only top hit survives fusion (test).
- [ ] Metadata/tenant filters apply to both retrieval legs (test).
- [ ] Gemini/secondary provider config constructs and fails over (test + chaos).
- [ ] Errors and "not found" responses are never cached (test).

**Evaluation**
- [ ] Single harness calls the production `Retriever`; every run stored with config + git SHA + per-question JSONL.
- [ ] All reported metrics have 95% CIs and win/loss counts; dev/test split respected.
- [ ] Baseline arm (A) is *measured*; no extrapolated numbers anywhere in README/docs/resume.
- [ ] Judge agreement with human labels reported (κ or % agreement) on ≥ 30 samples.

**Performance**
- [ ] Load-test report with step-load table and the identified bottleneck; QPS defined as completed/wall-clock at fixed concurrency.
- [ ] Cache-hit latency measured end-to-end; p50/p95 published.
- [ ] Reranker latency measured for the *configured* model on stated hardware.

**Operations**
- [ ] `docker compose up` brings up api + redis + otel + prometheus + grafana; `/readyz` reflects dependencies.
- [ ] CI green: lint, types, hermetic tests, retrieval regression gate.
- [ ] One trace screenshot showing root → retrieval → fuse → rerank → LLM spans with token/cost attributes.

**Security**
- [ ] Auth + rate limiting verified by tests; query length capped; Streamlit escapes untrusted text; red-team set passes at a stated rate.
