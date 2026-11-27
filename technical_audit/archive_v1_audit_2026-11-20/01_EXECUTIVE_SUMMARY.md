> **UNVERIFIED: written before the rebuild. Do not quote. See docs/CLAIMS_LOG.md.**

# 01 — Executive Summary

**Audit scope:** every Python module, config, tests, eval scripts, result CSVs, per-question checkpoints, README and `docs/`.
**Method:** static reading of all source, plus reproductions that did not touch application code (§5).
**Not done:** the app and test suite were **not executed**. Required dependencies (`torch`, `chromadb`, …) are not installed here and Chroma/Groq/Gemini credentials are not available. Anything that depends on live behaviour is labelled *unverified*.

---

## 1. Overall assessment

You have a **competent Level-2 project (lower half of the ₹25–30 LPA band)** with several real strengths and a few real problems:

* The strengths are an ablation mindset, a per-question checkpointed eval harness, a clean RRF implementation and thoughtful Redis failure handling.
* The problems are **evidence integrity** (headline numbers do not trace to raw data) and **correctness** (several silent bugs in the exact features the resume highlights).

An interviewer who clones the repo and spends 30 minutes will find the discrepancies listed in §3. Your docs invite them to ("verify the ablation").

> **Score: 5.0 / 10** for a ₹35 LPA+ portfolio. The self-assessment in `resume_bullets_v2.md` gave 7.5/10, including "Evaluation Rigor 9/10". I do not agree. The evaluation scaffolding is good, but its outputs contain an *estimated* baseline presented as measured, an ablation arm that is a bug, and a "throughput" number that is `60 / latency`.

| Dimension | Score | One-line reason |
|---|---|---|
| RAG architecture | 5.5 | Right ideas (hybrid, RRF, budgeting). Fusion output is not truncated, BM25 is in-memory over a possibly truncated corpus, and 3 chunking strategies are mixed in one collection. |
| LLM engineering | 3.5 | Gemini fallback is defective (reproduced), failures are cached, models are hard-coded, no retries/breaker, no structured output. |
| Backend engineering | 3 | Sync endpoints, no auth/rate-limit/limits, `print()` logging, import-time side effects, `KEYS` scans, no upload API. |
| Performance & scale | 4 | Fine at ~100 docs. Several O(N) or O(C×N) hot-path paths and a hard Chroma Cloud 10-concurrent-reads cap. |
| Reliability | 3.5 | Redis degradation is good. LLM, Chroma and startup failures are not handled. |
| Evaluation | 5.5 | Good structure and checkpoints, but the relevance metric is lexical, the eval path ≠ production path, and the judge is noisy. |
| Security / safety | 2 | No auth, destructive public endpoint, HTML injection in Streamlit, metadata spoofing via front-matter, no injection or PII handling. |
| DevOps | 2 | No Dockerfile, CI, `.env.example`. `requirements.txt` is a polluted `pip freeze` and is missing imported packages. |
| Test quality | 3 | About 35 tests, none for cache, API, context builder or loaders. One test is wrong by construction, and several need live cloud credentials at import. |
| Honesty of claims | 3 | See §3. This is the fastest dimension to fix and the highest-leverage one. |

## 2. Biggest strengths (keep and showcase)

1. **Ablation discipline.** The `evaluation/checkpoints/*.jsonl` per-question logs let me compute paired statistics. That is rare and valuable.
2. **Weighted RRF is correct and unit-tested** (`rag/retrieval.py::weighted_rrf`, `tests/unit/test_fusion.py`).
3. **The honest finding exists in your data:** Hybrid+RRF beats vector-only by **+6 pp Recall@5 (0.21 → 0.27)**. In a paired bootstrap this is 6 wins and 0 losses, 95% CI **[+2, +11] pp**. This is defensible. It is just not "+44%".
4. **Redis layer fails open** (`cache/cache.py` swallows errors, has timeouts, uses a connection pool).
5. **The pipeline is toggleable per call** (`retrieval_mode_override`, `use_rrf_override`, `reranker_override`), which is the right shape for A/B work.
6. **You already report a latency breakdown** (`latency_ms` dict) and Prometheus histograms.

## 3. Claims that are unsupported or misleading — do not put on a resume yet

| Claim | Verdict | Evidence |
|---|---|---|
| **+44% Recall@5** | **Unsupported.** | The 0.188 baseline is *derived*, not measured: `eval/feature_contrib.py` computes `rec_vec_est = rec.r5 × (base.r5 / prod.r5)` = 0.25 × 0.21 / 0.28 = 0.1875. No result file contains a measured "recursive + vector-only" run. Measured: 0.21 → 0.27 = **+28.6% relative / +6 pp**. |
| **P50 retrieval latency ≈ 1.7 s** | **Wrong metric.** | 1.680 s is the *mean end-to-end (retrieval + LLM)* for Hybrid+RRF in the 2026-06-19 run. Measured P50 *retrieval* is **0.505–0.551 s** (`observability_results.csv`; 0.537 s from checkpoints). The raw per-question data for the 06-19 run is not in the repo. |
| **CrossEncoder +12.6 s on CPU** | **Derived and mis-attributed.** | 12.6 = 14.284 − 1.680, a difference of two noisy mean end-to-end latencies. Directly measured reranker means are 11.714 s (06-19 CSV) and **2.994 s** (06-15 checkpoints, median 3.01). The benchmark reranker is `BAAI/bge-reranker-base` (`scripts/run_evaluation.py:RERANKER_MODEL`), **not** the `ms-marco-MiniLM-L-6-v2` that README, config and docs name. |
| **Streaming via SSE** | **Not SSE.** | `StreamingResponse(media_type="text/plain")` emits raw tokens. There is no `text/event-stream` and no `data:` framing. |
| **Gemini fallback** | **Broken.** | `config={"timeout": 30}` raises `ValidationError: Extra inputs are not permitted` on `google-genai==1.75.0` (your pin) and on 3.6.1 (latest). I reproduced this. The fallback therefore always ends in "Generation failed on both…". |
| **Two-tier cache (LRU + Redis)** | **Overstated.** | The LRU caches *query embeddings* and Redis caches *final answers*. Two different caches, not tiers of one. |
| **Throughput 4.2 / 14.3 QPM** | **Not a throughput measurement.** | `run_observability_eval` computes `60 / avg_total_latency` for a *sequential single stream*. This is the inverse of latency, not capacity. 14.3 QPM = 0.24 QPS. |
| **0% error rate** | **Hard-coded.** | `"error_rate": 0.0` literal. Generation errors become the string `"Generation failed."` and are scored (faithfulness defaults to 0.5 on judge failure). |
| **Cache hit < 50 ms** | **No evidence.** | No benchmark or test. `CACHE_HIT_LATENCY` only times the Redis GET and JSON decode. |
| **2.1 s → 1.7 s latency "improvement"** | **Unsupported.** | The 2.1 s baseline latency does not appear in any result file. |
| **Semantic chunking "+12% R@5 / +22% MRR"** | **Not statistically distinguishable.** | In the checkpoints, semantic vs recursive is 8 wins / 5 losses, 95% CI **[−4, +10] pp**. The docs also describe a spaCy sentence segmenter that does not exist in the code. |
| **Grafana / Phoenix** | **No artefacts.** | The auto-generated "resume summary" in `benchmark_summary.md` mentions them. No dashboards or configs exist. |
| **README P95 < 2.5 s** | **Ambiguous / contradicted.** | Hybrid+RRF end-to-end P95 is 4.91 s in the checkpoints. Retrieval-only P95 is 0.98 s. README doesn't say which it means. |

## 4. Biggest weaknesses (ranked)

1. **Evidence integrity** (above). Fix first. It costs the least and is the most damaging in an interview.
2. **The "hybrid" arm in your ablation is a bug.** `scripts/run_evaluation.py` caps the merged list to `candidate_k` *after* concatenating `vec + bm25`. The vector list is already `candidate_k` long, so BM25 never survives. Per-question R@5 and MRR are *identical* to vector-only (verified on all 100 questions). Your docs explain this as a score-scale problem. That is a post-hoc story for a slicing bug.
3. **Eval path ≠ production path.** The eval uses a separate `RAGRetriever` class: per-strategy BM25 corpus, BM25 top-k, a different reranker, and no context-builder step. The production `retrieve_and_build_context` does none of these. Benchmark numbers therefore do not describe the code that runs behind `/query-stream`.
4. **Production retrieval has silent correctness problems:**
   * Fused candidates are not truncated, so the reranker sees an unbounded list.
   * All three chunking strategies sit in one collection, with no default filter.
   * BM25 ignores `metadata_filter`.
   * The 300-result Chroma Cloud cap makes the `collection.get()` bulk load suspect.
5. **Recall metric is lexical and length-biased.** Relevance = unigram F1 ≥ 0.25 between gold evidence and the chunk. A 2000-char chunk that fully contains a 30-word evidence sentence scores about 0.18 and is called *irrelevant*. A short lexically similar chunk is called *relevant*. This favours BM25 and small chunks.
6. **Reliability and security basics are absent** (auth, rate limit, input caps, retries/backoff/circuit breaker, health vs readiness, graceful shutdown).
7. **No reproducible deployment.** There is no Dockerfile, compose file or CI, and `requirements.txt` does not install the app: `openinference-instrumentation-openai` and `langchain-huggingface` are imported but missing.

## 5. Independent reproductions (no app code modified)

| # | What | Result |
|---|---|---|
| 1 | Paired analysis of `evaluation/checkpoints/*.jsonl` (n=100, 5,000-sample paired bootstrap) | Vector → Hybrid(no RRF): **0 wins / 0 losses, identical**. Vector → Hybrid+RRF: **6 / 0, Δ=+0.060, CI [+0.020, +0.110]**. Hybrid+RRF → +Reranker: 9 / 8, Δ=+0.010, CI [−0.070, +0.090]. Recursive → Semantic: 8 / 5, CI [−0.040, +0.100]. Wilson 95% CI for R@5 = 0.27: [0.193, 0.364]. |
| 2 | Copy of `semantic_chunk` step 5 (small-chunk merge) run on a synthetic group list | 3,200 input chars → 2,950 output chars: **250 chars of content silently dropped**. |
| 3 | `GenerateContentConfig(timeout=30)` against pinned `google-genai==1.75.0` and 3.6.1 | `ValidationError … Extra inputs are not permitted`. |
| 4 | Latency outliers in raw checkpoints | Semantic-chunking run: mean total 191.7 s, **max 16,129 s**. Vector-only: mean 22.4 s, max 635 s. Means are dominated by a single outlier. Medians are 3–5 s. |

## 6. What prevents this from being a ₹35–50 LPA-quality portfolio project

| Level-3 criterion | Status |
|---|---|
| Evidence-backed engineering decisions | ❌ numbers don't trace to raw data; benchmark ≠ production |
| Measurable retrieval quality | ⚠️ measured, but lexical relevance and n=100 context-free questions (71/100 are ≤ 8 words, e.g. "What dataset do they use?" over a corpus-wide index) |
| Reliability / resilience | ❌ broken fallback, no retries/breaker, failures cached |
| Production backend | ❌ no auth, limits, upload API, async I/O, readiness |
| Advanced observability | ⚠️ metrics/tracing exist but are unlabeled and disconnected; no cost/token metrics |
| Performance & capacity testing | ❌ no load test; single-stream "QPM" only |
| Security & guardrails | ❌ none |
| Reproducible deployment | ❌ none |
| Scalability strategy | ❌ in-memory BM25 + `collection.get()` bulk load |

**The good news:** the gap is mostly *engineering follow-through*, not research. The five workstreams in [`05_IMPROVEMENT_ROADMAP.md`](05_IMPROVEMENT_ROADMAP.md) (about 150–190 hours) take this to a defensible 7.5–8/10.

## 7. Report index

| File | Purpose |
|---|---|
| [02_ARCHITECTURE_REVIEW.md](02_ARCHITECTURE_REVIEW.md) | Mermaid diagrams, claim verification matrix, component review, scale analysis, target architecture |
| [03_CODE_REVIEW.md](03_CODE_REVIEW.md) | Code-level findings with file/function/severity/fix |
| [04_MISSING_FEATURES.md](04_MISSING_FEATURES.md) | Gap analysis for the 22 features you listed |
| [05_IMPROVEMENT_ROADMAP.md](05_IMPROVEMENT_ROADMAP.md) | P0–P3 plan, effort, dependencies, top-5 ROI |
| [06_BENCHMARKING_STRATEGY.md](06_BENCHMARKING_STRATEGY.md) | Test/eval/load strategy and acceptance criteria |
| [07_INTERVIEW_READINESS.md](07_INTERVIEW_READINESS.md) | Hard questions, trade-offs, level placement |
