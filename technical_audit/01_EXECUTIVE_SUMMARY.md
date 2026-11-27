# 01 — Executive Summary

**Audit date:** 2026-11-23 · **Audited code fingerprint:** `d76f6665d2431c0a9efeba914d357f704ae658516553c0912754bc889b8d8188` (`python -m utils.fingerprint`; the pre-registered TEST run was produced by an earlier fingerprint, `3afc204d…`, which is how `docs/CLAIMS_LOG.md` correctly records it)
**Supersedes:** the audit of the *v1* code that used to live here. It is preserved unchanged in [`archive_v1_audit_2026-11-20/`](archive_v1_audit_2026-11-20/). Section 7 maps every v1 finding to its current status.

## 0. What I did, and what I did not do

| Done | Not done (and why) |
|---|---|
| Read the code of every live module (`rag/`, `rag_core/`, `app/`, `db/`, `eval/`) and every legacy module that your claims point at | No live call to Groq, Gemini, Langfuse or Chroma Cloud (CLAUDE.md rules 1, 3, 10, 22). `.env` was never opened |
| Ran the suite: `pytest --fast -p no:cacheprovider -q -rs --cov=…` → **1263 passed, 69 skipped, 0 failed in 251.8 s**; coverage measured | Did not run the full (non-`--fast`) suite, `scripts/run_evaluation.py`, or any TEST-split analysis beyond the pre-registered comparisons already on disk |
| Recomputed the legacy numbers from the raw v1 checkpoints; reproduced the TEST-run table from `runs/eval/…/summary.json` | No application code, config or doc was modified. The only filesystem change is moving the old audit into the archive folder |
| Ran four small read-only measurements (cache-hit path, semantic-window truncation, FTS5 scaling, embedding throughput). They are labelled **AUDIT MEASUREMENT** everywhere and reproduced in `06` Appendix A | Not measured, stated as such: vector-search scaling beyond 7K chunks, concurrent-user throughput (no load test exists), real Groq latency |

## 1. Overall assessment

You have **two projects in one folder**:

1. **The rebuilt core** (`rag/retriever.py`, `rag/lexical.py`, `rag/ingestion.py`, `rag_core/*`, `eval/*`, `app/*`): dependency-injected, import-safe, deterministic, and unusually well tested (live-path statement coverage **93.6%**, `rag_core` 96.8%). Its evaluation process (pre-registration with a hash check, paper-level dev/test split, one-shot test ledger, paper-clustered paired bootstrap with Holm correction, code fingerprints, a claims log, an artifact builder that refuses unsourced numbers) is genuinely senior-level and rare in portfolios.
2. **The v1 serving layer** (`app/main.py` FastAPI, `cache/cache.py` Redis, `observability/metrics.py` Prometheus, `rag/retrieval.py`, root `streamlit_app.py`): this is where **every claim on your list originates** (`resume_bullets_v2.md` §8). It is deprecated, **0% tested, and cannot be imported in your own venv** (`openinference` and `langchain_huggingface` are not installed; `rag/retrieval.py` also builds clients and a whole-corpus BM25 at import). `README.md`, `Makefile` and the resume bullets still advertise it.

> **Engineering maturity: upper Level 2 (₹25–30 LPA band) with Level-3 pockets.** Level-3 depth in evaluation integrity, LLM-gateway reliability and test discipline. **Level-1 gaps in serving, deployment, security and load evidence.** The profile is spiky, not flat, and the spikes are real, which is the good news.

### Scorecard (weights are mine; the per-dimension deltas versus the v1 audit are directly comparable)

| Dimension | Weight | v1 audit | **Now** | One-line evidence |
|---|---|---|---|---|
| RAG architecture | 18 | 5.5 | **7.0** | Injected `Retriever`, FTS5 BM25 + Chroma + weighted RRF, filter parity, deterministic ties, per-stage latency, versioned incremental ingestion. No query transform, no confidence gate, reranker off by default |
| LLM engineering | 14 | 3.5 | **7.0** | `LLMGateway`: bounded retry + jitter, per-provider breaker, failover only before first token, typed errors, prompt versioning + Langfuse validation + local fallback, citation parser, abstention. Single live provider, no structured output, no cost model (`prices: {}`) |
| Backend / serving | 14 | 3.0 | **3.5** | **No HTTP service on the live path.** Streamlit in-process, one global retrieval lock, sync only, no readiness/liveness, no background ingestion, no upload |
| Performance & scalability evidence | 10 | 4.0 | **4.5** | Good single-user per-stage latencies (p50 60 ms retrieval, TEST). No concurrency test. Scale-up *measured by me* shows the lexical leg growing ~linearly (single-paper p50 15 → 589 ms at 50× the corpus) |
| Reliability | 10 | 3.5 | **6.5** | Degraded modes, never-cache-errors, fail-open tracing, self-check. State is in-process (cache, budget), one SQLite connection |
| Evaluation & observability | 12 | 5.5 | **7.5** | Pre-registered, test-once, paired clustered bootstrap. But generation eval is EXPLORATORY (n=40; RAGAS scored 2 of 34 rows), no CI gate, Langfuse-only observability (sampled at 150 traces/day), no metrics export |
| Security & safety | 6 | 2.0 | **5.0** | Prompt-injection escaping + tests, secret allowlist, masked tracing, abuse caps. No authn/z, no PII control, per-session limits bypassable by reload, `.env` in a synced folder |
| DevOps & reproducibility | 8 | 2.0 | **2.0** | No Dockerfile, compose, CI, IaC, health checks. `requirements.txt` is a polluted freeze; three dependency sources disagree |
| Testing | 4 | 3.0 | **8.5** | 1,332 tests, hermetic network guard, Hypothesis properties, fakes using the real SDK exception classes. No live integration or load tests |
| Documentation & claims | 4 | 3.0 | **5.0** | `PROJECT_CONTEXT.md` and `CLAIMS_LOG.md` are excellent. README is `UNVERIFIED`/all-TBD with a placeholder CI badge; `resume_bullets_v2.md` still carries the inflated numbers |
| **Weighted overall** | 100 | 3.8† | **5.7 / 10** | †the v1 audit's headline was 5.0 on its own weighting; re-weighted here it is 3.8 |

**Projected 7.6 / 10 after the P0 + P1 work and the top-5 items in `05`** (≈ 110–140 engineering hours; a projection, not a measurement).

## 2. Biggest strengths (keep and show these first)

1. **Evidence discipline that survives hostile questioning.** Every row of `docs/CLAIMS_LOG.md` reproduces from `runs/eval/20261122T144233Z_3afc204d_test_paper/summary.json`: D nDCG@10 0.4562, MRR 0.4339, HitRate@5 0.6815, p50 59.8 ms, p95 286.1 ms (I checked each). You found, and published, that your own earlier headline numbers were wrong.
2. **A result you can defend because it is honest.** On the held-out TEST split (292 answerable questions, 124 papers) hybrid vs vector-only is nDCG@10 **+0.027, not significant after Holm** (clustered p = 0.032 unadjusted, m = 3); semantic vs recursive chunking is **+0.006, CI [−0.012, +0.024]**; the cross-encoder reranker is **+0.058, significant (p = 0.0004)** at **+1.19 s p50** on CPU. An interviewer can poke at that for an hour and it holds.
3. **`rag_core/gateway.py` is production-shaped.** Exponential backoff with jitter and `Retry-After`, a half-open breaker that admits one trial, mid-stream failures that are never spliced (`MidStreamError`), 4xx never retried, attempt records on every result: 48 gateway tests with a fake clock and the real `openai` exception classes; 98.0% line coverage.
4. **Cache correctness is better than most production code.** The key carries question, doc filter, retrieval-config hash, index version, embedder revision, prompt name+version+sha256, model, reasoning effort, top-k and mode; only validated, complete, non-degraded, cited answers are stored (`rag_core/cache.py:64`).
5. **Ingestion is incremental and verifiable**: deterministic chunk ids, `doc_version`, reverse-order batch upsert so a crash is detectable, ingest signature that refuses a mismatched store, post-run verification with vector/lexical id parity.

## 3. Biggest weaknesses (ranked by interview damage)

1. **The public face contradicts the engineering.** README opens with `UNVERIFIED… Do not quote`, every metric is `TBD`, the CI badge points at `your-org/advanced-rag`, it says "backed by ChromaDB Cloud" (CLAUDE.md forbids touching it), and its Quick Start runs a script whose first line is `raise SystemExit` (`scripts/ingest_qasper.py:1`) and a legacy app that cannot import. `resume_bullets_v2.md` still contains "+44%", "1.7 s P50", "~14 QPS" (a 60× error).
2. **No deployable service.** The only HTTP server (`app/main.py`) is dead. For GenAI/LLM-engineer roles the first question is "show me the API, auth, streaming, health checks, and what happens at 50 concurrent users". Today the answers are "none, none, in-process, none, and one global lock".
3. **No reproducible deployment or CI.** No Dockerfile, no GitHub Actions; `requirements.txt` has `nipype`, `nibabel`, `pyxnat`, `kubernetes`, `fitz==0.0.1.dev2`, `pathlib==1.0.1`, `openai==3.57.0` vs `pyproject` `<2.0`; the real environment is `docs/env_baseline.txt` (165 packages).
4. **Quality ceiling and an untested safety claim.** The only significant retrieval win (reranker) is off by default; there is no retrieval-confidence gate; behaviour on the **48 unanswerable dev questions** was never measured (generation eval used answerable questions only), so the hallucination rate is unknown.
5. **Semantic chunking is a measured design flaw.** 77.5% of its 7-paragraph windows exceed the embedder's 256-token limit; in 28.1% of all windows the centre paragraph is truncated away entirely (AUDIT MEASUREMENT, `06` Appendix A). It tied recursive on TEST and is not what you serve.
6. **The lexical leg will not scale as designed.** On synthetic copies of your corpus (AUDIT MEASUREMENT) FTS5 latency grows roughly linearly with corpus size, and the single-paper `doc_id` filter, your *primary* task, does not help (`doc_id` is an `UNINDEXED` column): p50 **15 ms → 589 ms** as the corpus grows 200 → 10,000 documents, with the same ~37-chunk answer set. Ingestion is also slow: 7,255 chunks took 781.8 s (9.3 chunks/s) in the saved manifest, so 10K documents is hours of CPU time.

## 4. The claims you asked me to verify (classes 1–6 as you defined them)

`1` verified in implementation · `2` verified by automated tests · `3` verified by evaluation results · `4` partially implemented · `5` not implemented · `6` cannot be verified from the repository. "Live" = the code path the current app uses; "legacy" = `rag/retrieval.py`, `app/main.py`, `cache/cache.py`, `observability/*`.

| # | Claim | Class | Evidence | On a resume? |
|---|---|---|---|---|
| 1 | Hybrid BM25 + Chroma | **1, 2**; quality gain weak (3) | `rag/retriever.py` (100% covered), FTS5 `bm25()` + Chroma + RRF. TEST: hybrid vs vector-only +0.027 nDCG@10, not significant after Holm | ✅ "implemented and evaluated"; ❌ "improves recall" |
| 2 | Parallel retrieval via `ThreadPoolExecutor` | **4** (legacy only) | Only `rag/retrieval.py:785`. Live `Retriever` is sequential **on purpose** (`rag/retriever.py:127`: loses OTel context, SQLite thread-bound). No parallel-vs-sequential benchmark exists | ❌ |
| 3 | RRF k = 60 | **1, 2, 3** | `rag/fusion.py` weighted RRF (weights 1.0 / 0.7), 21 tests, arms D/F | ✅ say "weighted RRF" |
| 4 | Semantic chunking (embeddings + cosine) | **1, 2**; benefit not supported | `rag/chunkers.py:234-251` (window embeddings, adjacent cosine ≥ 0.78); TEST F vs D +0.006 (ns); served config is recursive | ⚠️ "built and evaluated; did not beat recursive" |
| 5 | Automatic small-chunk merging | **1, 2, 3** | Recursive and semantic merge paths; v1 dropped 800 of 435,269 chars, current 0 (CLAIMS_LOG, property tests) | ✅ |
| 6 | Token-budgeted context | **1, 2** (live `rag_core/context.py`, 100%) | Budget is a conservative *estimate*, not the generator's tokenizer; legacy `context_builder.py` (the "1,500-token" version) is 0% tested | ⚠️ |
| 7 | Two-tier cache (LRU + Redis) | **4** | Live: one in-process TTL-LRU answer cache (256 entries, 1 h). Redis exists only in legacy `cache/cache.py` (0% tested). The v1 "LRU" cached query *embeddings*: two caches, not tiers | ❌ rewrite |
| 8 | Embedding cache, 512 entries | **4** (legacy only) | `rag/embeddings.py:42`; unused by the live `Retriever` (it embeds every query; p50 21.6 ms) | ❌ |
| 9 | Streaming FastAPI **SSE** | **5** (SSE) / **4** (FastAPI) | `app/main.py:164,319,395`: `media_type="text/plain"`, no `data:` framing; cannot import in your venv. Live streaming is `st.write_stream` over an in-process typed event generator | ❌ |
| 10 | Groq **LLaMA 3.1 8B** primary | **5** (live) | Live model is `openai/gpt-oss-120b` (`config.yaml`); `llama-3.1-8b-instant` is hard-coded only in legacy (`rag/retrieval.py:149`) | ❌ |
| 11 | Gemini fallback | **4** | Legacy fallback can **never** succeed: `config={"timeout": 30}` → `ValidationError` (reproduced on installed google-genai 3.6.1). New `GeminiProvider` is correct and tested with fakes, but `providers: ["groq"]`, so not enabled | ⚠️ "failover-capable gateway; Groq-only in the demo" |
| 12 | Regex + LLM query routing | **4** | Live router = allowlist of conversational phrases (`rag/router.py`, 100% covered). The LLM classifier is only in deprecated `rag/retrieval.py:255` and not even called by the legacy API | ⚠️ "rule-based" |
| 13 | Prometheus instrumentation | **4** (legacy) | `observability/metrics.py` defined; used only by legacy; 0% tested; no scrape config, dashboard or alert; "tokens saved" is `len(text.split())`, "latency saved" uses a hard-coded 6.0 s (`app/main.py:308`) | ❌ |
| 14 | OpenTelemetry distributed tracing | **4** | Legacy OTLP exporter hard-coded to `127.0.0.1:4317` + missing `openinference`. Live tracing is **Langfuse**, single process (97.5% covered, fakes) | ⚠️ say "Langfuse request tracing" |
| 15 | Custom eval, Recall@5 + MRR | **1, 2, 3** | Superset now: HitRate, MRR, nDCG@10, EvidenceRecall/F1. Legacy "Recall@5" was a lenient token-F1 hit rate: same TEST retrieval scores 0.863 under the old rule vs 0.6815 under span relevance | ✅ rename to HitRate@5 |
| 16 | 4-config ablation, 100 questions | **3** (legacy artifacts) | Reproduced: R@5 0.21 / 0.21 / 0.27 / 0.28. **Arm 2 is bit-identical to arm 1 on all 100 questions** (slicing bug), so only 3 distinct configs. Wilson 95% CI for 0.27 at n=100 = [0.193, 0.364] | ⚠️ superseded by the pre-registered run |
| 17 | **+44% Recall@5** | **6, and contradicted** | `0.27 ÷ (0.25×0.21÷0.28 = 0.1875) − 1 = 0.440`: the baseline was **estimated by a formula**, never measured. Measured v1 gain: +0.06 abs (+28.6% rel), 6 wins / 0 losses, CI [+0.02, +0.11]. On TEST the analogue is +0.027 nDCG (ns) | ❌ |
| 18 | **P50 retrieval ≈ 1.7 s** | **contradicted** | 1.680 s = *mean total (incl. LLM)* of Hybrid+RRF in the 06-19 table; its P50 retrieval was 0.505 s. Current TEST: 59.8 ms p50 / 286 ms p95 (paper mode, warm, CPU) | ❌ |
| 19 | Cache-hit < 50 ms | **6** | No artifact for the Redis path. AUDIT MEASUREMENT of the live in-process path with fakes: p50 0.20 ms, p99 0.41 ms (service overhead only, local prompts). Plausible, not evidenced | ⚠️ measure properly first |
| 20 | **CrossEncoder +12.6 s on CPU** | **contradicted** | 12.6 = 14.284 − 1.680 (two noisy means). The benchmark used `BAAI/bge-reranker-base` (`scripts/run_evaluation.py:87`), not the MiniLM model your docs name. Measured, pinned MiniLM: **rerank stage p50 1,191 ms / p95 1,848 ms** | ❌ use "≈1.2 s p50" |
| 21 | "~14 QPS" (resume) | **contradicted** | Source is **14.3 QPM = 60 ÷ 4.196 s** (`scripts/run_evaluation.py:1026`) = **0.238 QPS**. A sequential loop's 1/latency, not capacity. "~14 QPS" is 60× too high | ❌ |
| 22 | "0% error rate" | **contradicted** | Literal `"error_rate": 0.0  # no errors in eval run` (`scripts/run_evaluation.py:1034`) | ❌ |
| 23 | Grafana / Phoenix / Docker / K8s / CI | **5** | No dashboards, compose, Dockerfile, workflows, or manifests exist anywhere in the tree | ❌ |

## 5. What prevents this from being a ₹35–50 LPA portfolio project

1. **Serving**: no API, auth, SSE, health checks or concurrency story.
2. **Deployment evidence**: no container, CI, reproducible install, or load test.
3. **Operational observability**: no metrics/logs when Langfuse is off or sampled out (150/day cap).
4. **Quality loop**: reranker not shipped, no abstention measurement, generation eval exploratory.
5. **Presentation**: stale README and resume file undermine the strongest asset, the evidence discipline.

## 6. Top 5 improvements by return on time (details, effort and order in `05`)

1. **A real service boundary** (FastAPI + `text/event-stream` over the existing `RagService` events, API-key auth, per-key rate limits, request ids, `/healthz` `/readyz`) and deletion of the dead legacy server. *~18–24 h.*
2. **Reproducible deployment + CI**: Dockerfile, compose, a lockfile generated from the real import graph, GitHub Actions (ruff, tests, coverage gate, eval smoke). *~14–20 h.*
3. **Load and capacity test** (Locust, open- and closed-loop) plus removing the global retrieval lock, reported honestly as QPS with p95. *~12–18 h.*
4. **Close the quality loop**: ship the reranker behind a latency budget (or int8/ONNX it), add a retrieval-confidence gate, measure false-answer rate on the 48 unanswerable questions, finish RAGAS on n ≥ 100 with a small human-labelled calibration set. *~24–32 h.*
5. **Live-path observability**: RED metrics + per-stage histograms (the `Retriever` already returns them), OTLP traces, structured JSON logs, a committed Grafana dashboard and alert rules, token/cost accounting with configured prices. *~12–16 h.*

## 7. Status of the v1 audit findings (46 findings, 5 critical · 12 high · 17 medium · 12 low)

| v1 ID | Finding | Now | Evidence |
|---|---|---|---|
| R-01 | Fused list not truncated before rerank | **Fixed** | `rag/retriever.py`: `limit = rerank_k if rerank else top_k`; tested |
| I-01 | Semantic chunker dropped text | **Fixed** | CLAIMS_LOG: 0 of 435,269 chars uncovered; property tests |
| L-01 | Gemini fallback cannot succeed | **Fixed in gateway; legacy line remains** | `rag_core/gateway.py` `GeminiProvider`; `rag/retrieval.py:392` unchanged |
| L-03 | Failure text cached 1 h | **Fixed in live path; legacy still caches** | `is_cacheable`; `app/main.py:365` unconditional |
| S-01 | Unauthenticated `/cache/flush`, `/metrics` | **Open (dead code; `make run-api` points at it but it cannot start in your venv)** | `app/main.py:226` |
| R-02 / R-03 / R-09 | Bulk `collection.get()`, O(C×N) loop, stale collection handle | **Fixed in live path** | FTS5 index, `ChromaStore`, no module globals |
| R-04 / R-05 | Mixed strategies in one collection; filter bypasses BM25 | **Fixed** | `rag_{strategy}` collections; `doc_ids` filters both legs |
| E-01 / E-02 / E-03 / E-04 | No-RRF arm bug; eval ≠ prod; lexical metric; estimated numbers | **Fixed** | `naive_interleave`; eval calls `Retriever`; span relevance; claims log. Legacy scripts still hard-code |
| L-02 | Mid-stream failover splices answers | **Fixed** | `MidStreamError`; tested |
| S-02 | Prompt injection unmitigated | **Partly fixed** | Escaping + delimiters + rules + tests ("reduces, does not remove") |
| S-03 | Streamlit renders untrusted HTML | **Fixed in `app/`; open in root `streamlit_app.py`** (8× `unsafe_allow_html=True`) | grep |
| B-02 | No request validation / limits | **Partly fixed** | `validate_question` (500 chars), guards; no HTTP layer |
| D-01 | Does not install/build reproducibly | **Open** | §3.3 |
| I-02 | Semantic boundary detection misaligned | **Open, now measured** | 77.5% windows truncated |
| T-01 | Tests wrong by construction, not hermetic, missing coverage | **Fixed for the live path; legacy still 0%** | 35 → 862 test functions (1,332 collected); network guard in `tests/conftest.py`; legacy `app/main.py`, `cache/cache.py`, loaders, `context_builder.py` untested |

The remaining v1 medium/low items are classified in `03_CODE_REVIEW.md` §4.

## 8. Report index

| File | Purpose |
|---|---|
| [02_ARCHITECTURE_REVIEW.md](02_ARCHITECTURE_REVIEW.md) | Mermaid diagrams, lifecycle, component review, failure matrix, scale analysis, target architecture |
| [03_CODE_REVIEW.md](03_CODE_REVIEW.md) | Verified code-level findings with file, function, severity, fix |
| [04_MISSING_FEATURES.md](04_MISSING_FEATURES.md) | Capability matrix (sections A–H) and the 22 advanced features assessed |
| [05_IMPROVEMENT_ROADMAP.md](05_IMPROVEMENT_ROADMAP.md) | P0–P3 plan, effort, dependencies, order |
| [06_BENCHMARKING_STRATEGY.md](06_BENCHMARKING_STRATEGY.md) | Test, load and RAG-evaluation strategy; acceptance criteria; Appendix A with every audit measurement and its reproduction script |
| [07_INTERVIEW_READINESS.md](07_INTERVIEW_READINESS.md) | Hard questions with answers grounded in this repo; level placement |
