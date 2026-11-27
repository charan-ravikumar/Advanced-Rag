# 07 — Interview Readiness

**How to use this.** Every answer below is built only from things you can *open on screen*: a file, a test, a run folder. Where the honest answer is "I don't know yet", the answer says so and names the plan. That is a strength in an interview, if you say it before they find it.

## 1. What makes this project genuinely interesting

1. **You corrected yourself with evidence.** The v1 headline numbers were wrong (`+44%` used an *estimated* baseline; `1.7 s` was a mean end-to-end latency; `14.3 QPM` was `60 ÷ latency`). You found it, wrote it down (`PROJECT_CONTEXT.txt` §5), rebuilt the evaluation, and made it structurally impossible to publish an unsourced number (`eval/app_results.py` fails the build). Most candidates never audit themselves; interviewers remember the ones who do.
2. **A pre-registered, test-once evaluation** with a paper-level split, a ledger, a freeze file, paper-clustered paired bootstrap and Holm correction. It produced a result with a *sign flip*: on dev the reranker looked slightly worse (nDCG@10 0.481 vs 0.500, DEV only); on the untouched TEST split it was significantly better (0.514 vs 0.456, p = 0.0004). That single fact justifies the whole methodology.
3. **An LLM gateway designed around failure semantics** (retry only before the first token; mid-stream errors never spliced; 4xx not retried; half-open breaker; attempt records), with 48 tests on a fake clock and the real SDK exception classes.
4. **Cache and ingestion correctness** (`make_cache_key`, `is_cacheable`, deterministic chunk ids, reverse-order upsert, ingest-signature refusal).
5. **Honest decision records.** `PROJECT_CONTEXT.md` has "Alternatives considered" and "Honest limitations" for each decision. Interviewers can test any of them.

## 2. Where you stand, stated critically

| Level | What it looks like | You |
|---|---|---|
| **1: ₹15–20 LPA** | Basic RAG, LangChain/LlamaIndex, vector DB, simple API and UI, little evaluation | Far above this |
| **2: ₹25–30 LPA** | Custom retrieval, hybrid search, custom chunking, evaluation, caching, streaming, basic observability, some performance work | **Solidly here, and at the top of the band** |
| **3: ₹35–50 LPA** | Strong decisions, *measured* quality, production backend, resilience, automated evaluation, advanced observability, **capacity testing, security, reproducible deployment**, a scalability strategy, evidence-backed decisions | **Level 3 in:** evidence-backed decisions, measured retrieval quality, automated-evaluation design, resilience of the LLM path, test discipline. **Level 1 in:** production backend, capacity testing, reproducible deployment, advanced observability (metrics/alerts), security |

**Verdict: upper Level 2 with Level-3 pockets (5.7/10 on the scorecard in `01`).** I will not call it Level 3: an interviewer who asks for the API, the container, the load test or the dashboard will find none of them. Closing the P0 + P1 items in `05` gets the profile to a coherent Level 3.

## 3. Numbers you may say, and numbers you must never say

**Safe (each traces to a saved artifact; cite the run folder):**

| Statement | Source |
|---|---|
| TEST split, paper mode, 292 answerable questions from 124 papers: frozen configuration D nDCG@10 **0.456**, MRR 0.434, HitRate@5 0.682, EvidenceRecall@10 0.705 | `runs/eval/20261122T144233Z_3afc204d_test_paper/summary.json`, `docs/CLAIMS_LOG.md` |
| Hybrid vs vector-only: nDCG@10 0.456 vs 0.430, **not significant after Holm** (clustered 95% CI [−0.051, −0.002] on the unadjusted interval, p = 0.032 before correction) | `runs/analysis/20261122_test_freeze_compare/…` |
| Semantic vs recursive chunking: +0.006, CI [−0.012, +0.024], not significant | same |
| Cross-encoder reranker: nDCG@10 0.514 vs 0.456, **significant** (p = 0.0004), at **≈ 1.2 s p50 / 1.9 s p95** extra on a 4-core CPU | same; CLAIMS_LOG latency rows |
| Retrieval-only latency of D: **p50 60 ms, p95 286 ms, max 3.5 s** (paper mode, warm, after 5 warm-ups; excludes any LLM call) | `summary.json` |
| 1,332 tests (1,263 passed + 69 skipped in `--fast`); live-path statement coverage 93.6% | audit run, `01` §0 |
| EXPLORATORY only (dev, n = 40): short-answer F1 0.40, 95% CI [0.27, 0.53]; abstains on 20% of answerable questions; citation labels valid 50/50 | `runs/gen/…c5fac105…_dev_gen_R` |

**Never say** (each is contradicted by your own data): "+44% recall" · "1.7 s P50 retrieval" · "CrossEncoder adds 12.6 s" · "14 QPS" or "14.3 QPM throughput" as capacity · "0% error rate" · "SSE streaming" · "two-tier Redis cache" · "Gemini automatic failover" (disabled in the demo) · "Prometheus/OpenTelemetry/Grafana dashboards" · "LLaMA 3.1 8B primary" (it is `gpt-oss-120b`) · "parallel retrieval with ThreadPoolExecutor" (removed on purpose) · "cache hit under 50 ms" (plausible, not evidenced).

## 4. The hard questions, with answers grounded in this repository

### Retrieval and evaluation

1. **"Your title says semantic chunking, but you serve recursive. Why?"**
   *Answer:* "The pre-registered rule chose recursive on dev, and on TEST semantic equals recursive: +0.006 nDCG, CI [−0.012, +0.024]. I also measured why the semantic boundary detector is weak: 77.5% of its 7-paragraph windows exceed MiniLM's 256-token limit, and in 28% the centre paragraph is cut off entirely." *Weak spot:* the title still says "Semantic Chunking" (fix in P0-1), and the chunker is unfixed (P2-4).
2. **"Does hybrid retrieval beat vector search?"**
   *Answer:* "On the held-out set it is better by +0.027 nDCG@10 and +6.9 points HitRate@5, but after Holm correction over the three pre-registered comparisons the nDCG difference is not significant. It also costs latency: p50 31 → 60 ms, p95 51 → 286 ms. So: plausible, small, not established." *Do not* round it up.
3. **"Why is reranking off if it is your only significant win?"**
   *Answer:* "It costs about 1.2 s p50 on this CPU; the demo budget favours latency. The right fix is adaptive reranking under a latency budget, or an int8/ONNX cross-encoder (P1-4)." *Weak spot:* you have measured the win and not shipped it.
4. **"What is the false-answer rate on unanswerable questions?"**
   *Answer:* "I haven't measured it. The dev set has 48 unanswerable questions that no runner scores; the retrieval eval excludes them and the generation eval reports abstention only on answerable ones. Measuring it, with a confidence gate, is the first item of my quality plan." *This is the question to rehearse hardest.* A bluff here ends the interview.
5. **"Why pre-register? Isn't that overkill for a portfolio?"**
   *Answer:* "Because dev and test disagreed on the reranker's *sign*. Without a frozen configuration and a one-shot test, I would have tuned to dev and shipped the wrong conclusion."
6. **"Is your metric inflated?"**
   *Answer:* "The old rule (token-F1 ≥ 0.25) scored the same TEST retrieval at HitRate@5 0.863; the span rule (character overlap ≥ 0.5 × the shorter span) gives 0.682. I replaced it and keep the old number beside it only to show the gap." DocHit is 1.0 by construction in paper mode (the paper is given), so ignore it.
7. **"Is 292 questions enough?"**
   *Answer:* "Intervals are paper-clustered (124 clusters). The nDCG@10 difference CIs are about ±0.025 wide, so I can resolve differences of roughly 0.03 to 0.05 and not 0.027 after correction. That is what the numbers say; I report it that way."
8. **"Why MiniLM-L6 and not a larger embedder?"**
   *Answer:* "Pinned revision, CPU-only, offline reproducibility. A Gemini-embedding arm is designed (Stage 2b) but **not executed**, and I say so in the pre-registration." *Weak spot:* 256-token window and 384 dimensions limit quality; no embedder comparison exists.
9. **"How good is 'search all papers'?"**
   *Answer:* "Weak. That is the EXTENSION setting; dev Hit@5 ≈ 0.18 for the best arm. The canonical task is single-document, so it is labelled and kept out of headline claims."

### LLM engineering

10. **"Groq returns a 429, then dies halfway through a stream. Walk me through it."**
    *Answer:* before the first token: retry the same provider with exponential backoff, jitter and `Retry-After` (≤ 8 s cap), 4xx never retried, then fail over; after the first token: `MidStreamError` with the partial text, no retry, no splice, counted towards the breaker; every attempt recorded. Point at `tests/unit/test_rag_core_gateway.py` (`test_MIDSTREAM_failure_raises_a_typed_error_and_does_not_retry_or_fail_over`).
11. **"Does a valid `[S1]` mean the claim is supported?"**
    *Answer:* "No. It means the model cited a passage that was really in its context (50 of 50 valid in the exploratory run). Support is a faithfulness question; my RAGAS run scored 2 of 34 answers before the judge's daily quota ended, so I make no faithfulness claim." Propose the entailment-model check.
12. **"What is in your cache key and why?"**
    Question (NFKC, lower-cased), document filter, retrieval-config hash, index version, embedder revision, prompt name + version + template sha256, model, reasoning effort, top-k, mode. Only complete, validated, cited, non-degraded answers are stored. *Weak spot:* `index_version` is fixed at start-up (N-12).
13. **"How do you handle prompt injection?"**
    *Answer:* "Escape every piece of untrusted text, wrap sources in tags the model is told are data, validate fetched prompts, test with adversarial values. It reduces the risk; I have not measured an attack success rate yet (P1-7)."
14. **"Why `gpt-oss-120b` at low reasoning effort?"**
    *Answer:* "The earlier configured model (`llama-3.3-70b-versatile`) was no longer listed by the provider (`scripts/check_model_ids.py`, noted in `config.yaml`). Reasoning tokens count inside `completion_tokens`, so I cap with headroom and record the effort per answer; observed completions were about 67 to 116 tokens per call. 'Low' is a design default, not tuned." *Do not* claim a cost saving: `prices` is empty.

### Systems and scale

15. **"How many concurrent users can this serve?"**
    *Answer:* "I don't know, and I can say why: retrieval is serialised by a global lock, so with the reranker the ceiling is about 0.8 retrievals per second and without it, bounded by a 60 ms median and a 3.5 s worst case. Groq limits and an 80-requests/day demo cap dominate in practice. The load test in my plan measures it." Never quote the old 14.3 QPM.
16. **"Why Streamlit in-process instead of an API?"**
    *Answer:* the decision record: one consumer, one process, the seam is `RagService`. *Cost:* a slow request holds a thread, no health endpoints. "Extracting the API is the first item on my roadmap."
17. **"Your two retrieval legs run one after the other. Why?"**
    *Answer:* "The old `ThreadPoolExecutor` lost the tracing context and SQLite connections are thread-bound, so I measured instead of guessing: embed 21.6 ms, vector 14.3 ms, lexical 20.0 ms at p50. Parallelising the two legs saves at most about the smaller of them (≈ 14 ms of 60), so I left a documented TODO (`rag/retriever.py:127`)."
18. **"What happens at 1M chunks?"**
    See `02` §7: the HNSW index grows to gigabytes of RAM, FTS5 OR-queries slow roughly linearly (I measured the trend on synthetic copies of your corpus), and `verify_store` loads every chunk. "I would first prune lexical terms by IDF and page the verification; a dedicated search engine is justified only past a few million chunks or with concurrent writers."
19. **"What if you re-ingest while the app is running?"**
    *Answer:* ingestion is incremental and verified; but the cache key's `index_version` is computed at start-up, so answers can be stale for up to the 1 h TTL (N-12). Fix: derive it per request with a short TTL.
20. **"Why SQLite FTS5 instead of `rank_bm25` or Elasticsearch?"**
    *Answer:* persistent, filtered *before* ranking (filter parity is tested), porter stemming, `O(log N)` delete by document via a side table; limits are concurrent writers and multi-million-row OR-queries.

### Reliability and security

21. **"What if Langfuse is down?"** Tracing is fail-open (`SafeTracer`) and prompts fall back to the local template. *But* a failed prompt fetch is not remembered, so each request can wait up to the 2 s timeout (N-13).
22. **"What if Chroma is down?"** The service returns a typed `retrieval_failed`, degraded result, never a crash; there is no retry and no replica.
23. **"What stops someone burning your quota?"** Optional access code (`hmac.compare_digest`), per-session limits, a global daily cap that degrades to retrieved passages. *Weak spot:* the per-session pieces reset on reload (N-05).
24. **"Where do secrets live?"** Environment only; an allowlist loader; keys never logged or rendered (tests assert it); masked tracing. *Weak spot:* `.env` sits in a synced folder (N-11).
25. **"Is it multi-tenant?"** No. The design path: `tenant_id` metadata plus a mandatory server-side filter at the `Retriever` boundary, which is cheap because filter parity is already tested.

### Process and honesty

26. **"Your resume said +44%. What happened?"** *Best story you have:* "The baseline was estimated by a formula, never measured. I caught it, measured +28.6% on the old harness (6 wins, 0 losses, CI [+2, +11] points), then rebuilt the evaluation; on the held-out set the honest figure is a small, non-significant gain. I now cannot publish a number without a run folder."
27. **"Why is your README full of TBD?"** *Do not use this answer in an interview.* Fix it first (P0-1).
28. **"Show me a bug you found in your own system."** (a) The semantic chunker silently dropped text (800 of 435,269 characters in 5 of 20 papers), found by property tests and fixed (0 uncovered now). (b) The citation parser read `【S1】` as "no citations": found by reading raw answers, not the summary table. (c) The "no-RRF hybrid" arm was identical to vector-only on all 100 questions.
29. **"What is the weakest part of the project?"** "Serving and deployment (no API, container or load test), the unmeasured abstention behaviour, and a semantic chunker I measured to be flawed." Saying this first is stronger than being asked.

## 5. Trade-offs you should be able to defend

| Decision | Why | What it costs | When you would reverse it |
|---|---|---|---|
| Streamlit in-process, no API | One consumer; no second service to secure | No health checks, serialised retrieval, no HTTP clients | A second client or any real traffic |
| Chroma + SQLite FTS5 (two stores) | Filter-before-rank on both legs; persistent; simple | Two stores to keep in parity (verified after ingest) | Multi-million chunks or concurrent writers |
| Weighted RRF, k = 60 | Rank-based, no score calibration | Weights/`k` set by design, not tuned on a large set | A learned or score-normalised fusion with a bigger eval set |
| Recursive over semantic chunking | Won the pre-registered rule; cheaper | Gives up the "semantic" headline | The fixed boundary detector beating it on dev |
| Reranker off by default | 1.2 s p50 is too slow for the demo | Leaves a significant quality gain unused | Adaptive/ONNX reranking |
| One live LLM provider | Free-tier demo; fewer moving parts | No real failover | Any paying use |
| Langfuse (SaaS) for tracing | Fast to a useful span tree | Third-party data flow, sampling cap, no local fallback | A private-data deployment |
| In-process cache and budget | Zero dependencies | Lost on restart; not shared | Two replicas |
| Pre-registration and one-shot test | Prevents tuning on test | Slower iteration; one test run only | Never for headline claims |

## 6. Where your implementation is weak (so you are never surprised)

Serving, concurrency and deployment (no API, no container, no CI, no load test) · abstention/false-answer behaviour · semantic chunker design · corpus-wide retrieval quality · no cost model · no logs/metrics without Langfuse · per-session abuse limits · generation evaluation is exploratory (n = 40, RAGAS 2/34) · three conflicting dependency files · README and resume file.

## 7. Presentation advice for the repository itself

* An interviewer opens the README first, then `docs/CLAIMS_LOG.md`, then runs `pytest`. Make all three pleasant (P0-1, P0-4).
* `PROJECT_CONTEXT.md` is excellent engineering memory, but its "Likely interview questions and strong answers" sections read as canned if the repo is public. Keep the decision, alternatives, evidence and limitations; move the rehearsed answers into a private file.
* Keep career documents (`resume_bullets_v2.md`, `prompt.txt`, salary playbooks) out of the repository.
* Lead the 90-second story with the correction: *"I built a hybrid RAG system, caught that my own headline numbers were wrong, rebuilt the evaluation with pre-registration, and found that hybrid retrieval and semantic chunking give small or no significant gains while a cross-encoder gives a significant one at a measured 1.2 s cost."* It is true, specific, and shows the judgement that ₹35 LPA+ roles pay for.
