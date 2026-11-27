> **UNVERIFIED: written before the rebuild. Do not quote. See docs/CLAIMS_LOG.md.**

# 07 — Interview Readiness

Perspective: an interviewer who has cloned your repo, read the README, and then opened the code. They will test whether you *know* your system. This is written against **today's** code. Re-read it after you complete the roadmap, because many answers will change from "weakness" to "story".

---

## 1. What makes the project genuinely interesting

1. **You ran ablations and kept per-question data.** The result "RRF takes hybrid from 0.21 to 0.27 Recall@5 (6 wins, 0 losses)" is real and defensible. Lead with that, with the CI.
2. **You made an engineering decision from data:** the reranker adds ≤ 1 pp (not significant) at a high CPU latency, so it ships disabled by default and is configurable per request.
3. **Design for toggling:** per-call overrides for mode/fusion/reranker/filter allow experiments without code forks.
4. **Fail-open Redis layer** with pool, timeouts and explicit degradation.
5. **Custom evaluation harness with resumable checkpoints** shows you care about reproducibility and API-quota realities.
6. **A candid negative result:** "semantic chunking didn't significantly beat recursive; here's why I suspect my chunker (window truncation) and what I'd test." Interviewers love this once the claim is honest.

## 2. What interviewers will challenge (and what you should be able to say today)

### Retrieval and evaluation

**Q1. "Your README says +44% Recall@5. Show me the run that produced the baseline."**
Today: you can't. 0.188 is `0.25 × 0.21 / 0.28` in `eval/feature_contrib.py`. *Good answer:* "That number was an extrapolation I should not have published. The measured comparison is vector-only 0.21 vs Hybrid+RRF 0.27, +6 pp (+28.6% relative), paired-bootstrap CI [+2, +11] pp, 6 wins/0 losses on 100 questions. I've re-run the missing arm." (Only say the last sentence once it's true.)

**Q2. "Your 'Hybrid without RRF' result equals vector-only exactly. Why?"**
Trap. The honest answer: "Because my evaluation code truncated the concatenated list to `candidate_k` after the vector list already filled it, so BM25 results never survived. It's a bug in the harness, not a property of hybrid search. The docs' explanation about score scales was wrong."

**Q3. "How do you decide a chunk is relevant?"**
Unigram F1 ≥ 0.25 between gold evidence and chunk. Be ready to explain why it is length-biased (30-word evidence inside a 300-word chunk scores ≈ 0.18), why that favours BM25/short chunks, and what you'd do (span-overlap with `paper_id` + offsets).

**Q4. "71% of your questions are ≤ 8 words ("What dataset do they use?") over a multi-paper corpus. Is 0.27 a retrieval failure?"**
Partly an ill-posed task. Propose paper-scoped evaluation or question rewriting.

**Q5. "Is semantic chunking better than recursive?"**
Data: 0.28 vs 0.25, 8 wins/5 losses, CI [−4, +10] pp: not distinguishable. Also know the implementation caveats: paragraph-level, 7-paragraph windows, 256-token model limit, 2,000-char max chunk > window, and the small-chunk merge that drops content (fixed or not).

**Q6. "What's your embedding model's max sequence length and how does it interact with your chunk size?"**
MiniLM-L6: 256 tokens (~1,000 chars). Semantic chunks go to 2,000 chars and layout chunks are unbounded, so the vector index sees the first part only; BM25 sees all. (Be able to say how you'd verify: embed first-half vs full chunk and compare retrieval.)

**Q7. "Why RRF k=60 and weights 1.0/0.7?"**
k=60 from Cormack et al.; the 0.7 is a hand-picked number, not tuned. Say how you'd tune on a dev split and test on held-out.

**Q8. "In production, what does BM25 index?"**
Whatever `collection.get()` returns at start-up. Chroma Cloud documents a 300-result cap, and your eval script paginates in pages of 100. Know whether it truncates or raises, or that you've verified it. *Today you haven't.*

**Q9. "What happens to results when `metadata_filter` is set?"**
Vector leg filtered, BM25 not. Be ready for "so tenant isolation is broken?" → yes, until fixed.

**Q10. "Your reranker costs 12.6 s. Which model, which hardware, how many pairs?"**
Today's evidence: the benchmark used `bge-reranker-base` (not MiniLM as documented); measured means were 11.7 s and 3.0 s in two runs; production feeds *all* fused candidates (unbounded) to the reranker. The 12.6 s is a subtraction of two noisy means.

### LLM and reliability

**Q11. "Walk me through Groq failing mid-stream."**
Today: exception → Gemini → whole answer appended after the partial one → concatenation cached for an hour. Know it, and know the correct behaviour (typed error event; failover only before the first token; never cache).

**Q12. "Show me the Gemini fallback working."**
It can't: `config={"timeout": 30}` is rejected by pydantic. Say it before they do, and show the chaos test after the fix.

**Q13. "Why is the router an LLM call? What's its latency/cost? What does 'BERT architecture' route to?"**
Direct (≤ 3 words, no `?`), so hallucination risk. Also not wired to the API.

**Q14. "How do you prevent prompt injection from a malicious PDF?"**
Today: not at all. Describe the mitigation plan: delimiting, system role, citation validation, red-team set, plus honest limits (no complete defence).

**Q15. "What do you do about timeouts, retries, backoff, circuit breaking?"**
Today: SDK `max_retries=1`, per-call timeouts (up to 120 s), no backoff/jitter/breaker. Explain target design.

### Systems

**Q16. "Is `/query-stream` SSE?"**
No: `text/plain` chunked. SSE means `text/event-stream` + `data:` framing + event types. Say why SSE is better (typed errors, sources, heartbeats, resumability).

**Q17. "How many concurrent users can this serve?"**
No measurement. Reasoned limits: AnyIO's 40-thread default for sync endpoints, Chroma Cloud 10 concurrent reads/collection, LLM provider rate limits, GIL-bound CPU stages, per-worker BM25 copies. Don't answer with 14.3 QPM; that is `60/latency` for one client.

**Q18. "What does 'QPM 14.3' mean?"**
Single sequential stream; 14.3 QPM = 0.24 QPS. Not capacity.

**Q19. "What's your cache invalidation story?"**
Key = hash of lowercased query only. No config/model/prompt/index version. After re-ingest or model change you serve stale answers for up to an hour. Errors are cached (L-03). `KEYS` used for stats/flush (O(N)).

**Q20. "What breaks at 10k / 100k documents?"**
Use the table in `02_ARCHITECTURE_REVIEW.md` §4.4: bulk `get()`, pure-Python BM25 (O(|q|·N)), per-worker RAM, restart to re-index, Chroma caps; fix with SQLite FTS5 first; server-side hybrid index only at ≥ ~100k docs.

**Q21. "Thread-safety and tracing: your vector/BM25 spans run in a ThreadPoolExecutor."**
OTel context is contextvar-based; plain threads don't inherit it, so those spans are orphans. Also a new pool per request. Know the fix (`copy_context().run` / async clients).

**Q22. "Why is `cache_hit < 50 ms` true?"**
You have no benchmark; the histogram only times the Redis GET. Run it.

**Q23. "How do you re-index when a document changes?"**
Today: you don't. IDs are `uuid4`; ingest wipes the collection; BM25 needs a restart. Explain the deterministic-ID + upsert + `doc_version` design.

### Security

**Q24. "Anyone can POST `/cache/flush`. What stops cost abuse?"** Nothing today. Describe auth + rate limit + input caps.
**Q25. "Where does untrusted text become HTML?"** Streamlit `render_sources` with `unsafe_allow_html=True`.
**Q26. "Can a document spoof its `source`?"** In Markdown, front-matter keys are merged after `source`/`file_type`, so yes.

## 3. Architectural trade-offs you should be able to defend

| Decision | Defence | Counter-argument you must expect |
|---|---|---|
| Chroma Cloud for vectors | No index ops, persistence, upserts; quick to start | Rate/size quotas, network latency per query, vendor limits (300 results, 10 concurrent reads); local pgvector/Qdrant would be cheaper at scale |
| In-memory `rank_bm25` | Zero infra, simple, fast at ~10⁴ chunks | O(N) Python scoring, per-worker RAM, stale until restart → SQLite FTS5 |
| RRF over score interpolation | Scale-free, robust, tiny code | Throws away score magnitude; needs top-k lists of equal depth; weights untuned |
| Reranker off by default | Data shows ≤ 1 pp gain, large CPU cost | Maybe the model/pair count was wrong; measure a small cross-encoder with top-20 |
| Semantic chunking | Topic-coherent chunks in theory | No significant measured gain; chunker has known flaws; more ingestion compute |
| Redis exact-match answer cache | Simple, deterministic, big win on repeats | Low hit rate on paraphrases; invalidation; staleness; poisoning |
| Groq 8B primary + Gemini fallback | Fast and cheap | Quality ceiling of 8B; fallback must be proven; provider coupling |
| Custom eval instead of RAGAS | Control, per-stage latency, no vendor lock-in | You must validate your own metric (length bias) and judge |
| Sync FastAPI + threads | Simple with sync SDKs | Thread-pool ceiling; async would improve concurrency only if I/O-bound |
| Streaming plain text | Simplest | No typed errors/metadata; can't recover from mid-stream failure cleanly |

## 4. Weak areas today (ranked by how badly they'd hurt in an interview)

1. Headline metrics don't trace to artefacts (+44%, 1.7 s, 12.6 s, QPM, error rate).
2. Evaluation doesn't match production; hybrid ablation arm is a bug.
3. Failover claim is false in practice; errors cached.
4. No auth/limits/validation; no deployment/CI; project doesn't `pip install` into a working state.
5. No concurrency or capacity evidence.
6. No grounding/safety layer.
7. Docs describe a different system (FAISS/spaCy).

## 5. Where the project stands

| Level | Criteria | Your status |
|---|---|---|
| **Level 1** (₹15–20 LPA): basic RAG, LangChain/LlamaIndex, simple API/UI, limited eval | You exceed it: custom retrieval, own eval, caching |
| **Level 2** (₹25–30 LPA): custom retrieval architecture ✔, hybrid ✔, custom chunking ✔ (flawed), eval framework ✔ (validity issues), caching ✔, streaming ✔ (not SSE), basic observability ✔ (unlabelled), performance optimisation ⚠️ (parallel retrieval, unmeasured) | **You are here, in the lower half.** Several Level-2 items are present but defective or unverified. |
| **Level 3** (₹35–50 LPA): evidence-backed decisions ❌, measurable retrieval quality ⚠️, production backend ❌, reliability ❌, automated eval ❌ (no CI), advanced observability ⚠️, performance/capacity testing ❌, security ❌, reproducible deployment ❌, scalability strategy ❌ | **Not yet.** Roughly 2 of 10 Level-3 criteria are partly met. |

**Honest placement:** *Level 2 (lower half), 5/10.* After the **five workstreams** in the roadmap (specifically trustworthy evaluation, the LLM gateway with chaos tests, incremental ingestion, deployment with load testing, and grounding/safety), it credibly reaches **Level 3 (≈ 7.5–8/10)**. Level 3 is not about adding technologies; it is about being able to show a measurement, a failure experiment, and a trade-off for every claim.

## 6. A 5-minute narrative to aim for (after the fixes)

1. **Problem & constraints** (30 s): QA over research papers; latency budget; free-tier LLMs; Chroma Cloud limits.
2. **Design** (60 s): ingestion lifecycle → hybrid retrieval with RRF → budgeted, cited context → gateway with failover.
3. **Evidence** (90 s): ablation table with CIs; the finding that hybrid+RRF helps (+X pp) and the reranker does not; paper-scoped vs corpus-wide.
4. **Reliability & ops** (60 s): chaos test results, load-test knee, dashboard screenshot.
5. **What I got wrong and fixed** (60 s): the hybrid-arm bug, the estimated baseline, the chunker data loss. This is the most persuasive minute you have.

## 7. Resume guidance (until P0-2 is complete)

**Safe to say today (with the numbers attached):** "Built a hybrid BM25 + vector retrieval pipeline with weighted RRF; on a 100-question QASPER subset it improved Recall@5 from 0.21 to 0.27 over vector-only (paired-bootstrap 95% CI +2…+11 pp)." / "Per-question-checkpointed evaluation harness with latency breakdown." / "Redis answer cache with fail-open degradation."

**Do not say yet:** +44%; "P50 1.7 s"; "12.6 s reranker overhead"; "SSE"; "automatic Gemini failover"; "0% error rate"; "<50 ms cache hits"; "distributed tracing"; "14.3 QPM throughput"; "production-grade"; "Grafana/Phoenix".
