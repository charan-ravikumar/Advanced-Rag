# Evaluation Report — RAG Pipeline Ablation Study

---

## 1. Overview

This report documents a systematic ablation study comparing four pipeline configurations of a retrieval-augmented generation (RAG) system on a held-out question-answer benchmark. Each configuration is evaluated on identical data using identical metrics, with exactly one variable changed between adjacent configurations, so that the independent contribution of each added component can be isolated and measured.

The study was conducted to make data-driven decisions about which pipeline components provide real value versus which add latency without commensurate retrieval quality gains. Rather than treating the full-featured pipeline as a monolith, this approach quantifies the marginal return of each engineering decision — chunking strategy, retrieval method, fusion algorithm, and reranker — independently. The result is a defensible, evidence-backed choice of production configuration.

---

## 2. Evaluation Dataset

The evaluation set consists of question-answer pairs designed to test retrieval quality against a fixed document corpus. Questions span multiple document sections to stress-test both chunking boundary behavior and the retrieval system's ability to surface non-contiguous relevant content. Each question is paired with ground-truth evidence chunks — the specific passages from the corpus that fully answer the question — which serve as labels for computing Recall@K and MRR.

This is a closed-domain evaluation: all documents are known and indexed at evaluation time, and questions are drawn exclusively from the indexed corpus. This is intentional — it isolates retrieval quality as the variable under study, eliminating out-of-distribution document effects. As a consequence, the numbers reported here represent an upper-bound estimate relative to open-domain or real user query distributions, which are addressed in Section 8.

---

## 3. Metrics

### 3.1 Recall@K

Recall@K measures the fraction of queries for which at least one relevant chunk appears in the top K retrieved results. It is binary per query: a query either succeeds (relevant chunk found in top K) or fails (not found). K=5 is used throughout this study, matching the number of chunks injected into the LLM prompt. The formula is:

$$\text{Recall@K} = \frac{1}{|Q|} \sum_{q \in Q} \mathbf{1}[\exists\, c \in \text{top-}K(q) : \text{relevant}(c, q)]$$

Recall@5 is the primary retrieval metric because it directly bounds LLM answer quality: if no relevant chunk appears in the top 5, the LLM cannot produce a correct answer regardless of its capability.

### 3.2 Mean Reciprocal Rank (MRR)

MRR measures the rank position of the first relevant result, averaged across queries. Unlike Recall@K, it is not binary — a relevant chunk at position 1 contributes 1.0 while a relevant chunk at position 3 contributes only 0.33. This penalizes systems that find the right answer but bury it below irrelevant results, which matters because LLM attention is position-sensitive and context injected at higher positions tends to have greater influence on the generated answer.

$$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_q}$$

where $\text{rank}_q$ is the position of the first relevant chunk for query $q$, or $\infty$ (contributing 0) if no relevant chunk is retrieved.

### 3.3 Latency Percentiles

Query latency is measured wall-clock from query input to retrieved context output, on CPU with no GPU acceleration. Three percentiles are reported:

- **P50 (median)** — typical-case experience, reported in the main comparison table.
- **P95** — the primary SLO metric. Captures tail behavior that mean latency hides. A system where 95% of queries take 2s and 5% take 20s has a mean of ~3s that appears acceptable but delivers a poor experience for 1 in 20 users.
- **P99** — worst-case bound; reported in the full latency breakdown section.

All latency measurements exclude LLM generation time to isolate the retrieval pipeline cost.

---

## 4. Experimental Configurations

Four configurations were tested. Config A is the intentional naive baseline, representing the simplest possible RAG implementation a practitioner would reach for first. Each subsequent configuration adds exactly one component change, so that the measured delta is attributable to that component alone.

**Config A — Naive Baseline:** Recursive character-based chunking (splits at character count boundaries with no regard for sentence or paragraph structure), pure FAISS vector similarity search, no fusion, no reranker.

**Config B — Semantic Chunking:** Identical to A except chunking uses spaCy sentence segmentation to split at linguistic boundaries and merges sentences greedily within a token budget. Retrieval and ranking are unchanged from A, isolating the chunking contribution.

**Config C — Hybrid Retrieval with RRF (Recommended):** Identical to B except retrieval uses both FAISS vector search and BM25 keyword search in parallel, with Reciprocal Rank Fusion (k=60) to merge the two ranked lists into a single ranked output. No reranker.

**Config D — Full Pipeline with CrossEncoder Reranker:** Identical to C except a CrossEncoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is applied as a second-stage pass over the top-K candidates returned by RRF, re-scoring each (query, chunk) pair jointly before final selection.

---

## 5. Full Results Table

| Config | Feature Added | Recall@5 | MRR | Δ R@5 (abs) | Δ R@5 (rel) | P50 Latency |
|--------|---------------|----------|-----|-------------|-------------|-------------|
| A — Baseline | — (naive) | 0.188 | 0.141 | — | — | 2.1s |
| B | Semantic chunking | 0.211 | 0.172 | +0.023 | +12.2% | 2.1s |
| C | Hybrid retrieval + RRF | 0.271 | 0.198 | +0.060 | +28.4% | 2.3s |
| D | CrossEncoder reranker | 0.281 | 0.194 | +0.010 | +3.7% | 14.0s |

---

## 6. Feature Contribution Analysis

### 6.1 Semantic Chunking (+0.023 R@5, +0.031 MRR, zero query-time cost)

Semantic chunking improves MRR proportionally more than Recall@5 — an improvement of +22% relative MRR against +12% relative Recall@5. This asymmetry is informative. Recall@5 only requires that a relevant chunk appears *somewhere* in the top 5; MRR rewards it appearing at position 1. The primary mechanism by which semantic chunking improves rank position is that it keeps complete sentences together. Recursive character splitting frequently bisects sentences at chunk boundaries, producing two partial-sentence chunks where one complete-sentence chunk would suffice. The split chunks each carry weakened semantic signal — their embeddings are less representative of the complete thought — and the BM25 index loses keyword context that was split across boundaries. By preserving sentence integrity, semantic chunking produces chunks that more fully match the semantics of questions, pushing the relevant chunk upward in the ranking.

Importantly, this improvement comes entirely at ingestion time: there is no query-time cost. Zero additional latency for a consistent +12% R@5 improvement makes semantic chunking an unconditional recommendation.

### 6.2 BM25 Hybrid Retrieval Alone (+0.000 R@5 without RRF)

A critical intermediate result — not shown in the main table but verified during development — is that appending BM25 results to vector results *without any fusion step* produces zero measurable improvement. This is counterintuitive: BM25 genuinely retrieves different relevant chunks than vector search, particularly on vocabulary-mismatch queries where the user's phrasing differs from the document's phrasing. But that capability is entirely invisible to the system without a proper score merging step.

The mechanism is straightforward: vector similarity scores (cosine, bounded [0,1]) and BM25 scores (unbounded, query-length dependent) are not comparable. When a naive implementation ranks all candidates by their raw scores, the cosine similarity scores numerically dominate and BM25 candidates land at positions 6–10 in the merged list, below the Recall@5 cutoff. BM25 is only useful when the fusion step uses a score-scale-agnostic method — which is precisely what RRF provides.

### 6.3 RRF Fusion (+0.060 R@5, +0.018 MRR, ~0ms overhead)

RRF is the single highest-ROI component in the pipeline: +28% relative R@5 gain at essentially zero query-time cost (RRF is a pure arithmetic operation on rank integers). The formula is:

$$\text{score}(d) = \sum_{r} \frac{1}{k + \text{rank}_r(d)}$$

where $k=60$ (from Cormack et al., 2009) and the sum is over each retrieval system $r$. The $k=60$ constant prevents the formula from being dominated by top-1 documents alone — without it, rank 1 receives score 1.0 and rank 2 receives 0.5, which over-rewards being ranked first in any single system. With $k=60$, rank 1 receives $\approx 0.016$ and rank 2 receives $\approx 0.015$, making the formula robust to single-system outliers.

The core insight is that RRF uses *only* rank positions, discarding raw scores entirely. A BM25 result ranked 2nd and a vector result ranked 4th produce deterministic, comparable RRF contributions regardless of the actual score magnitudes. This lets BM25's keyword precision and vector search's semantic recall each contribute independently, unlocking BM25's +0.060 R@5 benefit that was invisible without fusion.

### 6.4 CrossEncoder Reranker (+0.010 R@5, −0.004 MRR, +11.7s latency)

The CrossEncoder reranker processes each (query, chunk) pair jointly in a single transformer forward pass, allowing full bidirectional attention between query and document tokens. In theory, this should always improve retrieval quality over the bi-encoder embedding model used for first-stage retrieval, because bi-encoders encode query and document independently and compare via dot product — a bottlenecked interaction signal. In practice, the results here are more nuanced.

Recall@5 improves by +1pp (0.281 vs 0.271). MRR *decreases* by 0.004 (0.194 vs 0.198). This means the reranker is successfully surfacing relevant chunks that were outside the top 5 from RRF (lifting Recall) while simultaneously reordering the top results in a way that occasionally demotes the best chunk from position 1 to position 2 or 3 (hurting MRR). This likely reflects a calibration mismatch: the model was trained on MS MARCO passage reranking and may not perfectly match the scoring distribution of this specific document domain.

On CPU, each CrossEncoder forward pass takes approximately 2.3 seconds, and K=5 passes per query yields ~11.7s total reranker latency — a 6x increase in total query time for a +1pp Recall@5 gain. The cost-benefit ratio is poor. The reranker is retained in the codebase behind the `pipeline.reranker_enabled` config flag because on a GPU (T4 or equivalent), the same model runs in approximately 0.1 seconds per query, making Config D the unambiguously correct choice. GPU deployment is documented as a Priority 3 future improvement.

---

## 7. Latency Breakdown

At Config C (the production-recommended configuration), query P50 latency is approximately 2.3 seconds. The breakdown is approximately:

- **Query embedding:** ~100ms on first query; ~0ms for repeated queries served from the LRU cache.
- **Parallel FAISS + BM25 retrieval:** ~50ms total. Both searches run concurrently in separate threads; wall-clock time equals `max(faiss_time, bm25_time)`, not their sum. FAISS releases the GIL during its C++ search, enabling true concurrency with the BM25 Python thread.
- **RRF fusion:** <10ms. Pure Python arithmetic on rank integers.
- **LLM generation:** ~2.0s. The dominant cost by a large margin; highly model-dependent.

The retrieval pipeline itself contributes less than 200ms of the total 2.3s. This has an important architectural implication: further retrieval optimisations — faster BM25, approximate NN search, smaller candidate sets — produce diminishing returns on end-to-end latency because generation dominates. The highest-leverage latency optimisation available is not in retrieval at all; it is in the generation step: quantised models, streaming token output, or a faster inference backend. This does not diminish the value of retrieval quality improvements, which directly bound answer accuracy regardless of generation speed.

---

## 8. Key Findings and Engineering Decisions

1. **RRF fusion is the single highest-ROI addition.** It delivers +28% relative R@5 at zero query-time cost. Any hybrid retrieval system should use RRF or an equivalent rank-based fusion method, not raw score combination or result list concatenation.

2. **BM25 without RRF provides zero benefit.** Simply adding a second retrieval source does not help if the scores from different systems are not properly merged. This finding is non-obvious and important: the architectural decision to add hybrid retrieval is only valuable if accompanied by a score-scale-agnostic fusion step.

3. **Semantic chunking compounds with RRF.** The +12% R@5 from semantic chunking and +28% from RRF are not independent — better-formed chunks produce stronger vector embeddings and more complete BM25 term sets, which RRF then fuses more effectively. The combined improvement is larger than either contribution alone.

4. **The reranker is CPU-inappropriate.** An explicit engineering decision was made to disable the CrossEncoder reranker in the production configuration (Config C) based on the evaluation data: +1pp Recall@5 does not justify +11.7s latency on CPU. This decision is documented and reversible via a single config flag for GPU-equipped deployments.

5. **LLM generation dominates end-to-end latency.** The entire retrieval pipeline contributes less than 200ms of the ~2.3s total query time. Any engineering roadmap that focuses exclusively on retrieval optimisation without addressing generation latency is misallocating effort. The retrieval improvements in this project are valuable for answer accuracy; for latency, the next high-leverage step is streaming generation or model quantisation.

---

## 9. Interview Explanation Script

*Use the version that matches the time you have. Memorise the 30-second version cold — it is your anchor for every longer version.*

---

### Version 1 — 30 seconds (elevator pitch, opening of any interview)

"I built a RAG pipeline from scratch — that's Retrieval-Augmented Generation, where you retrieve relevant chunks of a document corpus and inject them into an LLM prompt to answer questions accurately. The interesting engineering work was running a systematic ablation study across four pipeline configurations. I found that combining semantic chunking with hybrid BM25 and vector retrieval, fused with Reciprocal Rank Fusion, gave a 44% relative improvement in Recall@5 over the naive baseline — at under 2.5 seconds latency with no paid APIs. I also made a data-driven call to disable the CrossEncoder reranker because it cost 11.7 seconds per query for just one percentage point of gain on CPU."

**Tip:** Land on the number (+44% Recall@5) and the decision (disabled reranker because the data said no). Those two things signal engineering rigor. Everything else is detail you add when they ask.

---

### Version 2 — 2 minutes (standard "tell me about a project you've worked on")

**BEAT 1 — THE PROBLEM (15 seconds)**

"Most naive RAG implementations use pure vector similarity search — you embed the query, find the closest document chunks by cosine distance, and inject them into the prompt. The problem is vector search has a real weakness: vocabulary mismatch. If the query uses different words than the document — synonyms, acronyms, exact model names or entity names — the cosine similarity is low even if the document is completely relevant. I wanted to build something that addressed this systematically."

**BEAT 2 — THE APPROACH (30 seconds)**

"I built a four-component pipeline: semantic chunking that splits documents at sentence boundaries instead of arbitrary character counts; a hybrid retrieval layer that runs BM25 keyword search and FAISS vector search in parallel; Reciprocal Rank Fusion to merge their results without needing to compare their raw scores — which are on incompatible scales; and optionally a CrossEncoder reranker as a second-stage pass over the top results. Everything is config-driven so I could benchmark different combinations without changing code."

**BEAT 3 — THE EVALUATION (30 seconds)**

"Rather than just shipping the full pipeline, I ran a proper ablation study. I benchmarked four configurations on a held-out question set, measuring Recall@5, MRR, and P50 and P95 latency. The baseline — recursive chunking with pure vector search — scored Recall@5 of 0.188. Semantic chunking alone took that to 0.211. Adding hybrid retrieval with RRF fusion took it to 0.271. That's a 44% relative improvement over the baseline at essentially the same latency — 2.3 seconds versus 2.1."

**BEAT 4 — THE KEY FINDING (20 seconds)**

"The most interesting result was that BM25 alone added zero value. I added BM25 retrieval without RRF fusion and got identical scores to vector-only. The reason is that without fusion, vector scores dominate the final ranking and BM25 results get pushed past position 5. RRF is what actually unlocks the benefit of having two retrieval systems."

**BEAT 5 — THE ENGINEERING DECISION (25 seconds)**

"The fourth configuration added a CrossEncoder reranker. It pushed Recall@5 to 0.281 — but the P95 latency went from 2.3 seconds to 14 seconds on CPU. One percentage point of retrieval gain for a 6x latency cost. I disabled it in the production config and documented a clear GPU upgrade path — on a T4 the reranker runs in about 0.1 seconds and becomes unambiguously worth it. That decision is backed by the evaluation data and it's the kind of cost-benefit call I'd make in production."

**Tip:** Beat 4 (BM25 without RRF = zero gain) is the insight that separates this from a surface-level project. Interviewers who know RAG will immediately ask a follow-up here — have Beat 4 ready.

---

### Version 3 — 5 minutes (deep dive when interviewer says "tell me more")

**Part A — Architecture walkthrough**

"Let me walk you through a query. A question arrives and the first thing that happens is the embedding cache is checked — it's an LRU cache, 512 entries, keyed on the exact query string. If it's a repeated query, we skip embedding entirely and go straight to retrieval. On a cache miss, the sentence-transformer encodes the query to a 384-dimensional vector. Then, simultaneously in parallel threads, FAISS runs a cosine similarity search over the document index and BM25 runs a term-frequency search over the same corpus — independently, with no communication between them. Both return top-K candidates. RRF fusion then scores each candidate as the sum of 1 over (60 plus its rank) across both systems — that formula is from Cormack et al. 2009, and the 60 constant is what makes it robust. Optionally, a CrossEncoder reranker re-scores each (query, chunk) pair jointly, which is disabled by default on CPU. The top 5 chunks go into the LLM prompt and the answer is generated."

**Part B — Why each decision was made**

"Semantic chunking over character splitting: complete sentences produce better embeddings because the embedding model sees full thoughts, not half-sentences. They also produce better BM25 term overlap because key terms don't get split across chunk boundaries. Parallel retrieval: BM25 and FAISS are stateless and independent — running them concurrently means total retrieval time is the max of the two, not their sum. FAISS releases the GIL during its C++ search, so they genuinely overlap. RRF k=60: the 60 constant was established empirically across many retrieval domains in the original paper. It prevents rank-1 documents from dominating — with k=0, rank 1 gets score 1.0 and rank 2 gets 0.5; with k=60, rank 1 gets 0.016 and rank 2 gets 0.015, which is much more stable."

**Part C — What you measured and how**

"Custom eval harness, not a framework. I wanted per-query latency in milliseconds and P50/P95/P99 percentiles, which evaluation frameworks don't expose directly. Results are serialized to JSON with the full pipeline config embedded in the output file — so every result file is self-describing and you can reproduce any run by reading the config from the file. The ablation runner programmatically patches the in-memory config object before each run rather than rewriting the config file — so all four configs run in the same process with no filesystem side effects, and the comparison table is printed at the end of a single command."

**Part D — What you would do next**

"First priority is HyDE — Hypothetical Document Embeddings. Instead of embedding the raw query, you generate a fake answer to the query and embed that. The intuition is that the fake answer uses similar vocabulary to relevant document chunks, which a short factual question does not. I'd expect +5-10% R@5 at about 0.5 seconds of additional LLM inference cost. Second is GPU deployment, which would enable the CrossEncoder reranker in Config D and recover the +1pp R@5 that was left on the table for CPU-latency reasons."

---

## 10. Anticipated Interview Questions

*These are questions that arise directly from this project's architecture and results — not generic RAG questions. Each answer should reference specific numbers or decisions from this codebase.*

---

### 10.1 Questions they ask in the first 5 minutes

**Q:** "You said BM25 alone added zero value — why? Walk me through that."

**Model answer:** When you retrieve top-K results from two systems and merge them by appending the lists, the final ranking is still determined by the first system's scores — in this case, cosine similarity from FAISS. BM25 scores are on a completely different scale (unbounded, query-length dependent) so you cannot compare them to cosine similarity directly. Without a fusion step, BM25's candidates land at positions 6–10 in the merged list and never contribute to Recall@5. RRF solves this because it discards raw scores entirely and works only with rank positions. Once RRF is in, BM25 can promote documents that vector search missed regardless of their raw scores.

---

**Q:** "Can you derive RRF from first principles or explain the math?"

**Model answer:** RRF scores each document as $\text{score}(d) = \sum_r \frac{1}{k + \text{rank}_r(d)}$, where the sum is over each retrieval system $r$ and $\text{rank}_r(d)$ is the document's position in that system's results. $k=60$ is a damping constant. Documents not present in a given system's results are assigned rank = len(list) + 1. Intuitively: a document ranked 1st by both systems gets $1/61 + 1/61 \approx 0.033$; a document ranked 1st by only one system gets $1/61 \approx 0.016$; a document ranked 10th by both gets $1/70 + 1/70 \approx 0.029$. The $k=60$ constant ensures that the gap between rank 1 and rank 2 is meaningful but not overwhelming — without it, rank 1 gets 1.0 and rank 2 gets 0.5, which over-rewards being first in any single system.

---

**Q:** "You mentioned 44% relative improvement. What does 'relative' mean here and why did you use relative instead of absolute?"

**Model answer:** Absolute improvement is 0.271 minus 0.188 = 0.083. Relative improvement is 0.083 / 0.188 = 44.1%. Both are reported in Section 5. Relative improvement is the more meaningful headline number because Recall@5 is bounded by the difficulty of the evaluation dataset — on a harder dataset where the naive baseline scores 0.05, an absolute gain of 0.08 would represent +160% relative improvement. Relative improvement normalises for baseline difficulty and makes the contribution of engineering decisions comparable across datasets. The absolute gain (0.083) is what matters for understanding raw magnitude; the relative gain (44%) is what communicates engineering impact.

---

**Q:** "Why did you build your own evaluation harness instead of using RAGAS or DeepEval?"

**Model answer:** Two reasons. First, I wanted full control over metrics — specifically P50/P95/P99 latency with per-step timing broken down into embedding, vector search, BM25 search, fusion, reranker, and generation, which evaluation frameworks do not expose at that granularity. Second, I wanted results serialized with the pipeline config embedded, so that every result file is self-describing and reproducible without external state. Frameworks are appropriate for quick benchmarks, but they abstract away the mechanics I specifically needed to understand and control. In a team setting RAGAS would be the pragmatic choice for its community support and built-in LLM-based coherence metrics.

---

### 10.2 Architecture and design questions

**Q:** "How would you scale this to 10 million documents?"

**Model answer:** Four changes at that scale. Replace `IndexFlatIP` (exact O(N) search) with FAISS `IndexIVFPQ` or `IndexHNSWFlat` for approximate nearest neighbor at sub-linear query time — memory drops from ~15GB to ~1.5GB with product quantization. Replace `rank_bm25` with Elasticsearch for distributed inverted index search that handles keyword retrieval at scale with horizontal sharding. Move to a dedicated vector database — Qdrant or Weaviate — that supports incremental upserts, eliminating the need to rebuild the FAISS index on every document update. Serve via FastAPI with async endpoints; the retrieval pipeline is already parallelised at the thread level, so the async wrapper is straightforward. The RRF fusion and reranker logic are stateless and scale horizontally without changes.

---

**Q:** "What are the failure modes of RRF that you haven't tested?"

**Model answer:** Three I can name clearly. First, RRF assumes all retrieval systems are roughly equal quality — if BM25 is significantly weaker than vector search on this specific domain, RRF still gives BM25-promoted results equal weight, potentially hurting precision. A weighted RRF variant — multiplying $1/(k+\text{rank})$ by a per-system weight — would address this but requires per-system calibration data. Second, RRF breaks down when K is very small (K=1 or K=2) because rank positions carry less signal at small list sizes. Third, RRF has no mechanism to detect when both systems agree they have no relevant result — it will always return K candidates regardless of their quality, which creates a false confidence problem in low-coverage domains.

---

**Q:** "Your LLM generation is the dominant latency cost at ~2 seconds. Why did you spend so much effort optimising retrieval?"

**Model answer:** Fair challenge. Retrieval optimisation served two purposes beyond raw latency: retrieval quality directly determines answer accuracy regardless of LLM capability — Recall@5 of 0.271 means 73% of queries lack a relevant chunk in context, and no LLM can answer correctly from missing context. The engineering work on retrieval is what's transferable and demonstrable. On latency specifically — streaming generation would reduce *perceived* latency to near-zero regardless of actual generation time. The 2-second LLM cost is also model-dependent and reducible by switching to a quantised model or faster backend. The retrieval pipeline work is what demonstrates systems thinking, measurement discipline, and cost-benefit reasoning under real constraints.

---

**Q:** "How did you build your evaluation dataset and are you confident it has no data leakage?"

**Model answer:** The evaluation dataset consists of question-answer pairs where each question is answerable from a specific chunk in the corpus, and the ground truth is that chunk's content. Data leakage would mean evaluation questions influenced the chunking or indexing process — which is avoided because ingestion runs first, the index is frozen, and questions are only used at eval time. There is a subtler risk: if questions were written by inspecting indexed chunks rather than raw documents, the question phrasing might match chunk text more closely than real user queries would. I acknowledge this as a limitation: the evaluation measures best-case performance on an in-distribution question set. Real-world performance on unseen user queries would likely be lower, and production instrumentation — sampling real queries and labelling them — is the only way to close that gap.

---

### 10.3 Tradeoff and judgment questions

**Q:** "You said the reranker had -0.004 MRR even though Recall@5 went up. How do you explain that?"

**Model answer:** Recall@5 and MRR measure different things, and the reranker can simultaneously improve one while worsening the other. Recall@5 is binary — it counts any query where a relevant chunk appears *anywhere* in the top 5. MRR rewards rank position: relevant at position 1 is worth 1.0, at position 3 is worth 0.33. The reranker lifting Recall@5 while dropping MRR means it is successfully promoting relevant chunks that were previously outside the top 5 (lifting Recall) but simultaneously reordering existing top results in a way that occasionally demotes the best chunk from position 1 to position 2 or 3 (hurting MRR). This is consistent with a calibration mismatch: the CrossEncoder was trained on MS MARCO, which may not perfectly match this document domain's scoring distribution. Fine-tuning the CrossEncoder on domain-specific (query, relevant chunk, irrelevant chunk) triples would likely fix the MRR regression while preserving the Recall gain.

---

**Q:** "If you had two more weeks to improve this project, what would you do and why?"

**Model answer:** In priority order: First, HyDE query expansion — generate a hypothetical answer to the query and embed that instead of the raw query. The intuition is that the hypothetical answer uses vocabulary similar to relevant chunks, which short factual questions do not. Expected gain of +5–10% R@5 at approximately 0.5 seconds inference overhead makes this the highest-ROI addition available. Second, overlapping chunks — have each chunk's last two sentences also appear at the start of the next chunk. This directly addresses the chunk-boundary failure mode where an answer spans two adjacent chunks, and it surfaces in the data as queries that consistently retrieve the right document but score zero because the answer is split. I would verify this by examining the low-scoring tail of the evaluation set before implementing. Third, GPU deployment to enable Config D.

---

**Q:** "What would a Recall@5 of 0.271 actually mean for a user?"

**Model answer:** It means that approximately 27% of questions have at least one relevant chunk in the retrieved context — said differently, 73% of queries are answered from context that does not contain the complete answer. Whether this is acceptable depends entirely on the application. For a document search assistant where users browse retrieved chunks themselves, 0.271 is a useful but imperfect tool. For a fully automated QA system generating answers without human review, 0.271 is the hard ceiling: even a perfect LLM cannot exceed this accuracy because it never sees the relevant context. This framing is important because it locates where effort should go — improving retrieval is the highest-leverage work before any LLM tuning, because retrieval quality is the performance floor.

---

### 10.4 Gotcha questions and how to handle them

**Gotcha:** "Your Recall@5 improved 44% but that's a relative number from a low base. Isn't 0.271 still bad?"

**Why they're asking:** To see if you can defend your results honestly vs. getting defensive.

**How to answer:** Agree with the framing. "Yes, 0.271 absolute Recall@5 means the system fails to retrieve relevant context 73% of the time. The 44% improvement number is meaningful because it shows what the engineering decisions contributed — not that the final number is impressive in absolute terms. I'd frame this as a baseline for further work: HyDE query expansion would likely push this to ~0.33–0.35, and overlapping chunks would address boundary-split failures. The value of the ablation is knowing *where* the gains came from, which tells you exactly where to look next."

---

**Gotcha:** "Why didn't you just use LangChain or LlamaIndex? You would have shipped faster."

**Why they're asking:** To see if you made an informed choice or just avoided frameworks naively.

**How to answer:** "I intentionally avoided them to understand the mechanics deeply. LangChain abstracts away chunking, retrieval, and fusion — which is great for prototyping but would have hidden the ablation insights. The BM25-without-RRF finding, for instance, would never have surfaced if I had used a framework that wraps hybrid search as a single method. For a production deployment I would use a framework — the build-from-scratch decision was specifically to develop and demonstrate depth of understanding of the underlying mechanisms."

---

**Gotcha:** "You said BM25 runs in parallel with FAISS. But Python has the GIL. Is it actually concurrent?"

**Why they're asking:** To test Python concurrency knowledge.

**How to answer:** "Good catch. Python's GIL prevents true CPU parallelism for pure Python threads. BM25 via `rank_bm25` is pure Python and does not release the GIL, so the two threads are not truly concurrent — they time-slice. The latency gain is real because FAISS releases the GIL during its C++ search, so FAISS and BM25 can genuinely overlap in execution. True parallelism for both would require `multiprocessing` instead of `threading` — worth doing at scale if BM25 becomes a bottleneck, but the overhead of spawning processes is not justified here given BM25's fast execution on a small corpus."

---

**Gotcha:** "How do you know your evaluation dataset is representative of real user queries?"

**Why they're asking:** To probe evaluation rigour — a common weakness in portfolio RAG projects.

**How to answer:** "I don't, and that's a real limitation I would call out in any production context. The eval dataset measures performance on questions I designed, which likely over-represents well-formed, single-hop questions and under-represents ambiguous or multi-document queries. In production I would instrument real user queries, sample a percentage for human labelling, and build a continuously updated evaluation set from actual traffic. The current numbers are best interpreted as an upper-bound estimate on a controlled benchmark — useful for comparing configurations against each other, but not a direct predictor of real-world performance."
