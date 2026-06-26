# Technical Deep Dive — RAG Pipeline Architecture

---

## 1. Problem Statement

Large language models produce fluent, confident text — including fluently wrong text. On factual, domain-specific question answering, a vanilla LLM has no mechanism to distinguish what it knows accurately from what it has confabulated from training data patterns. Fine-tuning can partially address this by baking domain knowledge into model weights, but it is expensive, difficult to update when knowledge changes, and fundamentally opaque: there is no way to inspect which training examples contributed to a given output, making errors hard to diagnose and correct.

The goal of this project is to build a retrieval layer rigorous enough that retrieval quality — not generation quality — is the performance floor for domain QA. If the retrieval system reliably surfaces the right document chunks, even a modest LLM can produce correct, grounded answers by reading and summarising the context provided. The alternative — investing primarily in a better LLM — has a ceiling: no LLM can answer correctly from context it was never shown. This design priority is why the evaluation framework focuses on Recall@5 and MRR rather than generation-quality metrics. Naive vector-only retrieval has two well-documented failure modes that make it an insufficient foundation: vocabulary mismatch (the query and document use different words for the same concept) and chunk boundary artifacts (relevant content is split across adjacent chunks, degrading embedding quality and keyword overlap for both halves). Both are directly addressed by this pipeline.

---

## 2. Architecture Overview

The pipeline separates into two phases with a clean boundary between them.

**Ingestion** (offline, runs once): documents are chunked, each chunk is embedded and pushed to ChromaDB Cloud, and the full corpus is simultaneously indexed for BM25. This is expensive — encoding tens of thousands of chunks with a transformer model is the dominant cost — but it happens once and the result is persisted. The ingestion pipeline is parameterised by `config.yaml`; changing the chunking strategy requires a re-run of ingestion, not a code change.

**Query** (online, runs per request): a user query arrives, is embedded (with LRU cache), triggers parallel ChromaDB vector search and BM25 searches, passes through RRF fusion, optionally through a CrossEncoder reranker, and the top-K chunks are assembled into an LLM prompt. The entire retrieval path takes under 200ms; generation takes ~2s.

The config-driven design means all four ablation configurations share identical code. Switching from Config A to Config C changes three flags in `config.yaml` and nothing else. This property — that experimental configurations are not code variants but parameter variants — is essential for trustworthy ablation: it rules out the possibility that performance differences are caused by implementation differences between configurations.

---

## 3. Component Deep Dive

### 3.1 Semantic Chunking

The problem with recursive character splitting is fundamental: it splits documents at character count thresholds, which has no relationship to linguistic or semantic structure. A threshold of 512 characters will bisect sentences, split a heading from its first paragraph, and cut a bulleted argument mid-item. Each half of the split produces a chunk with a partial sentence at one of its boundaries. That partial sentence degrades the chunk's embedding in two ways: the sentence-transformer model receives malformed input that was never seen in training (well-formed sentences), and the embedding is pulled toward the partial sentence's incomplete semantic content.

Semantic chunking uses spaCy's sentence segmentation to identify sentence boundaries, then constructs chunks by greedily merging consecutive sentences until the chunk would exceed a token budget. Each chunk therefore begins at a sentence start and ends at a sentence end. The result is chunks that contain complete thoughts — a critical property because the embedding model and BM25 index both perform better when they operate on complete semantic units.

The trade-off is ingestion-time complexity: spaCy sentence segmentation is slower than character splitting, and the greedy merge logic adds implementation overhead. Both costs are amortised over all queries, however — ingestion is a one-time operation. The downstream reward is +12% R@5 and +22% MRR at zero query-time cost, which makes the trade-off straightforward.

### 3.2 Embedding Model and ChromaDB Cloud

`all-MiniLM-L6-v2` produces 384-dimensional embeddings. It is a distilled model trained on a large collection of semantic similarity tasks, fast enough to run on CPU without notable latency, and widely benchmarked as a strong performer on semantic search tasks at this model size.

Vector storage and search is handled by **ChromaDB Cloud** — a managed, serverless vector database. Document embeddings are persisted in ChromaDB Cloud; at query time, `collection.query()` performs approximate nearest-neighbor search over the stored embeddings. Using a managed service eliminates index lifecycle management (no serialisation, deserialisation, or memory management for the index), provides automatic persistence, and supports incremental upserts without full index rebuilds.

**LRU cache on query embeddings:** Queries are often repeated — in evaluation runs the same 100 questions are re-asked across multiple configurations, and in interactive demos users frequently re-ask the same question. Caching the query embedding eliminates the ~100ms embedding cost for repeated queries. The cache is keyed on the exact query string, has a maximum of 512 entries (configurable via `config.yaml`), and is implemented with `functools.lru_cache`. Cache hits return in <1ms. The cache does not persist across process restarts and does not handle semantically equivalent but lexically different queries — both are known limitations documented in the interview prep materials.

### 3.3 BM25 Retrieval

BM25 (Best Matching 25) is a probabilistic term-frequency model that ranks documents by the presence and frequency of query terms, with saturation (a term appearing 10 times in a document is not 10x better than appearing once) and length normalisation (longer documents are not unfairly rewarded for having more term occurrences). Vector search genuinely cannot replicate this: rare proper nouns, model names, version numbers, and technical acronyms may be underrepresented or inconsistently represented in the embedding space — they appear infrequently in training data, so the model has weak signal for them. A query containing "GPT-4" or "ResNet-50" or a specific paper title will retrieve exactly the documents containing those strings via BM25 with high precision, while vector search may return documents that are thematically similar but do not contain the specific term.

`rank_bm25` is used for the BM25 implementation: pure Python, zero external dependencies, no server, no configuration. For a corpus that fits in memory (which is the target scale for this project), it is fast and has no operational overhead. At larger scale, Elasticsearch provides a production-grade distributed inverted index with the same BM25 scoring.

### 3.4 Parallel Hybrid Retrieval

BM25 and ChromaDB vector searches are completely independent — neither requires the other's output, and they operate on separate data structures (the BM25 index and the ChromaDB Cloud collection). This makes them trivially parallelisable. The implementation uses `concurrent.futures.ThreadPoolExecutor` to submit both searches simultaneously:

```python
with ThreadPoolExecutor(max_workers=2) as executor:
    f_vec  = executor.submit(collection.query, query_embeddings=[query_embedding], n_results=top_k)
    f_bm25 = executor.submit(bm25_search,      query_tokens, top_k)
vector_results = f_vec.result()
bm25_results   = f_bm25.result()
```

Total retrieval wall-clock time equals `max(chroma_time, bm25_time)` rather than their sum. In practice, the ChromaDB Cloud request is slightly slower than BM25 on a corpus of this size, so the parallel execution means BM25 completes and waits while ChromaDB finishes — with near-zero idle time wasted.

A Python GIL nuance: `rank_bm25` is pure Python and does not release the GIL during execution. The ChromaDB Cloud client makes an HTTP request on a separate thread, genuinely running concurrently with the BM25 computation. For the current scale, the threading approach is correct and sufficient.

### 3.5 RRF Fusion

The fundamental problem with merging results from multiple retrieval systems by raw score is that the scores are on incompatible scales. Cosine similarity is bounded in [0, 1]. BM25 scores are unbounded, positively correlated with query length and document length, and vary by corpus. Normalising each system's scores by its maximum score is fragile: a single outlier document with an unusually high score compresses all other scores toward zero, distorting the ranking.

RRF discards raw scores entirely and uses only rank positions:

$$\text{score}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

where $R$ is the set of retrieval systems, $\text{rank}_r(d)$ is document $d$'s rank position in system $r$'s results (1-indexed), and $k=60$ is an empirically derived damping constant from Cormack et al. (2009), shown to be robust across many retrieval domains. Documents not returned by a given system are assigned $\text{rank} = |L_r| + 1$ where $|L_r|$ is the length of that system's result list.

The $k$ constant controls how sharply the formula rewards top-ranked documents. With $k=0$, rank 1 receives score 1.0 and rank 2 receives 0.5 — the formula is highly top-heavy. With $k=60$, rank 1 receives $\approx 0.0164$ and rank 2 receives $\approx 0.0161$ — the formula distributes weight more evenly across ranks, preventing any single system's top result from dominating the fused ranking.

Intuitively: a document that ranks 1st by both vector search and BM25 is almost certainly relevant. A document that ranks 1st by vector search but 15th by BM25 might be semantically similar to the query but lacks the exact keyword, suggesting moderate relevance. RRF captures exactly this intuition without any score normalisation.

### 3.6 CrossEncoder Reranker (Config-Gated)

The core architectural distinction is between **bi-encoders** (the embedding model) and **cross-encoders** (the reranker).

A bi-encoder encodes query $q$ and document $d$ independently: $f(q)$ and $f(d)$ are computed separately and compared with dot product or cosine similarity. This is fast — one forward pass per query, plus an indexed lookup — and scales to arbitrarily large document collections because document embeddings can be precomputed and cached. The limitation is that $q$ and $d$ never attend to each other during encoding. The relevance signal comes entirely from the geometric proximity of their independently-produced vectors.

A cross-encoder encodes the pair $(q, d)$ jointly in a single forward pass through a transformer. Every query token can attend to every document token and vice versa, producing a relevance score that captures interactions between specific query terms and specific document passages. This is substantially more powerful than the bi-encoder's bottlenecked dot product, but it requires one full forward pass per (query, document) pair — O(K) inference calls per query where K is the reranking candidate list size.

The operational consequence on CPU: `cross-encoder/ms-marco-MiniLM-L-6-v2` takes approximately 2.3 seconds per forward pass on CPU. With K=5 candidates, this is ~11.7 seconds of reranker latency per query — visible in the ablation as a 6x total query time increase. On a T4 GPU, the same model runs in approximately 0.02 seconds per pair, reducing total reranker latency to ~0.1 seconds and making Config D the unambiguously better choice.

The reranker is disabled by default via `pipeline.reranker_enabled: false` in `config.yaml`. This is a documented, reversible decision backed by cost-benefit data from the ablation study (see [Evaluation Report](EVALUATION_REPORT.md) Section 6.4).

---

## 4. Configuration System

All pipeline hyperparameters live in `config.yaml` at the project root. The config loader (`config.py`) reads the file once at startup using PyYAML and converts the nested dictionary to a `SimpleNamespace` object tree, enabling dot-access syntax (`cfg.pipeline.top_k`, `cfg.embedding.model`) without the syntactic noise of dictionary indexing.

The design principle is that **no magic constants exist in source code**. Model names, top-K values, RRF k, cache sizes, eval dataset paths, and all feature flags are config parameters. This enables three concrete engineering practices:

1. **Reproducible experiments:** Every result file produced by the eval harness embeds a copy of the full pipeline config, so any result can be reproduced exactly by restoring that config.
2. **Clean A/B testing:** Changing one flag and re-running produces a result that is directly comparable to the previous run, with no risk of code-level differences contaminating the comparison.
3. **Ablation automation:** The ablation runner (`eval/run_ablation.py`) overrides config attributes programmatically — mutating the in-memory `SimpleNamespace` object before each evaluation run and restoring it after — without touching the config file on disk. This means the ablation runs four configurations in a single process invocation with no filesystem side effects.

---

## 5. Evaluation Harness Design

The evaluation harness is custom-written rather than using an existing framework like RAGAS or DeepEval. This was a deliberate decision for two reasons: full control over the specific metrics that matter (P50/P95/P99 per-step latency, not just aggregate throughput), and result files that are self-describing (the pipeline config is embedded in every result JSON, not stored externally).

Per-query latency is measured with `time.perf_counter()` at microsecond resolution. Percentiles are computed with `numpy.percentile()` after all queries complete. The harness records the raw per-query latency array, not just summary statistics, enabling post-hoc analysis of the latency distribution shape.

The ablation runner wraps the eval harness: it defines four config override dicts, patches `cfg.pipeline.*` attributes before each run, calls `run_eval()`, and restores the original config. The result is a comparison table printed to stdout after all four configurations complete. The command `python -m eval.run_ablation --max-questions 20` runs a smoke test in under a minute; `python -m eval.run_ablation` runs the full 100-question evaluation for all four configurations.

---

## 6. Production Readiness Notes

### 6.1 What is production-grade in the current implementation

Several engineering properties here go beyond a typical portfolio RAG project. The config-driven pipeline separates concerns cleanly: changing pipeline behaviour requires editing a YAML file, not source code. The LRU embedding cache reduces latency to near-zero for repeated queries without any external caching infrastructure. Parallel retrieval correctly exploits the independence of BM25 and ChromaDB vector search. The formal evaluation harness with P95 latency tracking provides quantified performance guarantees. Every design decision is backed by ablation data rather than intuition, and the data is reproducible from the config and dataset alone.

### 6.2 What would change for production at scale

At production scale, several components would be replaced with more operationally appropriate alternatives. ChromaDB Cloud already handles managed vector storage with horizontal scaling; at very large scale (hundreds of millions of vectors), switching to Qdrant, Weaviate, or Pinecone may provide additional performance options. `rank_bm25` would be replaced with Elasticsearch or OpenSearch for distributed keyword search with inverted index persistence. The synchronous query pipeline would be wrapped in a FastAPI async endpoint with streaming response support. Logging would be structured (JSON) and shipped to a monitoring stack. The current implementation handles these concerns adequately for development and evaluation; at scale they become operational requirements.

### 6.3 GPU upgrade path

The most impactful single hardware upgrade is a GPU for the CrossEncoder reranker. On CPU, `cross-encoder/ms-marco-MiniLM-L-6-v2` takes ~2.3s per (query, chunk) pair; on a T4 GPU, the same model runs in approximately 0.02s per pair. With K=5 candidates, total reranker latency drops from ~11.7s to ~0.1s. This makes Config D (full pipeline with reranker) the correct production default, recovering the +1pp Recall@5 that was left on the table in the CPU-only evaluation. Enabling GPU requires setting `pipeline.reranker_enabled: true` in `config.yaml` and ensuring PyTorch has CUDA access — no code changes. The `all-MiniLM-L6-v2` embedding model also runs approximately 5x faster on GPU with batched document encoding during ingestion, reducing ingestion time significantly for large corpora.

### 6.4 Observability

The query pipeline emits a structured latency trace log for every request:

```
retrieval_trace | vector_search_ms=48 bm25_search_ms=12 fusion_ms=3 reranker_ms=0 total_retrieval_ms=53
```

Each field is the wall-clock duration of that pipeline stage in milliseconds. In production, this log line would be parsed and aggregated into a monitoring dashboard — graphing P95 vector search latency, BM25 latency, and reranker latency independently allows rapid diagnosis of which component is regressing when overall query latency increases. The per-step breakdown is also what revealed the reranker's cost during the ablation study; having it as a structured log from the start rather than added retroactively is the correct operational approach.

---

## 7. Future Work

**Priority 1 — HyDE (Hypothetical Document Embeddings)**

HyDE generates a hypothetical answer to the query using the LLM and embeds that answer as the retrieval query, rather than embedding the raw question. The intuition is that a hypothetical answer uses vocabulary and phrasing similar to the relevant document chunks, while a short factual question may use entirely different vocabulary than the documents. This is especially effective for questions where the query is short and abstract ("What is the proposed training objective?") while the relevant document passage is specific and technical. Expected gain: +5–10% R@5 with approximately 0.5 seconds of additional LLM inference cost for the hypothetical answer generation. This is the highest-ROI improvement available within the current architecture.

**Priority 2 — Alternative vector index strategies**

ChromaDB Cloud uses approximate nearest-neighbor search internally. For very large corpora (hundreds of millions of vectors), self-hosted alternatives such as Qdrant with HNSW indexing provide sub-linear query time and fine-grained control over the accuracy/speed trade-off. HNSW has less than 1% accuracy loss compared to exact search at typical parameter settings. Switching is an infrastructure change; the retrieval code calling `collection.query()` remains identical.

**Priority 3 — GPU deployment**

Enables the CrossEncoder reranker at acceptable latency, making Config D the production default and recovering the +1pp Recall@5 currently left on the table. Also accelerates document ingestion and query embedding as secondary benefits. Requires no code changes — only hardware and PyTorch CUDA configuration.

**Priority 4 — Streaming generation**

LLM generation is the dominant query latency cost at ~2 seconds. The user experience impact of streaming is disproportionate to the engineering effort: even though total time-to-answer remains 2 seconds, streaming the first tokens within ~100ms makes the system feel instant. Most LLM APIs support streaming natively; the change is confined to the generation layer and does not affect retrieval. This is the highest-leverage latency improvement available within the current architecture.
