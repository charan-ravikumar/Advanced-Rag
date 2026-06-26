# Benchmark Summary — QASPER Evaluation
Generated: 2026-06-19 17:09 UTC

---

## Evaluation 1: Chunking Strategy Comparison

| Chunking Strategy | Recall@5 | Recall@10 | MRR | Faithfulness | Answer Relevancy | Avg Retrieval Latency | Avg Total Latency |
|---|---|---|---|---|---|---|---|
| recursive | 0.2500 | 0.3400 | 0.1396 | 0.1700 | 0.1680 | 0.730s | 9.890s |
| layout | 0.2600 | 0.3500 | 0.1693 | 0.2290 | 0.2130 | 0.610s | 16.419s |
| semantic ⭐ | 0.2800 | 0.3600 | 0.1703 | 0.2300 | 0.2240 | 0.747s | 18.472s |

**Winner: `semantic`** (highest Recall@5 = 0.2800)

---

## Evaluation 2: Retrieval Pipeline Ablation (strategy = `semantic`)

| Config | Recall@5 | Recall@10 | MRR | Faithfulness | Answer Relevancy | Retrieval Lat | Reranker Lat | Total Lat | Recall@5 Δ% |
|---|---|---|---|---|---|---|---|---|---|
| Vector Only | 0.2100 | 0.3200 | 0.1568 | 0.2190 | 0.2210 | 0.390s | 0.000s | 3.585s | +0.0% |
| Hybrid (Vector+BM25) | 0.2100 | 0.3200 | 0.1568 | 0.2190 | 0.2210 | 0.603s | 0.000s | 3.894s | +0.0% |
| Hybrid+RRF | 0.2700 | 0.3300 | 0.1745 | 0.3980 | 0.0590 | 0.592s | 0.000s | 1.680s | +28.6% |
| Production (Full) | 0.2800 | 0.3600 | 0.1703 | 0.1500 | 0.0080 | 0.877s | 11.714s | 14.284s | +33.3% |

### Key Improvements (vs Vector-Only baseline)
- BM25 Hybrid improved Recall@5 by **+0.0%**
- RRF fusion improved MRR by **+11.3%**
- CrossEncoder reranking improved Faithfulness by **-62.3%**
- Full pipeline vs Vector-Only: Recall@5 **+33.3%**, MRR **+8.6%**, Faithfulness **-31.5%**

---

## Evaluation 3: RAGAS Benchmark

| Metric | Score |
|---|---|
| Faithfulness | N/A |
| Answer Relevancy | N/A |

---

## Evaluation 4: Observability & Performance

| Metric | Value |
|---|---|
| P50 Retrieval Latency | 0.505s |
| P95 Retrieval Latency | 1.305s |
| P50 Total Latency | 8.728s |
| P95 Total Latency | 20.087s |
| Throughput (QPM) | 4.2 |
| Error Rate | 0.00% |

---

## Resume-Ready Summary

> Built and evaluated an enterprise-grade RAG platform on QASPER, comparing 3 chunking strategies and 4 retrieval architectures. Best strategy: `semantic` with Recall@5=28.00% and MRR=0.1703. Improved Recall@5 by +33.3%, MRR by +8.6%, and Faithfulness by -31.5% using hybrid BM25 retrieval, RRF fusion, and CrossEncoder reranking. Added full observability with OpenTelemetry, Phoenix, Prometheus, and Grafana (P50=8.728s, P95=20.087s).
