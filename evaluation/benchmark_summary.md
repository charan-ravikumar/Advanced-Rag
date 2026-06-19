# Benchmark Summary — QASPER Evaluation
Generated: 2026-06-15 08:53 UTC

---

## Evaluation 1: Chunking Strategy Comparison

| Chunking Strategy | Recall@5 | Recall@10 | MRR | Faithfulness | Answer Relevancy | Avg Retrieval Latency | Avg Total Latency |
|---|---|---|---|---|---|---|---|
| recursive | 0.2500 | 0.3400 | 0.1396 | 0.1700 | 0.1680 | 1.178s | 4.279s |
| layout | 0.2600 | 0.3500 | 0.1693 | 0.2290 | 0.2130 | 0.679s | 4.864s |
| semantic ⭐ | 0.2800 | 0.3600 | 0.1703 | 0.2310 | 0.2240 | 14.437s | 191.724s |

**Winner: `semantic`** (highest Recall@5 = 0.2800)

---

## Evaluation 2: Retrieval Pipeline Ablation (strategy = `semantic`)

| Config | Recall@5 | Recall@10 | MRR | Faithfulness | Answer Relevancy | Retrieval Lat | Reranker Lat | Total Lat | Recall@5 Δ% |
|---|---|---|---|---|---|---|---|---|---|
| Vector Only | 0.2100 | 0.3200 | 0.1568 | 0.2110 | 0.2130 | 17.270s | 0.000s | 22.425s | +0.0% |
| Hybrid (Vector+BM25) | 0.2100 | 0.3200 | 0.1568 | 0.2110 | 0.2130 | 0.621s | 0.000s | 3.854s | +0.0% |
| Hybrid+RRF | 0.2700 | 0.3300 | 0.1745 | 0.2110 | 0.1970 | 0.616s | 0.000s | 3.805s | +28.6% |
| Production (Full) | 0.2800 | 0.3600 | 0.1703 | 0.2300 | 0.2240 | 0.563s | 2.994s | 4.196s | +33.3% |

### Key Improvements (vs Vector-Only baseline)
- BM25 Hybrid improved Recall@5 by **+0.0%**
- RRF fusion improved MRR by **+11.3%**
- CrossEncoder reranking improved Faithfulness by **+9.0%**
- Full pipeline vs Vector-Only: Recall@5 **+33.3%**, MRR **+8.6%**, Faithfulness **+9.0%**

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
| P50 Retrieval Latency | 0.551s |
| P95 Retrieval Latency | 1.154s |
| P50 Total Latency | 4.015s |
| P95 Total Latency | 6.306s |
| Throughput (QPM) | 14.3 |
| Error Rate | 0.00% |

---

## Resume-Ready Summary

> Built and evaluated an enterprise-grade RAG platform on QASPER, comparing 3 chunking strategies and 4 retrieval architectures. Best strategy: `semantic` with Recall@5=28.00% and MRR=0.1703. Improved Recall@5 by +33.3%, MRR by +8.6%, and Faithfulness by +9.0% using hybrid BM25 retrieval, RRF fusion, and CrossEncoder reranking. Added full observability with OpenTelemetry, Phoenix, Prometheus, and Grafana (P50=4.015s, P95=6.306s).
