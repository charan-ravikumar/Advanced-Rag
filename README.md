# Advanced-Rag




## Benchmark Results (QASPER) — 2026-06-15

Evaluated on a reproducible 100-question subset of the QASPER golden dataset (seed=42).
Full results: [`evaluation/benchmark_summary.md`](evaluation/benchmark_summary.md)

### Best Chunking Strategy

**Winner: `semantic`** · Recall@5 = 0.2800 · MRR = 0.1703 · Faithfulness = 0.2310

### Retrieval Improvements

| Upgrade | Improvement |
|---|---|
| + BM25 Hybrid | Recall@5 +0.0% |
| + RRF Fusion | MRR +11.3% |
| + CrossEncoder Reranking | Faithfulness +9.0% |

Full pipeline vs Vector-Only: **Recall@5 +33.3%**, **MRR +8.6%**, **Faithfulness +9.0%**

### Production Metrics

| Metric | Score |
|---|---|
| Faithfulness | 0.2300 |
| Answer Relevancy | 0.2240 |
| P50 Latency | 4.015s |
| P95 Latency | 6.306s |

### Resume-Ready Summary

> Built and evaluated an enterprise-grade RAG platform on QASPER, comparing 3 chunking strategies and 4 retrieval architectures. Best strategy: `semantic` with Recall@5=28.00% and MRR=0.1703. Improved Recall@5 by +33.3%, MRR by +8.6%, and Faithfulness by +9.0% using hybrid BM25 retrieval, RRF fusion, and CrossEncoder reranking. Added full observability with OpenTelemetry, Phoenix, Prometheus, and Grafana (P50=4.015s, P95=6.306s).
