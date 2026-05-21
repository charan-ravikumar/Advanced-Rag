# ============================================
# IMPORTS
# ============================================

from prometheus_client import (

    Counter,

    Histogram
)


# ============================================
# REQUEST COUNTER
# ============================================

REQUEST_COUNT = Counter(

    "rag_requests_total",

    "Total RAG requests"
)


# ============================================
# FAILURE COUNTER
# ============================================

GENERATION_FAILURES = Counter(

    "generation_failures_total",

    "Total failed generations"
)


# ============================================
# LATENCY HISTOGRAMS
# ============================================

RETRIEVAL_LATENCY = Histogram(

    "retrieval_latency_seconds",

    "Retrieval pipeline latency"
)

RERANK_LATENCY = Histogram(

    "rerank_latency_seconds",

    "Reranker latency"
)

GENERATION_LATENCY = Histogram(

    "generation_latency_seconds",

    "Generation latency"
)