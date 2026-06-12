# ============================================
# IMPORTS
# ============================================

from prometheus_client import (

    Counter,

    Histogram
)

# ============================================
# REQUESTS
# ============================================

REQUEST_COUNT = Counter(

    "rag_requests_total",

    "Total RAG requests"
)

# ============================================
# FAILURES
# ============================================

GENERATION_FAILURES = Counter(

    "generation_failures_total",

    "Total failed generations"
)

# ============================================
# LATENCIES
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

# ============================================
# CACHE METRICS
# ============================================

CACHE_HITS = Counter(

    "cache_hits_total",

    "Total cache hits"
)

CACHE_MISSES = Counter(

    "cache_misses_total",

    "Total cache misses"
)

TOKENS_SAVED = Counter(

    "tokens_saved_total",

    "Total tokens saved via caching"
)

LATENCY_SAVED = Counter(

    "latency_saved_seconds_total",

    "Total latency saved via caching"
)

# ============================================
# CACHE HIT LATENCY
# ============================================

CACHE_HIT_LATENCY = Histogram(

    "cache_hit_latency_seconds",

    "Time taken to serve a cache hit response"
)