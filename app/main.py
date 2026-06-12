# ============================================
# IMPORTS
# ============================================

import json
import time

from fastapi import FastAPI, HTTPException

from fastapi.responses import (

    StreamingResponse,

    Response
)

from schemas import (
    QueryRequest
)

from rag.query import (

    retrieve_and_build_context,

    generate_answer_stream
)

from cache.cache import (

    get_cached_response,

    set_cached_response,

    flush_cache,

    cache_size,

    ping_redis
)

from observability.metrics import (

    REQUEST_COUNT,

    CACHE_HITS,

    CACHE_MISSES,

    TOKENS_SAVED,

    LATENCY_SAVED,

    CACHE_HIT_LATENCY
)

# ============================================
# PHOENIX + OTEL
# ============================================

from openinference.instrumentation.openai import (
    OpenAIInstrumentor
)

from opentelemetry import trace

from opentelemetry.sdk.trace import (
    TracerProvider
)

from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor
)

from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter
)

# ============================================
# PROMETHEUS
# ============================================

from prometheus_client import (
    generate_latest
)

# ============================================
# OPENTELEMETRY SETUP
# ============================================

trace.set_tracer_provider(
    TracerProvider()
)

tracer_provider = (
    trace.get_tracer_provider()
)

span_processor = (

    BatchSpanProcessor(

        OTLPSpanExporter(

            endpoint=
            "http://127.0.0.1:4317",

            insecure=True
        )
    )
)

tracer_provider.add_span_processor(
    span_processor
)

# ============================================
# OPENINFERENCE
# ============================================

OpenAIInstrumentor().instrument()

# ============================================
# FASTAPI INIT
# ============================================

app = FastAPI(

    title="Advanced RAG API",

    version="3.0.0"
)

# ============================================
# HEALTH CHECK
# ============================================

@app.get("/")

def home():

    redis_ok = ping_redis()

    return {

        "message":
        "Advanced RAG API Running",

        "redis_status":
        "ok" if redis_ok else "unavailable"
    }

# ============================================
# PROMETHEUS METRICS
# ============================================

@app.get("/metrics")

def metrics():

    return Response(

        generate_latest(),

        media_type="text/plain"
    )

# ============================================
# CACHE STATS
# ============================================

@app.get("/cache/stats")

def cache_stats():
    """
    Returns a snapshot of cache activity:
    active key count plus Prometheus counters.
    The Prometheus counters are cumulative since
    process start (reset on restart).
    """

    from prometheus_client import (
        REGISTRY
    )

    def _counter_value(name):

        try:

            samples = REGISTRY.get_sample_value(
                name
            )

            return float(samples) if samples is not None else 0.0

        except Exception:

            return 0.0

    return {

        "redis_status":
        "ok" if ping_redis() else "unavailable",

        "active_cache_keys":
        cache_size(),

        "cache_hits_total":
        _counter_value("cache_hits_total"),

        "cache_misses_total":
        _counter_value("cache_misses_total"),

        "tokens_saved_total":
        _counter_value("tokens_saved_total"),

        "latency_saved_seconds_total":
        _counter_value(
            "latency_saved_seconds_total"
        )
    }

# ============================================
# CACHE FLUSH
# ============================================

@app.post("/cache/flush")

def cache_flush():
    """
    Deletes all rag:cache:* keys from Redis.
    Returns the number of keys deleted.
    """

    deleted = flush_cache()

    return {

        "status": "ok",

        "keys_deleted": deleted
    }

# ============================================
# STREAMING QUERY ENDPOINT
# ============================================

@app.post("/query-stream")

def query_rag_stream(

    request: QueryRequest
):

    print("\n===================================")
    print("NEW QUERY RECEIVED")
    print("===================================\n")

    print(f"Query: {request.query}")

    REQUEST_COUNT.inc()

    request_start_time = time.time()

    # ========================================
    # CACHE CHECK
    # ========================================

    cached_response = get_cached_response(
        request.query
    )

    if cached_response:

        print("\nCACHE HIT\n")

        CACHE_HITS.inc()

        cache_hit_latency = (
            time.time()
            - request_start_time
        )

        CACHE_HIT_LATENCY.observe(
            cache_hit_latency
        )

        # ====================================
        # TOKEN SAVINGS
        # ====================================

        cached_tokens = len(
            cached_response.split()
        )

        TOKENS_SAVED.inc(
            cached_tokens
        )

        # ====================================
        # REAL LATENCY SAVINGS
        # Estimated as: avg miss latency
        # minus the time taken for this hit.
        # We use a conservative 6s baseline
        # for an average uncached pipeline run.
        # ====================================

        LATENCY_SAVED.inc(
            max(0.0, 6.0 - cache_hit_latency)
        )

        def cached_generator():

            yield cached_response

        return StreamingResponse(

            cached_generator(),

            media_type="text/plain"
        )

    CACHE_MISSES.inc()

    print("\nCACHE MISS\n")

    # ========================================
    # RETRIEVAL + CONTEXT
    # ========================================

    retrieval_result = (

        retrieve_and_build_context(

            query=request.query
        )
    )

    context = retrieval_result[
        "context"
    ]

    # ========================================
    # STREAM + CACHE
    # ========================================

    def token_generator():

        full_response = ""

        for token in generate_answer_stream(

            query=request.query,

            context=context
        ):

            full_response += token

            yield token

        # ====================================
        # STORE CACHE
        # ====================================

        set_cached_response(

            request.query,

            full_response
        )

        # ====================================
        # REAL REQUEST LATENCY BASELINE
        # ====================================

        total_request_latency = (

            time.time()
            - request_start_time
        )

        print(
            f"\nRequest latency: "
            f"{total_request_latency:.2f}s"
        )

    # ========================================
    # STREAM RESPONSE
    # ========================================

    return StreamingResponse(

        token_generator(),

        media_type="text/plain"
    )
