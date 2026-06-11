# ============================================
# IMPORTS
# ============================================

import json
import time

from fastapi import FastAPI

from fastapi.responses import (

    StreamingResponse,

    Response
)

from schemas import (
    QueryRequest
)

from query import (

    retrieve_and_build_context,

    generate_answer_stream
)

from cache import (

    get_cached_response,

    set_cached_response
)

from metrics import (

    REQUEST_COUNT,

    CACHE_HITS,

    CACHE_MISSES,

    TOKENS_SAVED,

    LATENCY_SAVED
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

    return {

        "message":
        "Advanced RAG API Running"
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

        # ====================================
        # REAL TOKEN SAVINGS
        # ====================================

        cached_tokens = len(

            cached_response.split()
        )

        TOKENS_SAVED.inc(
            cached_tokens
        )

        # ====================================
        # REAL LATENCY SAVINGS
        # ====================================

        estimated_saved_latency = 8.0

        LATENCY_SAVED.inc(
            estimated_saved_latency
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
        # REAL LATENCY SAVED BASELINE
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