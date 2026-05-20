# ============================================
# IMPORTS
# ============================================

from fastapi import FastAPI

from fastapi.responses import (

    StreamingResponse
)

from schemas import (
    QueryRequest
)

from query import (

    retrieve_and_build_context,

    generate_answer_stream
)


# ============================================
# FASTAPI INIT
# ============================================

app = FastAPI(

    title="Advanced RAG API",

    version="2.0.0"
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
    # TOKEN GENERATOR
    # ========================================

    def token_generator():

        for token in generate_answer_stream(

            query=request.query,

            context=context
        ):

            yield token

    # ========================================
    # STREAM RESPONSE
    # ========================================

    return StreamingResponse(

        token_generator(),

        media_type="text/plain"
    )