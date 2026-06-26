# ============================================
# IMPORTS
# ============================================

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from google import genai

from openai import OpenAI

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from rank_bm25 import BM25Okapi

from db.vectordb import collection

from config import cfg

from rag.context_builder import (
    build_context
)

from rag.embeddings import encode_query, get_embedding_cache_stats

from schemas import (
    QueryResponse,
    SourceDocument,
    RagasMetrics
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from observability.logger_config import (
    get_logger
)

# ============================================
# PHOENIX / OPENTELEMETRY
# ============================================

from opentelemetry import trace

tracer = trace.get_tracer(__name__)


from observability.metrics import (

    RETRIEVAL_LATENCY,

    RERANK_LATENCY,

    GENERATION_LATENCY,

    GENERATION_FAILURES
)

# ============================================
# LOGGER
# ============================================

logger = get_logger(__name__)


# ============================================
# RAGAS TOGGLE
# ============================================

ENABLE_RAGAS = False


# ============================================
# OPTIONAL RAGAS IMPORTS
# ============================================

if ENABLE_RAGAS:

    from datasets import Dataset

    from ragas import evaluate

    from ragas.metrics import (
        faithfulness,
        answer_relevancy
    )

    from langchain_google_genai import (
        ChatGoogleGenerativeAI
    )


# ============================================
# LOAD ENV
# ============================================

load_dotenv()

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)


# ============================================
# MODELS  (config-driven)
# ============================================

PRIMARY_MODEL = "llama-3.1-8b-instant"
FALLBACK_MODEL = "gemini-2.0-flash-lite"


# ============================================
# INIT GROQ CLIENT
# ============================================

logger.info(
    "Initializing Groq client..."
)

groq_client = OpenAI(

    api_key=GROQ_API_KEY,

    base_url=
    "https://api.groq.com/openai/v1",

    timeout=30.0,

    max_retries=1,
)

logger.info(
    "Groq client initialized."
)


# ============================================
# INIT GEMINI CLIENT
# ============================================

logger.info(
    "Initializing Gemini client..."
)

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)

logger.info(
    "Gemini client initialized."
)


# ============================================
# LOAD RERANKER  (config-driven, optional)
# ============================================

reranker = None
if cfg.pipeline.reranker_enabled:
    logger.info(
        f"Loading reranker: {cfg.pipeline.reranker_model}"
    )
    reranker = CrossEncoder(cfg.pipeline.reranker_model)
    logger.info("Reranker loaded.")
else:
    logger.info(
        "Reranker disabled (reranker_enabled=false in config.yaml)."
    )


# ============================================
# LOAD DOCUMENTS
# ============================================

logger.info(
    "Loading documents from ChromaDB..."
)

all_data = collection.get(
    include=["documents", "metadatas"]
)

all_documents = all_data["documents"]
all_metadatas = all_data["metadatas"]

logger.info(
    f"Loaded {len(all_documents)} documents."
)


# ============================================
# BM25 INDEX
# ============================================

tokenized_corpus = [
    doc.lower().split()
    for doc in all_documents
]

bm25 = BM25Okapi(tokenized_corpus)

logger.info(
    "BM25 initialized."
)


# ============================================
# QUERY ROUTER
# ============================================

_CONVERSATIONAL_PATTERNS = [
    r"^\s*(hi|hello|hey|howdy|hiya|yo)\b",
    r"^\s*good\s+(morning|afternoon|evening|day)\b",
    r"^\s*how\s+are\s+you\b",
    r"^\s*what('?s|\s+is)\s+up\b",
    r"^\s*(thanks|thank\s+you|thx|ty)\b",
    r"^\s*(bye|goodbye|see\s+you|cya)\b",
    r"^\s*who\s+are\s+you\b",
    r"^\s*what\s+(are|can)\s+you\b",
    r"^\s*(help|help\s+me)\s*$",
    r"^\s*ok(ay)?\s*$",
    r"^\s*(sure|yes|no|nope|yep)\s*$",
]

import re as _re

def classify_query(query: str) -> str:
    """
    Returns 'direct' for conversational/chitchat queries that don't
    need document retrieval, or 'rag' for everything else.
    Uses fast regex first, then a lightweight Groq LLM call for ambiguous cases.
    """
    q = query.strip().lower()

    # Fast path: obvious conversational patterns
    for pattern in _CONVERSATIONAL_PATTERNS:
        if _re.match(pattern, q, _re.IGNORECASE):
            logger.info(f"Query classified as DIRECT (regex match): {pattern}")
            return "direct"

    # Fast path: very short queries that look like small talk (≤ 4 words, no "?")
    word_count = len(q.split())
    if word_count <= 3 and "?" not in q:
        logger.info("Query classified as DIRECT (short, no question mark)")
        return "direct"

    # LLM classification for everything else
    try:
        classification_prompt = (
            "You are a query router. Decide if the user message needs document retrieval "
            "from a knowledge base (answer: RAG) or can be answered directly as a "
            "conversational response without any documents (answer: DIRECT).\n\n"
            "Reply with exactly one word: RAG or DIRECT.\n\n"
            f"User message: {query}"
        )
        resp = groq_client.chat.completions.create(
            model=PRIMARY_MODEL,
            messages=[{"role": "user", "content": classification_prompt}],
            temperature=0,
            max_tokens=5,
            timeout=15.0,
        )
        verdict = resp.choices[0].message.content.strip().upper()
        result = "direct" if verdict == "DIRECT" else "rag"
        logger.info(f"Query classified as {result.upper()} (LLM verdict: {verdict})")
        return result
    except Exception:
        logger.exception("Query classification failed, defaulting to RAG.")
        return "rag"


# ============================================
# DIRECT CONVERSATIONAL ANSWER (streaming)
# ============================================

def generate_direct_answer_stream(query: str):
    """Stream a conversational answer without hitting the vector store."""
    logger.info("Generating direct conversational answer...")
    system = (
        "You are a helpful, friendly assistant. "
        "Answer the user's message naturally and concisely. "
        "Do not mention documents or knowledge bases unless directly asked."
    )
    try:
        response = groq_client.chat.completions.create(
            model=PRIMARY_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            temperature=0.7,
            stream=True,
            timeout=60.0,
        )
        for chunk in response:
            try:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
            except Exception:
                continue
    except Exception:
        logger.exception("Direct answer generation failed.")
        yield "Sorry, I couldn't generate a response right now."


# ============================================
# PROMPT BUILDER
# ============================================

def build_prompt(
    query,
    context
):
    """Construct the system+user prompt string for the LLM from a query and retrieved context."""
    system_prompt = """
You are an advanced enterprise RAG assistant.

Answer the user's question directly and concisely.

Answer ONLY using the retrieved context.

If the answer is not found in the context,
say:

'I could not find that information in the provided documents.'

Do not hallucinate.

Cite sources whenever possible.

Do not include unrelated supporting information.
"""

    prompt = f"""

SYSTEM:
{system_prompt}

USER QUESTION:
{query}

RETRIEVED CONTEXT:
{context}

ANSWER:
"""

    return prompt


# ============================================
# GEMINI FALLBACK
# ============================================

def generate_gemini_answer(
    prompt
):
    """Generate a non-streaming answer using the Gemini fallback model."""
    logger.info(
        "Using Gemini fallback model."
    )

    response = (

        gemini_client.models.generate_content(

            model=FALLBACK_MODEL,

            contents=prompt,

            config={"timeout": 30},
        )
    )

    return response.text


# ============================================
# STREAMING GENERATION
# ============================================

def generate_answer_stream(
    query,
    context
):
    """Stream the LLM answer token by token using Groq (Gemini fallback on failure)."""
    logger.info(
        "Generating streaming answer..."
    )

    prompt = build_prompt(

        query=query,

        context=context
    )

    generation_start = time.time()

    try:

        with tracer.start_as_current_span(
            "llm_generation"
        ):

            response = (

                groq_client.chat.completions.create(

                    model=PRIMARY_MODEL,

                    messages=[

                        {
                            "role": "user",

                            "content": prompt
                        }
                    ],

                    temperature=0,

                    stream=True,

                    timeout=120.0,
                )
            )

            total_tokens = 0

            for chunk in response:

                try:

                    delta = (

                        chunk.choices[0]
                        .delta
                        .content
                    )

                    if delta:

                        total_tokens += len(
                            delta.split()
                        )

                        yield delta

                except Exception:

                    continue

        generation_latency = (
            time.time()
            - generation_start
        )

        logger.info(
            f"Generation latency: "
            f"{generation_latency:.2f}s"
        )

        GENERATION_LATENCY.observe(
        generation_latency
        )

        logger.info(
            f"Approx streamed tokens: "
            f"{total_tokens}"
        )

    except Exception as e:

        logger.exception(
            "Groq streaming failed."
        )

        GENERATION_FAILURES.inc()

        # ====================================
        # GEMINI FALLBACK
        # ====================================

        try:

            with tracer.start_as_current_span(
                "gemini_fallback_generation"
            ):

                gemini_answer = (

                    generate_gemini_answer(
                        prompt
                    )
                )

            yield gemini_answer

        except Exception as fallback_error:

            logger.exception(
                "Gemini fallback failed."
            )

            yield (
                "\n\nGeneration failed on both "
                "Groq and Gemini."
            )


# ============================================
# OPTIONAL RAGAS
# ============================================

def run_ragas_evaluation(

    query,
    answer,
    contexts
):
    """Evaluate a RAG response with RAGAS faithfulness and answer_relevancy metrics."""
    if not ENABLE_RAGAS:
        return None

    logger.info(
        "Running RAGAS evaluation."
    )

    try:

        evaluator_llm = ChatGoogleGenerativeAI(

            model="gemini-2.0-flash-lite",

            google_api_key=GEMINI_API_KEY,

            temperature=0
        )

        evaluator_embeddings = (

            HuggingFaceEmbeddings(

                model_name=
                "sentence-transformers/all-MiniLM-L6-v2"
            )
        )

        sample = {

            "question": query,

            "answer": answer,

            "contexts": contexts
        }

        dataset = Dataset.from_list(
            [sample]
        )

        results = evaluate(

            dataset,

            metrics=[

                faithfulness,

                answer_relevancy
            ],

            llm=evaluator_llm,

            embeddings=evaluator_embeddings
        )

        faithfulness_score = None
        answer_relevancy_score = None

        try:

            raw_faithfulness = results[
                "faithfulness"
            ]

            if isinstance(
                raw_faithfulness,
                list
            ):

                if len(raw_faithfulness) > 0:

                    faithfulness_score = float(
                        raw_faithfulness[0]
                    )

            elif raw_faithfulness is not None:

                faithfulness_score = float(
                    raw_faithfulness
                )

        except Exception:

            logger.exception(
                "Faithfulness parsing failed."
            )

        try:

            raw_relevancy = results[
                "answer_relevancy"
            ]

            if isinstance(
                raw_relevancy,
                list
            ):

                if len(raw_relevancy) > 0:

                    answer_relevancy_score = float(
                        raw_relevancy[0]
                    )

            elif raw_relevancy is not None:

                answer_relevancy_score = float(
                    raw_relevancy
                )

        except Exception:

            logger.exception(
                "Answer relevancy parsing failed."
            )

        logger.info(
            f"Faithfulness Score: "
            f"{faithfulness_score}"
        )

        logger.info(
            f"Answer Relevancy Score: "
            f"{answer_relevancy_score}"
        )

        return RagasMetrics(

            faithfulness=
            faithfulness_score,

            answer_relevancy=
            answer_relevancy_score
        )

    except Exception:

        logger.exception(
            "RAGAS evaluation failed."
        )

        return None


# ============================================
# WEIGHTED RRF
# ============================================

def weighted_rrf(
    vector_rankings,
    bm25_rankings,
    vector_weight=1.0,
    bm25_weight=0.7,
    k=None
):
    """Merge two ranked lists using Weighted Reciprocal Rank Fusion and return a fused dict of scores."""
    if k is None:
        k = cfg.pipeline.rrf_k

    logger.info(
        "Applying Weighted RRF."
    )

    rrf_scores = {}

    for rank, doc_id in enumerate(
        vector_rankings
    ):

        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = 0

        rrf_scores[doc_id] += (

            vector_weight
            *
            (1 / (k + rank + 1))
        )

    for rank, doc_id in enumerate(
        bm25_rankings
    ):

        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = 0

        rrf_scores[doc_id] += (

            bm25_weight
            *
            (1 / (k + rank + 1))
        )

    return rrf_scores


# ============================================
# RETRIEVAL + CONTEXT
# ============================================

def retrieve_and_build_context(

    query,

    top_k=None,

    candidate_k=None,

    vector_threshold=None,

    bm25_threshold=None,

    vector_weight=None,

    bm25_weight=None,

    retrieval_mode_override=None,

    use_rrf_override=None,

    reranker_override=None,

    metadata_filter=None,
):
    """Run the full retrieval pipeline and return a dict with context, sources, and latency_ms."""
    # Fall back to config values when not explicitly overridden
    top_k            = top_k            if top_k            is not None else cfg.pipeline.top_k
    candidate_k      = candidate_k      if candidate_k      is not None else cfg.pipeline.candidate_k
    vector_threshold = vector_threshold if vector_threshold is not None else cfg.pipeline.vector_threshold
    bm25_threshold   = bm25_threshold   if bm25_threshold   is not None else cfg.pipeline.bm25_threshold
    vector_weight    = vector_weight    if vector_weight    is not None else cfg.pipeline.vector_weight
    bm25_weight      = bm25_weight      if bm25_weight      is not None else cfg.pipeline.bm25_weight

    logger.info(
        f"Query received: {query}"
    )

    retrieval_pipeline_start = time.time()

    # ========================================
    # PARALLEL VECTOR + BM25 SEARCH
    # Both searches run concurrently via ThreadPoolExecutor.
    # RRF fusion waits for both to complete.
    # ========================================

    logger.info("Generating query embedding (LRU cache).")
    query_embedding = encode_query(query)
    emb_cache_stats = get_embedding_cache_stats()
    logger.info(
        f"Embedding cache — hits: {emb_cache_stats['hits']}  "
        f"misses: {emb_cache_stats['misses']}  "
        f"size: {emb_cache_stats['current_size']}/{emb_cache_stats['max_size']}"
    )

    retrieval_mode = retrieval_mode_override if retrieval_mode_override is not None else cfg.pipeline.retrieval  # "vector", "bm25", or "hybrid"

    def _vector_search():
        t0 = time.time()
        with tracer.start_as_current_span("vector_retrieval"):
            query_kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": candidate_k,
            }
            if metadata_filter:
                query_kwargs["where"] = metadata_filter
            results = collection.query(**query_kwargs)
        return results, (time.time() - t0) * 1000  # ms

    def _bm25_search():
        t0 = time.time()
        tokenized_query = query.lower().split()
        with tracer.start_as_current_span("bm25_retrieval"):
            scores = bm25.get_scores(tokenized_query)
        return scores, (time.time() - t0) * 1000  # ms

    vector_results = None
    bm25_scores_raw = None
    vector_ms = 0.0
    bm25_ms = 0.0

    run_vector = retrieval_mode in ("vector", "hybrid")
    run_bm25   = retrieval_mode in ("bm25",   "hybrid")

    if run_vector and run_bm25:
        # Run both in parallel
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_vector = pool.submit(_vector_search)
            fut_bm25   = pool.submit(_bm25_search)
            vector_results, vector_ms = fut_vector.result()
            bm25_scores_raw, bm25_ms  = fut_bm25.result()
    elif run_vector:
        vector_results, vector_ms = _vector_search()
    elif run_bm25:
        bm25_scores_raw, bm25_ms = _bm25_search()

    logger.info(
        f"vector_search_ms={vector_ms:.1f}  "
        f"bm25_search_ms={bm25_ms:.1f}"
    )

    # ---- Process vector results ----
    vector_rankings = []
    vector_doc_map  = {}

    if vector_results is not None:
        vec_docs  = vector_results["documents"][0]
        vec_metas = vector_results["metadatas"][0]
        vec_dists = vector_results["distances"][0]
        for idx, doc in enumerate(vec_docs):
            sim = 1 / (1 + vec_dists[idx])
            if sim < vector_threshold:
                continue
            doc_id = hash(doc)
            vector_rankings.append(doc_id)
            vector_doc_map[doc_id] = {
                "document": doc,
                "metadata": vec_metas[idx],
                "vector_score": sim,
            }
        logger.info(f"Vector candidates kept: {len(vector_rankings)}")

    # ---- Process BM25 results ----
    bm25_rankings = []

    if bm25_scores_raw is not None:
        max_bm25 = max(bm25_scores_raw) or 1.0
        norm_scores = [s / max_bm25 for s in bm25_scores_raw]
        for idx, score in enumerate(norm_scores):
            if score < bm25_threshold:
                continue
            bm25_rankings.append(hash(all_documents[idx]))
        logger.info(f"BM25 candidates kept: {len(bm25_rankings)}")

    # ========================================
    # FUSION  (RRF or skip)
    # ========================================

    fusion_start = time.time()

    use_rrf = (
        use_rrf_override
        if use_rrf_override is not None
        else (cfg.pipeline.fusion == "rrf" and retrieval_mode == "hybrid")
    )

    if use_rrf:
        rrf_scores = weighted_rrf(
            vector_rankings=vector_rankings,
            bm25_rankings=bm25_rankings,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            k=cfg.pipeline.rrf_k,
        )
        ranked_results = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )
    else:
        # No fusion: just use vector rankings (or bm25 if vector-only disabled)
        source = vector_rankings if vector_rankings else bm25_rankings
        ranked_results = [(doc_id, 1.0) for doc_id in source]

    fusion_ms = (time.time() - fusion_start) * 1000
    logger.info(
        f"fusion_ms={fusion_ms:.1f}  candidates_after_fusion={len(ranked_results)}"
    )

    # ========================================
    # PREPARE RERANKING
    # ========================================

    rerank_inputs = []

    rerank_metadata = []

    for doc_id, rrf_score in ranked_results:

        if doc_id in vector_doc_map:

            doc_info = vector_doc_map[doc_id]

            document = doc_info["document"]

            metadata = doc_info["metadata"]

            vector_score = doc_info[
                "vector_score"
            ]

        else:

            found_idx = None

            for i, doc in enumerate(
                all_documents
            ):

                if hash(doc) == doc_id:

                    found_idx = i

                    break

            if found_idx is None:
                continue

            document = all_documents[
                found_idx
            ]

            metadata = all_metadatas[
                found_idx
            ]

            vector_score = 0

        rerank_inputs.append(
            [query, document]
        )

        rerank_metadata.append({

            "document": document,

            "metadata": metadata,

            "rrf_score": rrf_score,

            "vector_score":
                vector_score
        })

    # ========================================
    # RERANKING  (config-driven on/off)
    # ========================================

    rerank_ms = 0.0

    _reranker_active = reranker_override if reranker_override is not None else cfg.pipeline.reranker_enabled
    if _reranker_active and reranker is not None and rerank_inputs:
        rerank_start = time.time()
        with tracer.start_as_current_span("reranking"):
            rerank_scores = reranker.predict(rerank_inputs)
        rerank_ms = (time.time() - rerank_start) * 1000
        RERANK_LATENCY.observe(rerank_ms / 1000)
        logger.info(f"reranker_ms={rerank_ms:.1f}")
        for idx, score in enumerate(rerank_scores):
            rerank_metadata[idx]["rerank_score"] = float(score)
        final_results = sorted(
            rerank_metadata, key=lambda x: x["rerank_score"], reverse=True
        )
        logger.info(f"Top rerank score: {final_results[0]['rerank_score']:.4f}")
    else:
        # No reranker — use RRF score as proxy
        for item in rerank_metadata:
            item.setdefault("rerank_score", item.get("rrf_score", 0.0))
        final_results = rerank_metadata

    # ========================================
    # CONTEXT ENGINEERING
    # ========================================

    context_start = time.time()

    with tracer.start_as_current_span(
        "context_engineering"
    ):

        context = build_context(

            final_results[:top_k],

            max_tokens=1500
        )

    context_latency = (
        time.time()
        - context_start
    )

    logger.info(
        f"Context engineering latency: "
        f"{context_latency:.2f}s"
    )

    # ========================================
    # SOURCES
    # ========================================

    sources = []

    for result in final_results[:top_k]:

        metadata = result["metadata"]

        sources.append(

            SourceDocument(

                source=metadata.get(
                    "source"
                ),

                section_title=metadata.get(
                    "section_title"
                ),

                content=result[
                    "document"
                ],

                rerank_score=float(

                    result[
                        "rerank_score"
                    ]
                )
            )
        )

    total_ms = (time.time() - retrieval_pipeline_start) * 1000

    logger.info(
        f"retrieval_trace | "
        f"vector_search_ms={vector_ms:.1f} "
        f"bm25_search_ms={bm25_ms:.1f} "
        f"fusion_ms={fusion_ms:.1f} "
        f"reranker_ms={rerank_ms:.1f} "
        f"total_retrieval_ms={total_ms:.1f}"
    )

    RETRIEVAL_LATENCY.observe(total_ms / 1000)

    return {
        "context": context,
        "sources": sources,
        "latency_ms": {
            "vector_search_ms": round(vector_ms, 1),
            "bm25_search_ms":   round(bm25_ms, 1),
            "fusion_ms":        round(fusion_ms, 1),
            "reranker_ms":      round(rerank_ms, 1),
            "total_retrieval_ms": round(total_ms, 1),
        },
    }
