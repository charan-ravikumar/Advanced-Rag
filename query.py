# ============================================
# IMPORTS
# ============================================

import os
import time

from dotenv import load_dotenv

from google import genai

from openai import OpenAI

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from rank_bm25 import BM25Okapi

from vectordb import collection

from context_builder import (
    build_context
)

from schemas import (
    QueryResponse,
    SourceDocument,
    RagasMetrics
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from logger_config import (
    get_logger
)

# ============================================
# PHOENIX / OPENTELEMETRY
# ============================================

from opentelemetry import trace

tracer = trace.get_tracer(__name__)


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
# MODELS
# ============================================

PRIMARY_MODEL = (
    "llama-3.1-8b-instant"
)

FALLBACK_MODEL = (
    "gemini-2.0-flash-lite"
)


# ============================================
# INIT GROQ CLIENT
# ============================================

logger.info(
    "Initializing Groq client..."
)

groq_client = OpenAI(

    api_key=GROQ_API_KEY,

    base_url=
    "https://api.groq.com/openai/v1"
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
# LOAD EMBEDDING MODEL
# ============================================

logger.info(
    "Loading embedding model..."
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

logger.info(
    "Embedding model loaded."
)


# ============================================
# LOAD RERANKER
# ============================================

logger.info(
    "Loading reranker model..."
)

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)

logger.info(
    "Reranker loaded."
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
# PROMPT BUILDER
# ============================================

def build_prompt(
    query,
    context
):

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

    logger.info(
        "Using Gemini fallback model."
    )

    response = (

        gemini_client.models.generate_content(

            model=FALLBACK_MODEL,

            contents=prompt
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

                    stream=True
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

        logger.info(
            f"Approx streamed tokens: "
            f"{total_tokens}"
        )

    except Exception as e:

        logger.exception(
            "Groq streaming failed."
        )

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
    k=60
):

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

    top_k=2,

    candidate_k=20,

    vector_threshold=0.35,

    bm25_threshold=0.15,

    vector_weight=1.0,

    bm25_weight=0.7
):

    logger.info(
        f"Query received: {query}"
    )

    retrieval_pipeline_start = time.time()

    # ========================================
    # VECTOR SEARCH
    # ========================================

    logger.info(
        "Generating query embedding."
    )

    query_embedding = embedding_model.encode(
        query
    ).tolist()

    logger.info(
        "Running dense retrieval."
    )

    vector_start = time.time()

    with tracer.start_as_current_span(
        "vector_retrieval"
    ):

        vector_results = collection.query(

            query_embeddings=[
                query_embedding
            ],

            n_results=candidate_k
        )

    vector_latency = (
        time.time()
        - vector_start
    )

    logger.info(
        f"Vector retrieval latency: "
        f"{vector_latency:.2f}s"
    )

    vector_docs = vector_results[
        "documents"
    ][0]

    vector_metas = vector_results[
        "metadatas"
    ][0]

    vector_distances = vector_results[
        "distances"
    ][0]

    vector_rankings = []

    vector_doc_map = {}

    for idx, doc in enumerate(
        vector_docs
    ):

        similarity = 1 / (
            1 + vector_distances[idx]
        )

        if similarity < vector_threshold:
            continue

        doc_id = hash(doc)

        vector_rankings.append(doc_id)

        vector_doc_map[doc_id] = {

            "document": doc,

            "metadata":
                vector_metas[idx],

            "vector_score":
                similarity
        }

    logger.info(
        f"Vector candidates kept: "
        f"{len(vector_rankings)}"
    )

    # ========================================
    # BM25 SEARCH
    # ========================================

    bm25_start = time.time()

    tokenized_query = query.lower().split()

    with tracer.start_as_current_span(
        "bm25_retrieval"
    ):

        bm25_scores = bm25.get_scores(
            tokenized_query
        )

    bm25_latency = (
        time.time()
        - bm25_start
    )

    logger.info(
        f"BM25 latency: "
        f"{bm25_latency:.2f}s"
    )

    max_bm25 = max(bm25_scores)

    if max_bm25 > 0:

        bm25_scores = [
            score / max_bm25
            for score in bm25_scores
        ]

    bm25_rankings = []

    for idx, score in enumerate(
        bm25_scores
    ):

        if score < bm25_threshold:
            continue

        doc = all_documents[idx]

        doc_id = hash(doc)

        bm25_rankings.append(doc_id)

    logger.info(
        f"BM25 candidates kept: "
        f"{len(bm25_rankings)}"
    )

    # ========================================
    # WEIGHTED RRF
    # ========================================

    rrf_scores = weighted_rrf(

        vector_rankings=
            vector_rankings,

        bm25_rankings=
            bm25_rankings,

        vector_weight=
            vector_weight,

        bm25_weight=
            bm25_weight
    )

    ranked_results = sorted(

        rrf_scores.items(),

        key=lambda x: x[1],

        reverse=True
    )

    logger.info(
        f"Candidates after RRF: "
        f"{len(ranked_results)}"
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
    # RERANKING
    # ========================================

    rerank_start = time.time()

    with tracer.start_as_current_span(
        "reranking"
    ):

        rerank_scores = reranker.predict(
            rerank_inputs
        )

    rerank_latency = (
        time.time()
        - rerank_start
    )

    logger.info(
        f"Reranker latency: "
        f"{rerank_latency:.2f}s"
    )

    for idx, score in enumerate(
        rerank_scores
    ):

        rerank_metadata[idx][
            "rerank_score"
        ] = float(score)

    final_results = sorted(

        rerank_metadata,

        key=lambda x:
            x["rerank_score"],

        reverse=True
    )

    top_score = final_results[
        0
    ]["rerank_score"]

    logger.info(
        f"Top rerank score: "
        f"{top_score:.4f}"
    )

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

    total_latency = (
        time.time()
        - retrieval_pipeline_start
    )

    logger.info(
        f"Total retrieval pipeline latency: "
        f"{total_latency:.2f}s"
    )

    return {

        "context": context,

        "sources": sources
    }