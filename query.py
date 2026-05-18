# ============================================
# IMPORTS
# ============================================

import os

from dotenv import load_dotenv

from google import genai

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from rank_bm25 import BM25Okapi

from vectordb import collection

from context_builder import (
    build_context
)


# ============================================
# LOAD ENV
# ============================================

load_dotenv()

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)


# ============================================
# INIT GEMINI CLIENT
# ============================================

print("\nInitializing Gemini client...\n")

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)

print("Gemini client initialized.\n")


# ============================================
# MODELS
# ============================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.\n")


print("Loading reranker model...")

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)

print("Reranker loaded.\n")


# ============================================
# LOAD DOCUMENTS
# ============================================

print("Loading documents from ChromaDB...\n")

all_data = collection.get(
    include=["documents", "metadatas"]
)

all_documents = all_data["documents"]
all_metadatas = all_data["metadatas"]

print(
    f"Loaded {len(all_documents)} documents.\n"
)


# ============================================
# BM25 INDEX
# ============================================

tokenized_corpus = [
    doc.lower().split()
    for doc in all_documents
]

bm25 = BM25Okapi(tokenized_corpus)

print("BM25 initialized.\n")


# ============================================
# PROMPT BUILDER
# ============================================

def build_prompt(
    query,
    context
):

    system_prompt = """
You are an advanced enterprise RAG assistant.

Answer ONLY using the retrieved context.

If the answer is not found in the context,
say:

'I could not find that information in the provided documents.'

Do not hallucinate.

Cite sources whenever possible.
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
# GENERATE ANSWER
# ============================================

def generate_answer(
    query,
    context
):

    print("\nGenerating final answer...\n")

    prompt = build_prompt(
        query=query,
        context=context
    )

    response = gemini_client.models.generate_content(

        model="gemini-2.5-flash",

        contents=prompt
    )

    answer = response.text

    return answer


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

    print("\nApplying Weighted RRF...\n")

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
# HYBRID SEARCH
# ============================================

def hybrid_search(

    query,

    top_k=3,

    candidate_k=20,

    vector_threshold=0.35,

    bm25_threshold=0.15,

    vector_weight=1.0,

    bm25_weight=0.7
):

    print("\n===================================")
    print("ADVANCED RAG PIPELINE")
    print("===================================\n")

    print(f"Query: {query}")

    # ========================================
    # VECTOR SEARCH
    # ========================================

    print("\nGenerating query embedding...\n")

    query_embedding = embedding_model.encode(
        query
    ).tolist()

    print("Running dense retrieval...\n")

    vector_results = collection.query(

        query_embeddings=[
            query_embedding
        ],

        n_results=candidate_k
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

    # ========================================
    # VECTOR THRESHOLDING
    # ========================================

    print("Applying vector thresholding...\n")

    vector_rankings = []

    vector_doc_map = {}

    for idx, doc in enumerate(
        vector_docs
    ):

        similarity = 1 / (
            1 + vector_distances[idx]
        )

        print(
            f"Vector Similarity: "
            f"{similarity:.4f}"
        )

        if similarity < vector_threshold:

            print(
                "→ Below vector threshold."
            )

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

    print(
        f"\nVector candidates kept: "
        f"{len(vector_rankings)}"
    )

    # ========================================
    # BM25 SEARCH
    # ========================================

    print("\nRunning BM25 retrieval...\n")

    tokenized_query = query.lower().split()

    bm25_scores = bm25.get_scores(
        tokenized_query
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

    print(
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

    # ========================================
    # SORT RRF RESULTS
    # ========================================

    ranked_results = sorted(

        rrf_scores.items(),

        key=lambda x: x[1],

        reverse=True
    )

    print(
        f"\nCandidates after RRF: "
        f"{len(ranked_results)}"
    )

    # ========================================
    # PREPARE RERANKING
    # ========================================

    print("\nPreparing reranker inputs...\n")

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

    print("\nRunning cross-encoder reranking...\n")

    rerank_scores = reranker.predict(
        rerank_inputs
    )

    for idx, score in enumerate(
        rerank_scores
    ):

        rerank_metadata[idx][
            "rerank_score"
        ] = float(score)

    # ========================================
    # FINAL SORT
    # ========================================

    final_results = sorted(

        rerank_metadata,

        key=lambda x:
            x["rerank_score"],

        reverse=True
    )

    # ========================================
    # DISPLAY RETRIEVAL
    # ========================================

    print("===================================")
    print("FINAL RERANKED RESULTS")
    print("===================================\n")

    for idx, result in enumerate(
        final_results[:top_k]
    ):

        print(f"RESULT {idx + 1}\n")

        print(
            f"Rerank Score: "
            f"{result['rerank_score']:.4f}"
        )

        print(
            f"RRF Score: "
            f"{result['rrf_score']:.6f}"
        )

        print(
            f"Vector Score: "
            f"{result['vector_score']:.4f}"
        )

        print("\nMetadata:\n")

        for key, value in result[
            "metadata"
        ].items():

            print(f"{key}: {value}")

        print("\nContent:\n")

        print(
            result["document"][:1000]
        )

        print(
            "\n-----------------------------------\n"
        )

    # ========================================
    # CONTEXT ENGINEERING
    # ========================================

    print("\nBuilding optimized context...\n")

    context = build_context(

        final_results[:top_k],

        max_tokens=1500
    )

    # ========================================
    # GENERATION
    # ========================================

    answer = generate_answer(
        query=query,
        context=context
    )

    # ========================================
    # FINAL ANSWER
    # ========================================

    print("\n===================================")
    print("FINAL GENERATED ANSWER")
    print("===================================\n")

    print(answer)

    print("\n===================================")
    print("PIPELINE COMPLETE")
    print("===================================\n")


# ============================================
# INTERACTIVE LOOP
# ============================================

def interactive_search():

    print("\n===================================")
    print("ADVANCED RAG SYSTEM")
    print("===================================\n")

    print("Type 'exit' to quit.\n")

    while True:

        query = input("Enter Query: ")

        if query.lower() == "exit":

            print("\nExiting.\n")

            break

        hybrid_search(
            query=query
        )


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    interactive_search()