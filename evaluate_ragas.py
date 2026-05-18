# ============================================
# IMPORTS
# ============================================

from datasets import Dataset

from ragas import evaluate

from ragas.metrics import (

    faithfulness,

    answer_relevancy,

    context_precision,

    context_recall
)

from query import hybrid_search


# ============================================
# TEST QUERIES
# ============================================

evaluation_queries = [

    "What is RAG?",

    "What is Reciprocal Rank Fusion?",

    "What is semantic chunking?",

    "Who maintains the onboarding guide?",

    "What is the purpose of embeddings?",

    "What vector database is used?"
]


# ============================================
# GENERATE EVALUATION DATA
# ============================================

print("\n===================================")
print("GENERATING EVALUATION DATASET")
print("===================================\n")

samples = []

for query in evaluation_queries:

    print(f"\nRunning Query: {query}\n")

    result = hybrid_search(

        query=query,

        top_k=3
    )

    samples.append({

        "question":
            result["query"],

        "answer":
            result["answer"],

        "contexts":
            result["contexts"]
    })


# ============================================
# CREATE DATASET
# ============================================

print("\nCreating HuggingFace Dataset...\n")

dataset = Dataset.from_list(
    samples
)

print(
    f"Dataset created with "
    f"{len(samples)} samples.\n"
)


# ============================================
# RUN RAGAS
# ============================================

print("\n===================================")
print("RUNNING RAGAS EVALUATION")
print("===================================\n")

results = evaluate(

    dataset,

    metrics=[

        faithfulness,

        answer_relevancy,

        context_precision,

        context_recall
    ]
)


# ============================================
# FINAL RESULTS
# ============================================

print("\n===================================")
print("RAGAS RESULTS")
print("===================================\n")

print(results)

print("\n===================================")
print("EVALUATION COMPLETE")
print("===================================\n")