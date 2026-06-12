import json
import uuid
from pathlib import Path

from sentence_transformers import SentenceTransformer


# ============================================
# LOAD EMBEDDING MODEL
# ============================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.\n")


# ============================================
# GENERATE EMBEDDINGS
# ============================================

def generate_embeddings(chunks):

    print("\n===================================")
    print("STARTING EMBEDDING GENERATION")
    print("===================================\n")

    embedded_chunks = []

    total_chunks = len(chunks)

    print(f"Total chunks to embed: {total_chunks}\n")

    for idx, chunk in enumerate(chunks):

        print(
            f"Embedding chunk "
            f"{idx + 1}/{total_chunks}"
        )

        # ------------------------------------
        # GENERATE EMBEDDING
        # ------------------------------------

        embedding = embedding_model.encode(
            chunk.content
        )

        # ------------------------------------
        # CREATE EMBEDDING RECORD
        # ------------------------------------

        embedded_chunk = {

            "id": str(uuid.uuid4()),

            "content": chunk.content,

            "embedding": embedding.tolist(),

            "metadata": chunk.metadata
        }

        embedded_chunks.append(
            embedded_chunk
        )

        print(
            f"Embedding dimension: "
            f"{len(embedding)}"
        )

    print("\n===================================")
    print("EMBEDDING GENERATION COMPLETE")
    print("===================================\n")

    print(
        f"Total embedded chunks: "
        f"{len(embedded_chunks)}"
    )

    return embedded_chunks


# ============================================
# SAVE TO JSON
# ============================================

def save_embeddings_to_json(
    embedded_chunks,
    output_path="embedded_chunks.json"
):

    print("\nSaving embeddings to JSON...\n")

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            embedded_chunks,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"Embeddings saved to: "
        f"{output_path}"
    )

    file_size = (
        Path(output_path)
        .stat()
        .st_size
    )

    print(
        f"File size: "
        f"{round(file_size / 1024 / 1024, 2)} MB"
    )


# ============================================
# COMPLETE PIPELINE
# ============================================

def embedding_pipeline(chunks):

    embedded_chunks = generate_embeddings(
        chunks
    )

    save_embeddings_to_json(
        embedded_chunks
    )

    return embedded_chunks
