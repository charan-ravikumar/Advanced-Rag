import os

import chromadb
from dotenv import load_dotenv

from observability.logger_config import get_logger

logger = get_logger(__name__)

# ============================================
# LOAD ENV
# ============================================

load_dotenv()


# ============================================
# CHROMA CLOUD CLIENT
# ============================================

logger.info("Initializing Chroma Cloud Client...")

_chroma_port_raw = os.getenv("CHROMA_PORT")

client = chromadb.CloudClient(
    cloud_host=os.getenv("CHROMA_HOST"),
    cloud_port=int(_chroma_port_raw) if _chroma_port_raw else None,
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
)

logger.info("Chroma Cloud connected.")


# ============================================
# COLLECTION
# ============================================

COLLECTION_NAME = "rag_chunks"

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"description": "Advanced RAG chunk embeddings"},
)

logger.info(f"Using collection: {COLLECTION_NAME}")


# ============================================
# PUSH TO CHROMA
# ============================================

def push_to_chromadb(embedded_chunks):
    """Upload a list of embedded chunk dicts to the ChromaDB Cloud collection."""
    ids = []
    documents = []
    embeddings = []
    metadatas = []

    total = len(embedded_chunks)
    logger.info(f"Pushing {total} chunks to ChromaDB Cloud...")

    for idx, chunk in enumerate(embedded_chunks):
        logger.debug(f"Preparing chunk {idx + 1}/{total}")
        ids.append(chunk["id"])
        documents.append(chunk["content"])
        embeddings.append(chunk["embedding"])

        clean_metadata = {
            key: value if isinstance(value, (str, int, float, bool)) else str(value)
            for key, value in chunk["metadata"].items()
        }
        metadatas.append(clean_metadata)

    logger.info("Uploading embeddings to ChromaDB Cloud...")
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    logger.info(f"Upload complete. Collection count: {collection.count()}")


# ============================================
# CLEAR COLLECTION
# ============================================

def clear_collection():
    """Delete and recreate the collection. Returns the number of documents deleted."""
    global collection

    count_before = collection.count()
    logger.info(f"Clearing collection '{COLLECTION_NAME}' ({count_before} docs)...")

    client.delete_collection(name=COLLECTION_NAME)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Advanced RAG chunk embeddings"},
    )

    logger.info(f"Collection '{COLLECTION_NAME}' cleared and recreated.")
    return count_before


# ============================================
# LIVE COLLECTION COUNT
# ============================================

def get_collection_count() -> int:
    """Return the current document count for the live collection reference."""
    return collection.count()
