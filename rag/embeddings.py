import hashlib
import json
import uuid
from functools import lru_cache
from pathlib import Path

from sentence_transformers import SentenceTransformer

from config import cfg
from observability.logger_config import get_logger

logger = get_logger(__name__)

# ============================================
# LOAD EMBEDDING MODEL  (config-driven)
# ============================================

logger.info("Loading embedding model...")
embedding_model = SentenceTransformer(cfg.embedding.model)
logger.info("Embedding model loaded.")

# ============================================
# LRU QUERY EMBEDDING CACHE
# ============================================

_cache_hits = 0
_cache_misses = 0


@lru_cache(maxsize=cfg.embedding.lru_cache_size)
def _cached_encode(query_hash: str, query: str) -> tuple:
    """Inner function cached by (hash, query). Returns embedding as tuple."""
    return tuple(embedding_model.encode(query).tolist())


def encode_query(query: str) -> list:
    """Encode a query string, returning a list. Uses LRU cache."""
    global _cache_hits, _cache_misses
    query_hash = hashlib.sha256(query.encode()).hexdigest()
    info = _cached_encode.cache_info()
    result = _cached_encode(query_hash, query)
    new_info = _cached_encode.cache_info()
    if new_info.hits > info.hits:
        _cache_hits += 1
    else:
        _cache_misses += 1
    return list(result)


def get_embedding_cache_stats() -> dict:
    """Return LRU cache hit/miss stats."""
    info = _cached_encode.cache_info()
    return {
        "hits": info.hits,
        "misses": info.misses,
        "current_size": info.currsize,
        "max_size": info.maxsize,
    }


# ============================================
# GENERATE EMBEDDINGS
# ============================================

def generate_embeddings(chunks):
    """Embed a list of Chunk objects and return a list of embedding dicts."""
    embedded_chunks = []
    total_chunks = len(chunks)
    logger.info(f"Starting embedding generation for {total_chunks} chunks.")

    for idx, chunk in enumerate(chunks):
        logger.debug(f"Embedding chunk {idx + 1}/{total_chunks}")
        embedding = embedding_model.encode(chunk.content)
        embedded_chunks.append({
            "id": str(uuid.uuid4()),
            "content": chunk.content,
            "embedding": embedding.tolist(),
            "metadata": chunk.metadata,
        })

    logger.info(f"Embedding generation complete. Total embedded: {len(embedded_chunks)}")
    return embedded_chunks


# ============================================
# SAVE TO JSON
# ============================================

def save_embeddings_to_json(embedded_chunks, output_path="embedded_chunks.json"):
    """Persist embedded chunks as a JSON file and log the file size."""
    logger.info(f"Saving embeddings to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(embedded_chunks, f, indent=2, ensure_ascii=False)
    file_size = Path(output_path).stat().st_size
    logger.info(f"Embeddings saved. File size: {round(file_size / 1024 / 1024, 2)} MB")


# ============================================
# COMPLETE PIPELINE
# ============================================

def embedding_pipeline(chunks):
    """Run generate_embeddings then save_embeddings_to_json and return the embedded chunks."""
    embedded_chunks = generate_embeddings(chunks)
    save_embeddings_to_json(embedded_chunks)
    return embedded_chunks