"""
Qasper Ingestion Script
=======================
Clears ChromaDB and ingests papers from the official Qasper v0.3 dataset
(https://allenai.org/data/qasper) using ALL three chunking strategies:

    recursive  — fixed-size overlapping windows
    layout     — one chunk per paper section
    semantic   — embedding-guided paragraph grouping

Every chunk carries a `chunking_strategy` metadata field so
you can filter and compare strategies at query time.

The script downloads the official JSON directly from Allen AI's S3 bucket
(cached locally at .cache/qasper/) so it works with any datasets version.

Usage (run from project root):
    python -m scripts.ingest_qasper

Configuration (environment variables or edit below):
    QASPER_MAX_PAPERS   — how many papers to ingest (default: 30)
    QASPER_SPLIT        — train | dev              (default: train)
    QASPER_STRATEGIES   — comma-separated list     (default: recursive,layout,semantic)
"""

# ============================================
# IMPORTS
# ============================================

import io
import json
import os
import sys
import tarfile
import time
import uuid

import requests

# ------------------------------------
# Add project root to path so that
# package imports resolve correctly
# when running as a script.
# ------------------------------------

ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sentence_transformers import SentenceTransformer

from schemas import Document, Section, Chunk

from rag.chunkers import CHUNKER_MAP

from db.vectordb import (
    push_to_chromadb,
    clear_collection,
    get_collection_count,
)

from observability.logger_config import get_logger


# ============================================
# LOGGER
# ============================================

logger = get_logger(__name__)


# ============================================
# CONFIG
# ============================================

MAX_PAPERS = int(
    os.getenv("QASPER_MAX_PAPERS", 30)
)

SPLIT = os.getenv(
    "QASPER_SPLIT", "train"
)

_raw_strategies = os.getenv(
    "QASPER_STRATEGIES",
    "recursive,layout,semantic"
)

STRATEGIES = [
    s.strip()
    for s in _raw_strategies.split(",")
    if s.strip() in CHUNKER_MAP
]

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

BATCH_SIZE = 32   # chunks per embedding batch

# Official Qasper v0.3 release from Allen AI S3
QASPER_URL = (
    "https://qasper-dataset.s3.us-west-2.amazonaws.com"
    "/qasper-train-dev-v0.3.tgz"
)

# Local cache directory (gitignored)
CACHE_DIR = os.path.join(ROOT, ".cache", "qasper")


# ============================================
# EMBEDDING MODEL
# ============================================

logger.info(
    f"Loading embedding model "
    f"'{EMBEDDING_MODEL_NAME}'..."
)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

logger.info("Embedding model ready.")


# ============================================
# QASPER DATA LOADER
# ============================================

def _download_and_extract() -> None:
    """
    Downloads qasper-train-dev-v0.3.tar.gz from S3
    and extracts it into CACHE_DIR if not already present.
    """

    os.makedirs(CACHE_DIR, exist_ok=True)

    # Check if already extracted
    expected = os.path.join(CACHE_DIR, "qasper-train-v0.3.json")

    if os.path.isfile(expected):

        logger.info(
            f"Qasper data already cached at {CACHE_DIR}"
        )

        return

    logger.info(
        f"Downloading Qasper from S3...\n  {QASPER_URL}"
    )

    resp = requests.get(QASPER_URL, stream=True, timeout=120)

    resp.raise_for_status()

    raw = resp.content

    logger.info("Download complete. Extracting...")

    with tarfile.open(
        fileobj=io.BytesIO(raw),
        mode="r:gz"
    ) as tar:

        tar.extractall(path=CACHE_DIR)

    logger.info(f"Extracted to {CACHE_DIR}")


def load_qasper_split(split: str) -> list[dict]:
    """
    Returns a list of paper dicts for the given split.

    Qasper JSON is a dict keyed by paper_id. Each value:
        {
            "title": str,
            "abstract": str,
            "full_text": [
                { "section_name": str, "paragraphs": [str, ...] },
                ...
            ],
            "qas": [...]
        }
    """

    _download_and_extract()

    split_map = {
        "train": "qasper-train-v0.3.json",
        "dev":   "qasper-dev-v0.3.json",
    }

    filename = split_map.get(split)

    if filename is None:
        raise ValueError(
            f"Unknown split '{split}'. "
            f"Choose from: {list(split_map)}"
        )

    json_path = os.path.join(CACHE_DIR, filename)

    logger.info(f"Loading {json_path}...")

    with open(json_path, "r", encoding="utf-8") as f:
        raw: dict = json.load(f)

    papers = []

    for paper_id, data in raw.items():

        papers.append({
            "id":        paper_id,
            "title":     data.get("title", ""),
            "abstract":  data.get("abstract", ""),
            "full_text": data.get("full_text", []),
            "qas":       data.get("qas", [])
        })

    return papers


# ============================================
# QASPER → DOCUMENT CONVERTER
# ============================================

def paper_to_document(paper: dict) -> Document:
    """
    Convert a single Qasper paper record into a Document
    with one Section per paper section.
    """

    title    = paper.get("title", "Unknown")
    paper_id = paper.get("id", str(uuid.uuid4()))

    sections: list[Section] = []

    # ----------------------------------------
    # Abstract as first section
    # ----------------------------------------

    abstract = paper.get("abstract", "").strip()

    if abstract:

        sections.append(
            Section(
                content=abstract,
                section_title="Abstract",
                metadata={
                    "section_index": 0,
                    "section_name":  "Abstract"
                }
            )
        )

    # ----------------------------------------
    # Body sections
    # ----------------------------------------

    for idx, section in enumerate(
        paper.get("full_text", []),
        start=1
    ):

        name  = section.get("section_name", "") or f"Section {idx}"
        paras = section.get("paragraphs", [])

        section_text = "\n\n".join(
            p.strip() for p in paras if p.strip()
        )

        if not section_text:
            continue

        sections.append(
            Section(
                content=section_text,
                section_title=name,
                metadata={
                    "section_index": idx,
                    "section_name":  name
                }
            )
        )

    full_content = "\n\n".join(
        s.content for s in sections
    )

    return Document(
        content=full_content,
        metadata={
            "source":    title,
            "paper_id":  paper_id,
            "title":     title,
            "file_type": "qasper"
        },
        sections=sections
    )


# ============================================
# BATCH EMBEDDING
# ============================================

def embed_chunks(
    chunks: list[Chunk],
    strategy: str
) -> list[dict]:
    """
    Generate embeddings for a list of Chunk objects
    in batches. Returns dicts ready for push_to_chromadb().
    """

    texts = [c.content for c in chunks]

    logger.info(
        f"  Embedding {len(texts)} chunks "
        f"(strategy={strategy})..."
    )

    embeddings = embedding_model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False
    )

    return [
        {
            "id":        str(uuid.uuid4()),
            "content":   chunk.content,
            "embedding": emb.tolist(),
            "metadata":  chunk.metadata
        }
        for chunk, emb in zip(chunks, embeddings)
    ]


# ============================================
# MAIN INGESTION
# ============================================

def main():

    print("\n===================================")
    print("QASPER INGESTION PIPELINE")
    print("===================================\n")

    print(f"Split:      {SPLIT}")
    print(f"Max papers: {MAX_PAPERS}")
    print(f"Strategies: {STRATEGIES}\n")

    # ----------------------------------------
    # 1. CLEAR CHROMADB
    # ----------------------------------------

    deleted = clear_collection()

    logger.info(
        f"Collection cleared "
        f"({deleted} documents removed)."
    )

    # ----------------------------------------
    # 2. LOAD QASPER
    # ----------------------------------------

    papers = load_qasper_split(SPLIT)[:MAX_PAPERS]

    logger.info(
        f"Loaded {len(papers)} papers."
    )

    # ----------------------------------------
    # 3. PROCESS EACH PAPER × EACH STRATEGY
    # ----------------------------------------

    total_chunks_pushed = 0

    start_time = time.time()

    for paper_idx, paper in enumerate(papers, 1):

        title = paper.get("title", "Unknown")

        logger.info(
            f"\n[{paper_idx}/{len(papers)}] "
            f"{title[:70]}"
        )

        document = paper_to_document(paper)

        if not document.content.strip():

            logger.warning(
                "  Empty document — skipping."
            )

            continue

        for strategy in STRATEGIES:

            chunker = CHUNKER_MAP[strategy]

            try:

                chunks: list[Chunk] = chunker(document)

            except Exception as e:

                logger.error(
                    f"  Chunking failed "
                    f"(strategy={strategy}): {e}"
                )

                continue

            if not chunks:

                logger.warning(
                    f"  No chunks produced "
                    f"(strategy={strategy})."
                )

                continue

            embedded = embed_chunks(chunks, strategy)

            try:

                push_to_chromadb(embedded)

                total_chunks_pushed += len(embedded)

                logger.info(
                    f"  ✓ {strategy:10s} → "
                    f"{len(embedded)} chunks pushed"
                )

            except Exception as e:

                logger.error(
                    f"  Push failed "
                    f"(strategy={strategy}): {e}"
                )

    # ----------------------------------------
    # 4. SUMMARY
    # ----------------------------------------

    elapsed = time.time() - start_time

    final_count = get_collection_count()

    print("\n===================================")
    print("INGESTION COMPLETE")
    print("===================================\n")

    print(f"Papers processed : {len(papers)}")
    print(f"Strategies used  : {', '.join(STRATEGIES)}")
    print(f"Total chunks     : {total_chunks_pushed}")
    print(f"ChromaDB count   : {final_count}")
    print(f"Time elapsed     : {elapsed:.1f}s\n")

    print(
        "Metadata field 'chunking_strategy' is set on "
        "every chunk for comparison queries.\n"
    )


# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    main()
