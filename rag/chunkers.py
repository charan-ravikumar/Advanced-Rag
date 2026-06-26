import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import cfg
from schemas import Chunk
from observability.logger_config import get_logger

logger = get_logger(__name__)

# ============================================
# RECURSIVE CHUNKER
# ============================================

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


def recursive_chunk(document):
    """Split a document into overlapping fixed-size text chunks using recursive character splitting."""
    chunks = []
    split_chunks = recursive_splitter.split_text(document.content)
    for idx, chunk in enumerate(split_chunks):
        chunks.append(
            Chunk(
                content=chunk,
                metadata={
                    **document.metadata,
                    "chunk_id": idx,
                    "chunking_strategy": "recursive"
                }
            )
        )
    return chunks


# ============================================
# SECTION / LAYOUT CHUNKER
# ============================================

def layout_chunk(document):
    """Split a document into chunks aligned with its pre-parsed section boundaries."""
    chunks = []
    for idx, section in enumerate(document.sections):
        chunks.append(
            Chunk(
                content=section.content,
                metadata={
                    **document.metadata,
                    **section.metadata,
                    "section_title": section.section_title,
                    "chunk_id": idx,
                    "chunking_strategy": "layout"
                }
            )
        )
    return chunks


# ============================================
# EMBEDDING MODEL
# ============================================

logger.info("Loading embedding model for semantic chunker...")
embedding_model = SentenceTransformer(cfg.embedding.model)
logger.info("Embedding model loaded.")


# ============================================
# SEMANTIC CHUNKER
# ============================================

def semantic_chunk(
    document,
    similarity_threshold=0.78,
    max_chunk_chars=2000,
    min_chunk_chars=300,
    window_size=3
):
    """Split a document into semantically coherent chunks using embedding similarity."""
    logger.info(
        f"Starting semantic chunking for: {document.metadata.get('source')}"
    )

    # ----------------------------------------
    # STEP 1 - CLEAN + SPLIT
    # ----------------------------------------

    cleaned_text = re.sub(r"\n{3,}", "\n\n", document.content).strip()

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned_text) if p.strip()]

    logger.debug(f"Paragraphs found: {len(paragraphs)}")

    if not paragraphs:
        return []

    # ----------------------------------------
    # STEP 2 - SLIDING WINDOW
    # ----------------------------------------

    windows = []
    for i in range(len(paragraphs)):
        start = max(0, i - window_size)
        end = min(len(paragraphs), i + window_size + 1)
        windows.append(" ".join(paragraphs[start:end]))

    logger.debug(f"Windows created: {len(windows)}")

    # ----------------------------------------
    # STEP 3 - EMBEDDINGS
    # ----------------------------------------

    embeddings = embedding_model.encode(windows, show_progress_bar=False)
    logger.debug("Embeddings generated.")

    # ----------------------------------------
    # STEP 4 - SEMANTIC GROUPING
    # ----------------------------------------

    semantic_groups = []
    current_group = [paragraphs[0]]
    current_chunk_size = len(paragraphs[0])

    for i in range(1, len(paragraphs)):
        similarity = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
        next_para = paragraphs[i]
        projected_size = current_chunk_size + len(next_para)

        should_merge = (
            similarity >= similarity_threshold
            and projected_size <= max_chunk_chars
        )

        if should_merge:
            current_group.append(next_para)
            current_chunk_size += len(next_para)
        else:
            semantic_groups.append(current_group)
            current_group = [next_para]
            current_chunk_size = len(next_para)

    semantic_groups.append(current_group)
    logger.debug(f"Initial semantic groups: {len(semantic_groups)}")

    # ----------------------------------------
    # STEP 5 - SMALL CHUNK MERGING
    # ----------------------------------------

    balanced_groups = []
    temp_group = []

    for group in semantic_groups:
        chunk_text = "\n\n".join(group)
        chunk_size = len(chunk_text)

        if chunk_size < min_chunk_chars:
            temp_group.extend(group)
        else:
            if temp_group:
                temp_text = "\n\n".join(temp_group)
                combined_size = len(temp_text) + chunk_size
                if combined_size <= max_chunk_chars:
                    group = temp_group + group
                temp_group = []
            balanced_groups.append(group)

    if temp_group:
        if balanced_groups:
            combined_size = (
                len("\n\n".join(balanced_groups[-1])) + len("\n\n".join(temp_group))
            )
            if combined_size <= max_chunk_chars:
                balanced_groups[-1].extend(temp_group)
            else:
                balanced_groups.append(temp_group)
        else:
            balanced_groups.append(temp_group)

    logger.debug(f"Balanced groups: {len(balanced_groups)}")

    # ----------------------------------------
    # STEP 6 - CREATE CHUNKS
    # ----------------------------------------

    chunks = []
    for idx, group in enumerate(balanced_groups):
        chunk_text = "\n\n".join(group)
        chunk_size = len(chunk_text)
        chunks.append(
            Chunk(
                content=chunk_text,
                metadata={
                    **document.metadata,
                    "chunk_id": idx,
                    "chunking_strategy": "semantic",
                    "chunk_size_chars": chunk_size,
                    "num_paragraphs": len(group),
                    "similarity_threshold": similarity_threshold
                }
            )
        )

    logger.info(f"Semantic chunking complete. Final chunks: {len(chunks)}")
    return chunks


# ============================================
# CHUNKER MAP
# ============================================

CHUNKER_MAP = {
    "recursive": recursive_chunk,
    "layout": layout_chunk,
    "semantic": semantic_chunk
}