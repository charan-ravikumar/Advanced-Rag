from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from schemas import Chunk


# ============================================
# RECURSIVE CHUNKER
# ============================================

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


def recursive_chunk(document):

    chunks = []

    split_chunks = recursive_splitter.split_text(
        document.content
    )

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


import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from schemas import Chunk


# ============================================
# EMBEDDING MODEL
# ============================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.\n")


# ============================================
# SEMANTIC CHUNKER
# ============================================

def semantic_chunk(
    document,

    # Similarity threshold
    similarity_threshold=0.78,

    # Hard limits
    max_chunk_chars=2000,
    min_chunk_chars=300,

    # Windowing
    window_size=3
):

    print("\n===================================")
    print("STARTING PRODUCTION SEMANTIC CHUNKING")
    print("===================================\n")

    print(f"Document: {document.metadata.get('source')}")

    # ----------------------------------------
    # STEP 1 — CLEAN + SPLIT
    # ----------------------------------------

    print("\nCleaning document...")

    cleaned_text = re.sub(
        r"\n{3,}",
        "\n\n",
        document.content
    )

    cleaned_text = cleaned_text.strip()

    # ----------------------------------------
    # PARAGRAPH SPLITTING
    # ----------------------------------------

    print("Splitting into paragraphs...")

    paragraphs = re.split(
        r"\n\s*\n",
        cleaned_text
    )

    paragraphs = [
        p.strip()
        for p in paragraphs
        if p.strip()
    ]

    print(f"Paragraphs found: {len(paragraphs)}")

    if not paragraphs:
        return []

    # ----------------------------------------
    # STEP 2 — SLIDING WINDOW
    # ----------------------------------------

    print("\nCreating semantic windows...")

    windows = []

    for i in range(len(paragraphs)):

        start = max(0, i - window_size)
        end = min(
            len(paragraphs),
            i + window_size + 1
        )

        window_text = " ".join(
            paragraphs[start:end]
        )

        windows.append(window_text)

    print(f"Windows created: {len(windows)}")

    # ----------------------------------------
    # STEP 3 — EMBEDDINGS
    # ----------------------------------------

    print("\nGenerating embeddings...")

    embeddings = embedding_model.encode(
        windows,
        show_progress_bar=True
    )

    print("Embeddings generated.")

    # ----------------------------------------
    # STEP 4 — SEMANTIC GROUPING
    # ----------------------------------------

    print("\nStarting semantic grouping...\n")

    semantic_groups = []

    current_group = [paragraphs[0]]

    current_chunk_size = len(paragraphs[0])

    for i in range(1, len(paragraphs)):

        similarity = cosine_similarity(
            [embeddings[i - 1]],
            [embeddings[i]]
        )[0][0]

        next_para = paragraphs[i]

        projected_size = (
            current_chunk_size
            + len(next_para)
        )

        print(
            f"Paragraph {i-1} ↔ {i} | "
            f"Similarity: {similarity:.4f} | "
            f"Projected Size: {projected_size}"
        )

        # ------------------------------------
        # MERGE CONDITIONS
        # ------------------------------------

        should_merge = (
            similarity >= similarity_threshold
            and projected_size <= max_chunk_chars
        )

        if should_merge:

            print("→ Merging into current chunk.\n")

            current_group.append(next_para)

            current_chunk_size += len(next_para)

        else:

            reason = []

            if similarity < similarity_threshold:
                reason.append("topic shift")

            if projected_size > max_chunk_chars:
                reason.append("max size exceeded")

            print(
                f"→ Creating new chunk "
                f"({', '.join(reason)}).\n"
            )

            semantic_groups.append(current_group)

            current_group = [next_para]

            current_chunk_size = len(next_para)

    # Final group
    semantic_groups.append(current_group)

    print(
        f"\nInitial semantic groups: "
        f"{len(semantic_groups)}"
    )

    # ----------------------------------------
    # STEP 5 — SMALL CHUNK MERGING
    # ----------------------------------------

    print("\nBalancing small chunks...\n")

    balanced_groups = []

    temp_group = []

    for idx, group in enumerate(semantic_groups):

        chunk_text = "\n\n".join(group)

        chunk_size = len(chunk_text)

        print(
            f"Group {idx} | "
            f"Characters: {chunk_size}"
        )

        # ------------------------------------
        # TOO SMALL
        # ------------------------------------

        if chunk_size < min_chunk_chars:

            print("→ Too small. Holding.\n")

            temp_group.extend(group)

        else:

            if temp_group:

                temp_text = "\n\n".join(temp_group)

                combined_size = (
                    len(temp_text)
                    + chunk_size
                )

                # Merge only if still safe
                if combined_size <= max_chunk_chars:

                    print(
                        "→ Merging held content.\n"
                    )

                    group = temp_group + group

                temp_group = []

            balanced_groups.append(group)

    # Remaining temp content
    if temp_group:

        print("Adding remaining held content.")

        if balanced_groups:

            combined_size = (
                len(
                    "\n\n".join(
                        balanced_groups[-1]
                    )
                )
                +
                len("\n\n".join(temp_group))
            )

            if combined_size <= max_chunk_chars:

                balanced_groups[-1].extend(
                    temp_group
                )

            else:

                balanced_groups.append(
                    temp_group
                )

        else:

            balanced_groups.append(temp_group)

    print(
        f"\nBalanced groups: "
        f"{len(balanced_groups)}"
    )

    # ----------------------------------------
    # STEP 6 — CREATE CHUNKS
    # ----------------------------------------

    print("\nCreating final chunks...\n")

    chunks = []

    for idx, group in enumerate(balanced_groups):

        chunk_text = "\n\n".join(group)

        chunk_size = len(chunk_text)

        print(
            f"Chunk {idx} | "
            f"Characters: {chunk_size}"
        )

        chunks.append(
            Chunk(
                content=chunk_text,
                metadata={
                    **document.metadata,
                    "chunk_id": idx,
                    "chunking_strategy": "semantic",
                    "chunk_size_chars": chunk_size,
                    "num_paragraphs": len(group),
                    "similarity_threshold":
                        similarity_threshold
                }
            )
        )

    # ----------------------------------------
    # COMPLETE
    # ----------------------------------------

    print("\n===================================")
    print("SEMANTIC CHUNKING COMPLETE")
    print("===================================\n")

    print(
        f"Final chunks created: {len(chunks)}"
    )

    return chunks

# ============================================
# CHUNKER MAP
# ============================================

CHUNKER_MAP = {
    "recursive": recursive_chunk,
    "layout": layout_chunk,
    "semantic": semantic_chunk
}