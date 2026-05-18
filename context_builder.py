from typing import List

import tiktoken


# ============================================
# TOKENIZER
# ============================================

tokenizer = tiktoken.get_encoding(
    "cl100k_base"
)


# ============================================
# TOKEN COUNT
# ============================================

def count_tokens(text: str):

    return len(
        tokenizer.encode(text)
    )


# ============================================
# DEDUPLICATION
# ============================================

def deduplicate_chunks(results):

    print("\nDeduplicating chunks...\n")

    seen = set()

    unique_results = []

    for result in results:

        content = result[
            "document"
        ].strip()

        content_hash = hash(content)

        if content_hash in seen:
            continue

        seen.add(content_hash)

        unique_results.append(result)

    print(
        f"Chunks after deduplication: "
        f"{len(unique_results)}"
    )

    return unique_results


# ============================================
# GROUP BY SOURCE
# ============================================

def group_by_source(results):

    grouped = {}

    for result in results:

        source = result[
            "metadata"
        ].get(
            "source",
            "unknown"
        )

        if source not in grouped:

            grouped[source] = []

        grouped[source].append(result)

    return grouped


# ============================================
# MERGE NEIGHBORING CHUNKS
# ============================================

def merge_chunks(grouped_results):

    print("\nMerging neighboring chunks...\n")

    merged = []

    for source, chunks in grouped_results.items():

        # ------------------------------------
        # SORT BY CHUNK ID
        # ------------------------------------

        chunks = sorted(

            chunks,

            key=lambda x:
                x["metadata"].get(
                    "chunk_id",
                    0
                )
        )

        current_text = ""

        current_metadata = chunks[0][
            "metadata"
        ]

        previous_chunk_id = None

        for chunk in chunks:

            chunk_id = chunk[
                "metadata"
            ].get(
                "chunk_id",
                0
            )

            # --------------------------------
            # CONTIGUOUS CHUNK
            # --------------------------------

            if (

                previous_chunk_id
                is not None

                and

                chunk_id
                ==
                previous_chunk_id + 1
            ):

                current_text += (
                    "\n\n"
                    + chunk["document"]
                )

            else:

                if current_text:

                    merged.append({

                        "document":
                            current_text,

                        "metadata":
                            current_metadata
                    })

                current_text = chunk[
                    "document"
                ]

                current_metadata = chunk[
                    "metadata"
                ]

            previous_chunk_id = chunk_id

        # ------------------------------------
        # FINAL PUSH
        # ------------------------------------

        if current_text:

            merged.append({

                "document":
                    current_text,

                "metadata":
                    current_metadata
            })

    print(
        f"Merged chunks: "
        f"{len(merged)}"
    )

    return merged


# ============================================
# TOKEN BUDGETING
# ============================================

def enforce_token_budget(

    results,

    max_tokens=1500
):

    print(
        f"\nApplying token budget "
        f"({max_tokens} tokens)...\n"
    )

    final_results = []

    total_tokens = 0

    for result in results:

        text = result["document"]

        tokens = count_tokens(text)

        if (

            total_tokens + tokens
            >
            max_tokens
        ):

            break

        final_results.append(result)

        total_tokens += tokens

    print(
        f"Final token count: "
        f"{total_tokens}"
    )

    return final_results


# ============================================
# BUILD FINAL CONTEXT
# ============================================

def build_context(

    retrieval_results,

    max_tokens=4000
):

    print("\n===================================")
    print("CONTEXT ENGINEERING")
    print("===================================\n")

    # ----------------------------------------
    # STEP 1
    # DEDUPLICATE
    # ----------------------------------------

    deduped = deduplicate_chunks(
        retrieval_results
    )

    # ----------------------------------------
    # STEP 2
    # GROUP
    # ----------------------------------------

    grouped = group_by_source(
        deduped
    )

    # ----------------------------------------
    # STEP 3
    # MERGE
    # ----------------------------------------

    merged = merge_chunks(
        grouped
    )

    # ----------------------------------------
    # STEP 4
    # TOKEN BUDGET
    # ----------------------------------------

    final_results = enforce_token_budget(

        merged,

        max_tokens=max_tokens
    )

    # ----------------------------------------
    # STEP 5
    # BUILD FINAL TEXT
    # ----------------------------------------

    context_parts = []

    for result in final_results:

        source = result[
            "metadata"
        ].get(
            "source",
            "unknown"
        )

        section = result[
            "metadata"
        ].get(
            "section_title",
            ""
        )

        text = result[
            "document"
        ]

        formatted = (

            f"[SOURCE: {source}]\n"

            f"[SECTION: {section}]\n\n"

            f"{text}"
        )

        context_parts.append(
            formatted
        )

    final_context = "\n\n".join(
        context_parts
    )

    print(
        f"\nFinal context length: "
        f"{len(final_context)} chars"
    )

    print("\n===================================")
    print("CONTEXT ENGINEERING COMPLETE")
    print("===================================\n")

    return final_context