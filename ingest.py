import os

from loaders import load_document
from chunkers import CHUNKER_MAP
from embeddings import embedding_pipeline


from vectordb import push_to_chromadb

# ============================================
# CONFIG
# ============================================

DATA_FOLDER = "data"

CHUNKING_MAP = {
    ".pdf": "layout",
    ".docx": "layout",
    ".html": "layout",
    ".md": "layout",
    ".txt": "semantic",
    ".xlsx": "layout"
}


# ============================================
# INGESTION
# ============================================
def ingest_folder(folder_path):

    all_chunks = []

    for root, dirs, files in os.walk(folder_path):

        for file in files:

            # Skip hidden files
            if file.startswith("."):
                continue

            file_path = os.path.join(root, file)

            print(f"\nProcessing: {file_path}")

            try:

                # --------------------------------
                # LOAD DOCUMENT
                # --------------------------------

                document = load_document(file_path)

                # --------------------------------
                # SELECT CHUNKING STRATEGY
                # --------------------------------

                file_type = document.metadata.get(
                    "file_type"
                )

                CHUNKING_MAP = {
                    ".pdf": "layout",
                    ".docx": "layout",
                    ".html": "layout",
                    ".md": "layout",
                    ".txt": "semantic",
                    ".xlsx": "layout"
                }

                strategy = CHUNKING_MAP.get(
                    file_type,
                    "recursive"
                )

                print(
                    f"Using chunking strategy: "
                    f"{strategy}"
                )

                chunker = CHUNKER_MAP[strategy]

                # --------------------------------
                # CHUNK DOCUMENT
                # --------------------------------

                chunks = chunker(document)

                print(
                    f"Generated {len(chunks)} chunks"
                )

                all_chunks.extend(chunks)

            except Exception as e:

                print(f"Error: {e}")

    return all_chunks


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    chunks = ingest_folder(DATA_FOLDER)

    

    embedded_chunks = embedding_pipeline(
        chunks
    )

    push_to_chromadb(
        embedded_chunks
    )

    print("\n===================================")
    print(f"TOTAL CHUNKS: {len(chunks)}")
    print("===================================\n")

    if chunks:

        sample = chunks[0]

        print("SAMPLE CHUNK:\n")

        print(sample.content[:1000])

        print("\nMETADATA:\n")

        print(sample.metadata)