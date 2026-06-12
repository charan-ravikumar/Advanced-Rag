import os

import chromadb

from dotenv import load_dotenv


# ============================================
# LOAD ENV
# ============================================

load_dotenv()


# ============================================
# CHROMA CLOUD CLIENT
# ============================================

print("\nInitializing Chroma Cloud Client...")

client = chromadb.CloudClient(

    cloud_host=os.getenv("CHROMA_HOST"),

    cloud_port=int(
        os.getenv("CHROMA_PORT")
    ),

    api_key=os.getenv(
        "CHROMA_API_KEY"
    ),

    tenant=os.getenv(
        "CHROMA_TENANT"
    ),

    database=os.getenv(
        "CHROMA_DATABASE"
    )
)

print("Chroma Cloud connected.\n")


# ============================================
# COLLECTION
# ============================================

COLLECTION_NAME = "rag_chunks"

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={
        "description":
        "Advanced RAG chunk embeddings"
    }
)

print(
    f"Using collection: "
    f"{COLLECTION_NAME}\n"
)


# ============================================
# PUSH TO CHROMA
# ============================================

def push_to_chromadb(embedded_chunks):

    print("\n===================================")
    print("PUSHING TO CHROMADB CLOUD")
    print("===================================\n")

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    total = len(embedded_chunks)

    print(
        f"Total chunks to upload: "
        f"{total}\n"
    )

    for idx, chunk in enumerate(
        embedded_chunks
    ):

        print(
            f"Preparing chunk "
            f"{idx + 1}/{total}"
        )

        ids.append(chunk["id"])

        documents.append(
            chunk["content"]
        )

        embeddings.append(
            chunk["embedding"]
        )

        # ------------------------------------
        # CLEAN METADATA
        # ------------------------------------

        clean_metadata = {}

        for key, value in chunk[
            "metadata"
        ].items():

            if isinstance(
                value,
                (str, int, float, bool)
            ):

                clean_metadata[key] = value

            else:

                clean_metadata[key] = str(
                    value
                )

        metadatas.append(
            clean_metadata
        )

    print(
        "\nUploading embeddings "
        "to Chroma Cloud...\n"
    )

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print("\n===================================")
    print("UPLOAD COMPLETE")
    print("===================================\n")

    print(
        f"Collection Count: "
        f"{collection.count()}"
    )
