"""Eyeball check of the Retriever on the REAL local recursive store, real embedder, real FTS index.

    python scripts/smoke_retrieve.py

Runs 5 hand-written queries in all three modes and prints the top 3 hits with the legs that found
them and their ranks. Prints no scores, no latencies and no metrics, writes nothing to docs/ or
CLAIMS_LOG: this is for reading, not for quoting.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"

from config import cfg  # noqa: E402
from db.store import build_store  # noqa: E402
from rag.embedders import SentenceTransformerEmbedder  # noqa: E402
from rag.lexical import SqliteFtsIndex, lexical_index_path  # noqa: E402
from rag.retriever import Retriever  # noqa: E402

STRATEGY = "recursive"
QUERIES = [
    "What dataset was used to evaluate the model?",
    "How do they compute the BLEU score for translation quality?",
    "Which neural network architecture is used for sentiment classification?",
    "What are the annotation guidelines for labelling named entities?",
    "word embeddings trained on Wikipedia",
]
RUNS = [("hybrid", "rrf"), ("vector", "rrf"), ("lexical", "rrf")]


def one_line(text: str, width: int = 100) -> str:
    return " ".join(text.split())[:width]


def main() -> int:
    store = build_store(cfg, STRATEGY)
    lexical = SqliteFtsIndex(lexical_index_path(cfg, STRATEGY))
    print(f"store={store.name} chunks={store.count()}  lexical rows={lexical.count()}")
    retriever = Retriever(store, SentenceTransformerEmbedder.from_config(cfg), lexical, cfg=cfg)
    for query in QUERIES:
        print(f"\n=== {query}")
        for mode, fusion in RUNS:
            label = f"{mode}+{fusion}" if mode == "hybrid" else mode
            result = retriever.retrieve(query, mode=mode, fusion=fusion, top_k=3)
            print(f"  -- {label}")
            for hit in result.hits:
                legs = " ".join(f"{leg}#{info.rank}" for leg, info in sorted(hit.legs.items()))
                print(f"     {hit.rank}. [{hit.doc_id}] ({legs}) {one_line(hit.text)}")
    lexical.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
