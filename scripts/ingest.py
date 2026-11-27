"""Ingest the Qasper corpus into the LOCAL Chroma store for one chunking strategy.

    python scripts/ingest.py --strategy recursive [--corpus PATH] [--limit-docs N]

Idempotent and incremental: unchanged documents are skipped, a changed document has only its
own chunks replaced, and a whole collection is never cleared. The store lives under
%LOCALAPPDATA%/advrag/chroma (override with CHROMA_LOCAL_PATH) and is never in OneDrive.
Embeds with the pinned real model from the local HuggingFace cache only; never downloads.
A manifest is written to runs/ingest/<strategy>_<UTC timestamp>/manifest.json.
Exit code 1 if verification fails.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"

from config import cfg  # noqa: E402
from db.store import STRATEGIES, build_store, resolve_store_path  # noqa: E402
from rag.embedders import SentenceTransformerEmbedder  # noqa: E402
from rag.ingestion import IngestConfigMismatch, VerificationError, file_sha256, ingest_corpus  # noqa: E402
from rag.lexical import SqliteFtsIndex, lexical_index_path  # noqa: E402
from utils.fingerprint import code_fingerprint  # noqa: E402

DEFAULT_CORPUS = "eval/datasets/qasper_dev/corpus.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--strategy", required=True, choices=STRATEGIES)
    ap.add_argument("--limit-docs", type=int, default=None)
    ap.add_argument("--vectors", choices=["minilm", "gemini"], default="minilm",
                    help="model for the STORED vectors; the chunks are always cut with the pinned MiniLM tokenizer. gemini needs "
                         "CHROMA_LOCAL_PATH pointing at a separate store and GEMINI_API_KEY in the environment")
    args = ap.parse_args(argv)

    corpus_path = Path(args.corpus)
    if not corpus_path.is_absolute():
        corpus_path = ROOT / corpus_path
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))

    vector_embedder = None
    if args.vectors == "gemini":
        from rag.gemini_embedder import GeminiEmbedder, GeminiError, QuotaExhausted

        if not os.environ.get("CHROMA_LOCAL_PATH"):
            print("REFUSED: --vectors gemini needs CHROMA_LOCAL_PATH to point at its own store (for example "
                  "%LOCALAPPDATA%\\advrag\\chroma_gemini)", file=sys.stderr)
            return 2
        vector_embedder = GeminiEmbedder.from_config(cfg)

    store = build_store(cfg, args.strategy)
    lexical = SqliteFtsIndex(lexical_index_path(cfg, args.strategy))
    location = str(resolve_store_path(os.environ.get("CHROMA_LOCAL_PATH") or cfg.store.chroma_dir))
    embedder = SentenceTransformerEmbedder.from_config(cfg)

    print(f"strategy={args.strategy} docs={len(corpus)} store={location} collection={store.name} "
          f"existing_chunks={store.count()} lexical_index={lexical.path} existing_lexical_rows={lexical.count()} vectors={args.vectors}")
    try:
        manifest = ingest_corpus(
            corpus,
            args.strategy,
            store,
            embedder,
            cfg,
            lexical=lexical,
            limit_docs=args.limit_docs,
            corpus_path=str(corpus_path),
            dataset_sha256=file_sha256(corpus_path),
            code_fingerprint=code_fingerprint(),
            store_location=location,
            run_root=ROOT / "runs" / "ingest",
            vector_embedder=vector_embedder,
        )
    except IngestConfigMismatch as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except VerificationError as exc:
        print("VERIFICATION FAILED", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 1
    except Exception as exc:
        if args.vectors == "gemini" and isinstance(exc, (QuotaExhausted, GeminiError)):
            kind = "QUOTA EXHAUSTED" if isinstance(exc, QuotaExhausted) else "GEMINI ERROR"
            print(f"{kind}: {exc}\nEverything embedded so far is cached on disk and every finished document is stored; "
                  f"run the same command again later to resume.\nstats: {json.dumps(vector_embedder.stats())}", file=sys.stderr)
            return 75
        raise
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
