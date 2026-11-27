"""Backfill the SQLite FTS5 lexical index from an existing local vector store.

    python scripts/build_lexical.py --strategy recursive

Reads every chunk with ``store.get_all()`` and writes the lexical index (per document:
delete_by_doc then upsert, so it is idempotent). Never touches Chroma Cloud, never deletes anything
from the vector store, never embeds. Verifies that lexical.count() equals store.count() and that
the chunk_id sets are identical; exits non-zero otherwise.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"

from config import cfg  # noqa: E402
from db.store import STRATEGIES, build_store  # noqa: E402
from rag.ingestion import VerificationError, backfill_lexical  # noqa: E402
from rag.lexical import SqliteFtsIndex, lexical_index_path  # noqa: E402
from utils.fingerprint import code_fingerprint  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--strategy", required=True, choices=STRATEGIES)
    args = ap.parse_args(argv)

    store = build_store(cfg, args.strategy)
    if store.count() == 0:
        print(f"REFUSED: vector store collection {store.name} is empty; ingest first.", file=sys.stderr)
        return 2
    path = lexical_index_path(cfg, args.strategy)
    lexical = SqliteFtsIndex(path)
    print(f"strategy={args.strategy} store_chunks={store.count()} lexical_index={path} existing_rows={lexical.count()}")
    try:
        report = backfill_lexical(store, lexical)
    except VerificationError as exc:
        print("PARITY CHECK FAILED", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 1
    finally:
        lexical.close()
    started = datetime.now(timezone.utc)
    out_dir = ROOT / "runs" / "lexical" / f"{args.strategy}_{started.strftime('%Y%m%dT%H%M%S%fZ')}"
    out_dir.mkdir(parents=True)
    (out_dir / "parity.json").write_text(
        json.dumps({"created_utc": started.isoformat(), "code_fingerprint": code_fingerprint(), "strategy": args.strategy,
                    "lexical_index": str(path), "vector_collection": store.name, "report": report}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print(f"saved {out_dir / 'parity.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
