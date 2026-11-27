"""Compare two local stores chunk by chunk (read-only): counts, chunk ids, text and offsets.

    python scripts/compare_stores.py --old <chroma dir> --new <chroma dir> [--strategy recursive --strategy semantic]

Reports, per strategy: chunk counts, ids only in the old / only in the new store, and for ids present in both
whether the text, section, char_start, char_end, doc_version differ. Writes
runs/measurements/store_compare_<UTC>/results.json with the code fingerprint. Never writes to either store.
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
from utils.fingerprint import code_fingerprint  # noqa: E402


def compare(strategy: str, old_path: str, new_path: str) -> dict:
    old = {c.chunk_id: c for c in build_store(cfg, strategy, path=old_path).get_all()}
    new = {c.chunk_id: c for c in build_store(cfg, strategy, path=new_path).get_all()}
    both = set(old) & set(new)
    differ = {f: sum(getattr(old[i], f) != getattr(new[i], f) for i in both)
              for f in ("text", "section", "char_start", "char_end", "doc_version", "chunk_idx", "doc_id")}
    return {
        "chunks_old": len(old), "chunks_new": len(new),
        "ids_in_both": len(both), "ids_only_in_old": len(set(old) - set(new)), "ids_only_in_new": len(set(new) - set(old)),
        "ids_changed": len(set(old) ^ set(new)),
        "docs_old": len({c.doc_id for c in old.values()}), "docs_new": len({c.doc_id for c in new.values()}),
        "fields_differing_for_ids_in_both": differ,
        "examples_only_in_old": sorted(set(old) - set(new))[:5], "examples_only_in_new": sorted(set(new) - set(old))[:5],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--strategy", action="append", choices=STRATEGIES)
    args = ap.parse_args()
    out = {"created_utc": datetime.now(timezone.utc).isoformat(), "code_fingerprint": code_fingerprint(),
           "old_store": args.old, "new_store": args.new,
           "strategies": {s: compare(s, args.old, args.new) for s in (args.strategy or ["recursive", "semantic"])}}
    d = ROOT / "runs" / "measurements" / ("store_compare_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(d / "results.json")
    print(json.dumps(out["strategies"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
