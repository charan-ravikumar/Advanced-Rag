"""Audit every chunk stored in a local vector store against the source text (read-only).

    python scripts/audit_store_text.py [--strategy recursive --strategy semantic]

For every stored chunk: does its text equal the section slice [char_start:char_end] (modulo whitespace)?
And what share of the non-whitespace characters of the source sections is covered by some stored chunk?
(The ingest verification only samples 20 chunks; this checks all of them.) Writes
runs/measurements/store_audit_<UTC>/results.json with the code fingerprint. Never writes to the store.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"

from config import cfg  # noqa: E402
from db.store import STRATEGIES, build_store  # noqa: E402
from rag.chunk_adapter import qasper_to_document  # noqa: E402
from rag.ingestion import file_sha256  # noqa: E402
from utils.fingerprint import code_fingerprint  # noqa: E402

CORPUS = ROOT / "eval" / "datasets" / "qasper_dev" / "corpus.json"
_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip()


def audit(strategy: str, corpus: dict) -> dict:
    store = build_store(cfg, strategy)
    chunks = store.get_all()
    sections: dict[str, dict[str, list[str]]] = {}
    for pid, paper in corpus.items():
        by_title: dict[str, list[str]] = {}
        for s in qasper_to_document(pid, paper).sections:
            by_title.setdefault(s.section_title, []).append(s.content)
        sections[pid] = by_title
    cover = {pid: {t: [bytearray(len(c)) for c in cs] for t, cs in by.items()} for pid, by in sections.items()}
    mismatched = unlocated = 0
    for c in chunks:
        if c.char_start < 0:
            unlocated += 1
            continue
        ok = False
        for i, sec in enumerate(sections.get(c.doc_id, {}).get(c.section, [])):
            if _norm(sec[c.char_start:c.char_end]) == _norm(c.text):
                cover[c.doc_id][c.section][i][c.char_start:c.char_end] = b"\x01" * (c.char_end - c.char_start)
                ok = True
                break
        mismatched += not ok
    total = uncovered = docs_with_loss = 0
    for pid, by in sections.items():
        lost = 0
        for t, cs in by.items():
            for i, sec in enumerate(cs):
                cov = cover[pid][t][i]
                for j, ch in enumerate(sec):
                    if not ch.isspace():
                        total += 1
                        if not cov[j]:
                            uncovered += 1
                            lost += 1
        docs_with_loss += lost > 0
    return {"collection": store.name, "chunks": len(chunks), "chunks_whose_text_differs_from_section_slice": mismatched,
            "chunks_unlocated": unlocated, "nonwhitespace_source_chars": total, "uncovered_chars": uncovered,
            "docs_with_any_uncovered_char": docs_with_loss}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--strategy", action="append", choices=STRATEGIES)
    args = ap.parse_args()
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    out = {"created_utc": datetime.now(timezone.utc).isoformat(), "code_fingerprint": code_fingerprint(),
           "corpus_sha256": file_sha256(CORPUS), "docs": len(corpus),
           "strategies": {s: audit(s, corpus) for s in (args.strategy or ["recursive", "semantic"])}}
    d = ROOT / "runs" / "measurements" / ("store_audit_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(d / "results.json")
    print(json.dumps(out["strategies"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
