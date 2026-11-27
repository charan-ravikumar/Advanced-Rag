"""Export the served chunks as ``chunks.jsonl`` (plus derived dev relevance labels) and VERIFY that the chunk ids match the labels.

    $env:CHROMA_LOCAL_PATH = "$env:LOCALAPPDATA\\advrag\\chroma_v2"
    python scripts/export_chunks.py [--strategy recursive] [--out runs/exports] [--verify-only]

Read-only: it opens the vector store and the lexical index, never embeds, never writes to either (rule 14). It refuses to run without ``CHROMA_LOCAL_PATH`` (the old
store is never the default). Output folder ``runs/exports/<UTC>_chunks_<strategy>/``:

* ``chunks.jsonl``: one chunk per line, sorted by (paper, chunk_idx):
  ``{"chunk_id", "text", "metadata": {"paper_id", "doc_id", "section", "chunk_idx", "char_start", "char_end", "strategy", "text_sha256", "doc_version", "split"}}``.
  ``paper_id`` is the Qasper paper id (``doc_id`` is the same value, the name the store uses). ``split`` is ``dev``, ``test`` or ``none`` (a paper none of whose questions is in
  the split file), so that a consumer can keep the test papers out of anything it tunes.
* ``qrels_dev.jsonl``: for every answerable DEV question, ``{"question_id", "paper_id", "query", "relevant_chunk_ids"}``. The project's labels are evidence SPANS, not chunk ids:
  a chunk is relevant when it overlaps a span by >= 0.5 x the shorter of the two (``rag/eval_metrics.py``). These ids are derived from the very chunks written to
  ``chunks.jsonl`` with that rule, so they match by construction; the verification below proves it. Test labels are neither read nor exported by this script.
* ``manifest.json`` and ``SHA256SUMS``: counts, the store signature, hashes of the inputs, the code fingerprint.

Verification (any failure: nothing is written, exit code 1): ids unique and re-derivable from the content; vector store, lexical index and export hold the same ids; every
paper of the corpus has chunks, with ``chunk_idx`` running 0..n-1; every dev gold span is in a known paper and is covered by at least one chunk of THAT paper; every
relevant id exists in ``chunks.jsonl``; and the written files, read back from disk, still satisfy the same checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET_DIR = ROOT / "eval" / "datasets" / "qasper_dev"


def split_of_papers(split: dict, golden_by_id: dict) -> dict[str, str]:
    """paper id -> ``dev`` | ``test`` (the split file lists question ids; the split is by paper, so a paper is in one of them)."""
    out: dict[str, str] = {}
    for name in ("dev", "test"):
        for qid in split[name]:
            out[golden_by_id[qid]["paper_id"]] = name
    return out


def build_records(chunks, paper_split: dict[str, str]) -> list[dict]:
    records = []
    for c in sorted(chunks, key=lambda c: (c.doc_id, c.chunk_idx)):
        records.append({
            "chunk_id": c.chunk_id, "text": c.text,
            "metadata": {"paper_id": c.doc_id, "doc_id": c.doc_id, "section": c.section, "chunk_idx": c.chunk_idx, "char_start": c.char_start,
                         "char_end": c.char_end, "strategy": c.strategy, "text_sha256": c.text_hash, "doc_version": c.doc_version,
                         "split": paper_split.get(c.doc_id, "none")}})
    return records


def derive_qrels(chunks, questions: list[dict]) -> list[dict]:
    """Relevant chunk ids per question with the project's span rule (``rag.eval_metrics.is_relevant``), restricted to the question's own paper."""
    from eval.gold_spans import GoldSpan
    from rag.eval_metrics import is_relevant

    by_doc: dict[str, list] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    out = []
    for q in sorted(questions, key=lambda q: q["id"]):
        spans = [GoldSpan(**s) for s in q["gold_spans"]]
        rel = [c.chunk_id for c in sorted(by_doc.get(q["paper_id"], ()), key=lambda c: c.chunk_idx) if is_relevant(c, spans)]
        out.append({"question_id": q["id"], "paper_id": q["paper_id"], "query": q["question"], "relevant_chunk_ids": rel})
    return out


def verify(chunks, records: list[dict], qrels: list[dict], questions: list[dict], *, corpus_papers: set[str], lexical_ids: set[str], vector_count: int) -> list[str]:
    """Every problem found, as a list of sentences (empty = the export is consistent)."""
    from rag.schemas_ingest import make_chunk_id, sha256_text

    problems: list[str] = []
    ids = [c.chunk_id for c in chunks]
    id_set = set(ids)
    if len(id_set) != len(ids):
        problems.append(f"chunk ids are not unique ({len(ids) - len(id_set)} duplicates)")
    if vector_count != len(ids):
        problems.append(f"the vector store reports {vector_count} chunks but {len(ids)} were read")
    if lexical_ids != id_set:
        problems.append(f"lexical index and vector store hold different ids ({len(lexical_ids - id_set)} only lexical, {len(id_set - lexical_ids)} only in the store)")
    for c in chunks:
        if sha256_text(c.text) != c.text_hash:
            problems.append(f"{c.chunk_id}: text_hash does not match the text")
        elif make_chunk_id(c.doc_id, c.strategy, c.chunk_idx, c.text_hash) != c.chunk_id:
            problems.append(f"{c.chunk_id}: the id cannot be re-derived from doc_id, strategy, chunk_idx and the text hash")
        if not c.doc_id:
            problems.append(f"{c.chunk_id}: empty paper_id")
    papers = {c.doc_id for c in chunks}
    if corpus_papers - papers:
        problems.append(f"{len(corpus_papers - papers)} corpus papers have no chunk (for example {sorted(corpus_papers - papers)[:3]})")
    if papers - corpus_papers:
        problems.append(f"{len(papers - corpus_papers)} chunk papers are not in the corpus (for example {sorted(papers - corpus_papers)[:3]})")
    idx: dict[str, list[int]] = {}
    for c in chunks:
        idx.setdefault(c.doc_id, []).append(c.chunk_idx)
    for doc, values in sorted(idx.items()):
        if sorted(values) != list(range(len(values))):
            problems.append(f"{doc}: chunk_idx is not 0..{len(values) - 1}")
    paper_of = {c.chunk_id: c.doc_id for c in chunks}
    questions_by_id = {q["id"]: q for q in questions}
    if {r["question_id"] for r in qrels} != set(questions_by_id):
        problems.append("the labels do not cover exactly the dev questions")
    for r in qrels:
        q = questions_by_id.get(r["question_id"])
        if q is None:
            continue
        if any(s["doc_id"] != q["paper_id"] for s in q["gold_spans"]):
            problems.append(f"{q['id']}: a gold span belongs to another paper than the question's")
        if not r["relevant_chunk_ids"]:
            problems.append(f"{q['id']}: no chunk of paper {q['paper_id']} overlaps its gold evidence (the label is unreachable)")
        for cid in r["relevant_chunk_ids"]:
            if cid not in id_set:
                problems.append(f"{q['id']}: relevant id {cid} is not in the export")
            elif paper_of[cid] != q["paper_id"]:
                problems.append(f"{q['id']}: relevant id {cid} belongs to paper {paper_of[cid]}, not {q['paper_id']}")
    if [r["chunk_id"] for r in records] != [c.chunk_id for c in sorted(chunks, key=lambda c: (c.doc_id, c.chunk_idx))]:
        problems.append("the records are not the chunks in (paper, chunk_idx) order")
    if any(not r["metadata"].get("paper_id") for r in records):
        problems.append("a record has no metadata.paper_id")
    return problems


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main(argv=None, *, store=None, lexical=None, fingerprint: str | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--strategy", default="recursive")
    ap.add_argument("--out", default=str(ROOT / "runs" / "exports"))
    ap.add_argument("--verify-only", action="store_true", help="check everything, write nothing")
    ap.add_argument("--golden", default=str(DATASET_DIR / "golden_set_v2.json"))
    ap.add_argument("--corpus", default=str(DATASET_DIR / "corpus.json"))
    ap.add_argument("--split-file", default=str(DATASET_DIR / "split.json"))
    args = ap.parse_args(argv)

    if store is None:
        if not os.environ.get("CHROMA_LOCAL_PATH"):
            print("REFUSED: set CHROMA_LOCAL_PATH to the store to export (for example %LOCALAPPDATA%\\advrag\\chroma_v2); there is no default.", file=sys.stderr)
            return 2
        from config import cfg
        from db.store import build_store
        from rag.lexical import SqliteFtsIndex, lexical_index_path

        store = build_store(cfg, args.strategy)
        lexical = SqliteFtsIndex(lexical_index_path(cfg, args.strategy))
    golden_path, corpus_path, split_path = Path(args.golden), Path(args.corpus), Path(args.split_file)
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("golden_set_sha256") != _sha256(golden_path):
        print("REFUSED: split.json was made for a different golden set (sha256 mismatch).", file=sys.stderr)
        return 2
    by_id = {q["id"]: q for q in golden["questions"]}
    corpus_papers = set(json.loads(corpus_path.read_text(encoding="utf-8")))
    dev_questions = [by_id[i] for i in sorted(split["dev"]) if by_id[i]["answerable"]]

    chunks = store.get_all()
    records = build_records(chunks, split_of_papers(split, by_id))
    qrels = derive_qrels(chunks, dev_questions)
    problems = verify(chunks, records, qrels, dev_questions, corpus_papers=corpus_papers, lexical_ids=lexical.chunk_ids(), vector_count=store.count())
    if problems:
        print(f"VERIFICATION FAILED ({len(problems)} problems); nothing was written:", file=sys.stderr)
        for p in problems[:25]:
            print("  -", p, file=sys.stderr)
        return 1
    n_rel = [len(r["relevant_chunk_ids"]) for r in qrels]
    print(f"verified: {len(chunks)} chunks in {len({c.doc_id for c in chunks})} papers; {len(qrels)} dev questions, relevant chunks per question min {min(n_rel)} / "
          f"median {statistics.median(n_rel)} / max {max(n_rel)}; vector store, lexical index and export agree")
    if args.verify_only:
        return 0

    from utils.fingerprint import code_fingerprint

    started = datetime.now(timezone.utc)
    out = Path(args.out) / f"{started.strftime('%Y%m%dT%H%M%SZ')}_chunks_{args.strategy}"
    out.mkdir(parents=True)
    _write_jsonl(out / "chunks.jsonl", records)
    _write_jsonl(out / "qrels_dev.jsonl", qrels)
    # read back what is on disk and re-check it: the files, not the in-memory lists, are the deliverable
    disk_ids = {r["chunk_id"] for r in _read_jsonl(out / "chunks.jsonl")}
    disk_qrels = _read_jsonl(out / "qrels_dev.jsonl")
    stray = [cid for r in disk_qrels for cid in r["relevant_chunk_ids"] if cid not in disk_ids]
    if disk_ids != {c.chunk_id for c in chunks} or stray:
        print("VERIFICATION FAILED after writing: the files on disk do not match", file=sys.stderr)
        return 1
    per_split: dict[str, int] = {}
    for r in records:
        per_split[r["metadata"]["split"]] = per_split.get(r["metadata"]["split"], 0) + 1
    manifest = {
        "created_utc": started.isoformat(), "strategy": args.strategy, "code_fingerprint": fingerprint or code_fingerprint(),
        "store": {"collection": getattr(store, "name", None), "vector_count": store.count(), "lexical_count": lexical.count(),
                  "ingest_signature": store.get_signature() if hasattr(store, "get_signature") else None},
        "counts": {"chunks": len(records), "papers": len({r["metadata"]["paper_id"] for r in records}), "chunks_per_split": per_split,
                   "dev_questions": len(qrels), "relevant_pairs": sum(n_rel),
                   "relevant_per_question": {"min": min(n_rel), "median": statistics.median(n_rel), "max": max(n_rel)}},
        "inputs": {"golden_set_sha256": _sha256(golden_path), "split_file_sha256": _sha256(split_path), "corpus_sha256": _sha256(corpus_path)},
        "format": {"chunks.jsonl": "one JSON object per line: chunk_id, text, metadata{paper_id, doc_id, section, chunk_idx, char_start, char_end, strategy, text_sha256, doc_version, split}",
                   "qrels_dev.jsonl": "one JSON object per answerable dev question: question_id, paper_id, query, relevant_chunk_ids (derived with the span-overlap rule, >= 0.5 x min(len(span), len(chunk)))"},
        "notes": ["offsets are relative to the section text, not the whole paper", "test-split labels are not part of this export",
                  "metadata.split lets a consumer keep test papers out of anything it tunes"],
        "verification": "passed (ids unique and re-derivable; vector store, lexical index and export agree; every dev label reachable and inside its own paper; files re-read from disk)"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    names = ("chunks.jsonl", "qrels_dev.jsonl", "manifest.json")
    (out / "SHA256SUMS").write_text("".join(f"{_sha256(out / n)}  {n}\n" for n in names), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
