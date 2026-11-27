"""Idempotent, incremental ingestion of a Qasper corpus into a VectorStore, with verification.

* A document is skipped when the store already holds exactly its current ``doc_version``
  AND its chunk indices are complete (0..n-1). Otherwise its old chunks are deleted
  (``delete_by_doc``, never the collection) and it is re-chunked, re-embedded and upserted.
* After the run ``verify_store`` raises ``VerificationError`` on any inconsistency.
* A manifest (JSON-serialisable dict) records what was done; ``write_manifest`` saves it.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from rag.chunk_adapter import CONFIDENCES, adapt_document, qasper_to_document
from rag.schemas_ingest import sha256_text

_WS = re.compile(r"\s+")
EMBED_BATCH = 32
SAMPLE_SIZE = 20


class VerificationError(RuntimeError):
    """Raised when the store does not match what ingestion says it wrote."""

    def __init__(self, problems: list[str], report: dict):
        super().__init__("store verification failed:\n  - " + "\n  - ".join(problems))
        self.problems = problems
        self.report = report


class IngestConfigMismatch(RuntimeError):
    """The store was built with a different chunking/embedding setup than this run uses."""


_CHUNKING_CODE = ("rag/chunkers.py", "rag/chunk_adapter.py", "rag/tokens.py", "rag/splitting.py", "rag/schemas_ingest.py")


def chunking_code_hash() -> str:
    """sha256 over the files that decide what a chunk is (so a chunker change is noticed)."""
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for rel in _CHUNKING_CODE:
        h.update(rel.encode())
        h.update((root / rel).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def ingest_signature(strategy: str, embedder, cfg, vector_embedder=None) -> str:
    """Everything besides the document text that determines the stored chunks and vectors.

    ``embedder`` decides the chunks (token window and counter); ``vector_embedder`` (optional) is a different model used
    only for the stored vectors. Without it the payload is exactly the historical one, so existing stores stay valid.
    """
    payload = {
        "strategy": strategy,
        "chunking_config": dict(vars(cfg.chunking)),
        "embedder": [embedder.name, embedder.revision, embedder.max_seq_length, getattr(embedder, "token_counter", "")],
        "chunking_code": chunking_code_hash(),
    }
    if vector_embedder is not None:
        payload["vector_embedder"] = [vector_embedder.name, vector_embedder.revision, getattr(vector_embedder, "dims", None)]
    return sha256_text(json.dumps(payload, sort_keys=True))


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _summary(values: list[int]) -> dict:
    if not values:
        return {"min": 0, "median": 0, "max": 0}
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def verify_store(
    store,
    corpus: dict,
    doc_ids: list[str],
    per_doc_counts: dict[str, int],
    embedder,
    seed: int = 0,
    lexical=None,
) -> dict:
    """Check the store (and, when given, the lexical index) against the corpus.

    Returns a report; raises VerificationError if anything is inconsistent.
    """
    problems: list[str] = []
    report: dict = {}

    chunks = list(store.iter_chunks())
    stored_total = store.count()

    # 1. store count equals the sum of per-document chunk counts
    store_docs = Counter(c.doc_id for c in chunks)
    expected_total = sum(per_doc_counts.get(d, store_docs[d]) for d in store_docs)
    report["store_count"] = stored_total
    report["sum_of_chunk_counts"] = expected_total
    if stored_total != expected_total:
        problems.append(f"store.count()={stored_total} but chunk counts sum to {expected_total}")
    for doc_id, n in per_doc_counts.items():
        if store_docs[doc_id] != n:
            problems.append(f"{doc_id}: wrote {n} chunks but the store holds {store_docs[doc_id]}")
    orphans = sorted(set(store_docs) - set(corpus))
    report["orphan_doc_ids"] = len(orphans)
    if orphans:
        problems.append(f"{len(orphans)} stored doc_ids are not in the corpus, e.g. {orphans[:3]}")

    # 2. chunk ids are unique
    ids = [c.chunk_id for c in chunks]
    report["duplicate_chunk_ids"] = len(ids) - len(set(ids))
    if len(ids) != len(set(ids)) or len(ids) != stored_total:
        problems.append("chunk_ids are not unique or do not match store.count()")

    # 3. every processed corpus doc has at least one chunk, and complete indices
    missing = [d for d in doc_ids if store_docs[d] == 0]
    report["docs_without_chunks"] = len(missing)
    if missing:
        problems.append(f"{len(missing)} corpus docs have no chunk, e.g. {missing[:3]}")
    incomplete = [d for d in doc_ids if store_docs[d] and not store.is_doc_complete(d)]
    report["docs_with_missing_chunks"] = len(incomplete)
    if incomplete:
        problems.append(f"{len(incomplete)} docs have a hole in their chunk indices, e.g. {incomplete[:3]}")

    # 4. seeded sample round-trips: stored text == section text[char_start:char_end] (modulo whitespace)
    rng = random.Random(seed)
    sample = rng.sample(chunks, min(SAMPLE_SIZE, len(chunks)))
    mismatches, unlocated = [], 0
    section_cache: dict[str, dict[str, list[str]]] = {}
    for c in sample:
        if c.char_start < 0:
            unlocated += 1
            continue
        if c.doc_id not in section_cache:
            doc = qasper_to_document(c.doc_id, corpus[c.doc_id]) if c.doc_id in corpus else None
            by_title: dict[str, list[str]] = {}
            for s in (doc.sections if doc else []):
                by_title.setdefault(s.section_title, []).append(s.content)
            section_cache[c.doc_id] = by_title
        candidates = section_cache[c.doc_id].get(c.section, [])
        if not any(_norm(sec[c.char_start : c.char_end]) == _norm(c.text) for sec in candidates):
            mismatches.append(c.chunk_id)
    report["round_trip"] = {"sampled": len(sample), "unlocated_skipped": unlocated, "mismatches": len(mismatches)}
    if mismatches:
        problems.append(f"{len(mismatches)} sampled chunks do not match their section text, e.g. {mismatches[:3]}")

    # 5. no stored chunk exceeds the embedder's real window
    limit = embedder.max_seq_length
    over = [c.chunk_id for c in chunks if embedder.count_tokens(c.text) > limit]
    report["token_check"] = {
        "counter": getattr(embedder, "token_counter", "unknown"),
        "max_seq_length": limit,
        "chunks_checked": len(chunks),
        "chunks_over_max_seq_length": len(over),
    }
    if over:
        problems.append(f"{len(over)} stored chunks exceed {limit} tokens, e.g. {over[:3]}")

    # 6. the lexical index mirrors the vector store: same count, same chunk_id set
    if lexical is not None:
        vec_ids, lex_ids = set(ids), lexical.chunk_ids()
        missing_in_lex, extra_in_lex = sorted(vec_ids - lex_ids), sorted(lex_ids - vec_ids)
        report["lexical"] = {
            "count": lexical.count(),
            "store_count": stored_total,
            "count_parity": lexical.count() == stored_total,
            "id_set_parity": vec_ids == lex_ids,
            "missing_in_lexical": len(missing_in_lex),
            "extra_in_lexical": len(extra_in_lex),
            "fts_rows": lexical.fts_row_count() if hasattr(lexical, "fts_row_count") else None,
        }
        if lexical.count() != stored_total:
            problems.append(f"lexical.count()={lexical.count()} but store.count()={stored_total}")
        if vec_ids != lex_ids:
            problems.append(
                f"chunk_id sets differ: {len(missing_in_lex)} missing in lexical (e.g. {missing_in_lex[:3]}), "
                f"{len(extra_in_lex)} extra in lexical (e.g. {extra_in_lex[:3]})"
            )

    report["ok"] = not problems
    report["problems"] = problems
    if problems:
        raise VerificationError(problems, report)
    return report


def ingest_corpus(
    corpus: dict,
    strategy: str,
    store,
    embedder,
    cfg,
    *,
    limit_docs: int | None = None,
    corpus_path: str | None = None,
    dataset_sha256: str | None = None,
    code_fingerprint: str | None = None,
    store_location: str | None = None,
    lexical=None,
    run_root: str | Path | None = None,
    log=print,
    vector_embedder=None,
) -> dict:
    """Ingest ``corpus`` (paper_id -> paper) for one strategy. Returns the manifest.

    ``embedder`` chunks (token window and counter). ``vector_embedder``, when given, produces the stored vectors with its
    ``encode_documents`` (for example Gemini) while the chunks stay exactly those ``embedder`` yields. If it raises
    (for instance ``QuotaExhausted``) nothing of the current document is written; documents already stored are kept
    and the next run resumes with the remaining ones.

    When ``run_root`` is given the manifest is written to ``<run_root>/<strategy>_<UTC ts>/``
    even if verification fails; ``VerificationError`` is then raised after writing.
    """
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    doc_ids = list(corpus)[: limit_docs or None]

    signature = ingest_signature(strategy, embedder, cfg, vector_embedder)
    stored_signature = store.get_signature()
    if store.count() > 0 and stored_signature != signature:
        raise IngestConfigMismatch(
            "this collection was built with a different chunking/embedding setup "
            f"(stored signature {str(stored_signature)[:12]}, current {signature[:12]}). Unchanged "
            "documents would be skipped and stay stale. Ingest into a new store path "
            "(CHROMA_LOCAL_PATH) or ask the owner before deleting anything."
        )
    store.set_signature(signature)
    per_doc: dict[str, int] = {}
    skipped: list[str] = []
    ingested: list[str] = []
    replaced: list[str] = []
    lexical_backfilled: list[str] = []
    confidence = Counter({c: 0 for c in CONFIDENCES})
    searched = 0
    written = 0

    for n, doc_id in enumerate(doc_ids, 1):
        doc = qasper_to_document(doc_id, corpus[doc_id])
        version = sha256_text(doc.content)
        stored_versions = store.get_doc_versions(doc_id)
        if stored_versions == {version} and store.is_doc_complete(doc_id):
            vector_ids = store.doc_chunk_ids(doc_id)
            if lexical is not None and lexical.doc_chunk_ids(doc_id) != vector_ids:
                # vectors are fine (embedding is the expensive part): rebuild only this doc's lexical rows
                lexical.delete_by_doc(doc_id)
                lexical.upsert(store.get_chunks(sorted(vector_ids)))
                lexical_backfilled.append(doc_id)
            skipped.append(doc_id)
            per_doc[doc_id] = len(vector_ids)
            continue
        if stored_versions:
            store.delete_by_doc(doc_id)  # changed or incomplete: remove only this document
            replaced.append(doc_id)
        if lexical is not None:
            lexical.delete_by_doc(doc_id)  # also clears stale rows left by a crash between the two stores
        adapted = adapt_document(doc, doc_id, strategy, embedder=embedder)
        texts = [c.text for c in adapted.chunks]
        vectors = []
        if vector_embedder is None:
            for i in range(0, len(texts), EMBED_BATCH):
                vectors.extend(embedder.encode(texts[i : i + EMBED_BATCH], show_progress_bar=False))
        else:
            vectors = list(vector_embedder.encode_documents(texts))  # may raise QuotaExhausted: nothing of this doc is written
        store.upsert(adapted.chunks, vectors)
        if lexical is not None:
            lexical.upsert(adapted.chunks)  # after the vector write: a crash in between is repaired next run
        for info in adapted.offset_info.values():
            confidence[info.confidence] += 1
            searched += info.by_search
        per_doc[doc_id] = len(adapted.chunks)
        written += len(adapted.chunks)
        ingested.append(doc_id)
        if n % 20 == 0 or n == len(doc_ids):
            log(f"  [{n}/{len(doc_ids)}] ingested={len(ingested)} skipped={len(skipped)} chunks written={written}")

    manifest: dict = {
        "strategy": strategy,
        "started_utc": started.isoformat(),
        "dataset": {"path": corpus_path, "sha256": dataset_sha256, "docs_in_corpus": len(corpus), "docs_processed": len(doc_ids), "limit_docs": limit_docs},
        "code_fingerprint": code_fingerprint,
        "ingest_signature": signature,
        "embedder": {
            "class": type(embedder).__name__,
            "name": embedder.name,
            "revision": embedder.revision,
            "max_seq_length": embedder.max_seq_length,
            "token_counter": getattr(embedder, "token_counter", "unknown"),
        },
        "vector_embedder": None if vector_embedder is None else {
            "class": type(vector_embedder).__name__, "name": vector_embedder.name, "revision": vector_embedder.revision,
            "dims": getattr(vector_embedder, "dims", None),
            "stats": vector_embedder.stats() if hasattr(vector_embedder, "stats") else None,
        },
        "chunking_config": dict(vars(cfg.chunking)),
        "store": {"location": store_location, "collection": getattr(store, "name", None)},
        "lexical": None
        if lexical is None
        else {"path": str(getattr(lexical, "path", "")), "docs_backfilled_from_store": len(lexical_backfilled)},
        "docs": {
            "ingested": len(ingested),
            "skipped_unchanged": len(skipped),
            "replaced_old_version": len(replaced),
            "lexical_backfilled": len(lexical_backfilled),
        },
        "chunks": {
            "written_this_run": written,
            "total_in_store_for_processed_docs": sum(per_doc.values()),
            "per_doc": _summary(list(per_doc.values())),
        },
        "offset_confidence_ingested_docs_only": dict(confidence),
        "offsets_located_by_search_ingested_docs_only": searched,
    }

    verification_error = None
    try:
        manifest["verification"] = verify_store(
            store, corpus, doc_ids, {d: per_doc[d] for d in ingested}, embedder, lexical=lexical
        )
    except VerificationError as exc:
        manifest["verification"] = exc.report
        verification_error = exc
    manifest["store"]["count_after"] = store.count()
    if lexical is not None:
        manifest["lexical"]["count_after"] = lexical.count()
    manifest["duration_s"] = round(time.perf_counter() - t0, 2)

    if run_root is not None:
        manifest["manifest_path"] = str(write_manifest(manifest, run_root, started))
    if verification_error is not None:
        raise verification_error
    return manifest


def check_lexical_parity(store, lexical) -> dict:
    """Raise VerificationError unless lexical.count() == store.count() and the chunk_id sets are identical."""
    vec_ids = {c.chunk_id for c in store.get_all()}
    lex_ids = lexical.chunk_ids()
    missing, extra = sorted(vec_ids - lex_ids), sorted(lex_ids - vec_ids)
    report = {
        "store_count": store.count(),
        "lexical_count": lexical.count(),
        "fts_rows": lexical.fts_row_count() if hasattr(lexical, "fts_row_count") else None,
        "count_parity": store.count() == lexical.count(),
        "id_set_parity": vec_ids == lex_ids,
        "missing_in_lexical": len(missing),
        "extra_in_lexical": len(extra),
    }
    problems = []
    if not report["count_parity"]:
        problems.append(f"lexical.count()={lexical.count()} but store.count()={store.count()}")
    if not report["id_set_parity"]:
        problems.append(f"chunk_id sets differ: {len(missing)} missing in lexical (e.g. {missing[:3]}), "
                        f"{len(extra)} extra in lexical (e.g. {extra[:3]})")
    if report["fts_rows"] is not None and report["fts_rows"] != lexical.count():
        problems.append(f"FTS table has {report['fts_rows']} rows but the id map has {lexical.count()}")
    report["ok"] = not problems
    if problems:
        raise VerificationError(problems, report)
    return report


def backfill_lexical(store, lexical) -> dict:
    """Fill the lexical index from the vector store (``store.get_all()``). Idempotent.

    Per document: ``delete_by_doc`` then ``upsert``, so running it twice, or after a partial run,
    gives the same index. Nothing is ever deleted from the vector store and nothing is embedded.
    Ends with a parity check; rows for documents absent from the vector store make it fail loudly.
    """
    by_doc: dict[str, list] = {}
    for chunk in store.get_all():
        by_doc.setdefault(chunk.doc_id, []).append(chunk)
    for doc_id, chunks in by_doc.items():
        lexical.delete_by_doc(doc_id)
        lexical.upsert(chunks)
    return {"docs": len(by_doc), "chunks": sum(len(c) for c in by_doc.values()), "parity": check_lexical_parity(store, lexical)}


def delete_document(store, lexical, doc_id: str) -> dict:
    """Remove one document from BOTH the vector store and the lexical index (never a collection)."""
    return {
        "vector": store.delete_by_doc(doc_id),
        "lexical": lexical.delete_by_doc(doc_id) if lexical is not None else 0,
    }

def write_manifest(manifest: dict, run_root: str | Path, started: datetime) -> Path:
    run_dir = Path(run_root) / f"{manifest['strategy']}_{started.strftime('%Y%m%dT%H%M%S%fZ')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
