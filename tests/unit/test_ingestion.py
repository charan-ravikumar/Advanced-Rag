"""Ingestion tests: HashEmbedder, a tmp_path local Chroma, 5 fixture documents. No network."""

import copy
import json
from types import SimpleNamespace

import chromadb
import pytest
from chromadb.config import Settings

from config import cfg
from db.store import build_store
from rag.embedders import HashEmbedder
from rag.ingestion import (
    IngestConfigMismatch,
    VerificationError,
    file_sha256,
    ingest_corpus,
    verify_store,
)

EMB = HashEmbedder()
TOPICS = ["alpha model training dataset", "network latency server cache", "protein cell enzyme sequence", "river delta soil water", "market price trade supply"]


def _sentences(topic: str, n: int, tag: str) -> str:
    return " ".join(f"{topic} {tag} number {i} appears in this sentence." for i in range(n))


def _paper(i: int, tweak: str = "") -> dict:
    topic = TOPICS[i]
    return {
        "title": f"Paper {i}",
        "abstract": _sentences(topic, 6, f"abs{i}{tweak}"),
        "full_text": [
            {"section_name": "Introduction", "paragraphs": [_sentences(topic, 14, f"intro{i}"), _sentences(topic, 10, f"intro2{i}")]},
            {"section_name": "Methods", "paragraphs": [_sentences(topic, 30, f"meth{i}")]},
            {"section_name": "Results", "paragraphs": [_sentences(topic, 12, f"res{i}")]},
        ],
    }


@pytest.fixture
def corpus():
    return {f"1234.0000{i}": _paper(i) for i in range(5)}


@pytest.fixture
def client(tmp_path):
    return chromadb.PersistentClient(path=str(tmp_path / "chroma"), settings=Settings(anonymized_telemetry=False))


@pytest.fixture
def store(client):
    return build_store(cfg, "recursive", client=client)


def _ingest(corpus, store, **kw):
    return ingest_corpus(corpus, "recursive", store, EMB, cfg, log=lambda *_: None, **kw)


def _ids_by_doc(store):
    out = {}
    for c in store.iter_chunks():
        out.setdefault(c.doc_id, set()).add(c.chunk_id)
    return out


def test_first_run_ingests_everything_and_verifies(corpus, store):
    m = _ingest(corpus, store)
    assert m["docs"] == {"ingested": 5, "skipped_unchanged": 0, "replaced_old_version": 0, "lexical_backfilled": 0}
    assert m["chunks"]["written_this_run"] == store.count() > 5
    assert m["verification"]["ok"] is True
    assert m["verification"]["store_count"] == m["verification"]["sum_of_chunk_counts"]


def test_second_run_skips_all_and_upserts_nothing(corpus, store, monkeypatch):
    first = _ingest(corpus, store)
    before = _ids_by_doc(store)
    calls = []
    real_upsert = store.upsert
    monkeypatch.setattr(store, "upsert", lambda *a, **k: (calls.append(1), real_upsert(*a, **k)))
    second = _ingest(corpus, store)
    assert calls == []
    assert second["docs"] == {"ingested": 0, "skipped_unchanged": 5, "replaced_old_version": 0, "lexical_backfilled": 0}
    assert second["chunks"]["written_this_run"] == 0
    assert second["chunks"]["total_in_store_for_processed_docs"] == first["chunks"]["written_this_run"]
    assert _ids_by_doc(store) == before


def test_modifying_one_doc_changes_only_that_docs_chunks(corpus, store):
    _ingest(corpus, store)
    before = _ids_by_doc(store)
    changed = copy.deepcopy(corpus)
    target = "1234.00002"
    changed[target] = _paper(2, tweak="EDITED")
    m = _ingest(changed, store)
    after = _ids_by_doc(store)
    assert m["docs"] == {"ingested": 1, "skipped_unchanged": 4, "replaced_old_version": 1, "lexical_backfilled": 0}
    for doc_id in corpus:
        if doc_id != target:
            assert after[doc_id] == before[doc_id]
    assert after[target] != before[target]
    assert not (before[target] - after[target]) & {c.chunk_id for c in store.iter_chunks() if c.doc_id == target}
    assert m["verification"]["ok"] is True


def test_delete_by_doc_leaves_nothing_of_that_doc_retrievable(corpus, store):
    _ingest(corpus, store)
    target = "1234.00001"
    assert store.delete_by_doc(target) > 0
    assert store.get_doc_versions(target) == set()
    assert store.doc_chunk_idxs(target) == []
    hits = store.query(EMB.encode(TOPICS[1]), k=store.count())
    assert hits and all(h["metadata"]["doc_id"] != target for h in hits)


def test_verification_fails_loudly_when_a_chunk_is_deleted_by_hand(corpus, store):
    m = _ingest(corpus, store)
    target = "1234.00003"
    counts = {d: len(store.doc_chunk_idxs(d)) for d in corpus}
    victim = next(c for c in store.iter_chunks() if c.doc_id == target and c.chunk_idx == 1)
    store._col.delete(ids=[victim.chunk_id])
    with pytest.raises(VerificationError) as exc:
        verify_store(store, corpus, list(corpus), counts, EMB)
    msg = str(exc.value)
    assert "wrote" in msg and target in msg and "hole" in msg
    assert exc.value.report["ok"] is False
    # ... and the next ingest notices the incomplete document and repairs only that one
    repaired = _ingest(corpus, store)
    assert repaired["docs"]["ingested"] == 1 and repaired["docs"]["replaced_old_version"] == 1
    assert repaired["verification"]["ok"] is True
    assert m["chunks"]["written_this_run"] == store.count()


def test_crash_mid_document_leaves_a_detectable_hole_and_is_repaired(corpus, store, monkeypatch):
    _ingest({k: v for k, v in list(corpus.items())[:1]}, store)
    store._batch = 2  # force several batches per document
    store.delete_by_doc("1234.00000")
    real = store._col.upsert
    state = {"calls": 0}

    def flaky(**kw):
        state["calls"] += 1
        if state["calls"] == 2:
            raise RuntimeError("simulated crash")
        return real(**kw)

    monkeypatch.setattr(store._col, "upsert", flaky)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _ingest({k: v for k, v in list(corpus.items())[:1]}, store)
    assert store.count() > 0 and not store.is_doc_complete("1234.00000")  # tail written, head missing
    monkeypatch.setattr(store._col, "upsert", real)
    m = _ingest({k: v for k, v in list(corpus.items())[:1]}, store)
    assert m["docs"]["ingested"] == 1 and store.is_doc_complete("1234.00000")


def test_verification_detects_text_that_no_longer_matches_the_source(corpus, store):
    _ingest(corpus, store)
    counts = {d: len(store.doc_chunk_idxs(d)) for d in corpus}
    ids = [c.chunk_id for c in store.iter_chunks()]
    store._col.update(ids=ids, documents=["tampered text"] * len(ids), embeddings=EMB.encode(["tampered text"] * len(ids)).tolist())
    with pytest.raises(VerificationError, match="do not match their section text"):
        verify_store(store, corpus, list(corpus), counts, EMB)


def test_orphan_documents_fail_verification_and_manifest_is_still_written(corpus, store, tmp_path):
    _ingest(corpus, store)
    ghost = _paper(0, tweak="ghost")
    from rag.chunk_adapter import adapt_document, qasper_to_document

    res = adapt_document(qasper_to_document("ghost.doc", ghost), "ghost.doc", "recursive", embedder=EMB)
    store.upsert(res.chunks, EMB.encode([c.text for c in res.chunks]))
    with pytest.raises(VerificationError, match="not in the corpus"):
        _ingest(corpus, store, run_root=tmp_path / "runs")
    manifest = json.loads(next((tmp_path / "runs").glob("recursive_*/manifest.json")).read_text(encoding="utf-8"))
    assert manifest["verification"]["ok"] is False and manifest["verification"]["orphan_doc_ids"] == 1


def test_chunk_over_real_window_fails_verification_and_count_is_reported(corpus, store):
    _ingest(corpus, store)
    counts = {d: len(store.doc_chunk_idxs(d)) for d in corpus}
    strict = HashEmbedder(max_seq_length=20)
    with pytest.raises(VerificationError) as exc:
        verify_store(store, corpus, list(corpus), counts, strict)
    check = exc.value.report["token_check"]
    assert check["max_seq_length"] == 20
    assert check["chunks_over_max_seq_length"] > 0
    assert check["chunks_checked"] == store.count()


def test_manifest_contents_and_file(corpus, store, tmp_path):
    corpus_file = tmp_path / "corpus.json"
    corpus_file.write_text(json.dumps(corpus), encoding="utf-8")
    m = _ingest(corpus, store, corpus_path=str(corpus_file), dataset_sha256=file_sha256(corpus_file),
                code_fingerprint="f" * 64, store_location="tmp", run_root=tmp_path / "runs")
    on_disk = json.loads((tmp_path / "runs").glob("recursive_*/manifest.json").__next__().read_text(encoding="utf-8"))
    assert on_disk["dataset"]["sha256"] == file_sha256(corpus_file)
    assert on_disk["code_fingerprint"] == "f" * 64
    assert on_disk["embedder"]["name"] == EMB.name and on_disk["embedder"]["revision"] == EMB.revision
    assert on_disk["chunking_config"]["max_chunk_chars"] == cfg.chunking.max_chunk_chars
    assert set(on_disk["chunks"]["per_doc"]) == {"min", "median", "max"}
    assert on_disk["offset_confidence_ingested_docs_only"].keys() >= {"exact", "normalised", "ambiguous", "not_found"}
    assert sum(on_disk["offset_confidence_ingested_docs_only"].values()) == m["chunks"]["written_this_run"]
    assert on_disk["verification"]["token_check"]["chunks_over_max_seq_length"] == 0
    assert on_disk["verification"]["token_check"]["counter"] == "estimate"
    assert on_disk["duration_s"] >= 0 and on_disk["ingest_signature"]


def test_limit_docs(corpus, store):
    m = _ingest(corpus, store, limit_docs=2)
    assert m["docs"]["ingested"] == 2 and len(store.doc_ids()) == 2


def test_changed_chunking_config_is_refused_not_silently_skipped(corpus, store):
    _ingest(corpus, store)
    other = SimpleNamespace(**{**vars(cfg), "chunking": SimpleNamespace(**{**vars(cfg.chunking), "recursive_chunk_size": 500})})
    with pytest.raises(IngestConfigMismatch, match="different chunking/embedding setup"):
        ingest_corpus(corpus, "recursive", store, EMB, other, log=lambda *_: None)
    assert store.count() > 0  # nothing was deleted


def test_the_ingest_signature_covers_every_module_the_chunkers_depend_on():
    """A chunker that imports a helper module whose source is NOT in the signature would let a change to that helper
    slip past the stale-store check (this happened once: rag/splitting.py). rag.embedders is deliberately excluded:
    the embedder's name, revision and window are in the signature, not its source."""
    import ast
    from pathlib import Path

    from rag.ingestion import _CHUNKING_CODE

    root = Path(__file__).resolve().parents[2]
    covered = {Path(p).with_suffix("").as_posix().replace("/", ".") for p in _CHUNKING_CODE}
    needed = set()
    for rel in _CHUNKING_CODE:
        for node in ast.walk(ast.parse((root / rel).read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("rag."):
                needed.add(node.module)
            elif isinstance(node, ast.Import):
                needed |= {a.name for a in node.names if a.name.startswith("rag.")}
    assert needed - covered - {"rag.embedders"} == set()


def test_changed_embedder_revision_is_refused(corpus, store):
    _ingest(corpus, store)
    other = HashEmbedder()
    other.revision = "different-revision"
    with pytest.raises(IngestConfigMismatch):
        ingest_corpus(corpus, "recursive", store, other, cfg, log=lambda *_: None)


def test_upsert_batches_are_at_most_the_configured_size_and_last_batch_first(corpus, store, monkeypatch):
    store._batch = 7
    sizes, first_ids = [], []
    real = store._col.upsert
    monkeypatch.setattr(store._col, "upsert", lambda **kw: (sizes.append(len(kw["ids"])), first_ids.append(kw["metadatas"][0]["chunk_idx"]), real(**kw)))
    _ingest({k: v for k, v in list(corpus.items())[:1]}, store)
    assert max(sizes) <= 7 and len(sizes) > 1
    assert first_ids == sorted(first_ids, reverse=True)
