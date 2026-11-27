"""VectorStore / build_store tests: local Chroma in tmp_path, HashEmbedder, no network."""

import chromadb
import pytest
from chromadb.config import Settings

from config import cfg
from db.store import ChromaStore, VectorStore, build_store, resolve_store_path
from rag.chunk_adapter import adapt_document, qasper_to_document
from rag.embedders import Embedder, HashEmbedder

EMB = HashEmbedder()


def _paper(tag="a"):
    return {
        "title": "T",
        "abstract": f"Abstract about {tag} datasets and models. " * 3,
        "full_text": [
            {"section_name": "Intro", "paragraphs": [f"Introduction {tag} sentence one. " * 5]},
            {"section_name": "Methods", "paragraphs": [f"Methods {tag} alpha beta gamma. " * 8]},
        ],
    }


@pytest.fixture
def store(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"), settings=Settings(anonymized_telemetry=False))
    return build_store(cfg, "layout", client=client)


def _ingest(store, doc_id, tag="a"):
    res = adapt_document(qasper_to_document(doc_id, _paper(tag)), doc_id, "layout", embedder=EMB)
    store.upsert(res.chunks, EMB.encode([c.text for c in res.chunks]))
    return res


def test_protocols_are_satisfied(store):
    assert isinstance(EMB, Embedder)
    assert isinstance(store, ChromaStore)
    for method in ("upsert", "query", "get_doc_versions", "delete_by_doc", "count", "doc_ids"):
        assert callable(getattr(VectorStore, method))


def test_upsert_is_idempotent_and_round_trips(store):
    res = _ingest(store, "p1")
    n = store.count()
    assert n == len(res.chunks)
    _ingest(store, "p1")
    assert store.count() == n
    back = store.get_chunks([c.chunk_id for c in res.chunks])
    assert {c.chunk_id: c.text for c in back} == {c.chunk_id: c.text for c in res.chunks}


def test_query_returns_nearest_with_metadata(store):
    _ingest(store, "p1", "alpha")
    res = store.query(EMB.encode("Methods alpha beta gamma"), k=2)
    assert res and res[0]["metadata"]["doc_id"] == "p1"
    assert {"chunk_id", "text", "metadata", "distance"} <= set(res[0])


def test_delete_by_doc_removes_only_that_doc(store):
    _ingest(store, "p1")
    _ingest(store, "p2", "b")
    before = store.count()
    removed = store.delete_by_doc("p1")
    assert removed > 0 and store.count() == before - removed
    assert store.doc_ids() == {"p2"}
    assert store.get_doc_versions("p1") == set()
    assert all(r["metadata"]["doc_id"] == "p2" for r in store.query(EMB.encode("Introduction"), k=10))


def test_doc_version_is_stored(store):
    res = _ingest(store, "p1")
    assert store.get_doc_versions("p1") == {res.doc_version}


def test_collection_is_per_strategy(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "c"), settings=Settings(anonymized_telemetry=False))
    a, b = build_store(cfg, "layout", client=client), build_store(cfg, "recursive", client=client)
    assert a.name == "rag_layout" and b.name == "rag_recursive"
    res = adapt_document(qasper_to_document("p1", _paper()), "p1", "layout", embedder=EMB)
    a.upsert(res.chunks, EMB.encode([c.text for c in res.chunks]))
    assert a.count() > 0 and b.count() == 0


def test_unknown_strategy_rejected(tmp_path):
    with pytest.raises(ValueError):
        build_store(cfg, "nope", path=tmp_path)


def test_store_has_no_clear_collection_method(store):
    assert not any("clear" in name or "drop" in name for name in dir(store) if not name.startswith("_"))


def test_default_store_path_is_outside_onedrive_and_local():
    path = resolve_store_path(cfg.store.chroma_dir)
    assert "advrag" in path.parts
    assert not any("onedrive" in p.lower() for p in path.parts)


def test_store_path_inside_onedrive_refused(tmp_path):
    with pytest.raises(RuntimeError, match="OneDrive"):
        resolve_store_path(str(tmp_path / "OneDrive - X" / "chroma"))


def test_build_store_uses_given_path(tmp_path):
    store = build_store(cfg, "semantic", path=tmp_path / "chroma")
    assert store.count() == 0 and (tmp_path / "chroma").exists()


def test_chroma_local_path_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_LOCAL_PATH", str(tmp_path / "from_env"))
    store = build_store(cfg, "recursive")
    assert (tmp_path / "from_env").exists() and store.count() == 0


def test_explicit_path_beats_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_LOCAL_PATH", str(tmp_path / "from_env"))
    build_store(cfg, "recursive", path=tmp_path / "explicit")
    assert (tmp_path / "explicit").exists() and not (tmp_path / "from_env").exists()


def test_chroma_local_path_env_inside_onedrive_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_LOCAL_PATH", str(tmp_path / "OneDrive - Corp" / "chroma"))
    with pytest.raises(RuntimeError, match="OneDrive"):
        build_store(cfg, "recursive")
    assert not (tmp_path / "OneDrive - Corp").exists()
