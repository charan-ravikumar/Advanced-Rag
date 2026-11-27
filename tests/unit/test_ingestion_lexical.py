"""Dual-store ingestion: vector store + SQLite FTS5 lexical index stay in lockstep. HashEmbedder, tmp dirs."""

import chromadb
import pytest
from chromadb.config import Settings

from config import cfg
from db.store import build_store
from rag.embedders import HashEmbedder
from rag.ingestion import VerificationError, delete_document, ingest_corpus, verify_store
from rag.lexical import SqliteFtsIndex
from rag.schemas_ingest import Chunk

TOPICS = ["alpha model training dataset", "network latency server cache", "protein cell enzyme sequence",
          "river delta soil water", "market price trade supply"]


def _sentences(topic, n, tag):
    return " ".join(f"{topic} {tag} number {i} appears in this sentence." for i in range(n))


def _paper(i, tweak=""):
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


class CountingEmbedder(HashEmbedder):
    def __init__(self):
        super().__init__()
        self.encode_calls = 0

    def encode(self, texts, show_progress_bar=False, **kw):
        self.encode_calls += 1
        return super().encode(texts, show_progress_bar=show_progress_bar, **kw)


@pytest.fixture
def corpus():
    return {f"1234.0000{i}": _paper(i) for i in range(5)}


@pytest.fixture
def store(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"), settings=Settings(anonymized_telemetry=False))
    return build_store(cfg, "recursive", client=client)


@pytest.fixture
def lexical(tmp_path):
    ix = SqliteFtsIndex(tmp_path / "lexical_recursive.sqlite3")
    yield ix
    ix.close()


@pytest.fixture
def emb():
    return CountingEmbedder()


def _ingest(corpus, store, lexical, emb, **kw):
    return ingest_corpus(corpus, "recursive", store, emb, cfg, lexical=lexical, log=lambda *_: None, **kw)


def _per_doc_ids(chunks):
    out = {}
    for c in chunks:
        out.setdefault(c.doc_id, set()).add(c.chunk_id)
    return out


def _lexical_per_doc(lexical):
    return {d: lexical.doc_chunk_ids(d) for d in lexical.doc_ids()}


def test_first_ingest_fills_both_stores_with_the_same_ids(corpus, store, lexical, emb):
    m = _ingest(corpus, store, lexical, emb)
    assert store.count() == lexical.count() == m["chunks"]["written_this_run"] > 0
    assert lexical.chunk_ids() == {c.chunk_id for c in store.get_all()}
    assert lexical.fts_row_count() == lexical.count()
    par = m["verification"]["lexical"]
    assert par["count_parity"] and par["id_set_parity"] and par["missing_in_lexical"] == par["extra_in_lexical"] == 0
    assert m["lexical"]["count_after"] == store.count()


def test_reingest_unchanged_makes_no_writes_to_either_store(corpus, store, lexical, emb, monkeypatch):
    _ingest(corpus, store, lexical, emb)
    calls = []

    def spy(obj, name):
        real = getattr(obj, name)
        monkeypatch.setattr(obj, name, lambda *a, **k: (calls.append(f"{type(obj).__name__}.{name}"), real(*a, **k))[1])

    for name in ("upsert", "delete_by_doc"):
        spy(store, name)
        spy(lexical, name)
    encodes_before = emb.encode_calls
    m = _ingest(corpus, store, lexical, emb)
    assert calls == []
    assert emb.encode_calls == encodes_before  # nothing was re-embedded
    assert m["docs"]["skipped_unchanged"] == 5 and m["docs"]["ingested"] == 0 and m["chunks"]["written_this_run"] == 0


def test_changed_doc_replaces_its_chunks_in_both_stores(corpus, store, lexical, emb):
    _ingest(corpus, store, lexical, emb)
    before_v, before_l = _per_doc_ids(store.get_all()), _lexical_per_doc(lexical)
    target = "1234.00002"
    changed = dict(corpus)
    changed[target] = _paper(2, tweak="EDITED")
    m = _ingest(changed, store, lexical, emb)
    after_v, after_l = _per_doc_ids(store.get_all()), _lexical_per_doc(lexical)
    assert m["docs"]["ingested"] == 1 and m["docs"]["replaced_old_version"] == 1 and m["docs"]["skipped_unchanged"] == 4
    assert after_v == after_l  # the two stores agree document by document
    for doc_id in corpus:
        if doc_id != target:
            assert after_v[doc_id] == before_v[doc_id] and after_l[doc_id] == before_l[doc_id]
    assert after_v[target] != before_v[target] and after_l[target] != before_l[target]
    stale = before_l[target] - after_l[target]
    assert stale and not (stale & lexical.chunk_ids()) and not (stale & {c.chunk_id for c in store.get_all()})
    assert m["verification"]["lexical"]["id_set_parity"] is True


def test_delete_document_removes_it_from_both_stores(corpus, store, lexical, emb):
    _ingest(corpus, store, lexical, emb)
    target = "1234.00001"
    target_ids = lexical.doc_chunk_ids(target)
    assert target_ids and target_ids == store.doc_chunk_ids(target)
    assert lexical.search("network latency server cache", k=50)  # findable before
    removed = delete_document(store, lexical, target)
    assert removed["vector"] == removed["lexical"] == len(target_ids)
    assert store.get_doc_versions(target) == set() and lexical.doc_chunk_ids(target) == set()
    assert target not in store.doc_ids() and target not in lexical.doc_ids()
    assert store.count() == lexical.count()
    assert not target_ids & lexical.chunk_ids()
    assert not target_ids & {c.chunk_id for c in store.get_all()}
    assert not target_ids & {cid for cid, _ in lexical.search("network latency server cache", k=500)}
    assert all(h["metadata"]["doc_id"] != target for h in store.query(emb.encode(TOPICS[1]), k=store.count()))


def test_vector_only_store_is_backfilled_without_reembedding(corpus, store, lexical, emb):
    """The real recursive store was built before the lexical index existed: it must be completed
    from the stored chunks, not re-embedded."""
    _ingest_vector_only = ingest_corpus(corpus, "recursive", store, emb, cfg, log=lambda *_: None)
    assert lexical.count() == 0 and _ingest_vector_only["lexical"] is None
    encodes = emb.encode_calls
    m = _ingest(corpus, store, lexical, emb)
    assert emb.encode_calls == encodes  # no embedding happened
    assert m["docs"]["lexical_backfilled"] == 5 and m["docs"]["skipped_unchanged"] == 5 and m["docs"]["ingested"] == 0
    assert lexical.chunk_ids() == {c.chunk_id for c in store.get_all()}


def test_crash_between_the_two_writes_is_repaired_without_reembedding(corpus, store, lexical, emb, monkeypatch):
    one = {k: v for k, v in list(corpus.items())[:1]}
    real = lexical.upsert
    monkeypatch.setattr(lexical, "upsert", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("crash after vector write")))
    with pytest.raises(RuntimeError, match="crash after vector write"):
        _ingest(one, store, lexical, emb)
    assert store.count() > 0 and lexical.count() == 0
    monkeypatch.setattr(lexical, "upsert", real)
    encodes = emb.encode_calls
    m = _ingest(one, store, lexical, emb)
    assert emb.encode_calls == encodes and m["docs"]["lexical_backfilled"] == 1
    assert lexical.chunk_ids() == {c.chunk_id for c in store.get_all()}


def test_stale_lexical_rows_from_a_crash_during_replacement_are_cleaned(corpus, store, lexical, emb):
    _ingest(corpus, store, lexical, emb)
    target = "1234.00004"
    store.delete_by_doc(target)  # vector side deleted, lexical side crashed before deleting
    assert lexical.doc_chunk_ids(target)
    m = _ingest(corpus, store, lexical, emb)
    assert m["docs"]["ingested"] == 1
    assert _per_doc_ids(store.get_all()) == _lexical_per_doc(lexical)


def _counts(store, corpus):
    return {d: len(store.doc_chunk_idxs(d)) for d in corpus}


def test_verification_fails_when_lexical_lost_a_chunk(corpus, store, lexical, emb):
    _ingest(corpus, store, lexical, emb)
    victim = sorted(lexical.chunk_ids())[3]
    row = lexical._conn.execute("SELECT fts_rowid FROM chunk_map WHERE chunk_id=?", (victim,)).fetchone()[0]
    lexical._conn.execute("DELETE FROM chunk_map WHERE chunk_id=?", (victim,))
    lexical._conn.execute("DELETE FROM chunks WHERE rowid=?", (row,))
    lexical._conn.commit()
    with pytest.raises(VerificationError) as exc:
        verify_store(store, corpus, list(corpus), _counts(store, corpus), emb, lexical=lexical)
    assert "lexical.count()" in str(exc.value) and "missing in lexical" in str(exc.value)
    assert exc.value.report["lexical"]["count_parity"] is False


def test_verification_fails_on_equal_counts_but_different_ids(corpus, store, lexical, emb):
    """Count parity alone is not enough: swap one id for another so the counts still match."""
    _ingest(corpus, store, lexical, emb)
    victim = sorted(lexical.chunk_ids())[0]
    lexical._conn.execute("UPDATE chunk_map SET chunk_id=? WHERE chunk_id=?", ("ffffffffffffffff", victim))
    lexical._conn.commit()
    assert lexical.count() == store.count()
    with pytest.raises(VerificationError) as exc:
        verify_store(store, corpus, list(corpus), _counts(store, corpus), emb, lexical=lexical)
    assert exc.value.report["lexical"]["count_parity"] is True
    assert exc.value.report["lexical"]["id_set_parity"] is False
    assert "chunk_id sets differ" in str(exc.value)


def test_verification_fails_on_extra_lexical_chunk(corpus, store, lexical, emb):
    _ingest(corpus, store, lexical, emb)
    lexical.upsert([Chunk.create(doc_id="1234.00000", section="S", chunk_idx=999, char_start=0, char_end=3,
                                 strategy="recursive", text="ghost chunk", doc_version="v")])
    with pytest.raises(VerificationError, match="extra in lexical"):
        verify_store(store, corpus, list(corpus), _counts(store, corpus), emb, lexical=lexical)


def test_manifest_records_the_lexical_index(corpus, store, lexical, emb, tmp_path):
    m = _ingest(corpus, store, lexical, emb, run_root=tmp_path / "runs")
    assert m["lexical"]["path"].endswith("lexical_recursive.sqlite3") and m["lexical"]["docs_backfilled_from_store"] == 0
    assert m["verification"]["ok"] is True


def test_backfill_script_logic_is_idempotent_and_checks_parity(corpus, store, lexical, emb):
    from rag.ingestion import backfill_lexical

    ingest_corpus(corpus, "recursive", store, emb, cfg, log=lambda *_: None)  # vector only
    first = backfill_lexical(store, lexical)
    second = backfill_lexical(store, lexical)
    assert first["parity"]["ok"] and second["parity"]["ok"]
    assert lexical.count() == store.count() == lexical.fts_row_count()
    assert first["docs"] == 5 and first["chunks"] == store.count()


def test_backfill_fails_loudly_on_a_stale_lexical_doc(corpus, store, lexical, emb):
    from rag.ingestion import backfill_lexical

    ingest_corpus(corpus, "recursive", store, emb, cfg, log=lambda *_: None)
    before = store.count()
    lexical.upsert([Chunk.create(doc_id="gone.doc", section="S", chunk_idx=0, char_start=0, char_end=3,
                                 strategy="recursive", text="orphan row", doc_version="v")])
    with pytest.raises(VerificationError, match="extra in lexical"):
        backfill_lexical(store, lexical)
    assert store.count() == before > 0  # the vector store was not touched
