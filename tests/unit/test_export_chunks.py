"""scripts/export_chunks.py: the export and its verification, on a two-paper fixture with fake store and lexical index (no model, no Chroma, no network)."""

import dataclasses
import hashlib
import json

import pytest

from rag.schemas_ingest import Chunk
from scripts import export_chunks as ex


def chunk(doc, idx, start, end, section="Methods"):
    return Chunk.create(doc_id=doc, section=section, chunk_idx=idx, char_start=start, char_end=end, strategy="recursive", text=f"text {doc} {idx}", doc_version="v1")


CHUNKS = [chunk("pa", i, i * 500, (i + 1) * 500) for i in range(3)] + [chunk("pb", i, i * 500, (i + 1) * 500) for i in range(2)]


class FakeStore:
    name = "rag_recursive"

    def __init__(self, chunks):
        self.chunks = chunks

    def get_all(self):
        return list(reversed(self.chunks))  # the order the store returns is not the export order

    def count(self):
        return len(self.chunks)

    def get_signature(self):
        return "sig"


class FakeLexical:
    def __init__(self, ids, count=None):
        self._ids, self._count = set(ids), count

    def chunk_ids(self):
        return self._ids

    def count(self):
        return len(self._ids) if self._count is None else self._count


def span(doc, start, end):
    return {"doc_id": doc, "section": "Methods", "section_idx": 0, "char_start": start, "char_end": end, "evidence_idx": 0, "confidence": "exact"}


def question(qid, doc, start, end):
    return {"id": qid, "paper_id": doc, "question": f"question {qid}?", "answerable": True, "gold_spans": [span(doc, start, end)]}


@pytest.fixture
def files(tmp_path):
    golden = {"questions": [question("qa", "pa", 520, 700), question("qb", "pb", 10, 100), {**question("qu", "pa", 0, 1), "answerable": False}]}
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(json.dumps(golden), encoding="utf-8")
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"dev": ["qa", "qu"], "test": ["qb"], "golden_set_sha256": hashlib.sha256(golden_path.read_bytes()).hexdigest()}), encoding="utf-8")
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps({"pa": {}, "pb": {}}), encoding="utf-8")
    return ["--golden", str(golden_path), "--split-file", str(split_path), "--corpus", str(corpus_path), "--out", str(tmp_path / "out")]


def run(files, chunks=CHUNKS, lexical=None, extra=()):
    return ex.main([*files, *extra], store=FakeStore(chunks), lexical=lexical or FakeLexical(c.chunk_id for c in chunks), fingerprint="f" * 64)


def out_dir(tmp_path):
    (d,) = (tmp_path / "out").iterdir()
    return d


def test_the_export_has_paper_id_in_every_records_metadata_and_is_sorted_by_paper_and_chunk_idx(files, tmp_path):
    assert run(files) == 0
    rows = [json.loads(line) for line in (out_dir(tmp_path) / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(r["metadata"]["paper_id"], r["metadata"]["chunk_idx"]) for r in rows] == [("pa", 0), ("pa", 1), ("pa", 2), ("pb", 0), ("pb", 1)]
    assert all(r["metadata"]["paper_id"] == r["metadata"]["doc_id"] and r["text"] and r["chunk_id"] for r in rows)
    assert {r["metadata"]["paper_id"]: r["metadata"]["split"] for r in rows} == {"pa": "dev", "pb": "test"}


def test_the_dev_labels_use_the_exported_chunk_ids_and_the_test_label_is_not_exported(files, tmp_path):
    assert run(files) == 0
    d = out_dir(tmp_path)
    qrels = [json.loads(line) for line in (d / "qrels_dev.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["question_id"] for r in qrels] == ["qa"]  # the unanswerable dev question and the test question are not in it
    assert qrels[0]["relevant_chunk_ids"] == [CHUNKS[1].chunk_id] and qrels[0]["paper_id"] == "pa" and qrels[0]["query"] == "question qa?"
    ids = {json.loads(line)["chunk_id"] for line in (d / "chunks.jsonl").read_text(encoding="utf-8").splitlines()}
    assert set(qrels[0]["relevant_chunk_ids"]) <= ids and "qb" not in (d / "qrels_dev.jsonl").read_text(encoding="utf-8")


def test_the_manifest_counts_and_the_checksums_are_right(files, tmp_path):
    assert run(files) == 0
    d = out_dir(tmp_path)
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["chunks"] == 5 and manifest["counts"]["papers"] == 2 and manifest["counts"]["chunks_per_split"] == {"dev": 3, "test": 2}
    assert manifest["store"] == {"collection": "rag_recursive", "vector_count": 5, "lexical_count": 5, "ingest_signature": "sig"} and manifest["code_fingerprint"] == "f" * 64
    for line in (d / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ")
        assert hashlib.sha256((d / name).read_bytes()).hexdigest() == digest


def test_verify_only_writes_nothing(files, tmp_path):
    assert run(files, extra=["--verify-only"]) == 0
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("what", ["lexical_extra_id", "lexical_missing_id", "tampered_id", "duplicate", "paper_not_in_corpus", "hole_in_chunk_idx", "unreachable_label"])
def test_any_inconsistency_stops_the_export_and_writes_nothing(files, tmp_path, what, capsys):
    chunks, lexical = list(CHUNKS), None
    if what == "lexical_extra_id":
        lexical = FakeLexical([c.chunk_id for c in chunks] + ["ghost"])
    elif what == "lexical_missing_id":
        lexical = FakeLexical([c.chunk_id for c in chunks][:-1])
    elif what == "tampered_id":
        chunks[0] = dataclasses.replace(chunks[0], chunk_id="0" * 16)
    elif what == "duplicate":
        chunks.append(chunks[0])
    elif what == "paper_not_in_corpus":
        chunks.append(chunk("pz", 0, 0, 500))
    elif what == "hole_in_chunk_idx":
        chunks = [c for c in chunks if not (c.doc_id == "pa" and c.chunk_idx == 1)]
    elif what == "unreachable_label":
        chunks = [c for c in chunks if not (c.doc_id == "pa" and c.chunk_idx == 1)] + [chunk("pa", 1, 5000, 5500)]  # the evidence span [520, 700) now overlaps nothing
    assert run(files, chunks=chunks, lexical=lexical) == 1
    assert not (tmp_path / "out").exists() and "VERIFICATION FAILED" in capsys.readouterr().err


def test_it_refuses_to_run_without_a_named_store(monkeypatch, files):
    monkeypatch.delenv("CHROMA_LOCAL_PATH", raising=False)
    assert ex.main(files) == 2


def test_it_refuses_a_split_made_for_another_golden_set(files, tmp_path):
    split_path = tmp_path / "split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split["golden_set_sha256"] = "0" * 64
    split_path.write_text(json.dumps(split), encoding="utf-8")
    assert run(files) == 2
