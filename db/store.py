"""Vector store behind a small protocol, with a local Chroma implementation.

Local only: ``build_store`` opens a ``chromadb.PersistentClient`` under the directory
named by ``cfg.store.chroma_dir`` (default ``%LOCALAPPDATA%/advrag/chroma``, outside
OneDrive), one collection per chunking strategy. Importing this module does no I/O.

There is deliberately no "clear collection" method: the only deletion is
``delete_by_doc``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Protocol

from rag.schemas_ingest import Chunk

STRATEGIES = ("recursive", "layout", "semantic")


class VectorStore(Protocol):
    """What ingestion, verification and retrieval need from a store."""

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None: ...

    def query(self, embedding: Sequence[float], k: int, where: dict | None = None) -> list[dict[str, Any]]: ...

    def get_doc_versions(self, doc_id: str) -> set[str]: ...

    def doc_chunk_idxs(self, doc_id: str) -> list[int]: ...

    def doc_chunk_ids(self, doc_id: str) -> set[str]: ...

    def get_all(self) -> list[Chunk]: ...

    def is_doc_complete(self, doc_id: str) -> bool: ...

    def get_signature(self) -> str | None: ...

    def set_signature(self, signature: str) -> None: ...

    def delete_by_doc(self, doc_id: str) -> int: ...

    def count(self) -> int: ...

    def doc_ids(self) -> set[str]: ...

    def get_chunks(self, ids: Sequence[str]) -> list[Chunk]: ...

    def iter_chunks(self, batch_size: int = 1000) -> Iterator[Chunk]: ...


def resolve_store_path(raw: str) -> Path:
    """Expand %VARS% / ~ and refuse synced folders (SQLite files do not survive sync).

    Where ``LOCALAPPDATA`` is not set, ``%LOCALAPPDATA%`` means ``~/.local/share``.
    """
    expanded = raw
    if not os.environ.get("LOCALAPPDATA"):  # macOS / Linux: the Windows variable in config.yaml maps to the XDG data folder
        expanded = expanded.replace("%LOCALAPPDATA%", "~/.local/share")
    path = Path(os.path.expandvars(os.path.expanduser(expanded)))
    if "%" in str(path):
        raise RuntimeError(f"unresolved environment variable in store path: {raw!r}")
    if any("onedrive" in part.lower() for part in path.parts):
        raise RuntimeError(f"store path must be outside OneDrive: {path}")
    return path


class ChromaStore:
    """VectorStore over one Chroma collection (plus a tiny state collection for the ingest signature)."""

    STATE_COLLECTION = "ingest_state"

    def __init__(self, collection, batch_size: int = 300, state_collection=None):
        self._col = collection
        self._batch = batch_size
        self._state = state_collection

    @property
    def name(self) -> str:
        return self._col.name

    def get_signature(self) -> str | None:
        """Signature of the chunking/embedding setup this collection was built with."""
        if self._state is None:
            return None
        res = self._state.get(ids=[self.name], include=["metadatas"])
        return res["metadatas"][0]["signature"] if res["ids"] else None

    def set_signature(self, signature: str) -> None:
        if self._state is None:
            raise RuntimeError("store has no state collection")
        self._state.upsert(ids=[self.name], embeddings=[[0.0]], metadatas=[{"signature": signature}])

    def upsert(self, chunks, embeddings) -> None:
        """Upsert in batches of at most ``batch_size``, LAST batch first.

        Chunks arrive in document order, so a crash part-way leaves the tail present and the
        head (chunk_idx 0..) missing, which ``is_doc_complete`` detects. Forward order would
        leave a truncated tail that nothing could detect.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        for i in reversed(range(0, len(chunks), self._batch)):
            part = chunks[i : i + self._batch]
            self._col.upsert(
                ids=[c.chunk_id for c in part],
                documents=[c.text for c in part],
                embeddings=[list(map(float, e)) for e in embeddings[i : i + self._batch]],
                metadatas=[c.to_metadata() for c in part],
            )

    def query(self, embedding, k, where=None):
        res = self._col.query(
            query_embeddings=[list(map(float, embedding))],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {"chunk_id": i, "text": d, "metadata": m, "distance": dist}
            for i, d, m, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def get_doc_versions(self, doc_id) -> set[str]:
        res = self._col.get(where={"doc_id": doc_id}, include=["metadatas"])
        return {m["doc_version"] for m in res["metadatas"]}

    def doc_chunk_idxs(self, doc_id) -> list[int]:
        """Sorted ``chunk_idx`` values stored for the document."""
        res = self._col.get(where={"doc_id": doc_id}, include=["metadatas"])
        return sorted(int(m["chunk_idx"]) for m in res["metadatas"])

    def doc_chunk_ids(self, doc_id) -> set[str]:
        return set(self._col.get(where={"doc_id": doc_id}, include=[])["ids"])

    def get_all(self) -> list[Chunk]:
        """Every stored chunk (for backfilling the lexical index and for parity checks)."""
        return list(self.iter_chunks())

    def is_doc_complete(self, doc_id) -> bool:
        """True when the stored chunk indices are exactly 0..n-1 (no hole at the head or middle)."""
        idxs = self.doc_chunk_idxs(doc_id)
        return bool(idxs) and idxs == list(range(len(idxs)))

    def delete_by_doc(self, doc_id) -> int:
        ids = self._col.get(where={"doc_id": doc_id}, include=[])["ids"]
        for i in range(0, len(ids), self._batch):
            self._col.delete(ids=ids[i : i + self._batch])
        return len(ids)

    def count(self) -> int:
        return self._col.count()

    def doc_ids(self) -> set[str]:
        return {c.doc_id for c in self.iter_chunks()}

    def get_chunks(self, ids) -> list[Chunk]:
        res = self._col.get(ids=list(ids), include=["documents", "metadatas"])
        return [Chunk.from_metadata_and_text(m, d) for m, d in zip(res["metadatas"], res["documents"])]

    def iter_chunks(self, batch_size: int = 1000) -> Iterator[Chunk]:
        offset = 0
        while True:
            res = self._col.get(limit=batch_size, offset=offset, include=["documents", "metadatas"])
            if not res["ids"]:
                return
            for m, d in zip(res["metadatas"], res["documents"]):
                yield Chunk.from_metadata_and_text(m, d)
            offset += len(res["ids"])


def build_store(cfg, strategy: str, *, client=None, path: str | Path | None = None) -> ChromaStore:
    """Open (or create) the local collection for ``strategy``.

    The store directory is the ``path`` argument, else the ``CHROMA_LOCAL_PATH``
    environment variable, else ``cfg.store.chroma_dir``; a OneDrive path is always refused.
    ``client`` injects a ready Chroma client (tests use a ``tmp_path`` one).
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy!r}")
    if client is None:
        import chromadb
        from chromadb.config import Settings

        # precedence: explicit argument, then CHROMA_LOCAL_PATH, then config.yaml
        raw = str(path) if path else os.environ.get("CHROMA_LOCAL_PATH") or cfg.store.chroma_dir
        root = resolve_store_path(raw)
        root.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(root), settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(
        name=f"{cfg.store.collection_prefix}{strategy}",
        metadata={"hnsw:space": "cosine"},
    )
    state = client.get_or_create_collection(name=ChromaStore.STATE_COLLECTION)
    return ChromaStore(collection, batch_size=cfg.store.upsert_batch_size, state_collection=state)
