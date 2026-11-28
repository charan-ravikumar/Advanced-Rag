"""Persistent lexical index: SQLite FTS5 with Okapi BM25 ranking.

Importing this module does no I/O. ``SqliteFtsIndex(path)`` opens (and creates) one file.

Layout
  chunks(chunk_id UNINDEXED, doc_id UNINDEXED, text)   FTS5, tokenize='porter unicode61'
  chunk_map(chunk_id PK, doc_id, fts_rowid)            ordinary table: O(log N) delete by chunk or doc,
                                                       because FTS5 UNINDEXED columns cannot be looked up
                                                       by index (a delete by chunk_id would scan the table)

Scores returned by ``search`` are higher = better: FTS5's ``bm25()`` is negative (lower is better),
so it is negated. Ties are broken by chunk_id so the order is deterministic.
"""

from __future__ import annotations

import os
import queue
import re
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from db.store import resolve_store_path
from rag.schemas_ingest import Chunk

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
MAX_QUERY_TERMS = 64
MAX_DOC_FILTER = 30000  # stays under SQLite's bound-variable limit


class LexicalIndex(Protocol):
    def upsert(self, chunks: Sequence[Chunk]) -> None: ...

    def delete_by_doc(self, doc_id: str) -> int: ...

    def search(self, query: str, k: int, doc_ids: Sequence[str] | None = None) -> list[tuple[str, float]]: ...

    def count(self) -> int: ...

    def chunk_ids(self) -> set[str]: ...

    def doc_chunk_ids(self, doc_id: str) -> set[str]: ...


def escape_query(query: str) -> str | None:
    """Turn free text into a safe FTS5 MATCH expression, or ``None`` when it has no searchable word.

    Every word is emitted as a double-quoted FTS5 string and the words are OR-ed. A quoted string
    is always a literal, so quotes, hyphens, colons, parentheses, ``*``, ``^`` and the words
    AND / OR / NOT / NEAR can neither raise a syntax error nor act as operators. OR (not AND) because
    natural-language questions rarely have all their words in one chunk; BM25 ranks the partial matches.
    """
    seen: dict[str, None] = {}
    for word in _WORD.findall(query):
        seen.setdefault(word.lower(), None)
        if len(seen) >= MAX_QUERY_TERMS:
            break
    if not seen:
        return None
    return " OR ".join('"' + w.replace('"', '""') + '"' for w in seen)


def lexical_index_path(cfg, strategy: str, chroma_path: str | Path | None = None) -> Path:
    """The lexical index file, a sibling of the Chroma dir.

    For the default Chroma directory name ``chroma`` this is ``lexical_<strategy>.sqlite3``. For any other
    directory name (for example ``chroma_v2``) the name is part of the file name,
    ``lexical_<dirname>_<strategy>.sqlite3``, so two stores in the same parent never share an index.
    The Chroma dir is ``chroma_path``, else ``CHROMA_LOCAL_PATH``, else ``cfg.store.chroma_dir``.
    OneDrive paths are refused (same rule as the vector store).
    """
    raw = str(chroma_path) if chroma_path else os.environ.get("CHROMA_LOCAL_PATH") or cfg.store.chroma_dir
    chroma_dir = resolve_store_path(raw)
    name = f"lexical_{strategy}.sqlite3" if chroma_dir.name == "chroma" else f"lexical_{chroma_dir.name}_{strategy}.sqlite3"
    return chroma_dir.parent / name


class SqliteFtsIndex:
    """LexicalIndex over one SQLite file. Single-connection, not thread-safe: for concurrent readers use ``PooledFtsIndex``."""

    def __init__(self, path: str | Path, *, check_same_thread: bool = True):
        """``check_same_thread=False`` lets ONE index be used from several threads, one at a time (``PooledFtsIndex`` gives each caller
        a connection of its own). The default keeps SQLite's own thread check."""
        path = Path(path)
        if any("onedrive" in part.lower() for part in path.resolve().parts):
            raise RuntimeError(f"lexical index path must be outside OneDrive: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(str(path), check_same_thread=check_same_thread)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        try:
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5("
                "chunk_id UNINDEXED, doc_id UNINDEXED, text, tokenize='porter unicode61')"
            )
        except sqlite3.OperationalError as exc:
            raise RuntimeError(f"this SQLite build has no FTS5 support: {exc}") from exc
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS chunk_map("
            "chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, fts_rowid INTEGER NOT NULL)"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS chunk_map_doc ON chunk_map(doc_id)")
        self._conn.commit()

    # -- writes ----------------------------------------------------------------------------
    def upsert(self, chunks: Iterable[Chunk]) -> None:
        with self._conn:  # one transaction for the whole batch
            for c in chunks:
                row = self._conn.execute("SELECT fts_rowid FROM chunk_map WHERE chunk_id = ?", (c.chunk_id,)).fetchone()
                if row:
                    self._conn.execute("DELETE FROM chunks WHERE rowid = ?", (row[0],))
                    self._conn.execute("DELETE FROM chunk_map WHERE chunk_id = ?", (c.chunk_id,))
                cur = self._conn.execute(
                    "INSERT INTO chunks(chunk_id, doc_id, text) VALUES (?, ?, ?)", (c.chunk_id, c.doc_id, c.text)
                )
                self._conn.execute(
                    "INSERT INTO chunk_map(chunk_id, doc_id, fts_rowid) VALUES (?, ?, ?)",
                    (c.chunk_id, c.doc_id, cur.lastrowid),
                )

    def delete_by_doc(self, doc_id: str) -> int:
        with self._conn:
            rowids = [r[0] for r in self._conn.execute("SELECT fts_rowid FROM chunk_map WHERE doc_id = ?", (doc_id,))]
            self._conn.executemany("DELETE FROM chunks WHERE rowid = ?", [(r,) for r in rowids])
            self._conn.execute("DELETE FROM chunk_map WHERE doc_id = ?", (doc_id,))
        return len(rowids)

    # -- reads -----------------------------------------------------------------------------
    def search(self, query: str, k: int, doc_ids: Sequence[str] | None = None) -> list[tuple[str, float]]:
        """Top-``k`` ``(chunk_id, score)``, best first, higher = better. ``doc_ids`` restricts the
        candidate set BEFORE ranking and limiting (it is part of the SQL WHERE), never after."""
        match = escape_query(query)
        if match is None or k <= 0:
            return []
        sql = "SELECT chunk_id, bm25(chunks) AS s FROM chunks WHERE chunks MATCH ?"
        params: list = [match]
        if doc_ids is not None:
            doc_ids = list(dict.fromkeys(doc_ids))
            if not doc_ids:
                return []
            if len(doc_ids) > MAX_DOC_FILTER:
                raise ValueError(f"doc_ids filter too large ({len(doc_ids)} > {MAX_DOC_FILTER})")
            sql += " AND doc_id IN (" + ",".join("?" * len(doc_ids)) + ")"
            params += doc_ids
        sql += " ORDER BY s, chunk_id LIMIT ?"
        params.append(k)
        return [(cid, -float(s)) for cid, s in self._conn.execute(sql, params)]

    def count(self) -> int:
        return self._conn.execute("SELECT count(*) FROM chunk_map").fetchone()[0]

    def chunk_ids(self) -> set[str]:
        return {r[0] for r in self._conn.execute("SELECT chunk_id FROM chunk_map")}

    def doc_chunk_ids(self, doc_id: str) -> set[str]:
        return {r[0] for r in self._conn.execute("SELECT chunk_id FROM chunk_map WHERE doc_id = ?", (doc_id,))}

    def doc_ids(self) -> set[str]:
        return {r[0] for r in self._conn.execute("SELECT DISTINCT doc_id FROM chunk_map")}

    def fts_row_count(self) -> int:
        """Rows in the FTS table itself (must equal ``count()``; used by consistency checks)."""
        return self._conn.execute("SELECT count(*) FROM chunks").fetchone()[0]

    def close(self) -> None:
        self._conn.close()


class PooledFtsIndex:
    """A fixed pool of read connections to one index: concurrent searches each use their own connection (the file is in WAL mode, so
    readers do not block each other or a writer). A caller waits when every connection is busy. Read only: writes go through
    ``SqliteFtsIndex``; a pooled connection refuses them."""

    def __init__(self, path: str | Path, size: int = 4):
        if size < 1:
            raise ValueError("the pool needs at least one connection")
        self.path, self.size = Path(path), size
        self._all: list[SqliteFtsIndex] = []
        self._free: queue.Queue[SqliteFtsIndex] = queue.Queue()
        for _ in range(size):
            index = SqliteFtsIndex(path, check_same_thread=False)
            index._conn.execute("PRAGMA query_only=ON")
            self._all.append(index)
            self._free.put(index)

    @contextmanager
    def connection(self) -> Iterator[SqliteFtsIndex]:
        index = self._free.get()
        try:
            yield index
        finally:
            self._free.put(index)

    def search(self, query: str, k: int, doc_ids: Sequence[str] | None = None) -> list[tuple[str, float]]:
        with self.connection() as index:
            return index.search(query, k, doc_ids=doc_ids)

    def count(self) -> int:
        with self.connection() as index:
            return index.count()

    def chunk_ids(self) -> set[str]:
        with self.connection() as index:
            return index.chunk_ids()

    def doc_chunk_ids(self, doc_id: str) -> set[str]:
        with self.connection() as index:
            return index.doc_chunk_ids(doc_id)

    def close(self) -> None:
        for index in self._all:
            index.close()
