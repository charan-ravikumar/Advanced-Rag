"""Chunk contract for ingestion: one immutable, self-describing chunk record.

The chunk text is stored as the vector-store document; everything else is
stored as flat metadata (see ``Chunk.to_metadata``).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_chunk_id(doc_id: str, strategy: str, chunk_idx: int, text_hash: str) -> str:
    return hashlib.sha1(f"{doc_id}|{strategy}|{chunk_idx}|{text_hash}".encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str  # always the Qasper paper_id, never the title
    section: str
    chunk_idx: int  # order within the whole document
    char_start: int  # offsets into the section text; -1 when not located
    char_end: int
    strategy: str
    text: str
    text_hash: str  # sha256 of text
    doc_version: str  # sha256 of the full source document text

    @classmethod
    def create(
        cls,
        *,
        doc_id: str,
        section: str,
        chunk_idx: int,
        char_start: int,
        char_end: int,
        strategy: str,
        text: str,
        doc_version: str,
    ) -> "Chunk":
        text_hash = sha256_text(text)
        return cls(
            chunk_id=make_chunk_id(doc_id, strategy, chunk_idx, text_hash),
            doc_id=doc_id,
            section=section,
            chunk_idx=chunk_idx,
            char_start=char_start,
            char_end=char_end,
            strategy=strategy,
            text=text,
            text_hash=text_hash,
            doc_version=doc_version,
        )

    def to_metadata(self) -> dict[str, Any]:
        """Flat str/int/float/bool dict without None values; text is excluded."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "section": self.section,
            "chunk_idx": self.chunk_idx,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "strategy": self.strategy,
            "text_hash": self.text_hash,
            "doc_version": self.doc_version,
        }

    @classmethod
    def from_metadata_and_text(cls, metadata: dict[str, Any], text: str) -> "Chunk":
        return cls(
            chunk_id=str(metadata["chunk_id"]),
            doc_id=str(metadata["doc_id"]),
            section=str(metadata["section"]),
            chunk_idx=int(metadata["chunk_idx"]),
            char_start=int(metadata["char_start"]),
            char_end=int(metadata["char_end"]),
            strategy=str(metadata["strategy"]),
            text=text,
            text_hash=str(metadata["text_hash"]),
            doc_version=str(metadata["doc_version"]),
        )
