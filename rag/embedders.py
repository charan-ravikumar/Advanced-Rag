"""Embedders behind one small protocol.

* ``HashEmbedder``: deterministic and network-free, for tests.
* ``SentenceTransformerEmbedder``: the real model, loaded lazily (never at import),
  from the local HuggingFace cache only, at a pinned revision.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, runtime_checkable

import numpy as np

_TOKEN = re.compile(r"\w+", re.UNICODE)


@runtime_checkable
class Embedder(Protocol):
    """What the chunkers, ingestion and retrieval need from an embedder.

    ``encode_documents`` / ``encode_query`` are the asymmetric entry points: some models (Gemini) use different task
    types for stored passages and for questions. Symmetric models (MiniLM, the hash embedder) implement both as the
    same call as ``encode``.
    """

    name: str
    revision: str
    max_seq_length: int

    def encode(self, texts, show_progress_bar: bool = False, **kwargs): ...

    def encode_documents(self, texts: list[str]): ...

    def encode_query(self, text: str): ...


def embed_documents(embedder, texts):
    """Vectors for passages to be stored. Falls back to ``encode`` for embedders without the asymmetric API."""
    fn = getattr(embedder, "encode_documents", None)
    return fn(texts) if fn is not None else embedder.encode(texts, show_progress_bar=False)


def embed_query(embedder, text: str):
    """Vector for one question. Falls back to ``encode`` for embedders without the asymmetric API."""
    fn = getattr(embedder, "encode_query", None)
    return fn(text) if fn is not None else embedder.encode(text)


class HashEmbedder:
    """Bag-of-words hashing embedder: texts sharing words get similar vectors.

    Same interface subset as SentenceTransformer: ``encode`` and ``max_seq_length``.
    """

    name = "hash-embedder"
    revision = "n/a"
    token_counter = "estimate"

    def count_tokens(self, text: str) -> int:
        from rag.tokens import estimate_token_count

        return estimate_token_count(text)

    def __init__(self, dim: int = 64, max_seq_length: int = 256):
        self.dim = dim
        self.max_seq_length = max_seq_length

    def _one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in _TOKEN.findall(text.lower()):
            h = int.from_bytes(hashlib.md5(tok.encode("utf-8")).digest()[:8], "little")
            vec[h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def encode(self, texts, show_progress_bar: bool = False, **_kwargs):
        if isinstance(texts, str):
            return self._one(texts)
        return np.stack([self._one(t) for t in texts]) if len(texts) else np.zeros((0, self.dim), dtype=np.float32)

    def encode_documents(self, texts):
        return self.encode(list(texts))

    def encode_query(self, text):
        return self.encode(text)


class SentenceTransformerEmbedder:
    """all-MiniLM-L6-v2 (or any sentence-transformers model), loaded on first use.

    ``local_files_only=True`` by default: constructing or using this class never
    reaches the network. A missing cache raises a clear error instead of downloading.
    """

    def __init__(
        self,
        model_name: str,
        revision: str,
        max_seq_length: int = 256,
        local_files_only: bool = True,
    ):
        self.name = model_name
        self.revision = revision
        self.max_seq_length = max_seq_length
        self.token_counter = "wordpiece-real"
        self._local_only = local_files_only
        self._model = None

    @classmethod
    def from_config(cls, cfg, **kwargs) -> SentenceTransformerEmbedder:
        return cls(
            cfg.embedding.model,
            cfg.embedding.revision,
            max_seq_length=cfg.embedding.max_seq_length,
            **kwargs,
        )

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            try:
                self._model = SentenceTransformer(
                    self.name, revision=self.revision, local_files_only=self._local_only
                )
            except OSError as exc:
                raise RuntimeError(
                    f"{self.name}@{self.revision} is not in the local HuggingFace cache. "
                    "Download it once, explicitly, with huggingface_hub.snapshot_download."
                ) from exc
            self._model.max_seq_length = self.max_seq_length
        return self._model

    @property
    def tokenizer(self):
        """The model's own tokenizer: the source of truth for the token cap."""
        return self.model.tokenizer

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def count_tokens(self, text: str) -> int:
        """Tokens the model will see, including [CLS] and [SEP]."""
        return len(self.tokenizer(text, add_special_tokens=True)["input_ids"])

    def encode(self, texts, show_progress_bar: bool = False, **kwargs):
        return self.model.encode(texts, show_progress_bar=show_progress_bar, **kwargs)

    def encode_documents(self, texts):
        return self.encode(list(texts), show_progress_bar=False)

    def encode_query(self, text):
        return self.encode(text)
