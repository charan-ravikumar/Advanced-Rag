"""Answer cache: a protocol, an in-memory size-bounded LRU with TTL, and the key function.

Rules this module enforces (tests in tests/unit/test_rag_core_cache.py):
* the key carries EVERYTHING that can change the answer: the normalised question, the document filter, a hash of the retrieval
  and context settings, the index version (chunk count + manifest/ingest signature), the embedder revision, the prompt version,
  the generator model id, top_k and the answer mode. Change any one and the lookup misses. (The old cache keyed on the
  lower-cased query alone, so a re-ingested corpus or a new prompt kept serving stale answers.)
* only a successful, validated, complete, non-degraded answer is stored (``is_cacheable``): never an error, a degraded notice, a
  partial stream, an abstention, an answer with an invalid citation, a truncated one (finish_reason other than "stop"), or one
  produced by a fallback model rather than the model named in the key.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import unicodedata
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Protocol

from rag_core.schemas import AnswerResult


class CacheBackend(Protocol):
    def get(self, key: str) -> AnswerResult | None: ...

    def set(self, key: str, value: AnswerResult) -> None: ...

    def clear(self) -> None: ...

    def __len__(self) -> int: ...


def normalise_question(question: str) -> str:
    """NFKC, lower case, whitespace collapsed: 'What  IS attention?' and 'what is attention?' share an entry."""
    return " ".join(unicodedata.normalize("NFKC", question).lower().split())


def config_hash(settings: dict) -> str:
    return hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def make_cache_key(*, question: str, doc_ids: Sequence[str] | None, retrieval_config_hash: str, index_version: str,
                   embedder_revision: str, prompt: str, model_id: str, top_k: int, mode: str = "long", reasoning_effort: str | None = None) -> str:
    """``prompt`` is ``ResolvedPrompt.cache_id``: name, version AND the sha256 of the template text, so a prompt edited in place (same version
    number, new text) or a local template change both change the key."""
    payload = {
        "q": normalise_question(question),
        "docs": None if doc_ids is None else sorted(set(doc_ids)),
        "retrieval": retrieval_config_hash,
        "index": index_version,
        "embedder": embedder_revision,
        "prompt": prompt,
        "model": model_id,
        "effort": reasoning_effort,
        "top_k": top_k,
        "mode": mode,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def is_cacheable(result: AnswerResult, expected_model: str | None = None) -> bool:
    return bool(
        result.answer.strip()
        and not result.errors and not result.degraded and not result.partial and not result.abstained
        and not result.invalid_citations
        and result.finish_reason == "stop"
        and result.used_sources
        and result.mode in ("long", "short")
        and (result.mode == "short" or bool(result.citations))  # a long answer must cite at least one provided source
        and (expected_model is None or result.model == expected_model)
    )


class InMemoryTTLCache:
    """Size-bounded LRU with a time-to-live per entry. Thread-safe. Values are copied in and out so callers cannot mutate them."""

    def __init__(self, max_entries: int = 256, ttl_s: float = 3600.0, clock: Callable[[], float] = time.monotonic):
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.max_entries, self.ttl_s, self._clock = max_entries, ttl_s, clock
        self._data: OrderedDict[str, tuple[float, AnswerResult]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, key: str) -> AnswerResult | None:
        with self._lock:
            item = self._data.get(key)
            if item is None or self._clock() - item[0] >= self.ttl_s:
                if item is not None:
                    del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return item[1].model_copy(deep=True)

    def set(self, key: str, value: AnswerResult) -> None:
        with self._lock:
            self._data[key] = (self._clock(), value.model_copy(deep=True))
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
