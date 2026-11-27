"""Gemini embeddings (REST) behind the asymmetric Embedder API: ``encode_documents`` uses task type RETRIEVAL_DOCUMENT and
``encode_query`` RETRIEVAL_QUERY. Import-safe: no network, no key read and no file opened at import or construction.

* The API key is read from the environment variable ``GEMINI_API_KEY`` at call time, sent only in the ``x-goog-api-key`` header,
  and never stored on the object, logged, printed or put in an exception message (messages are scrubbed of it).
* Every embedding is cached on disk (SQLite) keyed by (model, version, task, dims, sha256(text)), so a repeated or resumed run
  never pays twice. Vectors are L2-normalised before they are cached (the API only normalises the full 3072-d output).
* HTTP 429/5xx are retried with backoff (honouring the server's retry delay). A per-day quota, or too many consecutive
  retries, raises ``QuotaExhausted``: ingestion catches it, keeps what is cached and exits with a "resume later" message.
* The HTTP transport is injectable, so tests never touch the network.

Only public text should be sent through this module (public Qasper papers and dev questions).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
KEY_ENV = "GEMINI_API_KEY"
TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"
MAX_BATCH = 100  # documented limit of batchEmbedContents


class GeminiError(RuntimeError):
    """Any failure talking to the API (never contains the API key)."""


class QuotaExhausted(GeminiError):
    """Quota used up (per-day, or repeated 429s): keep the cache and resume later."""


def default_cache_path() -> Path:
    root = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
    return Path(root) / "advrag" / "embcache" / "gemini.sqlite3"


def urllib_transport(method: str, url: str, headers: dict, body: dict | None, timeout: float):
    """(status, json, response headers). Network errors become GeminiError without the URL or headers."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers={**headers, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "{}"), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8") or "{}")
        except Exception:
            payload = {}
        return exc.code, payload, dict(exc.headers or {})
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise GeminiError(f"network error: {type(exc).__name__}") from None


class EmbeddingCache:
    """SQLite cache of embeddings. Outside OneDrive (like every store of this project)."""

    def __init__(self, path: str | Path):
        path = Path(path)
        if any("onedrive" in part.lower() for part in path.resolve().parts):
            raise RuntimeError(f"embedding cache path must be outside OneDrive: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(str(path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("CREATE TABLE IF NOT EXISTS emb(key TEXT PRIMARY KEY, dims INTEGER NOT NULL, vec BLOB NOT NULL)")
        self._conn.commit()

    def get_many(self, keys: Sequence[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for i in range(0, len(keys), 500):
            part = keys[i:i + 500]
            rows = self._conn.execute(f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(part))})", part)
            out.update({k: np.frombuffer(v, dtype=np.float32).copy() for k, v in rows})
        return out

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        with self._conn:
            self._conn.executemany("INSERT OR REPLACE INTO emb(key, dims, vec) VALUES (?, ?, ?)",
                                   [(k, len(v), np.asarray(v, dtype=np.float32).tobytes()) for k, v in items.items()])

    def count(self) -> int:
        return self._conn.execute("SELECT count(*) FROM emb").fetchone()[0]

    def close(self) -> None:
        self._conn.close()


def _percentile(xs: list[float], q: float) -> float | None:
    return None if not xs else float(np.percentile(np.asarray(xs, dtype=float), q))


class GeminiEmbedder:
    def __init__(self, model: str = "gemini-embedding-001", version: str = "001", dims: int = 768, *, cache: EmbeddingCache | None = None,
                 cache_path: str | Path | None = None, transport: Callable = urllib_transport, batch_size: int = MAX_BATCH,
                 min_interval_s: float = 0.0, max_retries: int = 6, timeout_s: float = 60.0, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.perf_counter):
        if not 1 <= batch_size <= MAX_BATCH:
            raise ValueError(f"batch_size must be in 1..{MAX_BATCH}")
        self.model, self.version, self.dims = model, str(version), int(dims)
        self.name = f"gemini:{model}"
        self.revision = f"{self.version}/dims{self.dims}"
        self.max_seq_length = 2048  # input token limit of the model; chunks are cut for MiniLM (256) long before this
        self.token_counter = "n/a"
        self._cache, self._cache_path, self._owns_cache = cache, cache_path, cache is None
        self._transport, self._batch, self._min_interval = transport, batch_size, min_interval_s
        self._max_retries, self._timeout, self._sleep, self._clock = max_retries, timeout_s, sleep, clock
        self._last_call = None
        self.reset_stats()

    # ---- statistics -------------------------------------------------------------------------------------------
    def reset_stats(self) -> None:
        self._s = {"texts_requested": 0, "cache_hits": 0, "cache_misses": 0, "network_calls": 0, "http_429": 0, "retries": 0,
                   "http_errors": 0, "texts_sent": 0, "tokens_reported": 0, "network_ms": []}

    def stats(self) -> dict:
        s = dict(self._s)
        ms = s.pop("network_ms")
        s["network_latency_ms"] = {"n": len(ms), "p50": _percentile(ms, 50), "p95": _percentile(ms, 95), "max": max(ms) if ms else None}
        s["model"], s["version"], s["dims"] = self.model, self.version, self.dims
        return s

    # ---- cache ------------------------------------------------------------------------------------------------
    @property
    def cache(self) -> EmbeddingCache:
        if self._cache is None:
            self._cache = EmbeddingCache(self._cache_path or default_cache_path())
        return self._cache

    def _key(self, task: str, text: str) -> str:
        return hashlib.sha256(f"{self.model}|{self.version}|{task}|{self.dims}|{hashlib.sha256(text.encode('utf-8')).hexdigest()}".encode()).hexdigest()

    # ---- HTTP -------------------------------------------------------------------------------------------------
    def _api_key(self) -> str:
        key = os.environ.get(KEY_ENV)
        if not key:
            raise GeminiError(f"{KEY_ENV} is not set in the environment")
        return key

    def _scrub(self, text: str) -> str:
        key = os.environ.get(KEY_ENV)
        return text.replace(key, "[redacted]") if key else text

    def _call(self, method: str, path: str, body: dict | None):
        """One attempt, with throttling BEFORE the timed window. Returns (status, json, ms)."""
        if self._min_interval and self._last_call is not None:
            wait = self._min_interval - (self._clock() - self._last_call)
            if wait > 0:
                self._sleep(wait)
        headers = {"x-goog-api-key": self._api_key()}
        t = self._clock()
        status, data, _hdr = self._transport(method, f"{API_ROOT}/{path}", headers, body, self._timeout)
        ms = (self._clock() - t) * 1000
        self._last_call = self._clock()
        self._s["network_calls"] += 1
        self._s["network_ms"].append(ms)
        return status, data, ms

    @staticmethod
    def _retry_after_s(data: dict) -> float | None:
        for d in (data.get("error") or {}).get("details", []) or []:
            m = re.fullmatch(r"([0-9.]+)s", str(d.get("retryDelay", "")))
            if m:
                return float(m.group(1))
        return None

    @staticmethod
    def _per_day(data: dict) -> bool:
        return "perday" in json.dumps((data.get("error") or {}).get("details", [])).lower().replace("_", "").replace("-", "")

    def _request(self, method: str, path: str, body: dict | None) -> dict:
        attempt = 0
        while True:
            status, data, _ = self._call(method, path, body)
            if status == 200:
                return data
            msg = self._scrub(str((data.get("error") or {}).get("message", "")))[:300]
            if status == 429:
                self._s["http_429"] += 1
                if self._per_day(data):
                    raise QuotaExhausted(f"daily quota exhausted (HTTP 429): {msg}")
            elif status < 500:
                self._s["http_errors"] += 1
                raise GeminiError(f"HTTP {status}: {msg}")
            attempt += 1
            if attempt > self._max_retries:
                raise QuotaExhausted(f"gave up after {self._max_retries} retries (last HTTP {status}): {msg}") if status == 429 \
                    else GeminiError(f"gave up after {self._max_retries} retries (last HTTP {status}): {msg}")
            self._s["retries"] += 1
            self._sleep(max(self._retry_after_s(data) or 0.0, min(60.0, 2.0 ** attempt)))

    # ---- embedding --------------------------------------------------------------------------------------------
    def _embed(self, texts: Sequence[str], task: str) -> np.ndarray:
        texts = list(texts)
        self._s["texts_requested"] += len(texts)
        if not texts:
            return np.zeros((0, self.dims), dtype=np.float32)
        keys = [self._key(task, t) for t in texts]
        have = self.cache.get_many(list(dict.fromkeys(keys)))
        n_missing = sum(k not in have for k in keys)
        self._s["cache_misses"] += n_missing
        self._s["cache_hits"] += len(texts) - n_missing
        todo, seen = [], set()  # first occurrence of every text that is not cached yet
        for i, k in enumerate(keys):
            if k not in have and k not in seen:
                seen.add(k)
                todo.append(i)
        for start in range(0, len(todo), self._batch):
            part = todo[start:start + self._batch]
            body = {"requests": [{"model": f"models/{self.model}", "content": {"parts": [{"text": texts[i]}]}, "taskType": task,
                                  "outputDimensionality": self.dims} for i in part]}
            data = self._request("POST", f"models/{self.model}:batchEmbedContents", body)
            embs = data.get("embeddings") or []
            if len(embs) != len(part) or any(len(e.get("values", [])) != self.dims for e in embs):
                raise GeminiError(f"unexpected response: {len(embs)} embeddings for {len(part)} texts (expected {self.dims} dims)")
            self._s["texts_sent"] += len(part)
            self._s["tokens_reported"] += int((data.get("usageMetadata") or {}).get("totalTokenCount", 0) or 0)
            vecs = np.asarray([e["values"] for e in embs], dtype=np.float32)
            vecs = vecs / np.where((n := np.linalg.norm(vecs, axis=1, keepdims=True)) == 0, 1.0, n)
            new = {keys[i]: v for i, v in zip(part, vecs)}
            self.cache.put_many(new)  # persisted per batch: a later quota error loses nothing already paid for
            have.update(new)
        return np.stack([have[k] for k in keys]).astype(np.float32)

    def encode_documents(self, texts) -> np.ndarray:
        return self._embed(texts, TASK_DOCUMENT)

    def encode_query(self, text: str) -> np.ndarray:
        return self._embed([text], TASK_QUERY)[0]

    def encode(self, texts, show_progress_bar: bool = False, **_kwargs):
        """Symmetric fallback for code that does not know the asymmetric API: a string is a query, a list are documents."""
        return self.encode_query(texts) if isinstance(texts, str) else self.encode_documents(texts)

    def count_tokens(self, text: str) -> int:
        raise NotImplementedError("use the remote countTokens only in the dry run (GeminiEmbedder.remote_token_count)")

    # ---- discovery (dry run only) -----------------------------------------------------------------------------
    def list_embedding_models(self) -> list[dict]:
        """Models that support embedContent for this key (id, version, display name, token limit, methods)."""
        out, token = [], ""
        while True:
            data = self._request("GET", "models?pageSize=200" + (f"&pageToken={token}" if token else ""), None)
            for m in data.get("models", []):
                methods = m.get("supportedGenerationMethods", [])
                if "embedContent" in methods or "batchEmbedContents" in methods:
                    out.append({"id": m["name"].split("/", 1)[-1], "version": m.get("version"), "displayName": m.get("displayName"),
                                "inputTokenLimit": m.get("inputTokenLimit"), "methods": methods})
            token = data.get("nextPageToken", "")
            if not token:
                return out

    def remote_token_count(self, texts: Sequence[str]) -> int:
        """Tokens of ``texts`` as counted by the API (one countTokens call; dry run only)."""
        body = {"contents": [{"parts": [{"text": t}]} for t in texts]}
        return int(self._request("POST", f"models/{self.model}:countTokens", body).get("totalTokens", 0))

    def describe(self) -> dict:
        return {"class": type(self).__name__, "model": self.model, "version": self.version, "dims": self.dims,
                "task_types": {"documents": TASK_DOCUMENT, "queries": TASK_QUERY}, "normalised": "L2, client-side",
                "cache_path": str(getattr(self._cache, "path", self._cache_path or default_cache_path()))}

    @classmethod
    def from_config(cls, cfg, **kwargs) -> GeminiEmbedder:
        g = cfg.embedding_gemini
        return cls(g.model, g.version, g.dims, **kwargs)
