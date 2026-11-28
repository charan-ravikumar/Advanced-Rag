"""Bootstrap of the Streamlit app and of the API server. One of the two modules (with the entry point) that may touch
the environment: ``prepare_environment`` calls ``rag_core.env.load_env`` (an allowlist, never overriding) and copies an allowlist of ``st.secrets`` into ``os.environ``.
Importing this module does nothing: no file is read, no client or model is created, ``load_env`` is not called, and Streamlit is imported only inside the functions that need it.

Two compositions share the ``Resources`` / ``SelfCheck`` types:

* ``build_resources`` (SERVER side, called by ``api.main`` at startup): opens the index, loads the models, builds the gateway, tracer and feedback store.
* ``build_client_resources`` (Streamlit side): opens NOTHING; it talks to the API (``ADVRAG_API_URL``, default http://127.0.0.1:8000; ``ADVRAG_API_TOKEN`` if the server has one)
  through the adapters in ``app.remote`` and mirrors the API's self-check.

Both NEVER raise: they return a ``Resources`` whose ``selfcheck`` lists every check (``Check(name, ok, detail, fatal)``). A failed FATAL check makes the UI
show an error screen with the failed checks, never a stack trace. Missing Gemini or Langfuse keys are warnings, not failures: the app still shows passages (degraded mode)
and the Observability tab has a ``no_keys`` state. Keys are only ever reported as booleans.

Index directory (server): ``ADVRAG_INDEX_DIR``, else ``CHROMA_LOCAL_PATH``, else a failed check (there is NO default: nothing opens a store you did not name).
The retriever is shared by all requests and there is NO global lock: the lexical leg uses a pool of SQLite connections, embedding and
reranking run in bounded worker pools (``rag.concurrency``; sizes in ``config.yaml`` ``app.*``), and the API refuses work beyond its queue limit.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.artifacts import Artifacts, load_artifacts
from app.guards import AccessGate, AppConfig, DailyBudget
from rag_core.env import load_env

ROOT = Path(__file__).resolve().parents[1]
SECRET_NAMES = ("GROQ_API_KEY", "GEMINI_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL", "APP_ACCESS_CODE", "ADVRAG_API_URL", "ADVRAG_API_TOKEN")
OVERRIDE_KEY = "__resources__"  # tests put a ready Resources here (dependency injection); in production nothing ever sets it
DEFAULT_API_URL = "http://127.0.0.1:8000"


# ---- environment -----------------------------------------------------------------------------------------------------------------

def copy_secrets_to_env(secrets: Mapping | None, env: dict | None = None, names=SECRET_NAMES) -> list[str]:
    """Copy the allowlisted ``names`` from ``secrets`` into ``env`` (default ``os.environ``) WITHOUT overriding a variable that is already set. Returns the NAMES copied.
    Any failure to read a secret (no secrets file: Streamlit raises on every access) is skipped. Values are never returned, printed or logged."""
    env = os.environ if env is None else env
    copied = []
    for name in names:
        if env.get(name):
            continue
        try:
            value = secrets[name] if secrets is not None and name in secrets else None
        except Exception:  # noqa: BLE001: StreamlitSecretNotFoundError, a malformed file, ...
            value = None
        if isinstance(value, str) and value.strip():
            env[name] = value.strip()
            copied.append(name)
    return copied


def prepare_environment() -> list[str]:
    """``.env`` (allowlist) and ``st.secrets`` (allowlist) into ``os.environ``. Idempotent; called by the entry point on every run."""
    import streamlit as st  # lazy: the API server imports this module for ``build_resources`` and must not need Streamlit

    load_env()
    try:
        secrets = st.secrets
    except Exception:  # noqa: BLE001
        secrets = None
    return copy_secrets_to_env(secrets)


def make_gate(settings: AppConfig, env: Mapping | None = None) -> AccessGate:
    env = os.environ if env is None else env
    return AccessGate(env.get("APP_ACCESS_CODE"), max_attempts=settings.access_code_attempts, lockout_s=settings.access_code_lockout_seconds)


# ---- the self-check ---------------------------------------------------------------------------------------------------------------

@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fatal: bool = True  # a failed fatal check means the app cannot serve answers


@dataclass
class SelfCheck:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str, fatal: bool = True) -> bool:
        self.checks.append(Check(name, bool(ok), detail, fatal))
        return bool(ok)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.fatal and not c.ok]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if not c.fatal and not c.ok]

    @property
    def ok(self) -> bool:
        return not self.failures


def _short(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc)[:200]}"


def fts5_available() -> bool:
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        conn.close()
        return True
    except sqlite3.Error:
        return False


def model_cached(name: str, revision: str) -> bool:
    """Is ``name`` at ``revision`` in the local HuggingFace cache (no download, no load)?"""
    try:
        from huggingface_hub import try_to_load_from_cache

        found = try_to_load_from_cache(name, "config.json", revision=revision)
        return isinstance(found, str)
    except Exception:  # noqa: BLE001
        return False


def _same_path(a: str, b: str) -> bool:
    norm = lambda p: os.path.normcase(os.path.normpath(os.path.expandvars(str(p))))  # noqa: E731
    return norm(a) == norm(b)


def find_manifest(manifest_dir: Path, strategy: str, index_dir: str | Path) -> tuple[Path, dict] | None:
    """The newest ingest manifest of ``strategy`` written for THIS index directory (``runs/ingest/<strategy>_<UTC>/manifest.json``), or None."""
    for path in sorted(Path(manifest_dir).glob(f"{strategy}_*/manifest.json"), key=lambda p: p.parent.name, reverse=True):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if _same_path(manifest["store"]["location"], str(index_dir)):
                return path, manifest
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


# ---- resources --------------------------------------------------------------------------------------------------------------------

@dataclass
class Factories:
    """The only places that touch the index, the models and the SDKs; tests replace them with fakes."""

    open_store: Callable[[Any, str, str], Any] = lambda cfg, strategy, path: __import__("db.store", fromlist=["build_store"]).build_store(cfg, strategy, path=path)
    open_lexical: Callable[[Any, str, str], Any] = lambda cfg, strategy, path: _open_lexical(cfg, strategy, path)
    make_embedder: Callable[[Any], Any] = lambda cfg: __import__("rag.embedders", fromlist=["SentenceTransformerEmbedder"]).SentenceTransformerEmbedder.from_config(cfg)
    make_reranker: Callable[[Any], Any] = lambda cfg: __import__("rag.rerankers", fromlist=["CrossEncoderReranker"]).CrossEncoderReranker.from_config(cfg)
    make_retriever: Callable[..., Any] = lambda store, embedder, lexical, reranker, cfg, executors: __import__("rag.retriever", fromlist=["Retriever"]).Retriever(
        store, embedder, lexical, reranker=reranker, cfg=cfg, executors=executors)
    make_gateway: Callable[[Any], Any] = lambda rag_cfg: __import__("rag_core.gateway", fromlist=["build_gateway"]).build_gateway(rag_cfg)
    set_torch_threads: Callable[[int], None] = lambda threads: __import__("rag.concurrency", fromlist=["set_torch_threads"]).set_torch_threads(threads)
    fts5_available: Callable[[], bool] = fts5_available
    model_cached: Callable[[str, str], bool] = model_cached


def _open_lexical(cfg, strategy: str, index_dir: str):
    from rag.lexical import PooledFtsIndex, lexical_index_path

    return PooledFtsIndex(lexical_index_path(cfg, strategy, chroma_path=index_dir), size=AppConfig.from_config(cfg).db_workers)


@dataclass
class Resources:
    settings: AppConfig
    selfcheck: SelfCheck
    service: Any = None
    daily: DailyBudget | None = None
    reader: Any = None
    artifacts: Artifacts = field(default_factory=Artifacts)
    record_feedback: Callable[..., Any] | None = None  # (request_id, trace_id, value, comment) -> rag_core.scores.FeedbackResult: local first, then Langfuse
    version_info: dict = field(default_factory=dict)
    gemini_key_present: bool = False
    langfuse_keys_present: bool = False
    tracing_on: bool = False
    prices_configured: bool = False
    reranker_available: bool = False
    index_dir: str | None = None
    warmup_ms: float | None = None
    executors: Any = None  # rag.concurrency.RetrievalExecutors; the API shuts them down when it stops

    @property
    def ready(self) -> bool:
        return self.selfcheck.ok and self.service is not None


def build_resources(cfg, env: Mapping | None = None, *, factories: Factories | None = None, root: Path = ROOT) -> Resources:
    """Open the index and build everything the app needs; every problem becomes a failed check. Never raises."""
    env = os.environ if env is None else env
    f = factories or Factories()
    settings = AppConfig.from_config(cfg)
    check = SelfCheck()
    res = Resources(settings=settings, selfcheck=check)
    try:
        res.artifacts = load_artifacts(settings.results_path, settings.papers_path, root)
    except Exception as exc:  # noqa: BLE001: load_artifacts is defensive; this is the last resort
        res.artifacts = Artifacts(problems=[f"artifacts could not be loaded ({type(exc).__name__})"])
    keys = {n: bool(env.get(n)) for n in ("GEMINI_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")}
    res.gemini_key_present = keys["GEMINI_API_KEY"]
    res.langfuse_keys_present = keys["LANGFUSE_PUBLIC_KEY"] and keys["LANGFUSE_SECRET_KEY"]
    res.daily = DailyBudget(settings.daily_request_cap)

    index_dir = env.get("ADVRAG_INDEX_DIR") or env.get("CHROMA_LOCAL_PATH")
    res.index_dir = index_dir
    index_ok = bool(index_dir) and Path(os.path.expandvars(str(index_dir))).is_dir()
    if not check.add("Index directory", index_ok, "set and found" if index_ok else (
            f"{index_dir} is not a folder" if index_dir else
            "not set: set ADVRAG_INDEX_DIR (or CHROMA_LOCAL_PATH) to the folder of the chroma store, for example %LOCALAPPDATA%\\advrag\\chroma_v2")):
        return _finish(res, keys, cfg)
    fts_ok = f.fts5_available()
    check.add("FTS5 (SQLite full-text search)", fts_ok, "available" if fts_ok else "this SQLite build has no FTS5: use another Python build")
    if not check.ok:
        return _finish(res, keys, cfg)

    from rag_core.config import RagConfig

    rag_cfg = RagConfig.from_config(cfg)
    try:
        store = f.open_store(cfg, settings.strategy, index_dir)
        lexical = f.open_lexical(cfg, settings.strategy, index_dir)
        n_store, n_lex = store.count(), lexical.count()
    except Exception as exc:  # noqa: BLE001
        check.add("Vector store and lexical index open", False, _short(exc))
        return _finish(res, keys, cfg)
    check.add("Vector store and lexical index open", True, f"{n_store} chunks in the vector store, {n_lex} in the lexical index")
    check.add("Chunk count equals the lexical index", n_store == n_lex and n_store > 0, "equal" if n_store == n_lex and n_store > 0 else f"vector store {n_store} vs lexical {n_lex}")
    found = find_manifest(Path(settings.manifest_dir) if Path(settings.manifest_dir).is_absolute() else root / settings.manifest_dir, settings.strategy, index_dir)
    manifest_count = None if found is None else (found[1].get("verification") or {}).get("store_count")
    if found is None:
        check.add("Chunk count equals the ingest manifest", False, f"no ingest manifest for this index under {settings.manifest_dir}")
    else:
        check.add("Chunk count equals the ingest manifest", manifest_count == n_store,
                  f"{n_store} = {manifest_count} ({found[0].parent.name})" if manifest_count == n_store else f"the store has {n_store} chunks but the manifest says {manifest_count}")
    emb_name, emb_rev = cfg.embedding.model, cfg.embedding.revision
    cached = f.model_cached(emb_name, emb_rev)
    check.add("Embedding model in the local cache", cached, f"{emb_name} @ {emb_rev[:10]}" if cached else f"{emb_name} @ {emb_rev[:10]} is not cached locally (the app never downloads models)")
    if not check.ok:
        return _finish(res, keys, cfg)

    embedder = f.make_embedder(cfg)
    reranker = None
    res.reranker_available = f.model_cached(cfg.pipeline.reranker_model, cfg.pipeline.reranker_revision)
    if res.reranker_available:
        reranker = f.make_reranker(cfg)  # loaded lazily, at the first use
    from rag.concurrency import RetrievalExecutors

    f.set_torch_threads(settings.torch_threads)
    res.executors = RetrievalExecutors(embed_workers=settings.embed_workers, db_workers=settings.db_workers, rerank_workers=settings.rerank_workers)
    retriever = f.make_retriever(store, embedder, lexical, reranker, cfg, res.executors)
    t0 = time.perf_counter()
    try:  # warm the embedder with one dummy query: the first real question must not pay the model load
        retriever.retrieve(settings.warmup_query, mode=rag_cfg.retrieval_mode, fusion=rag_cfg.fusion, top_k=1, candidate_k=rag_cfg.candidate_k, rrf_k=rag_cfg.rrf_k,
                           vector_weight=rag_cfg.vector_weight, lexical_weight=rag_cfg.lexical_weight)
    except Exception as exc:  # noqa: BLE001
        check.add("Embedder warm-up (one dummy query)", False, _short(exc))
        return _finish(res, keys, cfg)
    res.warmup_ms = (time.perf_counter() - t0) * 1000.0
    check.add("Embedder warm-up (one dummy query)", True, f"{res.warmup_ms:.0f} ms")

    from rag_core.cache import InMemoryTTLCache
    from rag_core.config import PromptsConfig, TracingConfig
    from rag_core.prompt_store import build_prompt_store
    from rag_core.prompts import PROMPT_VERSION
    from rag_core.scores import FeedbackStore, record_feedback
    from rag_core.service import RagService, RefreshingIndexVersion, index_version_for
    from rag_core.tracing import build_tracer

    manifest_path = None if found is None else found[0]
    manifest_root = Path(settings.manifest_dir) if Path(settings.manifest_dir).is_absolute() else root / settings.manifest_dir

    def current_index_version() -> str:  # a re-ingest adds a new manifest and changes the count: follow it without a restart
        newest = find_manifest(manifest_root, settings.strategy, index_dir)
        return index_version_for(store, manifest_path if newest is None else newest[0])

    tracing = TracingConfig.from_config(cfg)
    tracer = build_tracer(tracing, env=dict(env))
    index_provider = RefreshingIndexVersion(current_index_version, ttl_s=settings.index_version_ttl_seconds)
    index_version = index_provider()
    res.service = RagService(retriever, f.make_gateway(rag_cfg), InMemoryTTLCache(max_entries=rag_cfg.cache_max_entries, ttl_s=rag_cfg.cache_ttl_s), rag_cfg,
                             index_version=index_provider, prompts=build_prompt_store(PromptsConfig.from_config(cfg)), tracer=tracer, tracing=tracing)
    res.tracing_on = bool(getattr(tracer, "enabled", False))
    res.prices_configured = bool(rag_cfg.prices)
    try:  # the thumbs are saved here first; a path that cannot be used (for example inside OneDrive) only turns the local copy off
        from db.store import resolve_store_path

        feedback_store = FeedbackStore(resolve_store_path(settings.feedback_path))
    except Exception:  # noqa: BLE001
        feedback_store = None
    res.record_feedback = lambda request_id, trace_id, value, comment=None: record_feedback(feedback_store, request_id, trace_id, value, comment)
    res.version_info = {
        "index_dir": str(index_dir), "chunk_count": n_store, "index_version": index_version, "ingest_signature": (store.get_signature() or "")[:16] or None,
        "manifest": None if found is None else found[0].parent.name,
        "manifest_sha256": None if found is None else hashlib.sha256(found[0].read_bytes()).hexdigest(),
        "embedder": {"name": emb_name, "revision": emb_rev}, "reranker": {"name": cfg.pipeline.reranker_model, "revision": cfg.pipeline.reranker_revision,
                                                                       "cached": res.reranker_available},
        "prompt_version": PROMPT_VERSION, "prompt_mode": PromptsConfig.from_config(cfg).mode, "generator": {"model": rag_cfg.gemini_model, "reasoning_effort": rag_cfg.reasoning_effort},
        "retrieval": {"mode": rag_cfg.retrieval_mode, "fusion": rag_cfg.fusion, "rrf_k": rag_cfg.rrf_k, "vector_weight": rag_cfg.vector_weight,
                      "lexical_weight": rag_cfg.lexical_weight, "top_k": rag_cfg.top_k}}
    return _finish(res, keys, cfg)


def _finish(res: Resources, keys: dict, cfg) -> Resources:
    """The informational (non-fatal) checks and the observability reader, added whether or not the build got this far."""
    res.selfcheck.add("Gemini API key present", res.gemini_key_present, "present (value never shown)" if res.gemini_key_present
                      else "missing: answers will be unavailable (the app shows the retrieved passages instead)", fatal=False)
    res.selfcheck.add("Langfuse keys present", res.langfuse_keys_present, "present (value never shown)" if res.langfuse_keys_present
                      else "missing: no tracing; the Observability tab shows sample data only", fatal=False)
    from rag_core.observability_reader import ObservabilityReader

    res.reader = ObservabilityReader(ttl_s=float(res.settings.metrics_ttl_seconds))
    return res


# ---- the Streamlit entry points --------------------------------------------------------------------------------------------------------

def get_settings() -> AppConfig:
    from config import cfg

    return AppConfig.from_config(cfg)


def build_client_resources(cfg, env: Mapping | None = None, *, client: Any = None, root: Path = ROOT) -> Resources:
    """The resources of the Streamlit app as a CLIENT of the API: nothing is opened or loaded here (no index, no model, no key). Never raises.

    The self-check is the API's own (``GET /v1/ready``, every check mirrored with its fatal flag) plus one check that the API can be reached. The service, the daily budget, the
    metrics reader and the feedback recorder are adapters over the API (``app.remote``). The papers and evaluation artifacts are public files read locally, as before."""
    from api.client import ApiClient, ApiError, ApiUnavailable
    from app.remote import RemoteBudget, RemoteReader, RemoteService, make_record_feedback

    env = os.environ if env is None else env
    settings = AppConfig.from_config(cfg)
    check = SelfCheck()
    res = Resources(settings=settings, selfcheck=check)
    try:
        res.artifacts = load_artifacts(settings.results_path, settings.papers_path, root)
    except Exception as exc:  # noqa: BLE001
        res.artifacts = Artifacts(problems=[f"artifacts could not be loaded ({type(exc).__name__})"])
    url = (env.get("ADVRAG_API_URL") or DEFAULT_API_URL).strip()
    api = client or ApiClient(url, token=env.get("ADVRAG_API_TOKEN"))
    res.reader = RemoteReader(api)
    try:
        ready = api.ready()
    except ApiUnavailable as exc:
        check.add("Answer API reachable", False, f"{url} did not answer ({exc.reason}). Start it with: python -m uvicorn api.main:app --port 8000")
        return res
    except ApiError as exc:
        check.add("Answer API reachable", False, f"{url} refused the request (HTTP {exc.status}, {exc.code}): {exc.message}")
        return res
    check.add("Answer API reachable", True, url)
    for c in ready.checks:
        check.add(c.name, c.ok, c.detail, c.fatal)
    if not ready.ready:
        return res
    try:
        meta = api.meta()
    except (ApiError, ApiUnavailable) as exc:
        check.add("Answer API metadata", False, f"GET /v1/meta failed ({type(exc).__name__})")
        return res
    res.service = RemoteService(api, meta.limits)
    res.daily = RemoteBudget(api)
    res.record_feedback = make_record_feedback(api)
    res.version_info = dict(meta.version_info)
    res.gemini_key_present, res.langfuse_keys_present = meta.flags.gemini_key_present, meta.flags.langfuse_keys_present
    res.tracing_on, res.prices_configured, res.reranker_available = meta.flags.tracing_on, meta.flags.prices_configured, meta.flags.reranker_available
    res.warmup_ms = meta.warmup_ms
    return res


def get_resources() -> Resources:
    """The process-wide resources (built once, shared by all sessions). The entry point has already called ``prepare_environment``.
    Tests inject a ready ``Resources`` through ``st.session_state[OVERRIDE_KEY]``; in production nothing ever sets that key."""
    import streamlit as st

    override = st.session_state.get(OVERRIDE_KEY)
    if override is not None:
        return override
    return _cache_builder(st)()


_BUILDER: list = []


def _cache_builder(st):
    """The ``st.cache_resource`` function, created on first use (not at import, so this module needs no Streamlit to be imported). ``validate`` drops a failed build, so
    'Check again' tries again (the API may have started meanwhile)."""
    if not _BUILDER:
        @st.cache_resource(show_spinner="Connecting to the answer API…", validate=lambda resources: resources.selfcheck.ok)
        def build() -> Resources:
            from config import cfg

            return build_client_resources(cfg)

        _BUILDER.append(build)
    return _BUILDER[0]
