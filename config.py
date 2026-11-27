"""
config.py — Central config loader.

Reads config.yaml from the project root once at import time.
All pipeline components import from here; no hardcoded values elsewhere.

Usage:
    from config import cfg

    top_k     = cfg.pipeline.top_k
    emb_model = cfg.embedding.model
"""

from pathlib import Path
from types import SimpleNamespace

import yaml

_ROOT = Path(__file__).resolve().parent
_CONFIG_PATH = _ROOT / "config.yaml"

# The keys the code reads, per section. An unknown key (a typo, or a knob nothing reads) is an error, not a silent default.
# ``generation``, ``prompts``, ``tracing`` and ``app`` are owned by dataclasses (rag_core/config.py, app/guards.py): their keys are checked by
# ``from_config`` (``rag_core.config._overlay``), which every entry point calls.
_ALLOWED_KEYS: dict[str, set[str] | None] = {
    "pipeline": {"retrieval", "fusion", "reranker_enabled", "reranker_model", "reranker_revision", "top_k", "candidate_k", "rerank_k", "rrf_k",
                 "vector_weight", "bm25_weight"},
    "embedding": {"model", "revision", "max_seq_length"},
    "embedding_gemini": {"model", "version", "dims", "documents_task", "queries_task", "chroma_dir", "min_interval_s"},  # dormant Stage 2b section
    "store": {"chroma_dir", "collection_prefix", "upsert_batch_size"},
    "chunking": {"max_chunk_chars", "max_tokens", "recursive_chunk_size", "recursive_chunk_overlap", "semantic_similarity_threshold", "min_chunk_chars",
                 "semantic_window_size"},  # feeds the ingest signature: the values are never reshaped
    "generation": None,
    "prompts": None,
    "tracing": None,
    "app": None,
}


def _to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert a dict to SimpleNamespace for dot-access."""
    ns = SimpleNamespace()
    for key, value in d.items():
        setattr(ns, key, _to_namespace(value) if isinstance(value, dict) else value)
    return ns


def _validate(raw: dict, source: Path) -> None:
    if not isinstance(raw, dict):
        raise ValueError(f"{source.name}: the file must be a mapping of sections")
    unknown = sorted(set(raw) - set(_ALLOWED_KEYS))
    if unknown:
        raise ValueError(f"{source.name}: unknown section(s) {unknown}; known sections are {sorted(_ALLOWED_KEYS)}")
    for section, body in raw.items():
        if not isinstance(body, dict):
            raise ValueError(f"{source.name}: section {section!r} must be a mapping, got {type(body).__name__}")
        allowed = _ALLOWED_KEYS[section]
        if allowed is not None and set(body) - allowed:
            raise ValueError(f"{source.name}: unknown key(s) {sorted(set(body) - allowed)} in section {section!r}; allowed keys are {sorted(allowed)}")


def load_config(path: Path | None = None) -> SimpleNamespace:
    """Read and validate a config file (default: config.yaml at the project root)."""
    path = Path(path) if path is not None else _CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {path}. "
            "Copy config.yaml to the project root before running."
        )
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    _validate(raw, path)
    return _to_namespace(raw)


# Singleton — imported once, shared everywhere
cfg: SimpleNamespace = load_config()
