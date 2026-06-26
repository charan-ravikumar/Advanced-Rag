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


def _to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert a dict to SimpleNamespace for dot-access."""
    ns = SimpleNamespace()
    for key, value in d.items():
        setattr(ns, key, _to_namespace(value) if isinstance(value, dict) else value)
    return ns


def _load() -> SimpleNamespace:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {_CONFIG_PATH}. "
            "Copy config.yaml to the project root before running."
        )
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _to_namespace(raw)


# Singleton — imported once, shared everywhere
cfg: SimpleNamespace = _load()
