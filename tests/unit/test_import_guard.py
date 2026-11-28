"""Import-safety and hermetic-guard tests.

Importing the library modules must not load a model, open a network connection, read
credentials or touch a store. Subprocesses are used so that nothing is cached in
``sys.modules`` and the poisoned network cannot be bypassed by an earlier import.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.conftest import NetworkBlockedError

ROOT = Path(__file__).resolve().parents[2]

POISON = textwrap.dedent(
    """
    import socket, sys
    def _blocked(*a, **k):
        raise RuntimeError("NETWORK ACCESS AT IMPORT: %r" % (a[1:] if a else a,))
    socket.socket.connect = _blocked
    socket.socket.connect_ex = _blocked
    socket.create_connection = _blocked
    import chromadb
    chromadb.CloudClient = _blocked
    from sentence_transformers import SentenceTransformer
    SentenceTransformer.__init__ = _blocked
    """
)


def _run(code: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "ADVRAG_ALLOW_CLOUD"}
    env["PYTHONPATH"] = str(ROOT)
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-c", POISON + textwrap.dedent(code)],
        capture_output=True, text=True, cwd=ROOT, env=env, timeout=240,
    )


SAFE_MODULES = [
    "db.store",
    "rag.embedders",
    "rag.chunkers",
    "rag.chunk_adapter",
    "rag.tokens",
    "rag.schemas_ingest",
    "rag.fusion",
    "rag.router",
    "rag.lexical",
    "rag.retriever",
    "rag.concurrency",
    "rag.ingestion",
    "rag_core",
    "rag_core.schemas",
    "rag_core.errors",
    "rag_core.context",
    "rag_core.prompts",
    "rag_core.citations",
    "rag_core.gateway",
    "rag_core.cache",
    "rag_core.config",
    "rag_core.service",
    "rag_core.env",
    "rag_core.langfuse_client",
    "rag_core.prompt_store",
    "rag_core.prompt_sync",
    "rag_core.tracing",
    "rag_core.scores",
    "rag_core.observability_reader",
    "rag_core.claims",
    "rag_core.otel_tracing",
    "eval.answer_metrics",
    "eval.gen_arms",
    "eval.quota_plan",
    "eval.disk_cache",
    "eval.judge_core",
    "eval.ragas_adapter",
    "eval.run_generation_eval",
    "eval.run_ragas",
    "eval.app_results",
    "eval.heldout",
    "app.guards",
    "app.artifacts",
    "app.bootstrap",
    "app.ui_chat",
    "app.ui_observability",
    "app.ui_evaluation",
    "app.ui_about",
]

PROBE = textwrap.dedent(
    """
    import importlib, json
    out = {"imported": {}, "state": None}
    for m in %r:
        try:
            importlib.import_module(m); out["imported"][m] = "ok"
        except BaseException as exc:
            out["imported"][m] = repr(exc)
    import rag.chunkers as ch
    out["state"] = [ch._default_embedder is None]
    print("PROBE" + json.dumps(out))
    """
) % (SAFE_MODULES,)


@pytest.fixture(scope="module")
def probe():
    """Import every library module once in a poisoned subprocess (one torch import, not eleven)."""
    import json

    proc = _run(PROBE)
    assert proc.returncode == 0, proc.stderr[-800:]
    return json.loads(proc.stdout.split("PROBE", 1)[1])


@pytest.mark.slow
@pytest.mark.parametrize("module", SAFE_MODULES)
def test_module_imports_without_side_effects(probe, module):
    assert probe["imported"][module] == "ok"


@pytest.mark.slow
def test_importing_everything_leaves_no_state(probe):
    # importing every library module builds no embedder (the semantic chunker's default one is created lazily)
    assert probe["state"] == [True]


def test_embedder_construction_does_not_load_model():
    from config import cfg
    from rag.embedders import SentenceTransformerEmbedder

    emb = SentenceTransformerEmbedder.from_config(cfg)
    assert emb._model is None
    assert emb.revision == cfg.embedding.revision
    assert len(emb.revision) == 40


@pytest.mark.slow
def test_missing_model_cache_raises_clear_error_not_download():
    from rag.embedders import SentenceTransformerEmbedder

    emb = SentenceTransformerEmbedder("sentence-transformers/does-not-exist-xyz", "0" * 40)
    with pytest.raises(RuntimeError, match="not in the local HuggingFace cache"):
        emb.model


# ---- the hermetic guard itself ------------------------------------------------

def test_guard_blocks_outbound_connection():
    import socket

    with pytest.raises(NetworkBlockedError):
        socket.create_connection(("example.com", 80), timeout=1)


def test_guard_blocks_dns_lookup():
    import socket

    with pytest.raises(NetworkBlockedError):
        socket.getaddrinfo("huggingface.co", 443)


def test_guard_blocks_chroma_cloud_client():
    import chromadb

    with pytest.raises(NetworkBlockedError):
        chromadb.CloudClient(api_key="x")


def test_guard_sets_hf_offline():
    assert os.environ["HF_HUB_OFFLINE"] == "1"
