"""Static guarantees about rag_core: what it may import, that it never prints or logs (so keys cannot leak), and that nothing imports
the deleted legacy retrieval module."""

import ast
from pathlib import Path

import pytest

from rag_core.config import RagConfig

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = sorted((ROOT / "rag_core").glob("*.py"))
FORBIDDEN_PREFIXES = ("rag.retrieval", "rag.router_docs", "rag.gemini_embedder", "db.vectordb", "streamlit", "fastapi", "app", "scripts", "eval",
                      "chromadb", "sentence_transformers", "torch")
# rag.retrieval was deleted (Week 1, Step 1): no file may import it, so the set of allowed importers is empty.
LEGACY_IMPORTERS: set[str] = set()


def imports_of(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names |= {f"{node.module}.{a.name}" for a in node.names}
    return names


@pytest.mark.parametrize("path", PACKAGE, ids=lambda p: p.name)
def test_rag_core_imports_nothing_deprecated_dormant_or_heavy(path):
    bad = sorted(n for n in imports_of(path) if any(n == p or n.startswith(p + ".") for p in FORBIDDEN_PREFIXES))
    assert bad == [], f"{path.name} imports {bad}"


def test_the_package_has_the_documented_modules():
    assert {p.name for p in PACKAGE} >= {"__init__.py", "schemas.py", "context.py", "prompts.py", "citations.py", "gateway.py", "cache.py", "service.py",
                                         "errors.py", "config.py", "env.py", "langfuse_client.py", "prompt_store.py", "prompt_sync.py", "tracing.py",
                                         "scores.py", "observability_reader.py", "claims.py",
                                         "otel_tracing.py"}


SDK_TOP_LEVELS = {"openai", "google", "httpx", "langfuse", "opentelemetry", "groq", "dotenv", "requests"}


@pytest.mark.parametrize("path", PACKAGE, ids=lambda p: p.name)
def test_no_sdk_or_http_client_is_imported_at_module_level_anywhere_in_the_package(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top = {(n.module if isinstance(n, ast.ImportFrom) else a.name).split(".")[0] for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
           for a in (n.names if isinstance(n, ast.Import) else [n]) if (n.module if isinstance(n, ast.ImportFrom) else a.name)}
    assert not SDK_TOP_LEVELS & top, f"{path.name} imports {sorted(SDK_TOP_LEVELS & top)} at module level"


def test_provider_sdks_are_imported_only_inside_functions_never_at_module_level():
    tree = ast.parse((ROOT / "rag_core" / "gateway.py").read_text(encoding="utf-8"))
    top_level = {n.module if isinstance(n, ast.ImportFrom) else a.name for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
                 for a in (n.names if isinstance(n, ast.Import) else [n])}
    assert not {"openai", "google", "google.genai", "httpx"} & {t for t in top_level if t}


@pytest.mark.parametrize("path", PACKAGE, ids=lambda p: p.name)
def test_rag_core_never_prints_or_logs_so_a_key_cannot_end_up_in_output(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    printed = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) in ("print", "pprint")]
    logged = [n for n in imports_of(path) if n in ("logging", "loguru", "structlog")]
    assert printed == [] and logged == []


ENV_READERS = {"gateway.py": ("GROQ_API_KEY", "GEMINI_API_KEY"), "langfuse_client.py": ("LANGFUSE_",), "observability_reader.py": ("LANGFUSE_",),
               "env.py": ("GROQ_API_KEY", "GEMINI_API_KEY", "LANGFUSE_")}


def test_api_keys_are_read_from_the_environment_only_and_only_where_expected():
    for path in PACKAGE:
        text = path.read_text(encoding="utf-8")
        allowed = ENV_READERS.get(path.name)
        if allowed is None:
            assert "os.environ" not in text and "load_dotenv" not in text and "dotenv" not in text, path.name
            assert "GROQ_API_KEY" not in text and "GEMINI_API_KEY" not in text and "LANGFUSE_SECRET" not in text, path.name
        if path.name != "env.py":
            assert "load_dotenv" not in text and "dotenv_values" not in text, path.name  # only env.py touches .env
    assert "os.environ.get" in (ROOT / "rag_core" / "gateway.py").read_text(encoding="utf-8")


def test_env_py_loads_nothing_at_import_and_only_entry_points_call_it():
    tree = ast.parse((ROOT / "rag_core" / "env.py").read_text(encoding="utf-8"))
    module_level_calls = [n for n in tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
    assert module_level_calls == []
    callers = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*.py")
               if "load_env(" in p.read_text(encoding="utf-8", errors="ignore") and not p.relative_to(ROOT).as_posix().startswith(("tests/", "venv", ".venv"))}
    entry_points = {"rag_core/env.py", "eval/run_generation_eval.py", "eval/run_ragas.py", "app/bootstrap.py", "api/main.py"}  # eval runners are command-line entry points; app/bootstrap.py is the Streamlit app's entry-side caller; api/main.py loads .env once at server startup
    assert all(c.startswith("scripts/") or c in entry_points for c in callers), callers  # no library module calls it


def test_no_new_code_imports_the_deprecated_retrieval_module():
    importers = set()
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(("tests/", ".venv", "venv", "node_modules")) or "__pycache__" in rel:
            continue
        if any(n == "rag.retrieval" or n.startswith("rag.retrieval.") for n in imports_of(path)):
            importers.add(rel)
    assert importers <= LEGACY_IMPORTERS, f"new importers of rag.retrieval: {sorted(importers - LEGACY_IMPORTERS)}"
    for path in (ROOT / "tests").rglob("test_*.py"):
        if path.name == "test_retrieval.py":
            continue  # its legacy tests are skipped and patch the module by name; they do not import it at collection time
        assert "rag.retrieval" not in imports_of(path), path.name


def test_stage_2b_modules_are_not_imported_by_new_code():
    for rel in ("rag_core", "scripts/smoke_generate.py"):
        paths = list((ROOT / rel).glob("*.py")) if (ROOT / rel).is_dir() else [ROOT / rel]
        for path in paths:
            if path.exists():
                assert not {n for n in imports_of(path) if n.startswith(("rag.router_docs", "rag.gemini_embedder"))}, path.name


# ---- config -------------------------------------------------------------------------------------------------------------

def test_defaults_equal_the_generation_section_of_config_yaml_and_top_k_is_five():
    from config import cfg

    from_yaml, defaults = RagConfig.from_config(cfg), RagConfig()
    assert from_yaml.as_dict() == defaults.as_dict()
    assert defaults.top_k == 5 and defaults.providers == ["gemini"] and defaults.max_question_chars == 500
    assert defaults.judge == {"provider": "gemini", "model": "gemini-3.8-flash", "temperature": 0.0, "thinking_budget": 1024, "max_output_tokens": 4096}
    assert defaults.prices == {}  # the judge is for offline evaluation only; prices stay empty until a source is found (never estimated)


def test_the_retrieval_settings_that_feed_the_cache_key_change_with_the_config_and_with_rerank():
    a, b = RagConfig(), RagConfig(rrf_k=20)
    assert a.retrieval_settings() != b.retrieval_settings() and a.retrieval_settings() != a.retrieval_settings(rerank=True)
    assert RagConfig(top_k=4).retrieval_settings() != a.retrieval_settings()
    assert a.retrieval_settings()["mode"] == "hybrid" and a.retrieval_settings()["rrf_k"] == 60 and a.retrieval_settings()["lexical_weight"] == 0.7


def test_from_config_keeps_the_defaults_for_a_missing_section_and_rejects_unknown_keys():
    from types import SimpleNamespace

    assert RagConfig.from_config(SimpleNamespace()).as_dict() == RagConfig().as_dict()
    got = RagConfig.from_config(SimpleNamespace(generation=SimpleNamespace(max_output_tokens=7)))
    assert got.max_output_tokens == 7
    with pytest.raises(ValueError, match="not_a_setting"):
        RagConfig.from_config(SimpleNamespace(generation=SimpleNamespace(max_output_tokens=7, not_a_setting=1)))


def test_build_gateway_builds_only_the_configured_providers_lazily_without_keys_or_network():
    from rag_core.gateway import GeminiProvider, build_gateway

    gw = build_gateway(RagConfig())
    assert [p.name for p in gw.providers] == ["gemini"] and isinstance(gw.providers[0], GeminiProvider) and gw.primary_model == RagConfig().gemini_model
    assert all(p._client is None for p in gw.providers)
    with pytest.raises(ValueError):
        build_gateway(RagConfig(providers=["nope"]))
