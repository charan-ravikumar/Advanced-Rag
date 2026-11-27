"""Static checks of the Dockerfile, compose.yaml and .dockerignore. No Docker needed: building and running are verified by hand."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
COMPOSE_TEXT = (ROOT / "compose.yaml").read_text(encoding="utf-8")
COMPOSE = yaml.safe_load(COMPOSE_TEXT)
IGNORED = [line.strip() for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
CODE = "\n".join(line for line in DOCKERFILE.splitlines() if not line.lstrip().startswith("#"))


def test_the_build_needs_no_private_index_or_secret():
    assert "--mount=type=secret" not in CODE and "pipconf" not in CODE and "/etc/pip" not in CODE and "--extra-index-url" not in CODE
    assert re.findall(r"https?://[^\s\"']+", CODE.split("FROM python:3.13-slim\n")[0]) == ["https://download.pytorch.org/whl/cpu"]


def test_torch_is_the_cpu_build_and_the_build_fails_otherwise():
    assert CODE.index('--index-url "$TORCH_INDEX_URL"') < CODE.index("pip install -r runtime.lock")
    assert "torch.version.cuda is None" in CODE and "pip check" in CODE and "grep -iE '^nvidia'" in CODE
    assert CODE.count("FROM ") == 2 and "COPY --from=builder /opt/venv" in CODE


def test_every_copied_path_exists_and_the_image_does_not_run_as_root():
    for line in CODE.splitlines():
        if line.startswith("COPY --chown"):
            for source in line.split()[2:-1]:
                assert (ROOT / source).exists(), source
    assert "\nUSER app" in CODE


def test_secrets_local_state_and_virtualenvs_stay_out_of_the_build_context():
    assert {".env", ".env.*", "runs/", "*.sqlite3*", ".venv/", ".git/", ".cache/"} <= set(IGNORED)


def test_compose_runs_api_and_ui_from_one_image_and_ingest_only_on_request():
    services = COMPOSE["services"]
    assert set(services) == {"api", "ui", "ingest"} and len({s["image"] for s in services.values()}) == 1
    assert services["ingest"]["profiles"] == ["tools"] and "profiles" not in services["api"] and "profiles" not in services["ui"]
    assert services["ui"]["depends_on"]["api"]["condition"] == "service_healthy"
    assert services["ui"]["environment"]["ADVRAG_API_URL"] == "http://api:8000"


def test_ports_are_published_on_loopback_only():
    for name, service in COMPOSE["services"].items():
        assert all(p.startswith("127.0.0.1:") for p in service.get("ports", [])), name


def test_only_named_variables_reach_a_container_and_the_ui_gets_no_provider_key():
    for service in COMPOSE["services"].values():
        assert "env_file" not in service
        for key, value in service.get("environment", {}).items():
            assert value == "http://api:8000" or value == f"${{{key}:-}}", key  # never a literal value
    assert set(COMPOSE["services"]["ui"]["environment"]) == {"ADVRAG_API_URL", "ADVRAG_API_TOKEN", "APP_ACCESS_CODE"}
    assert "environment" not in COMPOSE["services"]["ingest"]  # ingestion needs no key


def test_the_model_cache_is_mounted_read_only_and_the_index_is_a_volume():
    mounts = COMPOSE["services"]["api"]["volumes"]
    assert "data:/data" in mounts and "data" in COMPOSE["volumes"]
    (cache,) = [m for m in mounts if isinstance(m, dict)]
    assert cache["target"] == "/models" and cache["read_only"] is True and cache["bind"]["create_host_path"] is False
