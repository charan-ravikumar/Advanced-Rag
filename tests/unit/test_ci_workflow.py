"""Static checks of the CI definition: the jobs that make the badge meaningful exist, and nothing silences them."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
TEXT = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
WORKFLOW = yaml.safe_load(TEXT)
SMOKE = (ROOT / "scripts" / "ci_smoke.sh").read_text(encoding="utf-8")


def runs(job: str) -> str:
    return "\n".join(step.get("run", "") for step in WORKFLOW["jobs"][job]["steps"])


def test_it_runs_on_push_and_pull_request_with_read_only_permissions():
    triggers = WORKFLOW[True]  # YAML 1.1 reads the key "on" as true
    assert set(triggers) == {"push", "pull_request"} and WORKFLOW["permissions"] == {"contents": "read"}


def test_the_five_jobs_run_the_real_checks():
    assert set(WORKFLOW["jobs"]) == {"lint", "types", "tests", "docker", "audit"}
    assert "ruff check ." in runs("lint") and re.search(r"^mypy$", runs("types"), re.M) and "pytest --fast" in runs("tests")
    assert "scripts/ci_smoke.sh" in runs("docker") and "pip-audit -r requirements.lock" in runs("audit")


def test_no_job_or_blocking_step_can_fail_silently():
    for name, job in WORKFLOW["jobs"].items():
        assert "continue-on-error" not in job and "if" not in job, name
        soft = [s["name"] for s in job["steps"] if s.get("continue-on-error")]
        assert soft == (["Audit the offline evaluation lock (report only)"] if name == "audit" else []), name
    assert "|| true" not in runs("tests") + runs("lint") + runs("types")


def test_the_smoke_test_requires_ready_and_only_removes_its_own_volume():
    assert "/v1/ready" in SMOKE and '["ready"] is True' in SMOKE and "set -euo pipefail" in SMOKE
    assert "-p advrag-smoke --env-file /dev/null" in SMOKE
    assert [line for line in SMOKE.splitlines() if "down -v" in line and not line.lstrip().startswith("#")] == \
        [line for line in SMOKE.splitlines() if line.strip().startswith("compose --profile tools down -v")]
    assert "docker compose down" not in SMOKE and "docker volume" not in SMOKE


def test_every_ignored_advisory_is_a_single_id_with_a_reason():
    lines = (ROOT / ".github" / "pip-audit-ignore.txt").read_text(encoding="utf-8").splitlines()
    entries = [line for line in lines if line.strip() and not line.startswith("#")]
    assert entries and all(re.fullmatch(r"PYSEC-\d{4}-\d+\s+# .{20,}", line) for line in entries)


def test_the_model_download_script_refuses_outside_ci(monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("ci_fetch_model", ROOT / "scripts" / "ci_fetch_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("CI", raising=False)
    assert module.main() == 2
