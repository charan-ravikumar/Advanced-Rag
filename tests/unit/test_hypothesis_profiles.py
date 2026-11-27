"""The Hypothesis profiles registered by tests/conftest.py."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from hypothesis import settings

ROOT = Path(__file__).resolve().parents[2]


def test_default_profile_is_the_derandomized_ci_profile():
    assert os.environ.get("ADVRAG_HYPOTHESIS_PROFILE", "ci") == "ci"
    assert settings.default.derandomize is True
    assert settings.default.database is None
    assert settings.default.deadline is None
    assert settings.default.max_examples == int(os.environ["ADVRAG_HYPOTHESIS_EXAMPLES"])


def test_both_profiles_are_registered_and_differ_in_seeding():
    ci, explore = settings.get_profile("ci"), settings.get_profile("explore")
    assert ci.derandomize is True and explore.derandomize is False
    assert ci.max_examples == explore.max_examples and explore.print_blob is True


def _run_pytest(env_extra: dict, *args) -> subprocess.CompletedProcess:
    env = {**os.environ, **env_extra}
    return subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "-o", "addopts=", *args],
                          capture_output=True, text=True, cwd=ROOT, env=env, timeout=300)


@pytest.mark.slow
def test_an_unknown_profile_name_is_a_usage_error():
    proc = _run_pytest({"ADVRAG_HYPOTHESIS_PROFILE": "bogus"}, "tests/unit/test_hypothesis_profiles.py::test_default_profile_is_the_derandomized_ci_profile")
    assert proc.returncode != 0 and "must be 'ci' or 'explore'" in (proc.stdout + proc.stderr)


@pytest.mark.slow
def test_the_explore_profile_is_selectable_by_environment_variable():
    probe = textwrap.dedent(
        """
        from hypothesis import settings
        def test_probe():
            assert settings.default.derandomize is False and settings.default.print_blob is True
        """
    )
    path = ROOT / "tests" / "unit" / "_probe_explore_profile.py"
    try:
        path.write_text(probe, encoding="utf-8")
        proc = _run_pytest({"ADVRAG_HYPOTHESIS_PROFILE": "explore"}, str(path.relative_to(ROOT)))
    finally:
        path.unlink(missing_ok=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
