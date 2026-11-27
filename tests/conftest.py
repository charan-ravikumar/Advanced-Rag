"""Shared pytest fixtures and the hermetic test guard.

Every test runs with:
  * HuggingFace offline (set before any transformers import),
  * outbound (non-loopback) connections and DNS lookups refused,
  * ``chromadb.CloudClient`` replaced by a function that raises,
  * ``ADVRAG_ALLOW_CLOUD`` removed.
A test that tries to reach the network fails loudly instead of passing by luck.
"""

import os
import socket

import pytest

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

FAST_EXAMPLES = 8
FULL_EXAMPLES = 200
PROFILE_ENV = "ADVRAG_HYPOTHESIS_PROFILE"


def pytest_addoption(parser):
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help=f"everyday profile: {FAST_EXAMPLES} hypothesis examples instead of {FULL_EXAMPLES}, "
        "and skip tests marked 'slow' (subprocess import probe, real-model tests)",
    )


def _register_hypothesis_profiles(max_examples: int) -> None:
    """Two profiles, same limits:

    ci       derandomize=True, no example database: the same examples every run FOR A FIXED SET OF COLLECTED TEST
             MODULES (Hypothesis also draws on literals found in imported modules). Default for normal and --fast runs.
             It is a regression net, not an exploration tool: it will not find anything new.
    explore  random seeds plus the example database, print_blob=True so a failure can be replayed. Use it to look for
             new bugs (ADVRAG_HYPOTHESIS_PROFILE=explore) and pin anything it finds with @example.
    """
    from hypothesis import HealthCheck, settings

    common = dict(max_examples=max_examples, deadline=None, suppress_health_check=list(HealthCheck))
    settings.register_profile("ci", derandomize=True, database=None, **common)
    settings.register_profile("explore", derandomize=False, print_blob=True, **common)


def pytest_configure(config):
    # runs before test modules are imported, so their @given decorators see the loaded profile
    fast = config.getoption("--fast")
    max_examples = FAST_EXAMPLES if fast else int(os.environ.get("ADVRAG_HYPOTHESIS_EXAMPLES", FULL_EXAMPLES))
    os.environ["ADVRAG_HYPOTHESIS_EXAMPLES"] = str(max_examples)
    if getattr(config.option, "hypothesis_profile", None):
        return  # the owner picked a profile explicitly with --hypothesis-profile
    name = os.environ.get(PROFILE_ENV, "ci")
    if name not in ("ci", "explore"):
        raise pytest.UsageError(f"{PROFILE_ENV} must be 'ci' or 'explore', got {name!r}")
    _register_hypothesis_profiles(max_examples)
    from hypothesis import settings

    settings.load_profile(name)


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--fast"):
        return
    skip_slow = pytest.mark.skip(reason="--fast profile: slow test (run without --fast)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

_LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0", ""}


class NetworkBlockedError(RuntimeError):
    """Raised when a test tries to open a network connection."""


def _host_of(address) -> str | None:
    if isinstance(address, tuple) and address:
        return str(address[0])
    return None  # AF_UNIX paths etc. are local


@pytest.fixture(autouse=True)
def hermetic_guard(monkeypatch):
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def guard_connect(self, address):
        host = _host_of(address)
        if host is not None and host not in _LOOPBACK:
            raise NetworkBlockedError(f"test tried to connect to {address!r}")
        return real_connect(self, address)

    def guard_connect_ex(self, address):
        host = _host_of(address)
        if host is not None and host not in _LOOPBACK:
            raise NetworkBlockedError(f"test tried to connect to {address!r}")
        return real_connect_ex(self, address)

    def guard_getaddrinfo(host, *args, **kwargs):
        if host is not None and str(host) not in _LOOPBACK:
            raise NetworkBlockedError(f"test tried to resolve {host!r}")
        return real_getaddrinfo(host, *args, **kwargs)

    def no_cloud(*_a, **_k):
        raise NetworkBlockedError("tests must never create a chromadb.CloudClient")

    monkeypatch.setattr(socket.socket, "connect", guard_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guard_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guard_getaddrinfo)
    monkeypatch.delenv("ADVRAG_ALLOW_CLOUD", raising=False)
    try:
        import chromadb

        monkeypatch.setattr(chromadb, "CloudClient", no_cloud)
    except ImportError:
        pass
    yield


@pytest.fixture
def sample_chunks():
    """Return a minimal list of embedded-chunk dicts for use in retrieval tests."""
    return [
        {"id": f"doc{i}", "content": f"chunk content {i}", "score": 1.0 / (i + 1)}
        for i in range(5)
    ]
