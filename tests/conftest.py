"""Shared pytest fixtures for the test suite."""

import pytest


@pytest.fixture
def sample_chunks():
    """Return a minimal list of embedded-chunk dicts for use in retrieval tests."""
    return [
        {"id": f"doc{i}", "content": f"chunk content {i}", "score": 1.0 / (i + 1)}
        for i in range(5)
    ]
