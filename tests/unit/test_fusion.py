"""Unit tests for the weighted RRF fusion function."""

import pytest


def _import_weighted_rrf():
    """Lazy import so the test file can be collected without loading the full pipeline."""
    from rag.retrieval import weighted_rrf
    return weighted_rrf


class TestWeightedRRF:
    """Tests for weighted_rrf."""

    def test_returns_dict(self):
        weighted_rrf = _import_weighted_rrf()
        vec = ["a", "b", "c"]
        bm25 = ["b", "c", "a"]
        result = weighted_rrf(vec, bm25, k=60)
        assert isinstance(result, dict)

    def test_all_doc_ids_present(self):
        weighted_rrf = _import_weighted_rrf()
        vec = ["a", "b", "c"]
        bm25 = ["b", "c", "d"]
        result = weighted_rrf(vec, bm25, k=60)
        # Every id from both lists must appear in the output
        for doc_id in vec + bm25:
            assert doc_id in result, f"{doc_id!r} missing from RRF scores"

    def test_higher_weight_boosts_rank(self):
        """A doc appearing first in the vector list should score higher when
        vector_weight > bm25_weight."""
        weighted_rrf = _import_weighted_rrf()
        # "vec_top" is rank-0 in vector, not in bm25
        # "bm25_top" is rank-0 in bm25, not in vector
        vec = ["vec_top", "shared"]
        bm25 = ["bm25_top", "shared"]
        result_vec_heavy = weighted_rrf(vec, bm25, vector_weight=2.0, bm25_weight=0.5, k=60)
        assert result_vec_heavy["vec_top"] > result_vec_heavy["bm25_top"]

    def test_scores_are_positive(self):
        weighted_rrf = _import_weighted_rrf()
        vec = ["x", "y"]
        bm25 = ["y", "z"]
        result = weighted_rrf(vec, bm25, k=60)
        assert all(v > 0 for v in result.values())

    def test_empty_lists(self):
        weighted_rrf = _import_weighted_rrf()
        result = weighted_rrf([], [], k=60)
        assert result == {}

    def test_single_list_populated(self):
        weighted_rrf = _import_weighted_rrf()
        result = weighted_rrf(["a", "b"], [], k=60)
        assert "a" in result and "b" in result

    @pytest.mark.parametrize("k", [10, 60, 100])
    def test_different_k_values(self, k):
        weighted_rrf = _import_weighted_rrf()
        vec = ["a", "b", "c"]
        bm25 = ["b", "c", "a"]
        result = weighted_rrf(vec, bm25, k=k)
        assert set(result.keys()) == {"a", "b", "c"}
