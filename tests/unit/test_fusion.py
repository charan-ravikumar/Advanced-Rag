"""Unit tests for rank fusion (rag/fusion.py): weighted RRF, naive interleave, deterministic RRF."""

import pytest

from rag.fusion import fuse_rrf, naive_interleave, weighted_rrf


def _import_weighted_rrf():
    """Kept so the original tests below read unchanged; the function now lives in rag/fusion.py."""
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

    def test_exact_scores(self):
        """Pin the formula: weight / (k + rank + 1) summed over the lists."""
        result = weighted_rrf(["a", "b"], ["b"], vector_weight=1.0, bm25_weight=0.7, k=60)
        assert result["a"] == pytest.approx(1 / 61)
        assert result["b"] == pytest.approx(1 / 62 + 0.7 / 61)

    def test_default_k_comes_from_config(self):
        from config import cfg

        assert weighted_rrf(["a"], []) == weighted_rrf(["a"], [], k=cfg.pipeline.rrf_k)


class TestFuseRRF:
    def test_orders_by_score_best_first(self):
        ranked = fuse_rrf(["a", "b"], ["b", "c"], k=60)
        assert [i for i, _ in ranked][0] == "b"  # in both lists

    def test_ties_break_by_id_deterministically(self):
        # symmetric inputs with equal weights give identical scores to x and y
        ranked = fuse_rrf(["y"], ["x"], vector_weight=1.0, lexical_weight=1.0, k=60)
        assert [i for i, _ in ranked] == ["x", "y"]
        assert [i for i, _ in fuse_rrf(["x"], ["y"], vector_weight=1.0, lexical_weight=1.0, k=60)] == ["x", "y"]

    def test_limit_applies_after_fusion_so_a_lexical_only_hit_is_kept_when_it_fits(self):
        vec = ["v0", "v1", "v2"]
        lex = ["lex_only"]
        ranked = fuse_rrf(vec, lex, limit=4, k=60)
        assert [i for i, _ in ranked] == ["v0", "v1", "v2", "lex_only"]
        assert len(fuse_rrf(vec, lex, limit=2, k=60)) == 2  # truncation happens after fusion

    def test_characterisation_vector_dominates_without_overlap_at_weights_1_and_0_7(self):
        """RRF with 1.0 / 0.7 and k=60: a lexical-only RANK-0 hit scores 0.7/61, below every vector
        rank up to ~26. With no overlap it only reaches a short top-k if the vector list is short.
        This pins the behaviour so a weight change is a visible, deliberate decision."""
        vec = [f"v{i}" for i in range(40)]
        ranked = [i for i, _ in fuse_rrf(vec, ["lex_only"], vector_weight=1.0, lexical_weight=0.7, k=60)]
        beaten_by = sum(1 for r in range(40) if 1 / (61 + r) > 0.7 / 61)
        assert ranked.index("lex_only") == beaten_by
        assert beaten_by > 20

    def test_no_limit_returns_everything(self):
        assert len(fuse_rrf(["a", "b"], ["c"], k=60)) == 3


class TestNaiveInterleave:
    def test_round_robin_order(self):
        assert naive_interleave(["v1", "v2", "v3"], ["l1", "l2", "l3"], k=6) == ["v1", "l1", "v2", "l2", "v3", "l3"]

    def test_dedupes_by_id(self):
        assert naive_interleave(["a", "b"], ["b", "c"], k=10) == ["a", "b", "c"]

    def test_unequal_lengths_keep_the_tail(self):
        assert naive_interleave(["a"], ["x", "y", "z"], k=10) == ["a", "x", "y", "z"]

    def test_empty_inputs(self):
        assert naive_interleave([], [], k=5) == []
        assert naive_interleave(["a"], [], k=5) == ["a"]
        assert naive_interleave([], ["a"], k=5) == ["a"]

    def test_no_k_means_no_truncation(self):
        assert len(naive_interleave(["a", "b"], ["c", "d"])) == 4

    def test_lexical_only_hits_survive_truncation(self):
        """Regression for the old harness: it truncated the CONCATENATED list, so once the vector
        leg alone filled k slots no lexical result could ever appear."""
        vec = [f"v{i}" for i in range(10)]
        lex = [f"l{i}" for i in range(10)]
        k = 5
        old_behaviour = (vec + lex)[:k]
        assert not any(i.startswith("l") for i in old_behaviour)  # the bug, demonstrated
        merged = naive_interleave(vec, lex, k)
        assert len(merged) == k
        assert {"l0", "l1"} <= set(merged)  # lexical-only hits are kept
        assert merged[:4] == ["v0", "l0", "v1", "l1"]

    def test_truncates_only_after_merging(self):
        # every lexical id is also a vector id: dedup must happen BEFORE the cut, so k unique ids come out
        vec = ["a", "b", "c", "d"]
        lex = ["a", "b", "c", "d"]
        assert naive_interleave(vec, lex, k=3) == ["a", "b", "c"]
