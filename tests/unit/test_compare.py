"""Paired bootstrap and Holm adjustment (eval/compare.py), on constructed data with known answers."""

import numpy as np
import pytest

from eval.compare import compare, format_report, holm_significant, paired_bootstrap


def test_ci_contains_a_known_true_difference():
    rng = np.random.default_rng(123)
    diffs = rng.normal(0.20, 0.10, size=150)  # true mean difference 0.20
    r = paired_bootstrap(diffs, 5000, seed=0)
    assert r["ci_low"] <= 0.20 <= r["ci_high"]
    assert r["ci_low"] > 0 and r["p_value"] < 0.01  # an obvious effect is detected
    assert r["wins"] + r["losses"] + r["ties"] == r["n"] == 150


def test_identical_arms_have_a_zero_ci_and_no_wins_or_losses():
    r = paired_bootstrap([0.0] * 80, 5000, seed=0)
    assert r["ci_low"] <= 0.0 <= r["ci_high"] and r["mean_diff"] == 0.0
    assert r["wins"] == r["losses"] == 0 and r["ties"] == 80
    assert r["p_value"] == 1.0


def test_no_true_difference_ci_contains_zero():
    rng = np.random.default_rng(7)
    r = paired_bootstrap(rng.normal(0.0, 0.3, size=120), 5000, seed=0)
    assert r["ci_low"] <= 0.0 <= r["ci_high"]


def test_deterministic_for_a_fixed_seed_and_different_for_another():
    d = list(np.random.default_rng(1).normal(0.05, 0.3, size=60))
    assert paired_bootstrap(d, 5000, seed=0) == paired_bootstrap(d, 5000, seed=0)
    assert paired_bootstrap(d, 5000, seed=0)["ci_low"] != paired_bootstrap(d, 5000, seed=1)["ci_low"]


def test_wins_losses_ties_are_counted_per_question():
    r = paired_bootstrap([1, 1, 1, -1, 0, 0], 1000, seed=0)
    assert (r["wins"], r["losses"], r["ties"]) == (3, 1, 2)
    assert r["mean_diff"] == pytest.approx(2 / 6)


def test_empty_input_is_an_error():
    with pytest.raises(ValueError):
        paired_bootstrap([], 100, seed=0)


def test_holm_step_down():
    assert holm_significant([0.001, 0.02, 0.04]) == [True, True, True]
    assert holm_significant([0.001, 0.03, 0.04]) == [True, False, False]  # .03 > .05/2: stop
    assert holm_significant([0.04, 0.001, 0.30]) == [False, True, False]  # unsorted input, alpha/3 = .0167 for .001
    assert holm_significant([0.5]) == [False] and holm_significant([0.01]) == [True]
    assert holm_significant([]) == []


def _rows(arm_values):
    rows = []
    for arm, values in arm_values.items():
        for i, v in enumerate(values):
            rows.append({"arm": arm, "question_id": f"q{i}", "metrics": {"HitRate@5": v, "MRR": v / 2, "nDCG@10": v / 3}})
    return rows


def test_compare_counts_every_comparison_and_flags_with_holm():
    rng = np.random.default_rng(0)
    a = list((rng.random(100) < 0.5).astype(float))
    b = [min(1.0, x + (1.0 if rng.random() < 0.3 else 0.0)) for x in a]  # B is better: it only ever adds hits
    rows = _rows({"A": a, "B": b, "C": a})
    rep = compare(rows, [("B", "A"), ("C", "A")])
    assert rep["comparisons_made"] == 6  # 2 pairs x 3 metrics
    by = {(r["arm_x"], r["metric"]): r for r in rep["results"]}
    assert by[("B", "HitRate@5")]["holm_significant"] is True and by[("B", "HitRate@5")]["mean_diff"] > 0
    assert by[("C", "HitRate@5")]["holm_significant"] is False  # identical to A
    assert by[("C", "HitRate@5")]["wins"] == by[("C", "HitRate@5")]["losses"] == 0


def test_compare_requires_the_same_questions():
    rows = _rows({"A": [1.0, 0.0, 1.0], "B": [1.0, 0.0]})
    with pytest.raises(ValueError, match="same questions"):
        compare(rows, [("A", "B")])
    with pytest.raises(ValueError, match="no rows"):
        compare(rows, [("A", "Z")])


def test_questions_where_a_metric_is_undefined_are_left_out_for_both_arms():
    rows = []
    for arm, vals in {"A": [1.0, None, 0.0, 1.0], "B": [0.0, None, 0.0, 1.0]}.items():
        for i, v in enumerate(vals):
            rows.append({"arm": arm, "question_id": f"q{i}", "paper_id": f"p{i}", "metrics": {"EvidenceF1@5": v}})
    (r,) = compare(rows, [("A", "B")], metrics=("EvidenceF1@5",), n_resamples=200)["results"]
    assert r["n"] == 3 and r["n_papers"] == 3 and (r["wins"], r["losses"], r["ties"]) == (1, 0, 2)


def test_report_never_prints_a_difference_without_ci_counts_and_total():
    rep = compare(_rows({"A": [1.0, 0.0, 1.0, 0.0], "B": [1.0, 1.0, 1.0, 0.0]}), [("B", "A")])
    text = format_report(rep)
    assert "comparisons made in this invocation: 3" in text
    for line in text.splitlines():
        if "B vs A" in line:
            assert line.count("[") == 2 and line.count("]") == 2  # primary CI and the secondary per-question CI
            assert line.count("/") >= 2  # wins/losses/ties
            assert "Holm" in text


# ---- paper-clustered bootstrap ---------------------------------------------------------------------------------------

from eval.compare import clustered_bootstrap  # noqa: E402


def _width(r):
    return r["ci_high"] - r["ci_low"]


def test_clustered_ci_is_wider_when_all_questions_of_a_paper_are_identical_copies():
    rng = np.random.default_rng(5)
    per_paper = rng.normal(0.10, 0.30, size=40)  # one true effect per paper
    diffs = np.repeat(per_paper, 5)  # five identical copies of each paper's question
    papers = [f"p{i}" for i in range(40) for _ in range(5)]
    per_q = paired_bootstrap(diffs, 5000, seed=0)
    clu = clustered_bootstrap(diffs, papers, 5000, seed=0)
    assert clu["mean_diff"] == pytest.approx(per_q["mean_diff"])
    assert _width(clu) > 1.5 * _width(per_q)  # five copies carry the information of one: ~sqrt(5) wider
    assert clu["n_clusters"] == 40 and clu["n"] == 200
    assert clu["ci_low"] <= per_paper.mean() <= clu["ci_high"]


def test_clustered_equals_per_question_when_every_paper_has_one_question():
    d = np.random.default_rng(1).normal(0.05, 0.3, size=80)
    per_q = paired_bootstrap(d, 5000, seed=3)
    clu = clustered_bootstrap(d, [f"p{i:03d}" for i in range(80)], 5000, seed=3)  # zero-padded: sorted order == input order
    assert clu["ci_low"] == pytest.approx(per_q["ci_low"], abs=1e-9) and clu["ci_high"] == pytest.approx(per_q["ci_high"], abs=1e-9)
    assert clu["p_value"] == pytest.approx(per_q["p_value"])


def test_papers_move_together_a_paper_whose_questions_cancel_never_adds_noise():
    """Every paper has one question the first arm wins and one it loses: each paper's contribution is exactly 0, so a
    paper-level resample can only ever give 0. A question-level bootstrap would show spurious spread."""
    d = [1.0, -1.0] * 30
    papers = [f"p{i}" for i in range(30) for _ in range(2)]
    clu = clustered_bootstrap(d, papers, 2000, seed=0)
    assert clu["ci_low"] == clu["ci_high"] == 0.0 and clu["p_value"] == 1.0
    assert paired_bootstrap(d, 2000, seed=0)["ci_high"] > 0.0


def test_clustered_statistic_is_the_question_weighted_mean_with_unequal_paper_sizes():
    d = [1.0] * 6 + [0.0] * 2  # paper a has 6 questions, paper b has 2
    res = clustered_bootstrap(d, ["a"] * 6 + ["b"] * 2, 1000, seed=0)
    assert res["mean_diff"] == pytest.approx(6 / 8)
    assert 0.0 <= res["ci_low"] <= res["ci_high"] <= 1.0


def test_clustered_bootstrap_is_deterministic_for_a_fixed_seed():
    d = list(np.random.default_rng(2).normal(0, 1, 60))
    papers = [f"p{i % 12}" for i in range(60)]
    assert clustered_bootstrap(d, papers, 3000, seed=7) == clustered_bootstrap(d, papers, 3000, seed=7)
    assert clustered_bootstrap(d, papers, 3000, seed=7)["ci_low"] != clustered_bootstrap(d, papers, 3000, seed=8)["ci_low"]


def test_clustered_bootstrap_validates_input():
    with pytest.raises(ValueError):
        clustered_bootstrap([], [], 100)
    with pytest.raises(ValueError, match="one cluster label"):
        clustered_bootstrap([1.0, 2.0], ["a"], 100)


def test_compare_uses_paper_ids_for_the_primary_interval_and_reports_both():
    rng = np.random.default_rng(3)
    per_paper_a = (rng.random(30) < 0.5).astype(float)
    rows = []
    for p in range(30):
        for q in range(4):  # four identical copies per paper
            rows.append({"arm": "A", "question_id": f"p{p}:{q}", "paper_id": f"p{p}", "metrics": {"HitRate@5": per_paper_a[p], "MRR": 0.0, "nDCG@10": 0.0}})
            rows.append({"arm": "B", "question_id": f"p{p}:{q}", "paper_id": f"p{p}", "metrics": {"HitRate@5": min(1.0, per_paper_a[p] + (p % 3 == 0)), "MRR": 0.0, "nDCG@10": 0.0}})
    r = compare(rows, [("B", "A")], ("HitRate@5",))["results"][0]
    assert r["n"] == 120 and r["n_papers"] == 30
    assert (r["qci_high"] - r["qci_low"]) < (r["ci_high"] - r["ci_low"])  # clustered (primary) is the wider one
    text = format_report(compare(rows, [("B", "A")], ("HitRate@5",)))
    assert "PRIMARY CI (clustered)" in text and "n_pap" in text and "  30 " in text
