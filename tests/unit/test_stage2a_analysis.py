"""Decision rule, error-analysis helpers and corpus diagnostics: hand-built cases, no model, no network."""

import re

import numpy as np

from eval.run_retrieval_eval import ARM_ALIASES, ARMS, PREREG_FILE
from scripts import corpus_diagnostics as cd
from scripts import decision_rule as dr
from scripts import error_analysis as ea

PREREG_2B = PREREG_FILE.with_name("PREREGISTRATION_2b.md")

# ---- decision rule -------------------------------------------------------------------------------------

def test_complexity_matches_the_table_in_the_preregistration():
    text = PREREG_FILE.read_text(encoding="utf-8")
    header = re.search(r"\| Arm \| ([A-Z0-9a-z |*]+) \|\n\|[-| ]+\|\n\| Complexity \| ([0-9 |]+) \|", text)
    names, values = header.group(1).split(" | "), [int(v) for v in header.group(2).split(" | ")]
    table = dict(zip(names, values))
    for name, arm in ARMS.items():
        if name not in ARM_ALIASES["all"]:
            continue  # Stage 2b arms are checked against their own table below
        key = "R*" if name.startswith("R") else name
        if name == "R60_05" or name == "R60_10" or name.startswith("R20"):
            assert dr.complexity(arm) == table["R*"], name
        else:
            assert dr.complexity(arm) == table[key], name


def test_stage_2b_complexity_matches_the_preregistration_2b_text():
    text = PREREG_2B.read_text(encoding="utf-8")
    assert "A: 1. D: 3. GA: 3. GD: 5. RT1, RT3, RT5: 5. GRT3: 7." in text
    got = {n: dr.complexity(ARMS[n]) for n in ("A", "D", "GA", "GD", "RT1", "RT3", "RT5", "GRT3")}
    assert got == {"A": 1, "D": 3, "GA": 3, "GD": 5, "RT1": 5, "RT3": 5, "RT5": 5, "GRT3": 7}


def _rows(values: dict[str, list[float]], papers=None):
    rows = []
    for arm, vals in values.items():
        for i, v in enumerate(vals):
            rows.append({"arm": arm, "question_id": f"q{i:03d}", "paper_id": f"p{i // 2:03d}", "metrics": {"nDCG@10": v}})
    return rows


def test_the_simplest_arm_statistically_tied_with_the_best_wins():
    rng = np.random.default_rng(1)
    base = 0.2 + 0.6 * rng.random(120)  # away from 0 and 1, so zero-mean noise is not distorted by clipping

    def noise(s):
        return list(base + rng.normal(0, s, 120))

    rows = _rows({"A": [0.2] * 120, "D": noise(0.1), "F": noise(0.1), "G": list(base)}, None)
    res = dr.apply_rule(rows, n_resamples=500)
    assert res["best"] in {"D", "F", "G"}
    assert "A" not in res["indistinguishable_unadjusted_CI_includes_0"]  # far worse than the best
    assert res["recommended_primary_rule"] == "D"  # D (complexity 3) beats F (4) and G (5) among the tied arms


def test_a_clearly_better_complex_arm_is_chosen_over_simpler_ones():
    rng = np.random.default_rng(2)
    low = list(np.clip(rng.random(120) * 0.3, 0, 1))
    high = list(np.clip(np.array(low) + 0.4, 0, 1))
    res = dr.apply_rule(_rows({"A": low, "D": low, "G": high}), n_resamples=500)
    assert res["best"] == "G" and res["recommended_primary_rule"] == "G" and res["indistinguishable_unadjusted_CI_includes_0"] == []
    assert res["n_comparisons"] == 2


# ---- error analysis helpers ----------------------------------------------------------------------------

def test_overlap_share_ignores_stopwords_and_case():
    assert ea.overlap_share("What is the Learning Rate?", "We tune the learning rate and batch size.") == 1.0
    assert ea.overlap_share("What dataset do they use?", "Completely unrelated sentence about cats.") == 0.0
    assert ea.overlap_share("what is it", "anything") is None  # no content words


def _hit(doc, sec, a, b, relevant):
    return {"doc_id": doc, "section": sec, "char_start": a, "char_end": b, "relevant": relevant}


def test_near_miss_needs_a_non_relevant_chunk_touching_a_gold_span_in_the_same_section():
    gold = [{"doc_id": "d", "section": "S", "char_start": 100, "char_end": 200}]
    assert ea.near_miss([_hit("d", "S", 150, 400, False)], gold)
    assert not ea.near_miss([_hit("d", "S", 200, 400, False)], gold)  # adjacent, no overlap
    assert not ea.near_miss([_hit("d", "T", 150, 400, False)], gold)  # other section
    assert not ea.near_miss([_hit("e", "S", 150, 400, False)], gold)  # other document
    assert not ea.near_miss([_hit("d", "S", 150, 400, True)], gold)  # already counted as relevant


def test_categories_are_multi_label_and_fall_back_to_none():
    q = {"gold_evidence": ["a", "b"], "gold_spans": [{"doc_id": "d", "section": "Appendix Z", "char_start": 0, "char_end": 5}],
         "references": [{"n_evidence_not_retrievable": 1}]}
    row = {"retrieved": [], "metrics": {"reachable": True}}
    cats = ea.categories_of(q, row, 0.1)
    assert {"vocabulary_mismatch", "multi_paragraph_evidence", "evidence_in_unusual_section", "table_or_figure_context_also_cited"} <= set(cats)
    plain = {"gold_evidence": ["a"], "gold_spans": [{"doc_id": "d", "section": "Introduction", "char_start": 0, "char_end": 5}], "references": []}
    assert ea.categories_of(plain, row, 0.9) == ["none_of_the_proposed"]
    assert "gold_not_reachable_by_any_chunk" in ea.categories_of(plain, {"retrieved": [], "metrics": {"reachable": False}}, 0.9)


# ---- corpus diagnostics --------------------------------------------------------------------------------

def _row(qid, paper, docs, hr5):
    return {"question_id": qid, "paper_id": paper, "retrieved": [{"doc_id": d} for d in docs], "metrics": {"HitRate@5": hr5}}


def test_conditional_hit_rate_splits_by_whether_the_paper_was_found():
    corpus = {"q1": _row("q1", "p", ["x", "p"], 1.0), "q2": _row("q2", "p", ["x", "y"], 0.0), "q3": _row("q3", "p", ["p"], 0.0),
              "q4": _row("q4", "p", ["y"] * 5 + ["p"], 0.0)}
    paper = {"q1": _row("q1", "p", [], 1.0), "q2": _row("q2", "p", [], 1.0), "q3": _row("q3", "p", [], 1.0), "q4": _row("q4", "p", [], 0.0)}
    out = cd.diagnose(corpus, paper, None)
    assert out["DocHit"]["@1"] == 0.25 and out["DocHit"]["@5"] == 0.5 and out["DocHit"]["@10"] == 0.75 and out["DocHit"]["@20"] is None
    assert out["doc_found_at_5"] == {"n": 2, "HitRate@5_corpus": 0.5, "HitRate@5_paper_same_questions": 1.0}
    assert out["doc_not_found_at_5"]["n"] == 2 and out["doc_not_found_at_5"]["HitRate@5_corpus"] == 0.0
    depth = {k: _row(k, "p", ["y"] * 12 + ["p"], 0.0) for k in corpus}
    assert cd.diagnose(corpus, paper, depth)["DocHit"]["@20"] == 1.0
