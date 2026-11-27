"""eval/answer_metrics.py: official-style Answer-F1 with hand-built cases. Pure, hermetic."""

import pytest

from eval.answer_metrics import (answer_f1, citation_counts, mean_with_interval, normalize_answer, score_long, score_short, strip_citation_markers,
                                 summarize_long, summarize_metric, summarize_short, token_f1, yes_no_accuracy, yes_no_label)


def ref(answer, typ="extractive", evidence=("e",), lost=0):
    return {"answer": answer, "type": typ, "mixed_types": False, "evidence": list(evidence), "n_evidence_not_retrievable": lost}


TABLE_ONLY = dict(evidence=(), lost=2)


# ---- normalisation ----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("The Quick, Brown Fox!", "quick brown fox"),
    ("an apple a day", "apple day"),
    ("bathe in the other river", "bathe in other river"),  # 'the' inside 'bathe' and 'other' is not an article
    ("U.S.A.", "usa"),
    ("state-of-the-art", "stateoftheart"),
    ("\u201cQuoted\u201d \u2013 text", "quoted text"),  # typographic quotes and an en dash are Unicode punctuation
    ("**Bold** answer", "bold answer"),
    ("topic model\u202fis LDA", "topic model is lda"),  # a narrow no-break space separates words
    ("  many   spaces\n\tand lines ", "many spaces and lines"),
    ("The", ""), ("the a an", ""), ("", ""), (None, ""), ("!!!", ""),
    ("Cost: $5 + 3 = 8", "cost 5 3 8"),  # ASCII symbols count as punctuation, as in the official evaluator
])
def test_normalisation(text, expected):
    assert normalize_answer(text) == expected


# ---- token F1 ---------------------------------------------------------------------------------------------------------------------

def test_f1_hand_computed_values():
    assert token_f1("red green blue", "red green blue") == 1.0
    assert token_f1("red green blue", "green blue yellow") == pytest.approx(2 / 3)  # overlap 2, P = R = 2/3
    assert token_f1("paris", "paris france") == pytest.approx(2 / 3)  # P = 1, R = 1/2
    assert token_f1("cat cat dog", "cat dog dog") == pytest.approx(2 / 3)  # multiset overlap: min counts 1 + 1
    assert token_f1("red", "blue") == 0.0


def test_f1_is_zero_when_either_side_is_empty_including_both_empty_as_in_the_official_evaluator():
    assert token_f1("", "paris") == 0.0 and token_f1("paris", "") == 0.0 and token_f1("", "") == 0.0 and token_f1(None, "x y") == 0.0 and token_f1("the", "a") == 0.0


def test_the_extractive_join_with_a_comma_is_identical_to_the_stored_space_join_after_normalisation():
    spans = ["crosslingual", "latent variables"]
    prediction = "crosslingual latent variables"
    assert token_f1(prediction, ", ".join(spans)) == token_f1(prediction, " ".join(spans)) == 1.0


# ---- max over references ----------------------------------------------------------------------------------------------------------

def test_answer_f1_takes_the_MAX_over_references_not_the_mean():
    refs = [ref("alpha beta"), ref("gamma delta", "abstractive")]
    score = answer_f1("alpha beta", refs)
    assert score.f1 == 1.0 and score.best_type == "extractive" and score.best_index == 0  # a mean would give 0.5


def test_the_best_reference_type_is_that_of_the_best_match_and_ties_go_to_the_first():
    refs = [ref("red green"), ref("blue yellow", "abstractive")]
    assert answer_f1("blue yellow", refs).best_type == "abstractive" and answer_f1("blue yellow", refs).best_index == 1
    tie = [ref("alpha beta"), ref("alpha gamma", "abstractive")]
    s = answer_f1("alpha", tie)
    assert s.f1 == pytest.approx(2 / 3) and s.best_index == 0 and s.best_type == "extractive"


def test_references_that_are_table_only_or_unanswerable_are_skipped_and_the_index_counts_usable_ones():
    table = ref("zzz", **TABLE_ONLY)
    una = ref(None, "unanswerable")
    good = ref("alpha")
    assert answer_f1("alpha", [table, una]) is None and answer_f1("alpha", []) is None and answer_f1("alpha", None) is None
    assert answer_f1("alpha", [table, una, good]).best_index == 0  # index into the usable references
    assert answer_f1("zzz", [table, good]).f1 == 0.0  # the table-only reference's answer never counts


# ---- yes / no ---------------------------------------------------------------------------------------------------------------------

def test_yes_no_labels_and_accuracy():
    assert yes_no_label("Yes, because of X") == "yes" and yes_no_label("No.") == "no" and yes_no_label("**Yes**") == "yes"
    assert yes_no_label("I think so") is None and yes_no_label("") is None and yes_no_label("Noted") is None and yes_no_label(None) is None
    yes, no = [ref("yes", "yes_no")], [ref("no", "yes_no")]
    assert yes_no_accuracy("Yes.", yes) == 1.0 and yes_no_accuracy("No.", yes) == 0.0 and yes_no_accuracy("maybe", yes) == 0.0 and yes_no_accuracy("", no) == 0.0
    assert yes_no_accuracy("no", [ref("yes", "yes_no"), ref("no", "yes_no")]) == 1.0  # matching any annotator counts


def test_yes_no_accuracy_is_undefined_without_a_usable_yes_no_reference():
    assert yes_no_accuracy("yes", [ref("something", "extractive")]) is None and yes_no_accuracy("yes", [ref("yes", "yes_no", **TABLE_ONLY)]) is None


# ---- abstention and the per-answer scores ------------------------------------------------------------------------------------------

def test_an_abstention_on_an_answerable_question_is_a_miss_with_f1_zero_and_is_counted():
    metrics, typ = score_short("INSUFFICIENT_CONTEXT", True, [ref("alpha"), ref("yes", "yes_no")])
    assert metrics == {"AnswerF1": 0.0, "YesNoAccuracy": 0.0, "Abstained": 1.0} and typ == "extractive"


def test_a_correct_short_answer_scores_its_best_reference_and_a_question_without_usable_references_has_no_f1():
    metrics, typ = score_short("red green", False, [ref("red green"), ref("other", "abstractive")])
    assert metrics == {"AnswerF1": 1.0, "YesNoAccuracy": None, "Abstained": 0.0} and typ == "extractive"
    metrics, typ = score_short("anything", False, [ref("x", **TABLE_ONLY)])
    assert metrics["AnswerF1"] is None and typ is None and metrics["Abstained"] == 0.0


def test_citation_markers_are_not_part_of_the_short_answer():
    assert strip_citation_markers("LDA [S1] model").split() == ["LDA", "model"] and strip_citation_markers("LDA [S1][ S2 ] model").split() == ["LDA", "model"]
    assert strip_citation_markers("LDA\u3010S1\u3011 model \uff3bS2, S3\uff3d").split() == ["LDA", "model"] and strip_citation_markers("x [S1, S2] y").split() == ["x", "y"]
    assert score_short("red green [S1]", False, [ref("red green")])[0]["AnswerF1"] == 1.0 and score_short("red green\u3010S1\u3011", False, [ref("red green")])[0]["AnswerF1"] == 1.0
    assert strip_citation_markers(None) == "" and strip_citation_markers("[S] stays") == "[S] stays"  # a malformed marker is not stripped


# ---- citation validity ------------------------------------------------------------------------------------------------------------

def test_citation_validity_is_valid_over_cited_and_none_when_nothing_is_cited():
    assert citation_counts(["S1", "S2"], ["S9"]) == (2, 3) and citation_counts(["S1", "S1"], []) == (1, 1) and citation_counts([], []) == (0, 0)
    assert score_long(["S1", "S2"], ["S9"], False) == {"CitationValidity": pytest.approx(2 / 3), "HasCitation": 1.0, "QuoteVerified": None, "Abstained": 0.0}
    assert score_long([], [], False) == {"CitationValidity": None, "HasCitation": 0.0, "QuoteVerified": None, "Abstained": 0.0}
    assert score_long([], ["S7"], False)["CitationValidity"] == 0.0
    assert score_long(["S1"], [], True) == {"CitationValidity": None, "HasCitation": None, "QuoteVerified": None, "Abstained": 1.0}


# ---- summaries with paper-clustered intervals ---------------------------------------------------------------------------------------

def rows_for(values, papers, extra=None):
    return [{"question_id": f"q{i}", "paper_id": p, "metrics": {"AnswerF1": v, "Abstained": 0.0, "YesNoAccuracy": None}, **(extra[i] if extra else {})}
            for i, (v, p) in enumerate(zip(values, papers))]


def test_mean_and_clustered_interval_hand_cases():
    m = mean_with_interval([1.0, 0.0, 0.5, 0.5, 1.0, 0.0], ["p1", "p1", "p2", "p2", "p3", "p3"])
    assert m["mean"] == pytest.approx(0.5) and m["n"] == 6 and m["n_papers"] == 3 and m["ci_low"] <= 0.5 <= m["ci_high"] and 0.0 <= m["ci_low"] and m["ci_high"] <= 1.0
    assert mean_with_interval([1.0, 1.0, 1.0], ["a", "b", "c"])["ci_low"] == pytest.approx(1.0)  # no variation, no uncertainty
    assert mean_with_interval([0.4, 0.6], ["a", "a"]) == {"n": 2, "n_papers": 1, "mean": pytest.approx(0.5), "ci_low": None, "ci_high": None}
    assert mean_with_interval([], []) == {"n": 0, "n_papers": 0, "mean": None, "ci_low": None, "ci_high": None}
    assert mean_with_interval([0.1, 0.9, 0.5], ["a", "b", "c"]) == mean_with_interval([0.1, 0.9, 0.5], ["a", "b", "c"])  # deterministic


def test_papers_not_questions_are_resampled_so_a_correlated_paper_widens_the_interval():
    independent = mean_with_interval([0, 1, 0, 1, 0, 1, 0, 1], ["a", "b", "c", "d", "e", "f", "g", "h"])
    clustered = mean_with_interval([0, 0, 0, 0, 1, 1, 1, 1], ["a", "a", "a", "a", "b", "b", "b", "b"])  # same mean, 2 papers
    assert independent["mean"] == clustered["mean"] == 0.5
    assert (clustered["ci_high"] - clustered["ci_low"]) > (independent["ci_high"] - independent["ci_low"])


def test_summarize_short_groups_f1_by_the_type_of_the_best_reference_and_skips_undefined_questions():
    rows = rows_for([1.0, 0.5, None, 0.0], ["p1", "p2", "p3", "p4"],
                    extra=[{"best_reference_type": "extractive"}, {"best_reference_type": "extractive"}, {"best_reference_type": None}, {"best_reference_type": "abstractive"}])
    rows[3]["metrics"]["YesNoAccuracy"] = 1.0
    s = summarize_short(rows)
    assert s["AnswerF1"]["n"] == 3 and s["AnswerF1"]["mean"] == pytest.approx(0.5) and s["n_questions"] == 4
    assert s["AnswerF1_by_best_reference_type"]["extractive"]["mean"] == pytest.approx(0.75) and s["AnswerF1_by_best_reference_type"]["abstractive"]["mean"] == 0.0
    assert s["YesNoAccuracy"]["n"] == 1 and s["AbstentionRate_on_answerable"]["mean"] == 0.0


def test_summarize_long_pools_citations_and_counts_answers_without_any():
    rows = [{"paper_id": "p1", "abstained": False, "citations": ["S1", "S2"], "invalid_citations": ["S9"], "metrics": {"CitationValidity": 2 / 3, "Abstained": 0.0}},
            {"paper_id": "p2", "abstained": False, "citations": [], "invalid_citations": [], "metrics": {"CitationValidity": None, "Abstained": 0.0}},
            {"paper_id": "p3", "abstained": True, "citations": [], "invalid_citations": [], "metrics": {"CitationValidity": None, "Abstained": 1.0}}]
    s = summarize_long(rows)
    assert s["CitationValidity_pooled"] == {"valid": 2, "cited": 3, "rate": pytest.approx(2 / 3)}
    assert s["answers_with_no_citation"] == {"n": 1, "of": 2} and s["AbstentionRate_on_answerable"]["mean"] == pytest.approx(1 / 3)
    assert summarize_long([])["CitationValidity_pooled"] is None


def test_summarize_metric_ignores_rows_where_the_metric_is_none():
    assert summarize_metric(rows_for([None, None], ["a", "b"]), "AnswerF1")["n"] == 0


def test_quote_verified_is_the_share_of_quotes_found_in_their_source_and_none_without_quotes():
    from eval.answer_metrics import quote_counts, score_long

    claims = [{"text": "a", "citations": [{"source": "S1", "verified": True}, {"source": "S2", "verified": False}]},
              {"text": "b", "citations": [{"source": "S1", "verified": True}]}]
    assert quote_counts(claims) == (2, 3)
    assert score_long(["S1", "S2"], [], False, claims)["QuoteVerified"] == 2 / 3
    assert score_long([], [], False, [])["QuoteVerified"] is None and score_long([], [], True, claims)["QuoteVerified"] is None


def test_summarize_long_pools_verified_quotes_over_answered_rows():
    from eval.answer_metrics import summarize_long

    claim = lambda *v: {"text": "x", "citations": [{"source": "S1", "verified": b} for b in v]}  # noqa: E731
    rows = [{"paper_id": "p1", "abstained": False, "citations": ["S1"], "invalid_citations": [], "claims": [claim(True, True)],
             "metrics": {"CitationValidity": 1.0, "QuoteVerified": 1.0, "Abstained": 0.0}},
            {"paper_id": "p2", "abstained": False, "citations": ["S1"], "invalid_citations": [], "claims": [claim(False)],
             "metrics": {"CitationValidity": 1.0, "QuoteVerified": 0.0, "Abstained": 0.0}}]
    s = summarize_long(rows, n_resamples=200)
    assert s["QuoteVerified_pooled"] == {"verified": 2, "quotes": 3, "rate": 2 / 3} and s["QuoteVerified_per_answer"]["mean"] == 0.5
