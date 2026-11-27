"""DELIBERATELY FAILING. Proves that CI goes red. This branch and its pull request are closed unmerged."""


def test_this_fails_on_purpose_to_prove_ci_goes_red():
    assert False, "deliberate failure: CI must report red"
