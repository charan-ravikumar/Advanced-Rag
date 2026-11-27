"""CrossEncoderReranker: lazy, local-only, injectable. The real-model test needs the pinned model in the local HF cache."""

import pytest

from config import cfg
from rag.rerankers import CrossEncoderReranker


def test_importing_and_constructing_loads_nothing():
    r = CrossEncoderReranker.from_config(cfg)
    assert r._model is None and r.describe()["local_files_only"] is True
    assert r.revision == cfg.pipeline.reranker_revision and len(r.revision) == 40
    assert r("q", []) == []  # nothing to score: no model load either
    assert r._model is None


def test_the_reranker_is_a_plain_callable_the_retriever_can_take():
    assert callable(CrossEncoderReranker("m", "r"))


@pytest.mark.slow
def test_a_missing_cache_raises_a_clear_error_and_never_downloads():
    r = CrossEncoderReranker("cross-encoder/this-model-does-not-exist", "0" * 40)
    with pytest.raises(RuntimeError, match="not in the local HuggingFace cache"):
        r("q", ["a"])


@pytest.mark.model
@pytest.mark.slow
def test_the_pinned_model_scores_a_relevant_passage_above_an_irrelevant_one():
    r = CrossEncoderReranker.from_config(cfg)
    good, bad = r("What dataset do they use?", ["We evaluate on the SQuAD dataset.", "The weather is nice today."])
    assert good > bad
    assert [round(s, 4) for s in r("What dataset do they use?", ["We evaluate on the SQuAD dataset."])] == [round(good, 4)]
