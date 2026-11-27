"""rag_core.cache: the key carries every ingredient, TTL, LRU, copies, and what is allowed in."""

import pytest

from rag_core.cache import (
    InMemoryTTLCache,
    config_hash,
    is_cacheable,
    make_cache_key,
    normalise_question,
)
from rag_core.schemas import AnswerResult, ErrorInfo, Source
from tests.unit.rag_core_fakes import FakeClock

BASE = dict(question="What is attention?", doc_ids=None, retrieval_config_hash="cfg1", index_version="7255:abc", embedder_revision="rev1",
            prompt="rag-answer-long|3|sha-aaa", model_id="model-a", top_k=5, mode="long")


def key(**over) -> str:
    return make_cache_key(**{**BASE, **over})


def good(**over) -> AnswerResult:
    base = dict(answer="Attention weighs tokens [S1].", citations=["S1"], used_sources=[Source(chunk_id="c1", doc_id="d", text="t", rank=1, score=1.0, label="S1")],
                provider="groq", model="model-a", finish_reason="stop", mode="long")
    return AnswerResult(**{**base, **over})


# ---- key ingredients ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("field,value", [
    ("question", "What is self-attention?"), ("doc_ids", ["1234.00001"]), ("retrieval_config_hash", "cfg2"), ("index_version", "7256:abc"),
    ("index_version", "7255:def"), ("embedder_revision", "rev2"), ("prompt", "rag-answer-long|4|sha-aaa"), ("prompt", "rag-answer-long|3|sha-bbb"),
    ("prompt", "rag-answer-short|3|sha-aaa"), ("model_id", "model-b"), ("top_k", 6), ("mode", "short"),
])
def test_changing_any_key_ingredient_changes_the_key(field, value):
    assert key(**{field: value}) != key()


def test_the_same_ingredients_give_the_same_key_and_a_question_is_normalised():
    assert key() == key()
    assert key(question="  what   IS attention? ") == key()
    assert key(question="What is attention?") == key(question="what is attention?")
    assert normalise_question("Ｗhat is  ATTENTION?") == "what is attention?"  # NFKC folds the full-width letter


def test_the_document_filter_is_order_and_duplicate_insensitive_but_none_differs_from_empty():
    assert key(doc_ids=["b", "a"]) == key(doc_ids=["a", "b", "a"])
    assert key(doc_ids=None) != key(doc_ids=[])
    assert key(doc_ids=["a"]) != key(doc_ids=["a", "b"])


def test_config_hash_depends_on_every_setting():
    base = {"mode": "hybrid", "rrf_k": 60, "top_k": 5}
    assert config_hash(base) == config_hash(dict(reversed(list(base.items()))))
    assert config_hash(base) != config_hash({**base, "rrf_k": 20}) and config_hash(base) != config_hash({**base, "rerank": True})


# ---- TTL and LRU ------------------------------------------------------------------------------------------------------

def test_an_entry_expires_after_the_ttl():
    clock = FakeClock()
    cache = InMemoryTTLCache(max_entries=10, ttl_s=60.0, clock=clock)
    cache.set("k", good())
    clock.advance(59.0)
    assert cache.get("k") is not None
    clock.advance(1.0)
    assert cache.get("k") is None and len(cache) == 0


def test_the_cache_is_size_bounded_and_evicts_the_least_recently_used_entry():
    cache = InMemoryTTLCache(max_entries=2, ttl_s=1000.0, clock=FakeClock())
    cache.set("a", good()), cache.set("b", good())
    cache.get("a")  # a is now the most recently used
    cache.set("c", good())
    assert cache.get("b") is None and cache.get("a") is not None and cache.get("c") is not None and len(cache) == 2


def test_values_are_copied_in_and_out():
    cache = InMemoryTTLCache(clock=FakeClock())
    original = good()
    cache.set("k", original)
    original.answer = "mutated after set"
    got = cache.get("k")
    assert got.answer == "Attention weighs tokens [S1]."
    got.citations.append("S99")
    assert cache.get("k").citations == ["S1"]


def test_hit_and_miss_counters_and_clear():
    cache = InMemoryTTLCache(clock=FakeClock())
    cache.set("k", good())
    cache.get("k"), cache.get("zzz")
    assert (cache.hits, cache.misses) == (1, 1)
    cache.clear()
    assert len(cache) == 0
    with pytest.raises(ValueError):
        InMemoryTTLCache(max_entries=0)


# ---- what may be cached -------------------------------------------------------------------------------------------------

def test_a_successful_validated_answer_is_cacheable():
    assert is_cacheable(good(), expected_model="model-a")
    assert is_cacheable(good(mode="short", citations=[], answer="yes"), expected_model="model-a")


@pytest.mark.parametrize("name,over", [
    ("error", dict(errors=[ErrorInfo(code="internal", message="x")])),
    ("degraded", dict(degraded=True)),
    ("partial (mid-stream failure)", dict(partial=True, finish_reason="error")),
    ("abstention", dict(abstained=True)),
    ("invalid citation", dict(invalid_citations=["S9"])),
    ("no citation in a long answer", dict(citations=[])),
    ("truncated by the token limit", dict(finish_reason="length")),
    ("unknown finish reason", dict(finish_reason=None)),
    ("empty answer", dict(answer="   ")),
    ("no sources were used", dict(used_sources=[])),
    ("small talk", dict(mode="direct")),
])
def test_these_results_are_never_cacheable(name, over):
    assert not is_cacheable(good(**over), expected_model="model-a"), name


def test_an_answer_from_a_fallback_model_is_not_cached_under_the_primary_models_key():
    assert not is_cacheable(good(model="fallback-model"), expected_model="model-a")
    assert is_cacheable(good(model="fallback-model"))  # without an expectation the check is skipped (unit use only)
