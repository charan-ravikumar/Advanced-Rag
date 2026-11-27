"""RagService.passages_only: retrieval without any LLM call (the daily-budget view) and the shared retrieval arguments."""

import pytest

from rag_core.config import RagConfig
from rag_core.errors import InvalidQuestion
from rag_core.service import BUDGET_MESSAGE, RETRIEVAL_FAILED_MESSAGE
from tests.unit.rag_core_fakes import FakeRetriever
from tests.unit.test_rag_core_service import Q, make


def test_passages_only_returns_the_retrieved_passages_with_a_typed_notice_and_makes_no_llm_call():
    svc, retriever, a, b, cache, _ = make()
    r = svc.passages_only(Q, doc_ids=["1234.00001"])
    assert r.degraded and r.answer == BUDGET_MESSAGE and [e.code for e in r.errors] == ["daily_budget_reached"] and r.errors[0].recoverable is False
    assert [s.chunk_id for s in r.retrieved_sources] == [f"c{i}" for i in range(1, 6)] and [s.label for s in r.retrieved_sources] == ["S1", "S2", "S3", "S4", "S5"]
    assert a.calls == [] and b.calls == [] and len(cache) == 0 and r.used_sources == [] and r.provider is None and not r.cached
    assert r.latency_ms.retrieve is not None and r.latency_ms.total >= r.latency_ms.retrieve and retriever.calls[0]["doc_ids"] == ["1234.00001"]


def test_passages_only_uses_the_same_retrieval_arguments_as_the_answer_pipeline():
    svc, retriever, *_ = make()
    svc.passages_only(Q, doc_ids=["p1"], rerank=True)
    list(svc.answer_stream(Q, doc_ids=["p1"], rerank=True))
    first, second = retriever.calls
    assert {k: v for k, v in first.items() if k != "query"} == {k: v for k, v in second.items() if k != "query"}
    assert first["rerank"] is True and first["top_k"] == RagConfig().top_k and first["rrf_k"] == 60


def test_passages_only_rejects_a_bad_question_like_the_other_entry_points():
    svc, retriever, *_ = make()
    for bad in ("", None, "x" * 501):
        with pytest.raises(InvalidQuestion):
            svc.passages_only(bad)
    assert retriever.calls == []


def test_a_retrieval_failure_is_a_typed_error_and_never_a_crash():
    svc, *_ = make(retriever=FakeRetriever(error=RuntimeError("index file locked: secret-path")))
    r = svc.passages_only(Q)
    assert r.degraded and r.answer == RETRIEVAL_FAILED_MESSAGE and [e.code for e in r.errors] == ["retrieval_failed"] and r.retrieved_sources == []
    assert "secret-path" not in r.model_dump_json()


def test_a_custom_code_and_message_can_be_given():
    svc, *_ = make()
    r = svc.passages_only(Q, code="session_cap", message="custom")
    assert r.answer == "custom" and r.errors[0].code == "session_cap"
