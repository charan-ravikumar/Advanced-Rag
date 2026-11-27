"""eval/gen_arms.py (arm contexts), eval/quota_plan.py (wall-time arithmetic and pacing) and RagService.answer_long. Hermetic: no network, no keys."""

import math
from types import SimpleNamespace

import pytest

from eval.gen_arms import (MAX_ORACLE_CHUNKS, ContextHit, FixedContextRetriever, context_hash, gold_paragraph_texts, oracle_context,
                           title_abstract_context, usable_references)
from eval.quota_plan import Limits, Pacer, plan_wall_time, scale_to_budget
from rag.eval_metrics import Paragraph


def para(section, si, pi, text):
    return Paragraph(section, si, pi, 0, len(text), text)


PARAS = [para("Abstract", 0, 0, "ABSTRACT TEXT"), para("Intro", 1, 0, "intro one"), para("Intro", 1, 1, "intro two"), para("Methods", 2, 0, "methods one"),
         para("Results", 3, 0, "results one"), para("Results", 3, 1, "results two"), para("Results", 3, 2, "results three"), para("Dup", 4, 0, "intro one")]


def ref(answer="a", typ="extractive", evidence=(), lost=0):
    return {"answer": answer, "type": typ, "mixed_types": False, "evidence": list(evidence), "n_evidence_not_retrievable": lost}


# ---- usable references (table-only references are skipped) ------------------------------------------------------------------------

def test_a_table_only_reference_is_not_usable_but_a_reference_with_text_evidence_or_no_evidence_at_all_is():
    table_only = ref(evidence=[], lost=2)
    mixed = ref(evidence=["intro one"], lost=1)
    no_evidence_none_lost = ref(typ="yes_no", answer="yes")
    assert usable_references([table_only]) == []
    assert usable_references([table_only, mixed, no_evidence_none_lost]) == [mixed, no_evidence_none_lost]


def test_unanswerable_references_and_references_without_an_answer_are_never_usable():
    assert usable_references([ref(typ="unanswerable", answer=None), ref(answer=None)]) == [] and usable_references([]) == [] and usable_references(None) == []


# ---- oracle context ---------------------------------------------------------------------------------------------------------------

def test_the_oracle_is_the_union_of_evidence_over_annotators_in_document_order_not_reference_order():
    refs = [ref(evidence=["results two", "intro two"]), ref(evidence=["methods one", "intro two"])]
    oc = oracle_context("p1", refs, PARAS)
    assert [h.text for h in oc.hits] == ["intro two", "methods one", "results two"]
    assert [h.rank for h in oc.hits] == [1, 2, 3] and oc.n_gold_paragraphs == 3 and oc.n_dropped_over_cap == 0 and oc.n_not_found == 0
    assert [h.chunk_id for h in oc.hits] == ["oracle:p1:1:1", "oracle:p1:2:0", "oracle:p1:3:1"] and {h.doc_id for h in oc.hits} == {"p1"}
    assert [h.section for h in oc.hits] == ["Intro", "Methods", "Results"]


def test_the_oracle_is_capped_at_five_paragraphs_keeping_the_first_in_document_order_and_counts_the_rest():
    refs = [ref(evidence=["results three", "results two", "results one", "methods one", "intro two", "intro one"])]
    oc = oracle_context("p1", refs, PARAS)
    assert MAX_ORACLE_CHUNKS == 5 and len(oc.hits) == 5 and oc.n_gold_paragraphs == 6 and oc.n_dropped_over_cap == 1
    assert [h.text for h in oc.hits] == ["intro one", "intro two", "methods one", "results one", "results two"]  # 'results three' is the one dropped


def test_a_text_that_occurs_twice_is_taken_once_at_its_first_occurrence():
    oc = oracle_context("p1", [ref(evidence=["intro one"])], PARAS)
    assert [h.chunk_id for h in oc.hits] == ["oracle:p1:1:0"]


def test_unanswerable_references_contribute_no_oracle_evidence_and_unknown_evidence_is_counted_not_invented():
    refs = [ref(typ="unanswerable", answer=None, evidence=["intro two"]), ref(evidence=["text that is not in the paper"])]
    oc = oracle_context("p1", refs, PARAS)
    assert oc.hits == [] and oc.n_not_found == 1 and oc.n_gold_paragraphs == 0
    assert gold_paragraph_texts(refs) == ["text that is not in the paper"]


def test_the_oracle_context_never_contains_a_retrieved_chunk_only_gold_paragraphs():
    oc = oracle_context("p1", [ref(evidence=["methods one"])], PARAS)
    assert all(h.chunk_id.startswith("oracle:") for h in oc.hits)


# ---- title and abstract -----------------------------------------------------------------------------------------------------------

def test_the_title_arm_is_one_source_with_the_title_and_the_abstract_and_nothing_from_the_body():
    (hit,) = title_abstract_context("p1", "A Title", "An abstract.")
    assert hit.text == "Title: A Title\n\nAbstract: An abstract." and hit.chunk_id == "title_abstract:p1" and hit.rank == 1 and hit.section == "Title and abstract"
    assert title_abstract_context("p1", "", "") == [] and title_abstract_context("p1", "Only title", "")[0].text.startswith("Title: Only title")


# ---- fixed-context retriever ------------------------------------------------------------------------------------------------------

def test_the_fixed_retriever_returns_exactly_the_context_it_was_given_and_honours_top_k():
    hits = [ContextHit(f"c{i}", "p", "S", f"t{i}", i) for i in range(1, 8)]
    r = FixedContextRetriever("oracle")
    r.set(hits)
    assert [h.chunk_id for h in r.retrieve("q", top_k=5, doc_ids=["p"]).hits] == ["c1", "c2", "c3", "c4", "c5"]
    assert len(r.retrieve("q").hits) == 7 and len(r.calls) == 2 and r.embedder.revision == "fixed-context:oracle"
    r.set([])
    assert r.retrieve("q", top_k=5).hits == []


def test_the_fixed_retriever_has_no_index_so_it_cannot_return_anything_else():
    r = FixedContextRetriever("title_abstract")
    assert not hasattr(r, "store") and not hasattr(r, "lexical") and r.retrieve("any question").hits == []


def test_context_hash_depends_on_ids_texts_and_order_only():
    a, b = ContextHit("c1", "p", "S", "one", 1), ContextHit("c2", "p", "S", "two", 2)
    assert context_hash([a, b]) == context_hash([ContextHit("c1", "other", "X", "one", 9), ContextHit("c2", "p", "S", "two", 2)])
    assert context_hash([a, b]) != context_hash([b, a]) != context_hash([a]) and context_hash([a]) != context_hash([ContextHit("c1", "p", "S", "ONE", 1)])


def test_an_oracle_service_feeds_the_generator_only_the_gold_paragraphs():
    from rag_core.cache import InMemoryTTLCache  # noqa: F401
    from rag_core.config import RagConfig
    from rag_core.gateway import LLMGateway
    from rag_core.service import RagService
    from tests.unit.rag_core_fakes import FakeClock, ScriptedProvider, answer_json, ok

    clock = FakeClock()
    provider = ScriptedProvider("groq", "m", [ok(answer_json(("Because of the method.", ["S1"])))])
    gateway = LLMGateway([provider], sleep=clock.sleep, clock=clock, rng=lambda: 1.0)
    fixed = FixedContextRetriever("oracle")
    oc = oracle_context("p1", [ref(evidence=["methods one", "intro two"])], PARAS)
    fixed.set(oc.hits)
    result = RagService(fixed, gateway, None, RagConfig()).answer_long("What?", doc_ids=["p1"])
    assert [s.chunk_id for s in result.used_sources] == ["oracle:p1:1:1", "oracle:p1:2:0"] and result.citations == ["S1"] and result.mode == "long"
    sent = provider.calls[0]["messages"][1]["content"]
    assert "intro two" in sent and "methods one" in sent and "results" not in sent and "ABSTRACT TEXT" not in sent


# ---- RagService.answer_long -------------------------------------------------------------------------------------------------------

def test_answer_long_returns_the_final_result_untraced_by_default_and_never_raises_for_provider_failures():
    from rag_core.config import RagConfig, TracingConfig
    from rag_core.errors import InvalidQuestion
    from rag_core.tracing import FakeTracer
    from tests.unit.rag_core_fakes import fail, ok
    from tests.unit.test_rag_core_service import make

    tracer = FakeTracer()
    svc, retriever, a, *_ = make(tracer=tracer, tracing=TracingConfig(enabled=True))
    r = svc.answer_long("What does the attention mechanism compute?", doc_ids=["1234.00001"])
    assert r.mode == "long" and r.citations == ["S1"] and tracer.traces == [] and retriever.calls[0]["doc_ids"] == ["1234.00001"]
    traced, *_ = make(tracer=tracer, tracing=TracingConfig(enabled=True, trace_evaluation=True))
    assert traced.answer_long("What does the attention mechanism compute?").trace_id == tracer.traces[0].trace_id
    down, *_ = make([[fail("server", 500)]], [[fail("server", 500)]])
    assert down.answer_long("What does the attention mechanism compute?").degraded
    with pytest.raises(InvalidQuestion):
        svc.answer_long("")


# ---- quota arithmetic -------------------------------------------------------------------------------------------------------------

LIM = Limits(rpm=30, rpd=1000, tpm=8000, tpd=200000)


def test_each_limit_gives_its_own_lower_bound_and_the_largest_binds():
    p = plan_wall_time({"a": (100, 2000.0)}, LIM)  # 100 calls, 200,000 tokens
    assert p["calls"] == 100 and p["tokens"] == 200000.0
    assert p["hours_by_limit"]["tpd"] == pytest.approx(24.0) and p["hours_by_limit"]["rpd"] == pytest.approx(2.4)
    assert p["hours_by_limit"]["tpm"] == pytest.approx(200000 / 8000 / 60) and p["hours_by_limit"]["rpm"] == pytest.approx(100 / 30 / 60)
    assert p["binding_limit"] == "tpd" and p["wall_hours"] == pytest.approx(24.0) and p["days_with_resets"] == 1


def test_requests_can_be_the_binding_limit_and_days_with_resets_counts_daily_allowances():
    p = plan_wall_time({"a": (2500, 10.0)}, LIM)
    assert p["binding_limit"] == "rpd" and p["days_with_resets"] == 3
    q = plan_wall_time({"a": (500, 1000.0)}, LIM)  # 500,000 tokens
    assert q["binding_limit"] == "tpd" and q["days_with_resets"] == 3 and q["wall_hours"] == pytest.approx(60.0)


def test_unknown_limits_are_left_out_and_an_empty_plan_is_zero():
    p = plan_wall_time({"a": (10, 100.0)}, Limits(rpm=5))
    assert list(p["hours_by_limit"]) == ["rpm"] and p["binding_limit"] == "rpm"
    e = plan_wall_time({}, LIM)
    assert e["calls"] == 0 and e["wall_hours"] == 0.0 and e["days_with_resets"] == 1
    assert plan_wall_time({"a": (1, 1.0)}, Limits()) ["binding_limit"] is None
    with pytest.raises(ValueError):
        plan_wall_time({"a": (-1, 1.0)}, LIM)


def test_scale_to_budget_gives_the_fraction_of_the_plan_that_fits_and_never_more_than_all_of_it():
    items = {"a": (100, 4000.0)}  # 400,000 tokens = 48 h by the daily limit
    assert scale_to_budget(items, LIM, 24.0) == pytest.approx(0.5)
    assert scale_to_budget({"a": (10, 100.0)}, LIM, 24.0) == 1.0 and scale_to_budget({}, LIM, 24.0) == 0.0


# ---- pacing -----------------------------------------------------------------------------------------------------------------------

class Clock:
    def __init__(self):
        self.t, self.slept = 0.0, []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def test_the_pacer_does_not_wait_while_the_window_has_room_and_waits_for_the_oldest_event_to_expire_when_it_does_not():
    c = Clock()
    p = Pacer(rpm=None, tpm=8000, clock=c, sleep=c.sleep)
    assert p.wait(2000) == 0.0
    p.record(3500)
    assert p.wait(3000) == 0.0  # 3500 + 3000 <= 7200 (0.9 of 8000)
    p.record(3000)
    waited = p.wait(2000)  # 6500 + 2000 > 7200: must wait for the first event (at t=0) to leave the 60 s window
    assert waited == pytest.approx(60.05, abs=0.2) and c.t == pytest.approx(60.05, abs=0.2)


def test_the_pacer_limits_requests_per_minute_and_forgets_events_older_than_a_minute():
    c = Clock()
    p = Pacer(rpm=5, tpm=None, safety=1.0, clock=c, sleep=c.sleep)
    for _ in range(5):
        assert p.wait() == 0.0
        p.record(1)
    assert p.wait() > 0  # the sixth request in the same minute waits
    for _ in range(5):
        p.record(1)
    c.t += 61
    assert p.wait() == 0.0  # everything has left the window


def test_a_single_call_larger_than_the_limit_is_not_deadlocked():
    c = Clock()
    p = Pacer(rpm=None, tpm=1000, clock=c, sleep=c.sleep)
    assert p.wait(5000) == 0.0 and c.slept == []
