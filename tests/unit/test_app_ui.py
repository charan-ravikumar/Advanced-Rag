"""The Streamlit app under ``streamlit.testing.v1.AppTest``: every tab, every chat state, the gate, the guards, feedback, the self-check screen, the numbers rendered, and the import rules.
Fakes everywhere: a ready ``Resources`` (fake retriever, scripted provider, fake tracer, fake reader) is injected through ``st.session_state``. No network, no keys, no real index."""

import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import app.bootstrap as bs
from app import ui_observability
from app.artifacts import Artifacts
from app.bootstrap import Check, Resources, SelfCheck
from rag_core.scores import FeedbackResult, FeedbackStore, record_feedback
from rag_core.service import BUDGET_MESSAGE, DEGRADED_MESSAGE, DIRECT_REPLY
from tests.unit.app_fakes import ENTRY, EXAMPLES, PAPER_ID, PAPERS, ROOT, FakeReader, all_text, ask, fixture_artifacts, make_resources, real_artifacts, run_app
from tests.unit.rag_core_fakes import ABSTAIN_JSON, FakeLangfuse, answer_json, default_quote, fail, ok

Q = "What does the attention mechanism compute?"
Q2 = "How are the heads combined in the model?"
Q3 = "What dataset do the authors evaluate on?"


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    """No real .env, no access code from the environment, empty process-wide caches before and after every test."""
    calls = []
    monkeypatch.setattr(bs, "load_env", lambda *a, **k: calls.append(1) or [])
    monkeypatch.delenv("APP_ACCESS_CODE", raising=False)
    st.cache_data.clear()
    st.cache_resource.clear()
    ui_observability._cached.cache_clear()
    yield calls
    st.cache_data.clear()
    st.cache_resource.clear()
    ui_observability._cached.cache_clear()


def clean(at: AppTest) -> AppTest:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    return at


def texts(elements) -> list[str]:
    return [str(e.value) for e in elements]


def provider_calls(res: Resources) -> int:
    return len(res._fake["primary"].calls)


# ---- every tab renders in every data state ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("artifacts", [Artifacts(), fixture_artifacts(), real_artifacts()], ids=["no-artifacts", "papers-only", "real-artifacts"])
@pytest.mark.parametrize("keys,reader_status", [(True, "ok"), (True, "no_data"), (True, "unavailable"), (False, "no_keys")], ids=["live", "no-data", "unavailable", "no-keys"])
def test_every_tab_renders_without_an_error_in_every_state(artifacts, keys, reader_status):
    at = run_app(make_resources(artifacts=artifacts, keys=keys, reader=FakeReader(reader_status)))
    clean(at)
    assert [t.label for t in at.tabs] == ["Chat", "Observability", "Evaluation", "About"]
    text = all_text(at)
    assert "What the data does NOT show" in text and "Architecture" in text and "System check" in text and "Observability" in text
    assert "masked" in text  # the observability note about masked traces
    if not artifacts.has_results:
        assert "No evaluation results are available" in text


def test_with_no_papers_artifact_the_chat_says_so_and_still_answers_across_all_papers():
    res = make_resources(artifacts=Artifacts())
    at = run_app(res)
    assert any("list of papers is not available" in w for w in texts(at.warning)) and [r.options for r in at.radio][0][1] == "All papers"
    at.radio[0].set_value("All papers").run()
    clean(ask(at, Q))
    assert res._fake["retriever"].calls[-1]["doc_ids"] is None


def test_observability_with_no_keys_defaults_to_labelled_sample_data_and_makes_no_metrics_request():
    reader = FakeReader("no_keys")
    at = clean(run_app(make_resources(keys=False, reader=reader)))
    assert at.toggle(key="obs_sample").value is True and reader.calls == 0
    text = all_text(at)
    assert "SAMPLE DATA" in text and "fake" in text
    at.toggle(key="obs_sample").set_value(False).run()
    clean(at)
    assert reader.calls == 1 and any("no Langfuse keys" in c for c in texts(at.caption)) and "SAMPLE DATA" not in " ".join(texts(at.warning))


@pytest.mark.parametrize("status,expected", [("no_data", "no data in the window"), ("unavailable", "unavailable"), ("no_keys", "no Langfuse keys")])
def test_every_panel_shows_its_status_and_never_an_exception(status, expected):
    at = clean(run_app(make_resources(reader=FakeReader(status))))
    at.toggle(key="obs_sample").set_value(False).run() if at.toggle(key="obs_sample").value else None
    clean(at)
    statuses = [c for c in texts(at.caption) if c.startswith("Status:")]
    assert len(statuses) == 9 and all(expected in c for c in statuses)


def test_live_panels_show_data_and_status_ok_with_an_as_of_time():
    at = clean(run_app(make_resources(reader=FakeReader("ok"))))
    assert at.toggle(key="obs_sample").value is False
    statuses = [c for c in texts(at.caption) if c.startswith("Status:")]
    assert len(statuses) == 9 and all(c.startswith("Status: ok · as of ") for c in statuses)
    frames = {tuple(df.value.columns) for df in at.dataframe}
    assert ("Stage", "p50 (ms)", "p95 (ms)", "p99 (ms)") in frames and ("Prompt", "Version", "Generations", "p95 latency (ms)") in frames
    assert any(m.label == "Hit rate" and m.value == "25.0%" for m in at.metric)  # from the fake Metrics API rows (3 hits, 9 misses)


def test_the_cost_panel_says_prices_are_not_configured_and_shows_no_cost_number_until_they_are():
    at = clean(run_app(make_resources(reader=FakeReader("ok"), prices=False)))
    assert any("Prices not configured" in i for i in texts(at.info)) and not any(m.label.startswith("Total cost") for m in at.metric)
    st.cache_data.clear()
    at = clean(run_app(make_resources(reader=FakeReader("ok"), prices=True)))
    assert not any("Prices not configured" in i for i in texts(at.info)) and any(m.label.startswith("Total cost") for m in at.metric)


def test_one_snapshot_is_shared_by_all_sessions_and_reruns_within_the_ttl():
    reader = FakeReader("ok")
    res = make_resources(reader=reader)
    a = clean(run_app(res))
    a.run()
    b = clean(run_app(make_resources(reader=reader)))  # another visitor, same process
    assert reader.calls == 1
    st.cache_data.clear()
    clean(run_app(make_resources(reader=reader)))
    assert reader.calls == 2  # the cache was emptied: a new read


# ---- chat: the happy path and the citation rendering --------------------------------------------------------------------------------------

def test_chat_happy_path_claims_render_with_bracket_markers_and_their_quote_on_the_source_card():
    res = make_resources()
    at = clean(ask(run_app(res), Q))
    md = texts(at.markdown)
    assert "Attention weighs the tokens. **[S1]**" in md  # rendered by the code from the structured claims
    assert not any('"status"' in m or '"claims"' in m for m in md)  # the JSON is never shown
    cards = [m for m in md if m.startswith("**[S") and "rank" in m]
    assert len(cards) == 5 and cards[0].startswith("**[S1]**") and "cited in the answer" in cards[0] and not any("cited in the answer" in c for c in cards[1:])
    assert [e.label for e in at.expander].count("Passage") == 5 and "Details of this answer" in [e.label for e in at.expander]
    captions = texts(at.caption)
    assert any("Citations in brackets match the source cards" in c and "1 of 1 quotes were found word for word" in c for c in captions)
    assert f"Quoted in the answer: \u201c{default_quote('S1')}\u201d (found word for word in this passage)" in captions
    turn = at.session_state["turn"]
    assert turn.kind == "answer" and turn.result.citations == ["S1"] and turn.question == Q


def test_source_cards_show_the_paper_title_section_rank_and_a_collapsed_passage():
    res = make_resources(hits=[__import__("tests.unit.rag_core_fakes", fromlist=["make_hit"]).make_hit(i, doc_id=PAPER_ID, text=f"Passage {i} costs $5.", section="Methods") for i in range(1, 4)])
    at = clean(ask(run_app(res), Q))
    cards = [m for m in texts(at.markdown) if m.startswith("**[S")]
    assert cards[0].startswith(f"**[S1]**  {PAPERS[0]['title']}  ·  *Methods*  ·  rank 1")
    assert "Passage 1 costs \\$5." in texts(at.markdown)  # dollar signs escaped so they do not start LaTeX
    assert all(not e.proto.expanded for e in at.expander)  # every passage and the details start collapsed


def test_an_invalid_citation_marker_is_left_out_of_the_text_and_flagged_quietly():
    res = make_resources(primary=[ok(answer_json(("Attention.", ["S1"]), ("And more.", ["S9"])))])
    at = clean(ask(run_app(res), Q))
    assert any(m.startswith("Attention. **[S1]** And more.") and "S9" not in m for m in texts(at.markdown))
    assert any("S9" in c and "left out" in c for c in texts(at.caption))


def test_a_quote_that_is_not_in_its_passage_is_labelled_not_found():
    res = make_resources(primary=[ok(answer_json(("Attention.", [("S2", "words the passage never used")])))])
    at = clean(ask(run_app(res), Q))
    captions = texts(at.caption)
    assert "Quoted in the answer: \u201cwords the passage never used\u201d (NOT found word for word in this passage)" in captions
    assert any("0 of 1 quotes were found word for word" in c for c in captions)


def test_an_answer_cut_off_by_the_output_limit_says_so():
    reply = answer_json(("First claim.", ["S1"]), ("Second", ["S2"]))
    res = make_resources(primary=[ok(reply[: reply.index("Second") + 6], finish="length")])
    at = clean(ask(run_app(res), Q))
    assert at.session_state["turn"].kind == "partial" and any("output limit" in w for w in texts(at.warning))


def test_each_question_is_independent_and_the_page_says_so():
    res = make_resources()
    at = clean(run_app(res))
    assert any("no conversation memory" in c for c in texts(at.caption))
    at = clean(ask(at, Q))
    at = clean(ask(at, Q2))
    assert at.session_state["turn"].question == Q2 and sum(m == Q2 for m in texts(at.markdown)) == 1 and Q not in texts(at.markdown)


# ---- scope, examples, sidebar ----------------------------------------------------------------------------------------------------------------

FIRST_PAPER = min(PAPERS, key=lambda p: p["title"].lower())["id"]  # the paper list is sorted by title: this is the default selection
OTHER_PAPER = next(p["id"] for p in PAPERS if p["id"] != FIRST_PAPER)


def test_scope_a_single_paper_is_the_default_and_filters_retrieval_all_papers_does_not():
    res = make_resources()
    at = clean(run_app(res))
    assert at.radio[0].value == "A single paper" and at.selectbox(key="paper_id").value == FIRST_PAPER
    clean(ask(at, Q))
    assert res._fake["retriever"].calls[-1]["doc_ids"] == [FIRST_PAPER]
    at.selectbox(key="paper_id").set_value(OTHER_PAPER).run()
    clean(ask(at, Q2))
    assert res._fake["retriever"].calls[-1]["doc_ids"] == [OTHER_PAPER]
    at.radio[0].set_value("All 2 papers").run()
    clean(ask(at, Q3))
    assert res._fake["retriever"].calls[-1]["doc_ids"] is None


def test_the_all_papers_caption_quotes_the_measured_weakness_from_the_artifact():
    arts = real_artifacts()
    at = clean(run_app(make_resources(artifacts=arts)))
    n = len(arts.papers)
    assert at.radio[0].options[1] == f"All {n} papers" and n == 200
    at.radio[0].set_value(f"All {n} papers").run()
    entries = {(e["setting"].split(" ")[0], e["arm"]): e["metrics"] for e in arts.results["setting_comparison"]["entries"]}
    corpus, paper = entries[("corpus-wide", "D")], entries[("paper", "D")]
    caption = next(c for c in texts(at.caption) if c.startswith("Searching all papers is measurably weaker"))
    assert f"{corpus['HitRate@5']['mean']:.0%}" in caption and f"{paper['HitRate@5']['mean']:.0%}" in caption and "DEV split" in caption


def test_the_four_example_questions_come_from_the_artifact_and_asking_one_sets_its_paper_and_answers():
    arts = real_artifacts()
    res = make_resources(artifacts=arts)
    at = clean(run_app(res))
    labels = [b.label for b in at.button if b.key.startswith("example_")]
    assert labels == [ex["question"] for ex in arts.examples] and len(labels) == 4
    at.button(key="example_1").click().run()
    clean(at)
    ex = arts.examples[1]
    assert at.session_state["turn"].question == ex["question"] and res._fake["retriever"].calls[-1]["doc_ids"] == [ex["paper_id"]]
    assert at.selectbox(key="paper_id").value == ex["paper_id"] and any("not representative" in c for c in texts(at.caption))


def test_the_reranker_toggle_is_off_by_default_shows_its_measured_cost_from_the_artifact_and_turns_reranking_on():
    arts = real_artifacts()
    res = make_resources(artifacts=arts)
    at = clean(run_app(res))
    toggle = at.toggle(key="rerank")
    assert toggle.value is False
    facts = next(g for g in arts.results["retrieval_test"]["arms"] if g["arm"] == "G")
    d = next(a for a in arts.results["retrieval_test"]["arms"] if a["arm"] == "D")
    caption = next(c for c in texts(at.sidebar.caption) if c.startswith("Measured on the TEST split"))
    assert f"nDCG@10 {d['metrics']['nDCG@10']['mean']:.3f} without" in caption and f"{facts['metrics']['nDCG@10']['mean']:.3f} with" in caption
    assert "1.2 s" in caption and f"{facts['latency']['rerank_p50_ms'] / 1000:.1f} s" in caption
    clean(ask(at, Q))
    assert res._fake["retriever"].calls[-1]["rerank"] is False
    toggle.set_value(True).run()
    clean(ask(at, Q2))
    assert res._fake["retriever"].calls[-1]["rerank"] is True


def test_without_a_cached_reranker_the_toggle_is_disabled_and_says_why():
    at = clean(run_app(make_resources(rerank_available=False)))
    assert at.toggle(key="rerank").proto.disabled and any("not in the local cache" in c for c in texts(at.sidebar.caption))


def test_top_k_has_no_slider_and_the_sidebar_names_the_default():
    res = make_resources()
    at = clean(run_app(res))
    assert len(at.slider) == 0 and any(f"Passages sent to the model per question: {res.service.cfg.top_k}" in c and "design default" in c for c in texts(at.sidebar.caption))


# ---- chat states -----------------------------------------------------------------------------------------------------------------------

def test_state_abstained_explains_and_shows_the_retrieved_passages():
    res = make_resources(primary=[ok(ABSTAIN_JSON)])
    at = clean(ask(run_app(res), Q))
    assert at.session_state["turn"].kind == "abstained" and any("do not contain enough information" in i for i in texts(at.info))
    assert len([m for m in texts(at.markdown) if m.startswith("**[S") and "rank" in m]) == 5 and len(at.feedback) == 1
    assert not any("cited in the answer" in m for m in texts(at.markdown))


def test_state_degraded_shows_a_banner_and_the_passages_only():
    res = make_resources(primary=[[fail("server", 500)]], secondary=[[fail("auth", 401)]])
    at = clean(ask(run_app(res), Q))
    assert at.session_state["turn"].kind == "degraded" and DEGRADED_MESSAGE in texts(at.warning)
    assert len([m for m in texts(at.markdown) if m.startswith("**[S") and "rank" in m]) == 5 and len(at.feedback) == 0
    assert any("no answer was generated" in c for c in texts(at.caption)) and res.daily.used == 0  # no tokens spent: the budget unit was given back


def test_state_mid_stream_error_keeps_the_partial_text_and_adds_a_notice():
    res = make_resources(primary=[['{"status": "answered", "claims": [{"text": "Attention weighs ', "the tok", fail("connection")]])
    at = clean(ask(run_app(res), Q))
    assert at.session_state["turn"].kind == "partial" and "Attention weighs the tok" in texts(at.markdown)
    assert any("incomplete" in w for w in texts(at.warning)) and len(at.feedback) == 1 and res.daily.used == 1  # the provider did answer partly: the unit is spent


def test_state_invalid_question_is_a_friendly_message_and_costs_nothing():
    res = make_resources()
    at = clean(ask(run_app(res), "   "))
    assert at.session_state["turn"].kind == "invalid" and "Please type a question." in texts(at.warning)
    assert provider_calls(res) == 0 and res._fake["retriever"].calls == [] and res.daily.used == 0 and at.session_state["__session_guard__"].total == 0 if "__session_guard__" in at.session_state else True


def test_state_daily_budget_reached_makes_no_llm_call_and_shows_the_passages():
    res = make_resources(daily_cap=1)
    at = clean(ask(run_app(res), Q))
    assert provider_calls(res) == 1 and res.daily.remaining == 0
    at = clean(ask(at, Q2))
    assert provider_calls(res) == 1  # still one: the second question never reached the model
    assert at.session_state["turn"].kind == "budget" and BUDGET_MESSAGE in texts(at.warning)
    assert len([m for m in texts(at.markdown) if m.startswith("**[S") and "rank" in m]) == 5 and len(at.feedback) == 0 and res._fake["retriever"].calls[-1]["query"] == Q2


def test_state_cached_says_so_and_makes_no_model_call_and_gives_the_budget_unit_back():
    res = make_resources(daily_cap=5)
    at = clean(ask(run_app(res), Q))
    assert not any("answer cache" in c for c in texts(at.caption))
    at = clean(ask(at, Q))
    assert any("Served from the answer cache (no model call)" in c for c in texts(at.caption)) and at.session_state["turn"].result.cached
    assert provider_calls(res) == 1 and res.daily.used == 1


def test_small_talk_gets_the_fixed_reply_with_no_sources_and_no_budget_used():
    res = make_resources()
    at = clean(ask(run_app(res), "hello"))
    assert DIRECT_REPLY in texts(at.markdown) and at.session_state["turn"].kind == "direct" and len(at.feedback) == 0 and provider_calls(res) == 0 and res.daily.used == 0


def test_the_details_expander_shows_this_answers_provider_model_effort_prompt_trace_latency_and_tokens():
    res = make_resources()
    at = clean(ask(run_app(res), Q))
    result = at.session_state["turn"].result
    details = next(m for m in texts(at.markdown) if m.startswith("- **Provider / model:**"))
    assert "groq / model-a" in details and "Prompt: rag-answer-long" in details.replace("**", "") and f"Trace id:** {result.trace_id}" in details and "fake-trace-1" in details
    assert "Tokens:** prompt 11 · completion 7" in details and "retrieval" in details and "first token" in details and "Served from the answer cache:** no" in details


# ---- the guards inside the page ----------------------------------------------------------------------------------------------------------

def test_the_per_session_rate_limit_stops_the_third_quick_question_before_any_retrieval_or_model_call():
    res = make_resources(rate=2)
    at = clean(ask(run_app(res), Q))
    at = clean(ask(at, Q2))
    at = clean(ask(at, Q3))
    assert at.session_state["turn"].kind == "refused" and any(i.startswith("Please wait") for i in texts(at.info))
    assert provider_calls(res) == 2 and len(res._fake["retriever"].calls) == 2


def test_the_per_session_cap_ends_the_session_after_its_questions():
    res = make_resources(session_cap=2)
    at = clean(ask(run_app(res), Q))
    at = clean(ask(at, Q2))
    at = clean(ask(at, Q3))
    assert any("reached its limit of 2 questions" in i for i in texts(at.info)) and provider_calls(res) == 2
    assert any("Questions left in this session: 0" in c for c in texts(at.sidebar.caption))


def test_the_daily_cap_is_shared_by_all_sessions_of_the_process():
    res = make_resources(daily_cap=2)
    a = clean(ask(run_app(res), Q))
    b = clean(ask(run_app(res), Q2))  # another browser session, same process-wide resources
    c = clean(ask(run_app(res), Q3))
    assert provider_calls(res) == 2 and a.session_state["turn"].kind == "answer" and b.session_state["turn"].kind == "answer" and c.session_state["turn"].kind == "budget"


def test_a_question_the_guards_refuse_is_not_counted_against_the_daily_budget():
    res = make_resources(daily_cap=3, rate=1)
    at = clean(ask(run_app(res), Q))
    clean(ask(at, Q2))  # refused by the session rate limit
    assert res.daily.used == 1


# ---- feedback ----------------------------------------------------------------------------------------------------------------------------

def test_a_thumb_sends_feedback_for_this_answers_trace_with_a_masked_comment(tmp_path):
    client = FakeLangfuse()
    local = FeedbackStore(tmp_path / "fb.sqlite3")
    res = make_resources(feedback=lambda request_id, trace_id, value, comment=None: record_feedback(local, request_id, trace_id, value, comment,
                                                                                                    client_factory=lambda: client))
    at = clean(ask(run_app(res), Q))
    trace_id = at.session_state["turn"].result.trace_id
    assert trace_id == "fake-trace-1"
    comment = next(t for t in at.text_input if t.label.startswith("Optional comment"))
    comment.set_value("not helpful, write to bob@corp.example.com or see https://secret.example.org/x").run()
    at.feedback[0].set_value(1).run()
    clean(at)
    (score,) = client.scores
    assert score["trace_id"] == trace_id and score["name"] == "user_feedback" and score["value"] == 1.0 and score["data_type"] == "BOOLEAN"
    assert "[EMAIL]" in score["comment"] and "[URL]" in score["comment"] and "bob@" not in score["comment"] and "secret.example.org" not in score["comment"]
    assert any("Thanks, your feedback was recorded" in c for c in texts(at.caption))
    at.feedback[0].set_value(0).run()  # changing one's mind updates the same score id
    assert client.scores[-1]["score_id"] == client.scores[0]["score_id"] and client.scores[-1]["value"] == 0.0


def test_without_a_trace_id_the_thumbs_still_work_and_nothing_is_sent_to_langfuse():
    calls = []
    res = make_resources(tracing=False, feedback=lambda request_id, trace_id, value, comment=None: calls.append((request_id, trace_id, value)) or FeedbackResult(True, False))
    at = clean(ask(run_app(res), Q))
    result = at.session_state["turn"].result
    assert result.trace_id is None and not at.feedback[0].proto.disabled
    assert not next(t for t in at.text_input if t.label.startswith("Optional comment")).proto.disabled
    at.feedback[0].set_value(1).run()
    clean(at)
    assert calls == [(result.request_id, None, 1)] and any("was saved" in c and "not traced" in c for c in texts(at.caption))


def test_a_failing_feedback_call_is_reported_quietly_and_never_breaks_the_page():
    res = make_resources(feedback=lambda *a, **k: FeedbackResult(False, False))
    at = clean(ask(run_app(res), Q))
    at.feedback[0].set_value(1).run()
    clean(at)
    assert any("could not be recorded" in c for c in texts(at.caption)) and at.session_state["turn"].kind == "answer"


def test_the_answer_survives_the_rerun_caused_by_a_thumb_without_calling_the_model_again():
    res = make_resources()
    at = clean(ask(run_app(res), Q))
    at.feedback[0].set_value(1).run()
    clean(at)
    assert provider_calls(res) == 1 and "Attention weighs the tokens. **[S1]**" in texts(at.markdown)


# ---- the access-code gate -------------------------------------------------------------------------------------------------------------

def test_without_an_access_code_the_app_runs_open_with_a_demo_mode_note():
    at = clean(run_app(make_resources()))
    assert len(at.tabs) == 4 and any("Demo mode" in c for c in texts(at.caption))


def test_a_wrong_code_blocks_the_right_code_opens_and_the_secret_is_never_rendered(monkeypatch):
    secret = "open-sesame-7731-xyz"
    monkeypatch.setenv("APP_ACCESS_CODE", secret)
    res = make_resources()
    at = clean(run_app(res))
    assert len(at.tabs) == 0 and [t.label for t in at.text_input] == ["Access code"] and secret not in all_text(at)
    # a form value is only committed together with the submit click, so set it and click in the same run
    at.text_input[0].set_value("nope")
    at.button[0].click().run()
    assert texts(at.error) == ["Wrong code."] and len(at.tabs) == 0 and secret not in all_text(at) and provider_calls(res) == 0
    at.text_input[0].set_value(secret)
    at.button[0].click().run()
    clean(at)
    assert len(at.tabs) == 4 and secret not in all_text(at) and not any("Demo mode" in c for c in texts(at.caption))
    assert secret not in json.dumps({k: str(at.session_state[k]) for k in at.session_state}, default=str)  # not kept in the session either
    clean(ask(at, Q))
    assert provider_calls(res) == 1


def test_repeated_wrong_codes_lock_the_session_and_the_right_code_is_refused_while_locked(monkeypatch):
    monkeypatch.setenv("APP_ACCESS_CODE", "the-right-code")
    at = clean(run_app(make_resources()))
    for _ in range(5):
        at.text_input[0].set_value("wrong")
        at.button[0].click().run()
    assert any("Locked for" in e or "Too many wrong codes" in e for e in texts(at.error) + texts(at.warning))
    assert at.text_input[0].proto.disabled and len(at.tabs) == 0


def test_the_access_code_can_come_from_st_secrets(monkeypatch):
    monkeypatch.setenv("APP_ACCESS_CODE", "")  # the monkeypatch restores the original (absent) state afterwards, undoing the copy below
    at = AppTest.from_file(ENTRY, default_timeout=60)
    at.session_state[bs.OVERRIDE_KEY] = make_resources()
    at.secrets["APP_ACCESS_CODE"] = "from-the-secrets-file"
    at.run()
    assert len(at.tabs) == 0 and [t.label for t in at.text_input] == ["Access code"] and "from-the-secrets-file" not in all_text(at)


# ---- self-check screen ----------------------------------------------------------------------------------------------------------------

def failing_resources() -> Resources:
    check = SelfCheck([Check("Index directory", False, "not set: set ADVRAG_INDEX_DIR to the folder of the chroma store"), Check("Gemini API key present", False, "missing", fatal=False)])
    return Resources(settings=make_resources().settings, selfcheck=check, service=None, artifacts=Artifacts())


def test_a_failed_self_check_shows_an_error_screen_with_the_failed_checks_and_no_tabs_and_no_stack_trace():
    at = run_app(failing_resources())
    assert not at.exception and len(at.tabs) == 0
    assert any("cannot start" in e for e in texts(at.error))
    md = texts(at.markdown)
    assert any(m.startswith("❌ **Index directory**") and "ADVRAG_INDEX_DIR" in m for m in md) and any(m.startswith("⚠️ **Gemini API key present**") for m in md)
    assert not any("Traceback" in m for m in md) and any(b.label == "Check again" for b in at.button)


def test_the_real_bootstrap_path_without_an_api_shows_the_error_screen_and_retries_on_every_run(monkeypatch):
    monkeypatch.setenv("ADVRAG_API_URL", "http://127.0.0.1:9")  # nothing listens on the discard port: connection refused, on loopback (the test harness allows only loopback)
    calls = []
    real_build = bs.build_client_resources
    monkeypatch.setattr(bs, "build_client_resources", lambda cfg, *a, **k: calls.append(1) or real_build(cfg, *a, **k))
    at = AppTest.from_file(ENTRY, default_timeout=60).run()
    assert not at.exception and len(at.tabs) == 0 and any(m.startswith("❌ **Answer API reachable**") and "did not answer" in m for m in texts(at.markdown))
    assert not any("Traceback" in m for m in texts(at.markdown))
    at.run()
    assert len(calls) == 2  # a failed bootstrap is not cached: starting the API and pressing the button works


def test_an_unexpected_error_in_one_tab_shows_its_class_name_not_a_stack_trace_and_the_other_tabs_still_render(monkeypatch):
    import app.ui_about as about

    monkeypatch.setattr(about, "render_about", lambda res: (_ for _ in ()).throw(KeyError("internal detail")))
    at = run_app(make_resources())
    assert not at.exception and any("unexpected error (KeyError)" in e for e in texts(at.error)) and not any("internal detail" in e for e in texts(at.error))
    assert any("What the data does NOT show" in m for m in texts(at.markdown))


# ---- the evaluation tab: numbers equal the artifact -----------------------------------------------------------------------------------------

def _frame(at, *columns):
    found = [df.value for df in at.dataframe if set(columns) <= set(df.value.columns)]
    assert found, f"no table with columns {columns}"
    return found[0]


def _ci(entry):
    return f"{entry['mean']:.3f} [{entry['ci_low']:.3f}, {entry['ci_high']:.3f}]"


def test_the_evaluation_tab_renders_exactly_the_numbers_of_the_artifact():
    arts = real_artifacts()
    at = clean(run_app(make_resources(artifacts=arts)))
    results = arts.results
    test = results["retrieval_test"]
    frame = _frame(at, "Arm", "Configuration", "Retrieval latency p50 / p95", "MRR")
    frame = frame[frame["Arm"].isin(["A", "D", "F", "G"])].reset_index(drop=True)  # the TEST table (the DEV table has 15 arms and no intervals)
    assert len(frame) == 4
    for arm, (_, row) in zip(test["arms"], frame.iterrows()):
        assert row["Arm"] == arm["arm"] and row["Configuration"] == arm["description"]
        for metric in ("nDCG@10", "MRR", "HitRate@5", "EvidenceRecall@10", "EvidenceF1@5"):
            assert row[metric] == _ci(arm["metrics"][metric]), (arm["arm"], metric)
    cmp_frame = _frame(at, "Comparison", "Difference (first − second)")
    for c, (_, row) in zip(test["comparisons"], cmp_frame.iterrows()):
        assert row["Comparison"] == c["pair"] and row["Difference (first − second)"] == f"{c['diff']:+.3f}" and row["95% CI (paper-clustered)"] == f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]"
        assert row["Holm-significant"] == ("yes" if c["holm_significant"] else "no") and row["Wins / losses / ties"] == f"{c['wins']} / {c['losses']} / {c['ties']}"
    gen = results["generation"]
    gframe = _frame(at, "Metric", "Value", "Questions")
    f1 = gframe[gframe["Metric"] == "Answer-F1 (short answers)"].iloc[0]
    assert f1["Value"] == _ci(gen["short_mode"]["AnswerF1"]) and f1["Questions"] == gen["short_mode"]["AnswerF1"]["n"]
    pooled = gen["long_mode"]["CitationValidity_pooled"]
    assert f"{pooled['valid']} of {pooled['cited']}" in set(gframe["Value"])


def test_every_table_has_a_scope_caption_with_its_label_run_folder_and_fingerprint():
    arts = real_artifacts()
    at = clean(run_app(make_resources(artifacts=arts)))
    captions = texts(at.caption)
    for key in ("retrieval_test", "retrieval_dev", "setting_comparison", "generation"):
        scope = arts.results[key]["scope"]
        line = next(c for c in captions if f"`{scope['run_folder']}`" in c)
        assert scope["label"] in line and scope["code_fingerprint"][:12] in line and f"{scope['n_questions']} questions" in line
    labels = {arts.results[k]["scope"]["label"] for k in ("retrieval_test", "retrieval_dev", "generation")}
    assert labels == {"TEST", "DEV", "EXPLORATORY"} and any(":blue-background[EXPLORATORY]" in c for c in captions)


def test_the_incomplete_ragas_run_is_reported_as_incomplete_and_shows_no_ragas_number():
    arts = real_artifacts()
    at = clean(run_app(make_resources(artifacts=arts)))
    r = arts.results["ragas"]
    assert r["complete"] is False
    note = next(i for i in texts(at.info) if i.startswith("The RAGAS scoring is incomplete"))
    assert f"{r['n_scored']} of {r['n_scorable']} answers were scored" in note and "uncalibrated" in note
    assert not any("Faithfulness" in df.value.to_csv() for df in at.dataframe)


def test_the_what_the_data_does_not_show_section_is_always_present_and_states_the_limits_plainly():
    at = clean(run_app(make_resources(artifacts=real_artifacts())))
    text = " ".join(texts(at.markdown))
    for phrase in ("No generation result on the TEST split", "No baseline and no noise floor", "uncalibrated", "Answer-F1 is word overlap", "One dataset", "No comparison with other systems"):
        assert phrase in text


def test_the_evaluation_tab_never_runs_anything(monkeypatch):
    import app.ui_evaluation as ev

    source = Path(ev.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {n.module.split(".")[0] if isinstance(n, ast.ImportFrom) and n.module else a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in getattr(n, "names", [])}
    assert not imported & {"subprocess", "eval", "rag", "rag_core", "db", "requests", "httpx", "openai", "groq", "langfuse"}  # only streamlit, pandas, altair and app.artifacts
    res = make_resources(artifacts=real_artifacts())
    clean(run_app(res))
    assert provider_calls(res) == 0 and res._fake["retriever"].calls == [] and res.reader.calls <= 1


# ---- the about tab ------------------------------------------------------------------------------------------------------------------------

def test_the_about_tab_has_the_diagram_decisions_limitations_privacy_attribution_and_versions():
    arts = real_artifacts()
    res = make_resources(artifacts=arts)
    at = clean(run_app(res))
    text = "\n".join(texts(at.markdown))
    assert len(at.get("graphviz_chart")) == 1
    dot = at.get("graphviz_chart")[0].proto.spec
    for node in ("Ingest", "Chunk", "Chroma", "SQLite FTS5", "Retrieve", "Fuse", "Context", "Gemini", "Validated answer", "Langfuse", "Evaluation runs"):
        assert node in dot
    for heading in ("#### Key decisions", "#### Limitations", "#### Privacy", "#### Data attribution", "#### What is running", "#### System check"):
        assert heading in text
    assert "Do not enter private or confidential data" in text and "masked" in text and "Text evidence only" in text and "single-user demo" in text
    assert "Qasper" in text and "CC BY 4.0" in text and "https://allenai.org/data/qasper" in text
    v = res.version_info
    assert v["manifest_sha256"][:16] in text and v["prompt_version"] in text and v["embedder"]["revision"][:12] in text and f"{v['chunk_count']} chunks" in text and st.__version__ in text


def test_the_decisions_quote_the_comparison_numbers_of_the_artifact_including_the_negative_result():
    arts = real_artifacts()
    at = clean(run_app(make_resources(artifacts=arts)))
    text = "\n".join(texts(at.markdown))
    cmp_ = {c["pair"]: c for c in arts.results["retrieval_test"]["comparisons"]}
    fd = cmp_["F vs D"]
    assert "Negative result: semantic chunking did not beat recursive chunking" in text and f"Difference {fd['diff']:+.3f}, 95% interval [{fd['ci_low']:+.3f}, {fd['ci_high']:+.3f}]" in text
    assert f"nDCG@10 difference {cmp_['A vs D']['diff']:+.3f}" in text and f"{-cmp_['D vs G']['diff']:+.3f}" in text


def test_the_system_check_panel_lists_every_check_with_a_status_icon_and_no_key_values():
    res = make_resources(keys=False)
    at = clean(run_app(res))
    md = texts(at.markdown)
    assert any(m.startswith("✅ **Index directory**") for m in md) and any(m.startswith("⚠️ **Gemini API key present**") for m in md)
    assert any("Keys are checked for presence only" in c for c in texts(at.caption))


# ---- import rules ----------------------------------------------------------------------------------------------------------------------------

APP_MODULES = ["guards", "artifacts", "bootstrap", "ui_chat", "ui_observability", "ui_evaluation", "ui_about", "streamlit_app"]
FORBIDDEN = {"openai", "groq", "google", "langfuse", "httpx", "requests", "chromadb", "torch", "sentence_transformers", "ragas", "instructor", "anthropic", "dotenv", "tiktoken"}


@pytest.mark.parametrize("module", APP_MODULES)
def test_app_modules_import_no_sdk_http_client_or_legacy_retrieval_module_only_streamlit_and_the_project(module):
    tree = ast.parse((ROOT / "app" / f"{module}.py").read_text(encoding="utf-8"))
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            names |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            names.add(n.module)
            names |= {f"{n.module}.{a.name}" for a in n.names}
    top = {n.split(".")[0] for n in names}
    assert not top & FORBIDDEN, sorted(top & FORBIDDEN)
    assert not {n for n in names if n == "rag.retrieval" or n.startswith("rag.retrieval.") or n.startswith("eval")}, "the app never imports the deprecated retrieval module or the evaluation runners"


def test_only_bootstrap_touches_the_environment_and_no_app_module_names_load_env_but_it():
    for module in APP_MODULES:
        text = (ROOT / "app" / f"{module}.py").read_text(encoding="utf-8")
        tree = ast.parse(text)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        if module != "bootstrap":
            assert "load_env" not in names and "environ" not in names and "getenv" not in names, module
    assert "load_env" in (ROOT / "app" / "bootstrap.py").read_text(encoding="utf-8")


@pytest.mark.slow
def test_importing_the_app_modules_never_calls_load_env_and_pulls_in_no_sdk_model_or_database():
    code = textwrap.dedent("""
        import sys
        import rag_core.env as env
        calls = []
        env.load_env = lambda *a, **k: calls.append(1) or []
        import app.guards, app.artifacts, app.bootstrap, app.ui_chat, app.ui_observability, app.ui_evaluation, app.ui_about
        bad = sorted(m for m in ("openai", "groq", "google.genai", "langfuse", "httpx", "chromadb", "torch", "sentence_transformers", "ragas", "instructor", "dotenv", "tiktoken", "rag.retrieval", "db.vectordb")
                     if m in sys.modules)
        print(calls, bad)
    """)
    envs = {k: v for k, v in os.environ.items() if not k.startswith(("LANGFUSE_", "GROQ_", "GEMINI_"))}
    envs["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, env=envs, timeout=180)
    assert proc.returncode == 0, proc.stderr[-800:]
    assert proc.stdout.strip().splitlines()[-1] == "[] []"


def test_load_env_runs_in_the_entry_point_run_not_at_import(isolate):
    assert isolate == []  # nothing called it while the modules were imported by this test file
    run_app(make_resources())
    assert isolate != []
