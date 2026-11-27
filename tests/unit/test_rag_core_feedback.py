"""Feedback is persisted locally first, keyed by request id, with the comment masked, and forwarded to Langfuse only when the request has a trace id (N-16).
Real SQLite files under tmp_path, FakeLangfuse: no network, no keys."""

import sqlite3
import threading

import pytest

from rag_core.scores import COMMENT_MAX_CHARS, FeedbackResult, FeedbackStore, record_feedback
from tests.unit.rag_core_fakes import FakeLangfuse
from tests.unit.test_rag_core_service import done_of, events, make


@pytest.fixture
def store(tmp_path):
    return FeedbackStore(tmp_path / "state" / "feedback.sqlite3")


def rows(store):
    with sqlite3.connect(store.path) as conn:
        return conn.execute("SELECT request_id, name, value, comment, trace_id, forwarded FROM feedback ORDER BY request_id, name").fetchall()


# ---- local first -------------------------------------------------------------------------------------------------------------

def test_without_a_trace_id_the_feedback_is_saved_locally_and_nothing_is_sent(store):
    client = FakeLangfuse()
    out = record_feedback(store, "req-1", None, 1, "helpful", client_factory=lambda: client)
    assert out == FeedbackResult(saved=True, forwarded=False)
    assert rows(store) == [("req-1", "user_feedback", 1, "helpful", None, 0)] and client.scores == []


def test_with_a_trace_id_it_is_saved_and_then_forwarded_and_the_row_says_so(store):
    client = FakeLangfuse()
    out = record_feedback(store, "req-1", "trace-9", 0, None, client_factory=lambda: client)
    assert out == FeedbackResult(saved=True, forwarded=True)
    assert rows(store) == [("req-1", "user_feedback", 0, None, "trace-9", 1)]
    assert [(s["trace_id"], s["value"]) for s in client.scores] == [("trace-9", 0.0)]


def test_changing_ones_mind_updates_the_same_row_instead_of_adding_one(store):
    client = FakeLangfuse()
    record_feedback(store, "req-1", None, 1, "first", client_factory=lambda: client)
    record_feedback(store, "req-1", None, 0, "second", client_factory=lambda: client)
    record_feedback(store, "req-2", None, 1, None, client_factory=lambda: client)
    assert rows(store) == [("req-1", "user_feedback", 0, "second", None, 0), ("req-2", "user_feedback", 1, None, None, 0)]


def test_the_comment_is_masked_and_truncated_before_it_is_written_to_disk(store):
    secret = "write to bob@corp.example.com or see https://secret.example.org/x key gsk_abcdefghijklmnop1234 " + "word " * 300
    record_feedback(store, "req-1", None, 0, secret)
    (row,) = rows(store)
    assert "[EMAIL]" in row[3] and "[URL]" in row[3] and len(row[3]) <= COMMENT_MAX_CHARS + 20
    raw = store.path.read_bytes()
    assert b"bob@corp" not in raw and b"secret.example.org" not in raw and b"gsk_abcdefghijklmnop1234" not in raw


# ---- fail open -----------------------------------------------------------------------------------------------------------------

def test_a_failing_langfuse_call_keeps_the_local_copy_unforwarded_and_never_raises(store):
    def boom():
        raise RuntimeError("langfuse is down")

    out = record_feedback(store, "req-1", "trace-9", 1, "x", client_factory=boom)
    assert out == FeedbackResult(saved=True, forwarded=False) and rows(store)[0][5] == 0


def test_an_unusable_local_store_does_not_stop_the_forwarding_and_never_raises(tmp_path):
    broken = FeedbackStore(tmp_path)  # a directory, not a file: sqlite cannot open it
    client = FakeLangfuse()
    out = record_feedback(broken, "req-1", "trace-9", 1, None, client_factory=lambda: client)
    assert out == FeedbackResult(saved=False, forwarded=True) and len(client.scores) == 1
    assert record_feedback(None, "req-1", None, 1) == FeedbackResult(False, False)  # no store, no trace: nothing could be recorded


@pytest.mark.parametrize("value", [7, -1, "up", None, float("nan")])
def test_an_invalid_value_is_neither_saved_nor_sent(store, value):
    client = FakeLangfuse()
    assert record_feedback(store, "req-1", "trace-9", value, "c", client_factory=lambda: client) == FeedbackResult(False, False)
    assert not store.path.exists() and client.scores == []


def test_a_missing_request_id_is_not_saved(store):
    assert record_feedback(store, "", None, 1) == FeedbackResult(False, False)


def test_the_database_is_created_lazily_so_building_the_app_writes_nothing(tmp_path):
    path = tmp_path / "deep" / "feedback.sqlite3"
    FeedbackStore(path)
    assert not path.exists() and not path.parent.exists()


def test_concurrent_saves_from_several_threads_all_land(store):
    errors = []

    def worker(i):
        try:
            assert record_feedback(store, f"req-{i}", None, i % 2, f"c{i}").saved
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == [] and len(rows(store)) == 20


# ---- the request id ----------------------------------------------------------------------------------------------------------

def test_every_answer_has_its_own_request_id_and_a_cache_hit_gets_a_fresh_one():
    svc, *_ = make()
    first = done_of(events(svc))
    second = done_of(events(svc))
    assert not first.cached and second.cached
    assert first.request_id and second.request_id and first.request_id != second.request_id and len(first.request_id) == 32


def test_a_result_without_any_trace_still_has_a_request_id_to_key_feedback_on():
    svc, *_ = make(tracer=None)
    r = done_of(events(svc))
    assert r.trace_id is None and r.request_id


# ---- the Streamlit page --------------------------------------------------------------------------------------------------------

def test_the_thumbs_are_enabled_without_a_trace_id_and_the_vote_is_saved_locally(tmp_path):
    from tests.unit.app_fakes import ask, make_resources, run_app
    from tests.unit.test_app_ui import Q, clean, texts

    local = FeedbackStore(tmp_path / "fb.sqlite3")
    client = FakeLangfuse()
    res = make_resources(tracing=False, feedback=lambda request_id, trace_id, value, comment=None: record_feedback(local, request_id, trace_id, value, comment,
                                                                                                                   client_factory=lambda: client))
    at = clean(ask(run_app(res), Q))
    result = at.session_state["turn"].result
    assert result.trace_id is None and not at.feedback[0].proto.disabled
    assert not next(t for t in at.text_input if t.label.startswith("Optional comment")).proto.disabled
    next(t for t in at.text_input if t.label.startswith("Optional comment")).set_value("mail me at bob@corp.example.com").run()
    at.feedback[0].set_value(1).run()
    clean(at)
    (row,) = rows(local)
    assert row[0] == result.request_id and row[2] == 1 and row[4] is None and row[5] == 0 and "[EMAIL]" in row[3] and "bob@" not in row[3]
    assert client.scores == [] and any("saved" in c for c in texts(at.caption))


def test_with_a_trace_id_the_vote_is_saved_and_forwarded(tmp_path):
    from tests.unit.app_fakes import ask, make_resources, run_app
    from tests.unit.test_app_ui import Q, clean

    local = FeedbackStore(tmp_path / "fb.sqlite3")
    client = FakeLangfuse()
    res = make_resources(feedback=lambda request_id, trace_id, value, comment=None: record_feedback(local, request_id, trace_id, value, comment,
                                                                                                    client_factory=lambda: client))
    at = clean(ask(run_app(res), Q))
    at.feedback[0].set_value(1).run()
    clean(at)
    (row,) = rows(local)
    assert row[4] == "fake-trace-1" and row[5] == 1 and client.scores[0]["trace_id"] == "fake-trace-1"
