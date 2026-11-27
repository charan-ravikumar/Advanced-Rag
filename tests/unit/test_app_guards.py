"""app/guards.py: access code, per-session limits, the process-wide daily cap. Pure and hermetic: injected clocks, no streamlit."""

import threading
from datetime import datetime, timezone

import pytest

from app.guards import AccessGate, AppConfig, DailyBudget, SessionGuard, check_question
from rag_core.errors import InvalidQuestion


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


# ---- config -----------------------------------------------------------------------------------------------------------------------

def test_config_yaml_app_section_matches_the_dataclass_defaults_and_the_documented_daily_cap():
    from config import cfg

    from_yaml = AppConfig.from_config(cfg)
    assert from_yaml == AppConfig() and from_yaml.daily_request_cap == 80 and from_yaml.metrics_ttl_seconds == 3600
    assert AppConfig.from_config(object()).daily_request_cap == 80  # no section: defaults


# ---- access code -------------------------------------------------------------------------------------------------------------------

def test_without_a_secret_the_app_is_open_and_any_code_passes():
    gate = AccessGate(None)
    state = gate.new_state()
    assert not gate.required and gate.is_open(state) and gate.attempt(state, "anything") == (True, "ok")
    assert not AccessGate("   ").required and not AccessGate("").required  # a blank secret is "not set", never "empty code accepted as secret"


def test_the_right_code_opens_the_session_and_a_wrong_one_does_not():
    gate = AccessGate("s3cret-Code")
    state = gate.new_state()
    assert gate.required and not gate.is_open(state)
    assert gate.attempt(state, "wrong") == (False, "Wrong code.") and not gate.is_open(state)
    assert gate.attempt(state, None)[0] is False and gate.attempt(state, "")[0] is False and gate.attempt(state, "s3cret-code")[0] is False  # case matters
    assert gate.attempt(state, "s3cret-Code") == (True, "ok") and gate.is_open(state)


def test_the_secret_is_stripped_like_a_pasted_value_and_unicode_codes_work():
    assert AccessGate(" abc \n").attempt(AccessGate(" abc ").new_state(), "abc")[0] is True
    gate = AccessGate("pässwörd-漢字")
    assert gate.attempt(gate.new_state(), "pässwörd-漢字")[0] is True and gate.attempt(gate.new_state(), "passwoerd")[0] is False


def test_compare_digest_is_used_on_bytes_so_the_comparison_is_constant_time_and_length_safe(monkeypatch):
    seen = []
    import hmac as real

    original = real.compare_digest
    monkeypatch.setattr("app.guards.hmac.compare_digest", lambda a, b: seen.append((type(a), type(b))) or original(a, b))
    gate = AccessGate("abc")
    gate.attempt(gate.new_state(), "abcd")
    assert seen == [(bytes, bytes)]


def test_after_the_allowed_attempts_the_session_is_locked_out_without_looking_at_the_code_and_unlocks_later():
    clock = Clock()
    gate = AccessGate("right", max_attempts=3, lockout_s=60, clock=clock)
    state = gate.new_state()
    assert [gate.attempt(state, "x")[0] for _ in range(3)] == [False, False, False]
    ok, message = gate.attempt(state, "right")  # even the RIGHT code is refused while locked
    assert ok is False and message == "Too many wrong codes. Try again in 61 seconds."
    clock.advance(30)
    assert gate.attempt(state, "right")[0] is False and 0 < gate.seconds_locked(state) <= 30
    clock.advance(31)
    assert gate.attempt(state, "right") == (True, "ok")


def test_the_secret_never_appears_in_repr_or_in_any_message():
    gate = AccessGate("TOPSECRET-123")
    state = gate.new_state()
    messages = [gate.attempt(state, c)[1] for c in ("TOPSECRET-12", "x", "TOPSECRET-123 ")] + [repr(gate), str(gate)]
    assert all("TOPSECRET" not in m and "123" not in m for m in messages)
    assert "TOPSECRET" not in repr(state)  # the per-session state holds only counters and a flag


def test_one_sessions_failures_do_not_lock_another():
    gate = AccessGate("right", max_attempts=2)
    a, b = gate.new_state(), gate.new_state()
    gate.attempt(a, "x")
    gate.attempt(a, "x")
    assert gate.attempt(b, "right")[0] is True and gate.attempt(a, "right")[0] is False


# ---- session limits ------------------------------------------------------------------------------------------------------------------

def test_the_rate_limit_is_a_sliding_window_that_reports_how_long_to_wait():
    clock = Clock()
    guard = SessionGuard(max_requests=3, window_s=60, session_cap=100, clock=clock)
    for _ in range(3):
        assert guard.check().allowed
        guard.record()
        clock.advance(10)
    d = guard.check()  # t = 30: three requests at t = 0, 10, 20 are inside the window
    assert not d.allowed and d.reason == "rate_limited" and d.retry_after_s == pytest.approx(30) and "31 seconds" in d.message
    clock.advance(30.5)  # the first request (t = 0) has left the window
    assert guard.check().allowed


def test_the_session_cap_blocks_for_good_and_counts_only_recorded_requests():
    clock = Clock()
    guard = SessionGuard(max_requests=100, window_s=60, session_cap=3, clock=clock)
    for _ in range(5):
        guard.check()  # checking never counts
    assert guard.total == 0 and guard.remaining == 3
    for _ in range(3):
        guard.record()
        clock.advance(100)
    d = guard.check()
    assert not d.allowed and d.reason == "session_cap" and "3 questions" in d.message and guard.remaining == 0


def test_the_session_cap_wins_over_the_rate_limit_message():
    guard = SessionGuard(max_requests=1, window_s=60, session_cap=1, clock=Clock())
    guard.record()
    assert guard.check().reason == "session_cap"


# ---- the daily cap -------------------------------------------------------------------------------------------------------------------

def test_the_daily_cap_is_shared_stops_at_the_cap_and_resets_on_the_next_utc_day():
    now = [datetime(2026, 11, 22, 23, 0, tzinfo=timezone.utc).timestamp()]
    budget = DailyBudget(3, clock=lambda: now[0])
    assert [budget.try_acquire() for _ in range(5)] == [True, True, True, False, False] and budget.used == 3 and budget.remaining == 0
    now[0] += 2 * 3600  # past midnight UTC
    assert budget.try_acquire() is True and budget.used == 1


def test_release_gives_a_unit_back_but_never_below_zero():
    budget = DailyBudget(2)
    assert budget.try_acquire()
    budget.release()
    budget.release()
    assert budget.used == 0 and budget.try_acquire() and budget.try_acquire() and not budget.try_acquire()


def test_the_daily_cap_is_atomic_under_concurrent_sessions():
    budget = DailyBudget(50)
    granted = []

    def worker():
        for _ in range(20):
            if budget.try_acquire():
                granted.append(1)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(granted) == 50 and budget.used == 50  # 160 attempts, never more than the cap granted


def test_a_zero_cap_grants_nothing():
    assert DailyBudget(0).try_acquire() is False


# ---- question length --------------------------------------------------------------------------------------------------------------------

def test_check_question_is_the_services_own_validator_so_the_cap_cannot_differ():
    from rag_core.config import RagConfig
    from tests.unit.test_rag_core_service import make

    svc, *_ = make(cfg=RagConfig(max_question_chars=30))
    assert check_question(svc, "  What is attention?  ") == "What is attention?"
    for bad, code in (("", "empty"), ("x" * 31, "too_long"), (None, "invalid_type")):
        with pytest.raises(InvalidQuestion) as exc:
            check_question(svc, bad)
        assert exc.value.code == code
