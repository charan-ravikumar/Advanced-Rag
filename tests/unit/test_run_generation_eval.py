"""eval/run_generation_eval.py with a fake generator: sample, cache, token governor, budget exit and resume (no duplicate calls), rate limit, guards, row shape.
No network, no keys, no real store."""

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import eval.run_generation_eval as rg
from eval.compare import load_rows, metric_by_question
from eval.run_generation_eval import (GenerationCache, GuardRefused, TokenGovernor, TokenLedger, build_retrieve, check_test_guard, eligible_questions, generation_key,
                                      select_sample)
from rag_core.config import RagConfig
from rag_core.gateway import Delta, Final, LLMGateway
from rag_core.prompts import SYSTEM_SHORT
from rag_core.service import RagService
from tests.unit.rag_core_fakes import FakeClock, FakeRetriever, fail, make_hit

START = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _fast_environment_probes(monkeypatch):
    """The fingerprint hashes every source file and the hardware probe asks the OS: neither is under test here and each run would pay for them."""
    monkeypatch.setattr("utils.fingerprint.code_fingerprint", lambda *a, **k: "f" * 64)
    monkeypatch.setattr("eval.run_retrieval_eval.hardware_info", lambda: {"stub": True})


def refs(answer="red green", typ="extractive", evidence=("ev",), lost=0):
    return [{"answer": answer, "type": typ, "mixed_types": False, "evidence": list(evidence), "n_evidence_not_retrievable": lost}]


def make_data(tmp_path: Path, papers=4, per=3):
    questions = []
    for p in range(papers):
        for k in range(per):
            questions.append({"id": f"p{p}:q{k}", "paper_id": f"p{p}", "question": f"Question number {p}{k} about the method?", "answerable": True, "references": refs()})
    questions.append({"id": "p9:una", "paper_id": "p9", "question": "Unanswerable question?", "answerable": False, "references": [{"answer": None, "type": "unanswerable", "evidence": [], "n_evidence_not_retrievable": 0}]})
    questions.append({"id": "p9:table", "paper_id": "p9", "question": "A table question?", "answerable": True, "references": refs(evidence=(), lost=2)})
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({"questions": questions}), encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"dev": [q["id"] for q in questions], "test": [], "golden_set_sha256": hashlib.sha256(golden.read_bytes()).hexdigest()}), encoding="utf-8")
    return golden, split, questions


class ModeProvider:
    """Answers 'red green' in short mode and 'Because [S1].' in long mode; fails on chosen call numbers; records every call."""

    name, model = "gemini", "gemini-3.5-flash-lite"

    def __init__(self, effort="low", fail_calls=(), fail_kind="rate_limit", status=429, tokens=(1000, 100, 30)):
        self.reasoning_effort, self.fail_calls, self.fail_kind, self.status, self.tokens, self.calls = effort, set(fail_calls), fail_kind, status, tokens, []

    def stream(self, messages, *, timeout_s, max_output_tokens, temperature):
        self.calls.append({"messages": messages, "max_output_tokens": max_output_tokens})
        if len(self.calls) in self.fail_calls:
            raise fail(self.fail_kind, self.status)
        short = messages[0]["content"] == SYSTEM_SHORT
        question = messages[1]["content"].split("<question>\n")[1].split("\n")[0]  # long answers differ per question, so the judge prompts differ too
        yield Delta("red green" if short else f"Because [S1]. About: {question}")
        yield Final("stop", self.tokens[0], self.tokens[1], self.tokens[2])

    def keys(self):
        """(question text, is_short) of every call: a duplicate means a question was paid for twice."""
        return [(c["messages"][1]["content"].split("<question>\n")[1].split("\n")[0], c["messages"][0]["content"] == SYSTEM_SHORT) for c in self.calls]


class DateClock:
    def __init__(self):
        self.t = START

    def __call__(self):
        return self.t

    def advance(self, hours):
        self.t += timedelta(hours=hours)


def run(tmp_path, provider, *extra, retriever=None, clock=None, n=6, log=None, golden=None, split=None, budget=None):
    g, s, _ = (golden, split, None) if golden else make_data(tmp_path) if not (tmp_path / "golden.json").exists() else (tmp_path / "golden.json", tmp_path / "split.json", None)
    vclock = FakeClock()
    gateway = LLMGateway([provider], sleep=vclock.sleep, clock=vclock, rng=lambda: 1.0, max_attempts=1)
    lines = [] if log is None else log
    date_clock = clock or DateClock()
    args = ["--split", "dev", "--n", str(n), "--out", str(tmp_path / "gen"), "--golden", str(g), "--split-file", str(s), "--token-estimate", "1100", *extra]
    if budget is not None:
        args += ["--daily-token-budget", str(budget)]
    code = rg.main(args, retriever=retriever or FakeRetriever(), gateway=gateway, index_version="idx", clock=date_clock,
                   sleep=lambda secs: date_clock.advance(secs / 3600.0), log=lines.append)
    return code, lines


def run_dirs(tmp_path):
    return sorted((tmp_path / "gen").glob("*_dev_gen_R*"))


def read_rows(run_dir):
    return [json.loads(line) for line in (run_dir / "per_question.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


# ---- the sample -------------------------------------------------------------------------------------------------------------------

def test_the_sample_is_seeded_deterministic_and_only_answerable_questions_with_a_usable_reference(tmp_path):
    _, _, qs = make_data(tmp_path)
    assert {q["id"] for q in eligible_questions(qs)} == {q["id"] for q in qs if q["id"] not in ("p9:una", "p9:table")}
    a, b, c = select_sample(qs, 5, 0), select_sample(qs, 5, 0), select_sample(qs, 5, 1)
    assert [q["id"] for q in a] == [q["id"] for q in b] != [q["id"] for q in c] and len(a) == 5 and len({q["id"] for q in a}) == 5
    assert "p9:una" not in {q["id"] for q in a} and "p9:table" not in {q["id"] for q in a}
    assert len(select_sample(qs, 100, 0)) == 12  # fewer eligible questions than asked: all of them


def test_the_sample_does_not_depend_on_the_order_of_the_input(tmp_path):
    _, _, qs = make_data(tmp_path)
    assert [q["id"] for q in select_sample(qs, 5, 3)] == [q["id"] for q in select_sample(list(reversed(qs)), 5, 3)]


# ---- the cache key and the cache ---------------------------------------------------------------------------------------------------

BASE = dict(model="m", reasoning_effort="low", prompt_id="rag-answer-short|v|abc", arm="R", mode="short", question_id="q1", context_hash_="h",
            max_output_tokens=512, temperature=0.0, top_k=5)


@pytest.mark.parametrize("field,value", [("model", "other"), ("reasoning_effort", "high"), ("reasoning_effort", None), ("prompt_id", "rag-answer-short|v|xyz"), ("arm", "O"),
                                         ("mode", "long"), ("question_id", "q2"), ("context_hash_", "h2"), ("max_output_tokens", 1024), ("temperature", 0.5), ("top_k", 10)])
def test_every_ingredient_changes_the_generation_key(field, value):
    assert generation_key(**{**BASE, field: value}) != generation_key(**BASE)
    assert generation_key(**BASE) == generation_key(**dict(BASE))


def test_the_cache_round_trips_survives_a_corrupt_file_and_leaves_no_temporary_files(tmp_path):
    cache = GenerationCache(tmp_path / "c")
    assert cache.get("ab" + "0" * 62) is None and len(cache) == 0
    cache.set("ab" + "1" * 62, {"x": [1, 2]})
    assert cache.get("ab" + "1" * 62) == {"x": [1, 2]} and len(cache) == 1 and not list((tmp_path / "c").rglob("*.tmp"))
    (tmp_path / "c" / "ab" / ("ab" + "2" * 62 + ".json")).write_text("{ not json", encoding="utf-8")
    assert cache.get("ab" + "2" * 62) is None  # corrupt = miss, not an error


# ---- the governor -------------------------------------------------------------------------------------------------------------------

def test_the_ledger_counts_only_the_trailing_24_hours_and_ignores_damaged_lines(tmp_path):
    clock = DateClock()
    ledger = TokenLedger(tmp_path / "l.jsonl", clock)
    ledger.record(1000)
    clock.advance(23)
    ledger.record(500)
    assert ledger.used() == 1500
    clock.advance(2)  # the first entry is now 25 h old
    assert ledger.used() == 500
    with open(tmp_path / "l.jsonl", "a", encoding="utf-8") as f:
        f.write("garbage\n{\"utc\": \"not a date\", \"tokens\": 1}\n\n")
    assert ledger.used() == 500


def test_the_governor_stops_before_the_call_that_would_exceed_the_budget_and_counts_declared_outside_use(tmp_path):
    gov = TokenGovernor(TokenLedger(tmp_path / "l.jsonl", DateClock()), 3000, already_used=500)
    assert gov.can_spend(2500) and not gov.can_spend(2501)
    gov.spend(1000)
    assert gov.used() == 1500 and gov.can_spend(1500) and not gov.can_spend(1501)


# ---- one complete session -----------------------------------------------------------------------------------------------------------

def test_a_complete_session_writes_the_run_folder_with_config_rows_summary_and_checksums(tmp_path):
    provider = ModeProvider()
    code, lines = run(tmp_path, provider)
    assert code == 0 and len(provider.calls) == 12  # 6 questions x short and long
    (run_dir,) = run_dirs(tmp_path)
    assert sorted(p.name for p in run_dir.iterdir()) == ["SHA256SUMS", "config.json", "per_question.jsonl", "summary.json"]
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["complete"] is True and config["sample"]["seed"] == 0 and len(config["sample"]["question_ids"]) == 6 and config["arm"] == "R"
    assert config["generator"]["reasoning_effort"] is None and config["generator"]["model"] == "gemini-3.5-flash-lite" and config["generator"]["tracing"].startswith("off")
    assert len(config["code_fingerprint"]) == 64 and config["preregistration"]["recorded_sha256"] == config["preregistration"]["sha256_original_text"]
    assert set(config["prompts"]) == {"rag-answer-long", "rag-answer-short"} and all(v["source"] == "local" for v in config["prompts"].values())
    assert config["label"].startswith("EXPLORATORY") and config["token_governor"]["daily_token_budget"] == 180000
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["label"].startswith("EXPLORATORY") and summary["short_mode"]["AnswerF1"]["mean"] == 1.0 and summary["long_mode"]["CitationValidity_pooled"]["rate"] == 1.0
    assert summary["session"]["tokens_used_this_session"] == 12 * 1100 and any("SESSION REPORT" in line for line in lines)
    from eval.run_retrieval_eval import verify_checksums

    assert verify_checksums(run_dir)


def test_rows_have_the_documented_fields_and_compare_py_can_read_them_per_mode(tmp_path):
    run(tmp_path, ModeProvider())
    (run_dir,) = run_dirs(tmp_path)
    rows = load_rows(run_dir)
    assert len(rows) == 12 and {r["arm"] for r in rows} == {"R_short", "R_long"} and {r["arm_base"] for r in rows} == {"R"}
    one = next(r for r in rows if r["arm"] == "R_short")
    for field in ("question_id", "paper_id", "answer", "short_answer", "used_sources", "citations", "abstained", "provider", "model", "reasoning_effort", "usage",
                  "latency_ms", "finish_reason", "errors", "metrics", "from_cache", "mode"):
        assert field in one, field
    assert one["usage"] == {"prompt_tokens": 1000, "completion_tokens": 100, "reasoning_tokens": 30} and one["reasoning_effort"] == "low" and one["short_answer"] == "red green"
    assert one["metrics"] == {"AnswerF1": 1.0, "YesNoAccuracy": None, "Abstained": 0.0}
    short = metric_by_question(rows, "R_short", "AnswerF1")
    assert len(short) == 6 and set(short.values()) == {1.0} and metric_by_question(rows, "R_long", "CitationValidity")  # compare.py's reader works unchanged
    long_row = next(r for r in rows if r["arm"] == "R_long")
    assert long_row["citations"] == ["S1"] and long_row["metrics"]["CitationValidity"] == 1.0 and long_row["metrics"]["HasCitation"] == 1.0


def test_arm_R_gives_both_modes_exactly_the_retrieved_passages_and_nothing_else(tmp_path):
    retriever = FakeRetriever([make_hit(i) for i in range(1, 9)])
    provider = ModeProvider()
    run(tmp_path, provider, "--limit", "1", retriever=retriever)
    short, long_ = provider.calls
    for call in (short, long_):
        text = call["messages"][1]["content"]
        assert all(f"Passage number {i} says" in text for i in range(1, 6)) and "Passage number 6 says" not in text  # the top 5, in order, no more
    assert short["messages"][1]["content"].split("<question>")[0] == long_["messages"][1]["content"].split("<question>")[0]  # the same context in both modes
    assert len(retriever.calls) == 1 and retriever.calls[0]["doc_ids"] == ["p0"] or retriever.calls[0]["doc_ids"][0].startswith("p")  # retrieved once, in the question's paper
    assert retriever.calls[0]["rerank"] is False and retriever.calls[0]["top_k"] == 5 and retriever.calls[0]["candidate_k"] == 20


def test_the_retrieval_arguments_equal_the_ones_the_service_uses_so_the_hashed_context_is_the_generated_context():
    cfg = RagConfig()
    seen = []

    class Spy(FakeRetriever):
        def retrieve(self, query, **kw):
            seen.append(kw)
            return super().retrieve(query, **kw)

    spy = Spy()
    build_retrieve(spy, cfg)({"question": "Q?", "paper_id": "p1"})
    clock = FakeClock()
    RagService(spy, LLMGateway([ModeProvider()], sleep=clock.sleep, clock=clock, rng=lambda: 1.0, max_attempts=1), None, cfg).answer_short("Q?", doc_ids=["p1"])
    assert seen[0] == seen[1]


# ---- budget exit and resume -----------------------------------------------------------------------------------------------------------

def test_the_budget_stops_the_session_cleanly_and_the_next_sessions_finish_it_with_no_duplicate_calls(tmp_path):
    provider, clock = ModeProvider(), DateClock()
    codes, reports = [], []
    for _ in range(12):  # every session may spend two calls (budget 2500, each call 1100): at most 6 sessions are needed
        code, lines = run(tmp_path, provider, "--daily-token-budget", "2500", clock=clock)
        codes.append(code)
        reports.append([line for line in lines if "SESSION REPORT" in line or "STOPPED" in line])
        if code == 0:
            break
        clock.advance(25)  # the next day: the ledger window is empty again
    assert codes[0] == 3 and codes[-1] == 0 and all(c == 3 for c in codes[:-1])
    assert any("STOPPED: daily token budget reached" in line for line in reports[0]) and any("resume later" in line for line in reports[0])
    keys = provider.keys()
    assert len(provider.calls) == 12 and len(set(keys)) == 12  # every (question, mode) was generated exactly once across all sessions
    first_dir = run_dirs(tmp_path)[0]
    assert json.loads((first_dir / "config.json").read_text(encoding="utf-8"))["complete"] is False and len(read_rows(first_dir)) == 2
    final = read_rows(run_dirs(tmp_path)[-1])
    assert len(final) == 12 and sum(1 for r in final if r["from_cache"]) == 10  # the last session made 2 new calls and served 10 from the cache


def test_the_first_session_stops_before_the_call_that_would_exceed_the_budget_and_records_what_it_used(tmp_path):
    provider = ModeProvider()
    code, lines = run(tmp_path, provider, "--daily-token-budget", "2500")
    assert code == 3 and len(provider.calls) == 2
    (run_dir,) = run_dirs(tmp_path)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["session"]
    assert summary["tokens_used_this_session"] == 2200 and summary["tokens_used_in_window_after"] == 2200 and summary["complete"] is False and len(summary["remaining"]) == 10
    assert "daily token budget reached" in summary["stopped_reason"]
    ledger = [json.loads(line) for line in (tmp_path / "gen" / "token_ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["tokens"] for e in ledger] == [1100, 1100]


def test_declared_outside_use_counts_against_the_budget(tmp_path):
    provider = ModeProvider()
    code, _ = run(tmp_path, provider, "--daily-token-budget", "2500", "--already-used-tokens", "1500")
    assert code == 3 and provider.calls == []  # 1500 declared + the 1100 estimate = 2600 > 2500: not even the first call is made
    code, _ = run(tmp_path, provider, "--daily-token-budget", "2700", "--already-used-tokens", "1500")
    assert code == 3 and len(provider.calls) == 1  # 2600 <= 2700 allows one call; then 2600 + 1100 > 2700


def test_resuming_in_the_same_window_with_a_larger_budget_continues_from_the_cache(tmp_path):
    provider = ModeProvider()
    run(tmp_path, provider, "--daily-token-budget", "2500")
    code, _ = run(tmp_path, provider, "--daily-token-budget", "100000")
    assert code == 0 and len(provider.calls) == 12 and len(set(provider.keys())) == 12


def test_a_fully_cached_rerun_costs_nothing(tmp_path):
    provider = ModeProvider()
    run(tmp_path, provider)
    code, lines = run(tmp_path, provider)
    assert code == 0 and len(provider.calls) == 12  # no new call
    last = read_rows(run_dirs(tmp_path)[-1])
    assert all(r["from_cache"] for r in last) and any("tokens used this session 0" in line for line in lines)
    first = read_rows(run_dirs(tmp_path)[0])
    assert [r["metrics"] for r in first] == [r["metrics"] for r in last]  # cached answers score identically


# ---- failures ------------------------------------------------------------------------------------------------------------------------

def test_a_provider_rate_limit_stops_the_session_cleanly_caches_nothing_for_the_failed_call_and_resumes(tmp_path):
    provider = ModeProvider(fail_calls=(3,), fail_kind="rate_limit", status=429)
    code, lines = run(tmp_path, provider)
    assert code == 3 and len(provider.calls) == 3 and any("HTTP 429" in line and "resume later" in line for line in lines)
    assert len(GenerationCache(tmp_path / "gen" / "cache")) == 2  # the two finished generations only
    provider.fail_calls = set()
    code, _ = run(tmp_path, provider)
    assert code == 0 and len(provider.calls) == 3 + 10  # the failed call and the nine others; the two finished ones came from the cache


def test_a_non_quota_failure_is_recorded_not_cached_and_retried_on_the_next_run(tmp_path):
    provider = ModeProvider(fail_calls=(3,), fail_kind="server", status=500)
    code, lines = run(tmp_path, provider)
    assert code == 3 and len(provider.calls) == 12  # the session went on to the other questions
    summary = json.loads((run_dirs(tmp_path)[0] / "summary.json").read_text(encoding="utf-8"))["session"]
    assert len(summary["failed"]) == 1 and summary["stopped_reason"] is None and summary["complete"] is False and any("FAILED" in line for line in lines)
    provider.fail_calls = set()
    code, _ = run(tmp_path, provider)
    assert code == 0 and len(provider.calls) == 13  # exactly one more call: the failed one


def test_a_changed_reasoning_effort_or_context_is_a_cache_miss_and_never_serves_a_stale_answer(tmp_path):
    run(tmp_path, ModeProvider())
    high = ModeProvider(effort="high")
    run(tmp_path, high)
    assert len(high.calls) == 12  # a different reasoning effort: nothing is reused
    other = ModeProvider()
    run(tmp_path, other, retriever=FakeRetriever([make_hit(i, text=f"different passage {i}") for i in range(1, 9)]))
    assert len(other.calls) == 12  # a different context: nothing is reused


# ---- limits and guards -------------------------------------------------------------------------------------------------------------

def test_limit_processes_only_the_first_questions_of_the_sample(tmp_path):
    provider = ModeProvider()
    code, _ = run(tmp_path, provider, "--limit", "2")
    assert code == 0 and len(provider.calls) == 4
    config = json.loads((run_dirs(tmp_path)[0] / "config.json").read_text(encoding="utf-8"))
    assert len(config["sample"]["question_ids"]) == 6 and len(config["sample"]["processed_this_session"]) == 2
    assert config["complete"] is False  # two of the six sampled questions: not a complete run of the sample


def test_mode_short_only_runs_only_short_answers(tmp_path):
    provider = ModeProvider()
    run(tmp_path, provider, "--mode", "short")
    assert len(provider.calls) == 6 and all(short for _, short in provider.keys())


def test_the_test_split_is_refused_without_the_flag_and_without_a_freeze_file(tmp_path):
    with pytest.raises(GuardRefused, match="final"):
        check_test_guard(confirmed=False, freeze_path=tmp_path / "gen_freeze.json")
    with pytest.raises(GuardRefused, match="no generation freeze file"):
        check_test_guard(confirmed=True, freeze_path=tmp_path / "gen_freeze.json")
    check_test_guard(confirmed=True, freeze_path=Path(__file__))  # an existing file passes the guard itself
    assert not rg.FREEZE_FILE.exists()  # and the real freeze file does not exist: it is created only when the owner asks


def test_main_refuses_the_test_split_and_runs_nothing(tmp_path):
    provider = ModeProvider()
    g, s, _ = make_data(tmp_path)
    with pytest.raises(GuardRefused):
        rg.main(["--split", "test", "--golden", str(g), "--split-file", str(s), "--out", str(tmp_path / "gen")], retriever=FakeRetriever(),
                gateway=LLMGateway([provider], max_attempts=1))
    with pytest.raises(GuardRefused):
        rg.main(["--split", "test", "--i-understand-this-is-final", "--freeze-file", str(tmp_path / "missing.json"), "--golden", str(g), "--split-file", str(s),
                 "--out", str(tmp_path / "gen")], retriever=FakeRetriever(), gateway=LLMGateway([provider], max_attempts=1))
    assert provider.calls == [] and not (tmp_path / "gen").exists()


def test_only_arm_R_exists():
    with pytest.raises(SystemExit):
        rg.main(["--split", "dev", "--arm", "O"])


def test_a_changed_preregistration_text_refuses_the_run(tmp_path):
    prereg = tmp_path / "PREREG.md"
    prereg.write_bytes(b"# original\r\ntext\r\n")  # bytes: write_text would translate the newlines on Windows
    hash_file = tmp_path / "PREREG.sha256"
    hash_file.write_text(hashlib.sha256(b"# original\r\ntext\r\n").hexdigest() + "  PREREG.md\n", encoding="utf-8")
    provider = ModeProvider()
    code, _ = run(tmp_path, provider, "--prereg", str(prereg), "--prereg-hash-file", str(hash_file), "--limit", "1")
    assert code == 0  # unchanged text: runs, and the hash is recorded
    prereg.write_bytes(b"# original\r\nTEXT CHANGED\r\n")
    with pytest.raises(SystemExit, match="REFUSED"):
        run(tmp_path, ModeProvider(), "--prereg", str(prereg), "--prereg-hash-file", str(hash_file))
    prereg.write_bytes(b"# original\r\ntext\r\n\r\n## Deviations\r\nappended later\r\n")
    assert run(tmp_path, ModeProvider(), "--prereg", str(prereg), "--prereg-hash-file", str(hash_file), "--limit", "1")[0] == 0  # appending a Deviations section is allowed


def test_a_split_made_for_another_golden_set_is_refused(tmp_path):
    g, s, _ = make_data(tmp_path)
    split = json.loads(s.read_text(encoding="utf-8"))
    split["golden_set_sha256"] = "0" * 64
    s.write_text(json.dumps(split), encoding="utf-8")
    with pytest.raises(SystemExit, match="different golden set"):
        run(tmp_path, ModeProvider(), golden=g, split=s)
