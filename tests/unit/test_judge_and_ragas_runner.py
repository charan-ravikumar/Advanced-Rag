"""eval/judge_core.py and eval/run_ragas.py with a FAKE judge: cache key, cached/resumable calls, quota handling, usage, run folder. No ragas, no google-genai, no network."""

import asyncio
import json

import pytest
from pydantic import BaseModel

import eval.run_ragas as rr
from eval.disk_cache import DiskCache
from eval.judge_core import (CachingJudge, JudgeError, JudgeQuotaExhausted, JudgeRateLimited, JudgeSettings, JudgeStats, RawJudgeResult, classify_http_error,
                             judge_cache_key, parse_reply)
from tests.unit.rag_core_fakes import ABSTAIN_JSON, answer_json, make_hit
from tests.unit.test_run_generation_eval import ModeProvider, run as run_generation

SETTINGS = JudgeSettings(model="gemini-2.5-flash", temperature=0.0, thinking_budget=1024, max_output_tokens=4096)


class Verdict(BaseModel):
    verdict: int


class Other(BaseModel):
    verdict: int
    reason: str


@pytest.fixture(autouse=True)
def _fast_environment_probes(monkeypatch):
    monkeypatch.setattr("utils.fingerprint.code_fingerprint", lambda *a, **k: "f" * 64)
    monkeypatch.setattr("eval.run_retrieval_eval.hardware_info", lambda: {"stub": True})


class FakeRaw:
    """An async raw judge: returns canned JSON, counts calls, can fail on chosen call numbers."""

    def __init__(self, replies=None, fail_on=None, tokens=(100, 20, 50)):
        self.replies, self.fail_on, self.tokens, self.calls = replies or [], dict(fail_on or {}), tokens, []

    async def __call__(self, prompt, response_model):
        self.calls.append(prompt)
        n = len(self.calls)
        if n in self.fail_on:
            raise self.fail_on[n]
        text = self.replies[n - 1] if n - 1 < len(self.replies) else json.dumps({"verdict": 1})
        return RawJudgeResult(text=text, prompt_tokens=self.tokens[0], output_tokens=self.tokens[1], thinking_tokens=self.tokens[2], finish_reason="STOP")


def judge(raw, tmp_path, metric="faithfulness", version="0.4.4", settings=SETTINGS, stats=None, **kw):
    stats = stats or JudgeStats()
    return CachingJudge(raw, DiskCache(tmp_path / "jc"), settings, metric, version, stats, sleep=kw.pop("sleep", lambda s: None), **kw), stats


# ---- the cache key ----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("change", [dict(settings=JudgeSettings("other-model")), dict(settings=JudgeSettings("gemini-2.5-flash", temperature=0.2)),
                                    dict(settings=JudgeSettings("gemini-2.5-flash", thinking_budget=0)), dict(settings=JudgeSettings("gemini-2.5-flash", max_output_tokens=2048)),
                                    dict(settings=JudgeSettings("gemini-2.5-flash", api_version="v1")), dict(metric="answer_relevancy"), dict(ragas_version="0.4.5"),
                                    dict(prompt="another prompt"), dict(response_model=Other)])
def test_every_ingredient_changes_the_judge_cache_key(change):
    base = dict(settings=SETTINGS, metric="faithfulness", ragas_version="0.4.4", prompt="p", response_model=Verdict)
    assert judge_cache_key(**{**base, **change}) != judge_cache_key(**base)
    assert judge_cache_key(**base) == judge_cache_key(**dict(base))


# ---- caching ------------------------------------------------------------------------------------------------------------------------

def test_a_second_identical_call_is_served_from_the_disk_cache_even_by_a_new_judge_object(tmp_path):
    raw = FakeRaw()
    j, stats = judge(raw, tmp_path)
    assert asyncio.run(j.agenerate("prompt", Verdict)).verdict == 1 and len(raw.calls) == 1
    assert j.generate("prompt", Verdict).verdict == 1 and len(raw.calls) == 1 and stats.usage("faithfulness").cache_hits == 1
    j2, stats2 = judge(raw, tmp_path)  # a later session: new objects, same directory
    assert j2.generate("prompt", Verdict).verdict == 1 and len(raw.calls) == 1 and stats2.usage("faithfulness").cache_hits == 1 and stats2.total_calls == 0


@pytest.mark.parametrize("kw", [dict(metric="answer_relevancy"), dict(version="0.4.5"), dict(settings=JudgeSettings("another-judge"))])
def test_changing_the_metric_the_ragas_version_or_the_judge_is_a_cache_miss(tmp_path, kw):
    raw = FakeRaw()
    judge(raw, tmp_path)[0].generate("prompt", Verdict)
    judge(raw, tmp_path, **kw)[0].generate("prompt", Verdict)
    assert len(raw.calls) == 2  # the first verdict was NOT reused for another metric, ragas version or judge


def test_a_different_response_schema_for_the_same_prompt_is_a_miss(tmp_path):
    raw = FakeRaw([json.dumps({"verdict": 1}), json.dumps({"verdict": 0, "reason": "x"})])
    j, _ = judge(raw, tmp_path)
    j.generate("prompt", Verdict)
    assert j.generate("prompt", Other).reason == "x" and len(raw.calls) == 2


def test_only_valid_replies_are_cached_and_invalid_ones_are_retried(tmp_path):
    raw = FakeRaw(["not json at all", json.dumps({"verdict": 1})])
    j, stats = judge(raw, tmp_path)
    assert j.generate("prompt", Verdict).verdict == 1 and len(raw.calls) == 2 and stats.usage("faithfulness").retries == 1
    assert len(DiskCache(tmp_path / "jc")) == 1  # the invalid reply was never stored
    judge(raw, tmp_path)[0].generate("prompt", Verdict)
    assert len(raw.calls) == 2


def test_a_judge_that_never_gives_a_usable_reply_raises_after_the_attempts_and_caches_nothing(tmp_path):
    raw = FakeRaw(["", "```", '{"verdict": "x"}', "{}"])
    j, _ = judge(raw, tmp_path, max_attempts=4)
    with pytest.raises(JudgeError, match="no usable reply after 4 attempts"):
        j.generate("prompt", Verdict)
    assert len(raw.calls) == 4 and len(DiskCache(tmp_path / "jc")) == 0


def test_a_reply_wrapped_in_a_code_fence_parses():
    assert parse_reply('```json\n{"verdict": 1}\n```', Verdict).verdict == 1
    with pytest.raises(JudgeError, match="empty"):
        parse_reply("  ", Verdict)


def test_a_damaged_cache_entry_is_a_miss(tmp_path):
    raw = FakeRaw()
    j, _ = judge(raw, tmp_path)
    key = j.key("prompt", Verdict)
    DiskCache(tmp_path / "jc").set(key, {"text": "garbage"})
    assert j.generate("prompt", Verdict).verdict == 1 and len(raw.calls) == 1


# ---- quota ------------------------------------------------------------------------------------------------------------------------------

def test_a_per_minute_429_is_waited_out_with_the_delay_the_api_named_and_then_succeeds(tmp_path):
    slept = []
    raw = FakeRaw(fail_on={1: JudgeRateLimited(12.0)})
    j, stats = judge(raw, tmp_path, sleep=slept.append)
    assert j.generate("prompt", Verdict).verdict == 1 and slept == [13.0] and stats.usage("faithfulness").retries == 1 and len(raw.calls) == 2


def test_a_judge_that_stays_rate_limited_is_a_clean_quota_exhaustion_not_a_crash(tmp_path):
    raw = FakeRaw(fail_on={i: JudgeRateLimited(1.0) for i in range(1, 10)})
    j, _ = judge(raw, tmp_path, max_attempts=3)
    with pytest.raises(JudgeQuotaExhausted, match="rate limited"):
        j.generate("prompt", Verdict)
    assert len(DiskCache(tmp_path / "jc")) == 0


def test_the_session_call_budget_stops_before_a_network_call_but_cached_replies_stay_free(tmp_path):
    raw = FakeRaw()
    stats = JudgeStats(max_calls=2)
    j, _ = judge(raw, tmp_path, stats=stats)
    j.generate("a", Verdict)
    j.generate("b", Verdict)
    with pytest.raises(JudgeQuotaExhausted, match="budget"):
        j.generate("c", Verdict)
    assert len(raw.calls) == 2 and j.generate("a", Verdict).verdict == 1  # cached: allowed even after the budget


def test_a_daily_quota_error_propagates_and_is_not_cached(tmp_path):
    raw = FakeRaw(fail_on={1: JudgeQuotaExhausted("per-day limit")})
    j, _ = judge(raw, tmp_path)
    with pytest.raises(JudgeQuotaExhausted, match="per-day"):
        j.generate("prompt", Verdict)
    assert len(DiskCache(tmp_path / "jc")) == 0


def test_usage_is_accumulated_per_metric_from_the_replies(tmp_path):
    raw = FakeRaw(tokens=(100, 20, 50))
    stats = JudgeStats()
    judge(raw, tmp_path, "faithfulness", stats=stats)[0].generate("a", Verdict)
    judge(raw, tmp_path, "faithfulness", stats=stats)[0].generate("b", Verdict)
    judge(raw, tmp_path, "answer_relevancy", stats=stats)[0].generate("a", Verdict)
    d = stats.as_dict()
    assert d["faithfulness"]["calls"] == 2 and d["faithfulness"]["prompt_tokens"] == 200 and d["faithfulness"]["thinking_tokens"] == 100
    assert d["answer_relevancy"]["calls"] == 1 and d["_total_network_calls"] == 3


@pytest.mark.parametrize("code,message,expected,delay", [
    (429, "Quota exceeded ... requests per minute limit exceeded on Free Tier. Please retry in 33.5s.", JudgeRateLimited, 33.5),
    (429, "RESOURCE_EXHAUSTED {'retryDelay': '12s'}", JudgeRateLimited, 12.0),
    (429, "no delay named here", JudgeRateLimited, None),
    (429, "Quota exceeded: Generate content requests per day limit exceeded", JudgeQuotaExhausted, None),
    (429, "GenerateRequestsPerDayPerProjectPerModel-FreeTier", JudgeQuotaExhausted, None),
    (503, "overloaded", JudgeRateLimited, 10.0), (400, "bad request", JudgeError, None), (None, "weird", JudgeError, None)])
def test_http_errors_are_classified(code, message, expected, delay):
    exc = classify_http_error(code, message)
    assert isinstance(exc, expected) and (delay is None or exc.retry_after_s == delay)


# ---- the scoring runner ---------------------------------------------------------------------------------------------------------------

class AbstainingProvider(ModeProvider):
    """Abstains in long mode for questions whose text contains a marker."""

    def __init__(self, marker, **kw):
        super().__init__(**kw)
        self.marker = marker

    def stream(self, messages, *, timeout_s, max_output_tokens, temperature, response_schema=None):
        from rag_core.gateway import Delta, Final

        self.calls.append({"messages": messages, "max_output_tokens": max_output_tokens})
        from rag_core.prompts import SYSTEM_SHORT

        text = messages[1]["content"]
        question = text.split("<question>\n")[1].split("\n")[0]
        short = messages[0]["content"] == SYSTEM_SHORT
        if not short and self.marker in question:
            yield Delta(ABSTAIN_JSON)
        else:
            yield Delta("red green" if short else answer_json((f"Because of it. About: {question}", ["S1"])))
        yield Final("stop", 1000, 100, 30)


def generation_run(tmp_path, provider=None):
    code, _ = run_generation(tmp_path, provider or ModeProvider())
    assert code == 0
    (run_dir,) = sorted((tmp_path / "gen").glob("*_dev_gen_R*"))
    return run_dir


class FakeScorer:
    """Three judge calls per answer (two faithfulness, one relevancy), each through a real CachingJudge: the runner's caching and resume logic is the real one."""

    version = "fake-ragas-1"

    def __init__(self, settings, cache, stats, pacer, sleep, raw):
        kw = dict(pacer=pacer, sleep=sleep or (lambda s: None))
        self.f = CachingJudge(raw, cache, settings, "faithfulness", self.version, stats, **kw)
        self.r = CachingJudge(raw, cache, settings, "answer_relevancy", self.version, stats, **kw)

    def score(self, question, answer, contexts):
        a = self.f.generate(f"statements of: {answer}", Verdict)
        b = self.f.generate(f"verdicts of: {answer} against {contexts}", Verdict)
        c = self.r.generate(f"question for: {answer}", Verdict)
        return {"Faithfulness": (a.verdict + b.verdict) / 2, "AnswerRelevancy": float(c.verdict)}


def score(tmp_path, gen_run, raw, *extra, judge_cfg=None, log=None):
    from tests.unit.test_run_generation_eval import DateClock

    lines = [] if log is None else log
    clock = DateClock()  # the pacer reads it and the injected sleep advances it, so pacing costs no real time
    code = rr.main(["--run", str(gen_run), "--out", str(tmp_path / "gen"), *extra], scorer_factory=lambda s, c, st, p, sl: FakeScorer(s, c, st, p, sl, raw),
                   judge=judge_cfg or {"provider": "gemini", "model": "gemini-2.5-flash", "temperature": 0.0, "thinking_budget": 1024, "max_output_tokens": 4096},
                   log=lines.append, sleep=lambda secs: clock.advance(secs / 3600.0), clock=clock)
    return code, lines


def scoring_dirs(tmp_path):
    return sorted((tmp_path / "gen").glob("*_gen_ragas*"))


def test_a_full_scoring_run_labels_everything_exploratory_and_records_the_judge(tmp_path):
    gen = generation_run(tmp_path)
    raw = FakeRaw()
    code, lines = score(tmp_path, gen, raw)
    assert code == 0 and len(raw.calls) == 18  # 6 answers x 3 judge calls
    (run_dir,) = scoring_dirs(tmp_path)
    assert sorted(p.name for p in run_dir.iterdir()) == ["SHA256SUMS", "config.json", "per_question.jsonl", "summary.json"]
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["label"].startswith("EXPLORATORY") and "UNCALIBRATED" in config["label"] and config["complete"] is True
    assert config["judge"]["model"] == "gemini-2.5-flash" and config["judge"]["thinking_budget"] == 1024 and config["judge"]["temperature"] == 0.0
    assert config["judge"]["max_output_tokens"] == 4096 and config["judge"]["strictness"] == 1 and config["ragas_version"] == "fake-ragas-1"
    assert config["preregistration"]["recorded_sha256"] == config["preregistration"]["sha256_original_text"] and len(config["code_fingerprint"]) == 64
    assert config["generation_run"]["per_question_sha256"] and len(config["generation_run"]["sample"]["question_ids"]) == 6 and "citation markers" in config["judge_input_preprocessing"]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["label"].startswith("EXPLORATORY") and summary["n_scored_rows"] == 6
    assert summary["metrics"]["Faithfulness"]["mean"] == 1.0 and summary["metrics"]["AnswerRelevancy"]["n"] == 6 and summary["metrics"]["Faithfulness"]["n_papers"] >= 2
    assert summary["judge_usage"]["faithfulness"]["calls"] == 12 and summary["judge_usage"]["answer_relevancy"]["calls"] == 6 and summary["judge_usage"]["_total_network_calls"] == 18
    rows = [json.loads(line) for line in (run_dir / "per_question.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 6 and all(r["arm"] == "R_long" and set(r["metrics"]) == {"Faithfulness", "AnswerRelevancy"} for r in rows)
    assert any("SESSION REPORT" in line for line in lines)
    from eval.run_retrieval_eval import verify_checksums

    assert verify_checksums(run_dir)


def test_the_judge_sees_the_answer_without_citation_markers_and_the_contexts_that_were_sent(tmp_path):
    gen = generation_run(tmp_path, ModeProvider())
    raw = FakeRaw()
    score(tmp_path, gen, raw, "--limit", "1")
    statements_prompt = next(p for p in raw.calls if p.startswith("statements of:"))
    assert "[S1]" not in statements_prompt and "Because" in statements_prompt
    verdicts_prompt = next(p for p in raw.calls if p.startswith("verdicts of:"))
    assert "Passage number 1 says" in verdicts_prompt and "Passage number 5 says" in verdicts_prompt


def test_abstained_answers_are_not_scored_and_are_counted(tmp_path):
    gen = generation_run(tmp_path, AbstainingProvider("Question number 0"))
    raw = FakeRaw()
    code, _ = score(tmp_path, gen, raw)
    summary = json.loads((scoring_dirs(tmp_path)[0] / "summary.json").read_text(encoding="utf-8"))
    assert code == 0 and summary["selection"]["skipped_abstained"] >= 1 and summary["n_scored_rows"] == summary["selection"]["scored_candidates"]
    assert len(raw.calls) == 3 * summary["n_scored_rows"]


def test_a_quota_stop_exits_cleanly_and_the_resume_makes_no_duplicate_judge_calls(tmp_path):
    gen = generation_run(tmp_path)
    shared = []

    class Counting(FakeRaw):
        def __init__(self, fail_on=None):
            super().__init__(fail_on=fail_on)
            self.calls = shared

    code, lines = score(tmp_path, gen, Counting({8: JudgeQuotaExhausted("per-day limit reached")}))
    assert code == 3 and len(shared) == 8 and any("STOPPED" in line and "resume later" in line for line in lines)
    first = json.loads((scoring_dirs(tmp_path)[0] / "config.json").read_text(encoding="utf-8"))
    assert first["complete"] is False
    code, _ = score(tmp_path, gen, Counting())
    assert code == 0 and len(shared) == 18 + 1  # the 18 needed calls plus the one attempt that was refused (never cached)
    assert len(set(shared)) == 18 and len(DiskCache(tmp_path / "gen" / "judge_cache")) == 18  # every distinct judge call was paid for exactly once


def test_the_session_call_budget_ends_the_session_and_the_next_one_continues(tmp_path):
    gen = generation_run(tmp_path)
    raw = FakeRaw()
    code, lines = score(tmp_path, gen, raw, "--max-judge-calls", "5")
    assert code == 3 and len(raw.calls) == 5 and any("budget" in line for line in lines)
    code, _ = score(tmp_path, gen, raw, "--max-judge-calls", "100")
    assert code == 0 and len(raw.calls) == 18 and len(set(raw.calls)) == 18


def test_a_judge_error_on_one_answer_leaves_its_metrics_empty_and_scoring_goes_on(tmp_path):
    gen = generation_run(tmp_path)
    raw = FakeRaw(["not json"] * 4)  # the first answer's first call fails four times
    code, lines = score(tmp_path, gen, raw)
    summary = json.loads((scoring_dirs(tmp_path)[0] / "summary.json").read_text(encoding="utf-8"))
    assert len(summary["errors"]) == 1 and summary["n_scored_rows"] == 6 and summary["metrics"]["Faithfulness"]["n"] == 5 and any("JUDGE ERRORS" in line for line in lines)


def test_a_modified_generation_run_a_foreign_folder_and_a_non_gemini_judge_are_refused(tmp_path):
    gen = generation_run(tmp_path)
    with pytest.raises(SystemExit, match="not a generation run folder"):
        score(tmp_path, tmp_path, FakeRaw())
    judge_cfg = {"provider": "openai", "model": "x", "thinking_budget": 0, "max_output_tokens": 10}
    with pytest.raises(SystemExit, match="only the Gemini judge"):
        score(tmp_path, gen, FakeRaw(), judge_cfg=judge_cfg)
    (gen / "per_question.jsonl").write_text((gen / "per_question.jsonl").read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="checksums"):
        score(tmp_path, gen, FakeRaw())


def test_a_changed_preregistration_refuses_the_scoring_run(tmp_path):
    gen = generation_run(tmp_path)
    prereg = tmp_path / "P.md"
    prereg.write_bytes(b"# original\r\n")
    import hashlib

    (tmp_path / "P.sha256").write_text(hashlib.sha256(b"# original\r\n").hexdigest() + "  P.md\n", encoding="utf-8")
    prereg.write_bytes(b"# changed\r\n")
    with pytest.raises(SystemExit, match="REFUSED"):
        score(tmp_path, gen, FakeRaw(), "--prereg", str(prereg), "--prereg-hash-file", str(tmp_path / "P.sha256"))


def test_every_ragas_entry_point_disables_ragas_analytics_before_any_non_standard_library_import():
    import ast
    import sys
    from pathlib import Path

    for name in ("run_ragas.py", "ragas_adapter.py", "run_generation_eval.py"):
        tree = ast.parse((Path(__file__).resolve().parents[2] / "eval" / name).read_text(encoding="utf-8"))

        def third_party(node):
            mods = [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module else [a.name for a in node.names] if isinstance(node, ast.Import) else []
            return any(m.split(".")[0] not in sys.stdlib_module_names for m in mods)

        first = next(i for i, n in enumerate(tree.body) if isinstance(n, (ast.Import, ast.ImportFrom)) and third_party(n))
        setdefaults = [i for i, n in enumerate(tree.body) if isinstance(n, ast.Expr) and "RAGAS_DO_NOT_TRACK" in ast.dump(n)]
        assert setdefaults and setdefaults[0] < first, name


@pytest.mark.slow
def test_importing_the_evaluation_modules_pulls_in_no_ragas_no_provider_sdk_and_no_http_client():
    import os
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LANGFUSE_", "GROQ_", "GEMINI_"))}
    env["PYTHONPATH"] = str(root)
    code = textwrap.dedent("""
        import sys
        import eval.answer_metrics, eval.gen_arms, eval.quota_plan, eval.disk_cache, eval.judge_core, eval.ragas_adapter, eval.run_generation_eval, eval.run_ragas
        print(sorted(m for m in ("ragas", "google", "openai", "groq", "instructor", "langfuse", "httpx", "datasets", "torch", "chromadb") if m in sys.modules))
    """)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root, env=env, timeout=180)
    assert proc.returncode == 0, proc.stderr[-600:]
    assert proc.stdout.strip() == "[]"


# ---- the judge's relation to the generator (recorded in config.json) ------------------------------------------------------------

@pytest.mark.parametrize("judge, generator, expected", [
    ("gemini-3.8-flash", "gemini-3.8-flash", "SAME model"),
    ("gemini-3.8-flash", "gemini-3.5-flash-lite", "SAME family"),
    ("gemini-3.8-flash", "openai/gpt-oss-120b", "different model families"),
    ("gemini-3.8-flash", None, "unknown"),
])
def test_the_family_note_says_how_the_judge_relates_to_the_generator(judge, generator, expected):
    note = rr.family_note(judge, generator)
    assert expected in note and "NOT calibrated" in note


def test_the_configured_judge_is_not_the_generator_model():
    from config import cfg
    from rag_core.config import RagConfig

    rc = RagConfig.from_config(cfg)
    assert rc.judge["model"] == "gemini-3.8-flash" and rc.judge["model"] != rc.gemini_model
