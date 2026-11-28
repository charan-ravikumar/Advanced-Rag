"""scripts/load_test.py: the pass rule and the sustained rate (no server), and its row in the run index."""

import json

import pytest

from scripts.load_test import dev_questions, percentile, summarise_step, sustained_qps
from scripts.runs_index import key_metrics


def records(latencies, status=200, lag=1.0):
    return [{"status": status, "latency_ms": ms, "send_lag_ms": lag} for ms in latencies]


def test_percentile_is_nearest_rank():
    values = list(range(1, 101))
    assert (percentile(values, 50), percentile(values, 95), percentile(values, 100), percentile([7], 95), percentile([], 95)) == (50, 95, 100, 7, None)


@pytest.mark.parametrize("recs, passed", [
    (records([100.0] * 100), True),
    (records([100.0] * 90 + [900.0] * 10), False),  # p95 above the target
    (records([100.0] * 99) + records([5.0], status=503), True),  # 1% failed: allowed
    (records([100.0] * 98) + records([5.0, 5.0], status=503), False),  # fast rejections do not make a rate pass
    (records([100.0] * 98) + records([30000.0, 30000.0], status="ReadTimeout"), False),
    (records([100.0] * 100, lag=50.0), False),  # the generator fell behind: the step says nothing about the server
    (records([5.0] * 10, status=503), False),
])
def test_a_rate_passes_only_with_low_latency_few_failures_and_a_generator_that_kept_up(recs, passed):
    step = summarise_step(10.0, 10.0, recs, slo_ms=300.0, max_failed=0.01)
    assert step["passed"] is passed and step["sent"] == len(recs) and step["ok"] == sum(r["status"] == 200 for r in recs)


def test_the_sustained_rate_stops_at_the_first_failing_rate():
    steps = [{"offered_qps": r, "passed": p} for r, p in [(30.0, True), (10.0, True), (20.0, False), (40.0, True)]]
    assert sustained_qps(steps) == 10.0 and sustained_qps([{"offered_qps": 5.0, "passed": False}]) == 0.0


def test_the_questions_are_the_answerable_dev_questions():
    assert len(dev_questions()) == 174


def test_a_load_run_shows_its_sustained_rate_and_latencies_in_the_run_index(tmp_path):
    step = summarise_step(10.0, 10.0, records([100.0] * 100), 300.0, 0.01)
    (tmp_path / "summary.json").write_text(json.dumps({"slo_ms": 300.0, "max_failed_fraction": 0.01, "sustained_qps": 10.0, "steps": [step]}), encoding="utf-8")
    got = key_metrics(tmp_path)
    assert got["sustained_qps"] == 10.0 and got["10qps.p95_ms"] == 100.0 and got["10qps.failed_fraction"] == 0.0
