"""scripts/runs_index.py: one row per run folder, byte-identical rebuilds, and a check that fails on a stale index."""

import json
import shutil
from pathlib import Path

import pytest

from scripts.runs_index import INDEX, ROOT, build, main

EVAL_ID = "20260101T000000Z_abcd1234_dev_paper"


def write(root: Path, relative: str, data) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture()
def repo(tmp_path):
    run = f"runs/eval/{EVAL_ID}"
    write(tmp_path, f"{run}/config.json", {"created_utc": "2026-01-01T00:00:00+00:00", "code_fingerprint": "abcd1234", "split": "dev", "eval_mode": "paper"})
    write(tmp_path, f"{run}/summary.json", {"arms": {"D": {"n_questions": 3, "means": {"nDCG@10": 0.5, "Recall@50": 0.9, "old_MRR": 0.1},
                                                          "retrieval_latency_ms": {"total": {"p50": 4.0, "p95": 9.0}, "rerank": None}}}})
    write(tmp_path, "runs/analysis/20260102_compare/table.txt", "text only\n")
    write(tmp_path, "runs/measurements/audit_20260103T040506Z/results.json", {"docs": 2, "source_run": rf"runs\eval\{EVAL_ID}", "by": {"ok": True}})
    write(tmp_path, "runs/gen/cache/00/x.json", {"paid": "work"})
    write(tmp_path, "runs/gen/token_ledger.jsonl", "{}\n")
    return tmp_path


def rows(root: Path) -> dict:
    return {r["run_id"]: r for r in map(json.loads, build(root).decode("utf-8").splitlines())}


def test_one_row_per_run_folder_and_caches_and_loose_files_are_not_runs(repo):
    got = rows(repo)
    assert list(got) == ["20260102_compare", EVAL_ID, "audit_20260103T040506Z"]  # ordered by kind, then name
    ev = got[EVAL_ID]
    assert (ev["path"], ev["kind"], ev["split"], ev["code_fingerprint"], ev["config_hash_source"]) == (f"runs/eval/{EVAL_ID}", "eval", "dev", "abcd1234", "config.json")
    assert ev["metrics"] == {"D.n_questions": 3, "D.nDCG@10": 0.5, "D.Recall@50": 0.9, "D.latency_ms.total.p50": 4.0, "D.latency_ms.total.p95": 9.0}


def test_what_a_run_did_not_record_is_null_and_the_split_follows_the_source_run(repo):
    got = rows(repo)
    text = got["20260102_compare"]
    assert text["timestamp"] == "2026-01-02" and text["code_fingerprint"] is None and text["config_hash"] is None and text["split"] is None and text["metrics"] == {}
    audit = got["audit_20260103T040506Z"]
    assert audit["timestamp"] == "2026-01-03T04:05:06+00:00" and audit["split"] == "dev" and audit["metrics"] == {"docs": 2, "by.ok": True}


def test_rebuilding_gives_the_same_bytes_wherever_the_repository_is(repo, tmp_path_factory):
    assert main([], repo) == 0
    first = (repo / INDEX).read_bytes()
    assert main([], repo) == 0 and (repo / INDEX).read_bytes() == first and b"\r\n" not in first
    copy = tmp_path_factory.mktemp("elsewhere") / "copy"
    shutil.copytree(repo, copy)
    write(copy, "runs/.DS_Store", "junk")
    write(copy, f"runs/eval/{EVAL_ID}/.DS_Store", "junk")
    assert build(copy) == first


@pytest.mark.parametrize("change", ["new run", "removed run", "edited number", "hand edit", "no index"])
def test_check_fails_on_a_stale_index_and_does_not_write(repo, capsys, change):
    main([], repo)
    assert main(["--check"], repo) == 0
    if change == "new run":
        write(repo, "runs/eval/20260104T000000Z_ffff0000_dev_paper/config.json", {"split": "dev"})
    elif change == "removed run":
        shutil.rmtree(repo / "runs/analysis/20260102_compare")
    elif change == "edited number":
        write(repo, f"runs/eval/{EVAL_ID}/per_question.jsonl", "{}\n")
    elif change == "hand edit":
        (repo / INDEX).write_bytes((repo / INDEX).read_bytes().replace(b"0.5", b"0.6"))
    else:
        (repo / INDEX).unlink()
    before = (repo / INDEX).read_bytes() if (repo / INDEX).exists() else None
    assert main(["--check"], repo) == 1 and "STALE" in capsys.readouterr().out
    assert ((repo / INDEX).read_bytes() if (repo / INDEX).exists() else None) == before


def test_a_claimed_run_that_does_not_exist_fails_both_modes(repo, capsys):
    write(repo, "docs/CLAIMS_LOG.md", f"| x | 1 | `runs/eval/{EVAL_ID}` |\n")
    assert main([], repo) == 0 and main(["--check"], repo) == 0
    write(repo, "README.md", "see runs/eval/20990101T000000Z_dead_dev_paper/summary.json\n")
    assert main([], repo) == 1 and main(["--check"], repo) == 1 and "20990101T000000Z_dead_dev_paper" in capsys.readouterr().out


def test_broken_json_and_duplicate_run_ids_are_errors(repo):
    write(repo, f"runs/measurements/{EVAL_ID}/results.json", {})
    with pytest.raises(ValueError, match="unique"):
        build(repo)
    write(repo, f"runs/measurements/{EVAL_ID}/results.json", "{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        build(repo)


def test_the_committed_index_is_up_to_date_and_ci_checks_it():
    assert main(["--check"], ROOT) == 0
    assert "python scripts/runs_index.py --check" in (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
