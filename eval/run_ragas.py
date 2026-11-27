"""RAGAS scoring of the long answers of a generation run (EXPLORATORY: n=40, one arm, an UNCALIBRATED judge; Phase 4 minimal scope).

    # in the EVALUATION venv (ragas lives only there):
    & "$env:LOCALAPPDATA\\advrag\\venv_eval\\Scripts\\python.exe" eval/run_ragas.py --run runs/gen/<generation run folder>

Scores faithfulness and answer relevancy (ragas 0.4.4 collections API) of every NON-ABSTAINED long-mode answer of the generation run. Judge: Gemini ``gemini-3.8-flash`` (``generation.judge.model``; a different model from the generator, the same family) at temperature 0 with
the thinking budget and output limit from ``config.yaml`` (``generation.judge``), through ``eval/ragas_adapter.py``; answer relevancy uses the local pinned MiniLM embeddings. The answers are scored
with their ``[S#]`` citation markers removed; the contexts are exactly the passages that were sent to the generator (``used_sources``). Only public Qasper text goes to Gemini.

EVERY judge call is cached on disk (eval/judge_core.py: the key holds the judge model, thinking budget, metric, ragas version and the prompt), so the run is resumable: a quota stop exits cleanly
(exit code 3, "quota exhausted, resume later") and running the same command again reads what is done from the cache. ``--max-judge-calls`` bounds the network calls of one session.

Outputs a run folder like the other runners (config.json, per_question.jsonl, summary.json, SHA256SUMS) with paper-clustered 95% bootstrap intervals. No human labels exist: the scores are the
judge's opinion until a calibration against human labels is done (cut from this phase).
"""

from __future__ import annotations

import os

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")  # first, before anything can import ragas

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable, Sequence  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.answer_metrics import strip_citation_markers, summarize_metric  # noqa: E402
from eval.disk_cache import DiskCache  # noqa: E402
from eval.judge_core import JudgeError, JudgeQuotaExhausted, JudgeSettings, JudgeStats  # noqa: E402
from eval.quota_plan import Pacer  # noqa: E402

LABEL = "EXPLORATORY: n=40 long answers of one arm (R), an UNCALIBRATED LLM judge (no human labels): the scores are the judge's opinion, not a measurement of correctness"
PREREG_FILE = ROOT / "eval" / "PREREGISTRATION_gen.md"
PREREG_HASH_FILE = ROOT / "eval" / "PREREGISTRATION_gen.sha256"
EXIT_QUOTA = 3
GEMINI_FREE_TIER_RPM = 5  # observed (HTTP 429 named the Free Tier limit) in runs/gen_dryrun/*_quota_dry_run
METRICS = ("Faithfulness", "AnswerRelevancy")


def load_generation_run(run_dir: Path) -> tuple[dict, list[dict]]:
    from eval.run_retrieval_eval import verify_checksums

    run_dir = Path(run_dir)
    if not (run_dir / "per_question.jsonl").exists():
        raise SystemExit(f"REFUSED: {run_dir} is not a generation run folder (no per_question.jsonl).")
    if not verify_checksums(run_dir):
        raise SystemExit(f"REFUSED: the checksums of {run_dir} do not match: the run folder was modified.")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (run_dir / "per_question.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return config, rows


def select_rows(rows: Sequence[dict]) -> tuple[list[dict], dict]:
    """The long-mode rows that can be scored, and why the others cannot. An abstention (INSUFFICIENT_CONTEXT) is noncommittal by construction and says nothing about faithfulness."""
    long_rows = [r for r in rows if r.get("mode") == "long"]
    skipped = {"abstained": 0, "no_contexts": 0, "empty_answer": 0}
    keep = []
    for r in long_rows:
        if r.get("abstained"):
            skipped["abstained"] += 1
        elif not r.get("used_sources"):
            skipped["no_contexts"] += 1
        elif not strip_citation_markers(r.get("answer")).strip():
            skipped["empty_answer"] += 1
        else:
            keep.append(r)
    return keep, {"long_mode_rows": len(long_rows), "scored_candidates": len(keep), **{f"skipped_{k}": v for k, v in skipped.items()}}


def score_rows(rows: Sequence[dict], scorer, *, log: Callable[[str], None] = print) -> dict:
    """Score each row; a quota stop ends the loop cleanly, a judge error on one answer leaves its metrics None and goes on."""
    out, errors, stopped = [], [], None
    for r in rows:
        contexts = [s["text"] for s in r["used_sources"]]
        try:
            metrics = scorer.score(r["question"], strip_citation_markers(r["answer"]), contexts)
        except JudgeQuotaExhausted as exc:
            stopped = exc.reason
            break
        except JudgeError as exc:
            errors.append({"question_id": r["question_id"], "error": str(exc)[:200]})
            metrics = {m: None for m in METRICS}
        out.append({"question_id": r["question_id"], "paper_id": r["paper_id"], "arm": "R_long", "arm_base": "R", "mode": "long", "n_contexts": len(contexts),
                    "metrics": metrics})
        log(f"  scored {r['question_id']}  " + "  ".join(f"{k}={'-' if v is None else round(v, 3)}" for k, v in metrics.items()))
    return {"rows": out, "errors": errors, "stopped_reason": stopped}


def family_note(judge_model: str, generator_model: str | None) -> str:
    """How the judge relates to the generator that wrote the answers it scores (recorded in config.json)."""
    calibration = "; NOT calibrated against human labels"
    if generator_model is None:
        return "the generator's model is not recorded in the generation run, so the judge's relation to it is unknown" + calibration
    if judge_model == generator_model:
        return f"the judge is the SAME model as the generator ({judge_model}): self-preference bias is possible" + calibration
    if judge_model.split("-")[0] == generator_model.split("-")[0]:
        return (f"the judge ({judge_model}) is a different model of the SAME family as the generator ({generator_model}): "
                "family-level self-preference bias is still possible" + calibration)
    return f"the judge ({judge_model}) and the generator ({generator_model}) are from different model families" + calibration


def summarize(rows: Sequence[dict], selection: dict, result: dict, stats: JudgeStats) -> dict:
    return {"label": LABEL, "n_scored_rows": len(rows), "selection": selection,
            "metrics": {m: summarize_metric(rows, m) for m in METRICS},
            "interval": "95% paper-clustered bootstrap, 5,000 resamples, seed 0 (eval/compare.py::clustered_bootstrap); n is small, the intervals are wide",
            "judge_usage": stats.as_dict(), "errors": result["errors"], "stopped_reason": result["stopped_reason"]}


def main(argv=None, *, scorer_factory=None, judge: dict | None = None, log: Callable[[str], None] = print, sleep: Callable[[float], None] | None = None,
         clock: Callable[[], datetime] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="the generation run folder (its long-mode rows are scored)")
    ap.add_argument("--out", default=str(ROOT / "runs" / "gen"))
    ap.add_argument("--cache-dir", default=None, help="default: <out>/judge_cache")
    ap.add_argument("--max-judge-calls", type=int, default=200, help="network calls this session may make (a quota guard; cached replies are free)")
    ap.add_argument("--limit", type=int, default=None, help="score only the first LIMIT answers (a smoke test)")
    ap.add_argument("--prereg", default=str(PREREG_FILE))
    ap.add_argument("--prereg-hash-file", default=str(PREREG_HASH_FILE))
    args = ap.parse_args(argv)

    from eval.run_retrieval_eval import hardware_info, preregistration_info, verify_checksums, write_run

    prereg = preregistration_info(Path(args.prereg), Path(args.prereg_hash_file))  # refuses if the pre-registered text changed
    gen_config, gen_rows = load_generation_run(Path(args.run))
    candidates, selection = select_rows(gen_rows)
    todo = candidates[: args.limit] if args.limit else candidates

    if judge is None:
        from config import cfg
        from rag_core.config import RagConfig

        judge = RagConfig.from_config(cfg).judge
    if judge.get("provider") != "gemini":
        raise SystemExit("REFUSED: the configured judge is not Gemini (generation.judge.provider); only the Gemini judge is supported.")
    settings = JudgeSettings.from_config(judge)

    out_root = Path(args.out)
    cache = DiskCache(Path(args.cache_dir) if args.cache_dir else out_root / "judge_cache")
    stats = JudgeStats(max_calls=args.max_judge_calls)
    now = clock or (lambda: datetime.now(timezone.utc))
    pacer = Pacer(rpm=GEMINI_FREE_TIER_RPM, tpm=None, safety=0.8, clock=(lambda: now().timestamp()) if clock else None, sleep=sleep)  # one clock for pacer and report
    versions: dict = {}
    if scorer_factory is None:
        from eval.ragas_adapter import build_real_scorer, package_version
        from rag_core.env import load_env, present

        load_env()
        if not present("GEMINI_API_KEY")["GEMINI_API_KEY"]:
            print("SKIPPED: GEMINI_API_KEY is not available; nothing was sent anywhere.")
            return 0
        scorer_factory = build_real_scorer
        versions = {p: package_version(p) for p in ("ragas", "google-genai", "instructor", "openai", "sentence-transformers")}
    scorer = scorer_factory(settings, cache, stats, pacer, sleep)
    ragas_version = versions.get("ragas") or getattr(scorer, "version", None)

    from utils.fingerprint import code_fingerprint

    started = now()
    log(f"{LABEL}\nscoring {len(todo)} of {len(candidates)} scorable long answers ({selection}); judge {settings.model} (thinking budget {settings.thinking_budget}, "
        f"max output {settings.max_output_tokens}, temperature {settings.temperature}); at most {args.max_judge_calls} network calls this session")
    result = score_rows(todo, scorer, log=log)
    if hasattr(scorer, "close"):
        scorer.close()
    rows = result["rows"]
    complete = result["stopped_reason"] is None and len(rows) == len(todo)
    gen_rows_path = Path(args.run) / "per_question.jsonl"
    config = {
        "created_utc": started.isoformat(), "label": LABEL, "kind": "ragas_scoring", "preregistration": prereg,
        "scope_note": "Phase 4 minimal scope: faithfulness and answer relevancy only; no context precision/recall, no calibration (see the dated Deviations in eval/PREREGISTRATION_gen.md)",
        "generation_run": {"path": str(args.run), "per_question_sha256": hashlib.sha256(gen_rows_path.read_bytes()).hexdigest(), "label": gen_config.get("label"),
                           "sample": {k: gen_config.get("sample", {}).get(k) for k in ("seed", "n_drawn", "question_ids")},
                           "generator": gen_config.get("generator"), "code_fingerprint": gen_config.get("code_fingerprint")},
        "judge": {**settings.as_dict(), "provider": "gemini", "family_note": family_note(settings.model, (gen_config.get("generator") or {}).get("model")),
                  "strictness": 1, "strictness_note": "answer relevancy asks for one generated question (at temperature 0 more would be near-identical)"},
        "ragas_version": ragas_version, "libraries": versions,
        "embeddings": "local sentence-transformers/all-MiniLM-L6-v2 at its pinned revision (local cache only), used by answer relevancy",
        "judge_input_preprocessing": "the answer is scored with [S#] citation markers removed; contexts = the passages sent to the generator (used_sources), in order",
        "judge_cache": {"dir": str(cache.dir), "key": "sha256(judge model, temperature, thinking budget, output limit, API version, metric, ragas version, prompt, response schema)"},
        "max_judge_calls_this_session": args.max_judge_calls, "n_candidates": len(candidates), "n_attempted_this_session": len(todo),
        "code_fingerprint": code_fingerprint(), "hardware": hardware_info(), "complete": complete,
    }
    summary = summarize(rows, selection, result, stats)
    run_dir = write_run(out_root, config, rows, summary, started=started, split=gen_config.get("split", "dev"), eval_mode="gen_ragas")
    log(f"wrote {run_dir}  (checksums ok: {verify_checksums(run_dir)})")
    usage = stats.as_dict()
    log(f"SESSION REPORT: scored {len(rows)} of {len(todo)}; judge network calls {usage['_total_network_calls']} (budget {args.max_judge_calls}); remaining {len(todo) - len(rows)}")
    if result["errors"]:
        log(f"JUDGE ERRORS (metrics left empty): {result['errors']}")
    if result["stopped_reason"]:
        log(f"STOPPED: {result['stopped_reason']}. Quota exhausted, resume later: run the same command again (finished judge calls are cached).")
        return EXIT_QUOTA
    return 0 if complete else EXIT_QUOTA


if __name__ == "__main__":
    sys.exit(main())
