"""Builds ``app_data/results.json`` and ``app_data/papers.json``: the ONLY data the app's Evaluation and Chat tabs read about results.

Everything is computed from saved run folders (``runs/...``: ``config.json``, ``per_question.jsonl``, ``summary.json``, ``SHA256SUMS``) and parsed from
``docs/CLAIMS_LOG.md``; nothing is typed by hand. Each section carries a ``scope`` (label TEST / DEV / EXPLORATORY, split, setting, n, run folder, the run's code
fingerprint). ``validate`` FAILS the build if a section has no existing run folder, if its fingerprint does not match the run's ``config.json``, or if a number sits outside a sourced
section; ``crosscheck_claims`` FAILS it if a TEST value differs from the corresponding row of the claims log (3 decimals). The builder runs nothing: no retrieval, no LLM.

    python scripts/build_app_artifacts.py            # writes app_data/results.json and app_data/papers.json
    python scripts/build_app_artifacts.py --check    # fails if the committed files are stale (differ from a fresh build, ignoring the build time)
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA = 1
SCOPE_LABELS = ("TEST", "DEV", "EXPLORATORY")
RETRIEVAL_METRICS = ["nDCG@10", "MRR", "HitRate@5", "EvidenceRecall@10", "EvidenceF1@5"]
TEST_ARMS = ["A", "D", "F", "G"]
PREREGISTERED_TEST_PAIRS = [("A", "D"), ("F", "D"), ("D", "G")]  # eval/PREREGISTRATION.md: A vs frozen, F vs frozen, frozen vs frozen + reranker
# The run folders the app displays (the published TEST run, the dev ablation and its corpus-wide counterpart). A later run is shown only after this list is edited.
TEST_RUN = "runs/eval/20261122T144233Z_3afc204d_test_paper"
DEV_PAPER_RUN = "runs/eval/20261121T125301Z_0d717273_dev_paper"
DEV_CORPUS_RUN = "runs/eval/20261121T131902Z_0d717273_dev_corpus"
CORPUS = "eval/datasets/qasper_dev/corpus.json"
EXAMPLES = 4
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class BuildError(SystemExit):
    """The build is refused (a message, exit status 1)."""

    def __init__(self, message: str):
        super().__init__(f"BUILD REFUSED: {message}")


# ---- run folders -------------------------------------------------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_run(root: Path, rel: str) -> dict:
    """``{"rel", "dir", "config", "summary", "rows"}`` of a run folder; refuses a folder whose checksums do not match (it was modified after the run)."""
    run_dir = Path(root) / rel
    if not run_dir.is_dir():
        raise BuildError(f"run folder not found: {rel}")
    sums = run_dir / "SHA256SUMS"
    if sums.exists():
        for line in sums.read_text(encoding="utf-8").splitlines():
            digest, name = line.split("  ", 1)
            if sha256_file(run_dir / name) != digest:
                raise BuildError(f"{rel}/{name} does not match SHA256SUMS: the run folder was modified")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8")) if (run_dir / "summary.json").exists() else {}
    rows = []
    if (run_dir / "per_question.jsonl").exists():
        rows = [json.loads(line) for line in (run_dir / "per_question.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return {"rel": rel, "dir": run_dir, "config": config, "summary": summary, "rows": rows}


def scope_of(run: dict, label: str, *, setting: str, n_questions: int, n_papers: int, note: str = "") -> dict:
    cfg = run["config"]
    return {"label": label, "split": cfg.get("split"), "setting": setting, "n_questions": n_questions, "n_papers": n_papers, "run_folder": run["rel"],
            "code_fingerprint": cfg["code_fingerprint"], "created_utc": cfg.get("created_utc"), "note": note}


def arm_description(definition: dict) -> str:
    parts = [definition.get("strategy", "?"), definition.get("mode", "?") + (f" ({definition['fusion']})" if definition.get("fusion") else "")]
    if definition.get("fusion") == "rrf":
        parts.append(f"k={definition['rrf_k']}, weights {definition['vector_weight']:g}/{definition['lexical_weight']:g}")
    if definition.get("lexical") and definition["lexical"] != "fts":
        parts.append("stopword-filtered lexical")
    if definition.get("rerank"):
        parts.append("+ cross-encoder rerank")
    return ", ".join(parts)


def _clustered(values: list[float], papers: list[str]) -> dict:
    from eval.answer_metrics import mean_with_interval

    return mean_with_interval(values, papers)


def arm_metrics(run: dict, arm: str, metrics: Iterable[str], *, intervals: bool) -> dict:
    """Mean (and paper-clustered 95% CI) of each metric over the rows of ``arm``; the mean must equal the run's own ``summary.json`` mean."""
    rows = [r for r in run["rows"] if r["arm"] == arm]
    out: dict[str, Any] = {}
    for m in metrics:
        pairs = [(float(r["metrics"][m]), r["paper_id"]) for r in rows if r["metrics"].get(m) is not None]
        mean = sum(v for v, _ in pairs) / len(pairs)
        summary_mean = run["summary"]["arms"][arm]["means"][m]
        if abs(mean - summary_mean) > 1e-9:
            raise BuildError(f"{run['rel']} arm {arm} {m}: the mean of the rows ({mean}) differs from summary.json ({summary_mean})")
        entry = {"mean": mean, "n": len(pairs)}
        if intervals:
            ci = _clustered([v for v, _ in pairs], [p for _, p in pairs])
            entry.update(ci_low=ci["ci_low"], ci_high=ci["ci_high"], n_papers=ci["n_papers"])
        out[m] = entry
    return out


def latency(run: dict, arm: str) -> dict:
    stages = run["summary"]["arms"][arm]["retrieval_latency_ms"]
    out = {"total_p50_ms": stages["total"]["p50"], "total_p95_ms": stages["total"]["p95"], "n_measured": stages["total"].get("n_measured")}
    if "rerank" in stages and stages["rerank"].get("p50"):
        out.update(rerank_p50_ms=stages["rerank"]["p50"], rerank_p95_ms=stages["rerank"]["p95"])
    return out


# ---- sections -----------------------------------------------------------------------------------------------------------------------

def retrieval_test_section(root: Path) -> dict:
    run = load_run(root, TEST_RUN)
    cfg = run["config"]
    if cfg.get("split") != "test":
        raise BuildError(f"{TEST_RUN} is not a test-split run")
    from eval.compare import compare

    report = compare(run["rows"], PREREGISTERED_TEST_PAIRS, ("nDCG@10",))
    arms = [{"arm": a, "description": arm_description(cfg["arms"][a]), "metrics": arm_metrics(run, a, RETRIEVAL_METRICS, intervals=True),
             "latency": latency(run, a)} for a in TEST_ARMS]
    comparisons = [{"pair": f"{r['arm_x']} vs {r['arm_y']}", "metric": r["metric"], "n": r["n"], "n_papers": r["n_papers"], "mean_x": r["mean_x"], "mean_y": r["mean_y"],
                    "diff": r["mean_diff"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p_value": r["p_value"], "holm_significant": r["holm_significant"],
                    "wins": r["wins"], "losses": r["losses"], "ties": r["ties"]} for r in report["results"]]
    return {"scope": scope_of(run, "TEST", setting=cfg["setting"], n_questions=cfg["scored_questions"], n_papers=cfg["scored_papers"],
                              note="The frozen configuration D and the pre-registered comparisons; this split was run once."),
            "arms": arms, "comparisons": comparisons, "comparison_method": {"interval": "95% paper-clustered paired bootstrap", "resamples": report["resamples"], "seed": report["seed"],
                                                                             "holm_family_size": report["comparisons_made"], "alpha": report["alpha"],
                                                                             "difference": "first arm minus second arm"},
            "reranker": cfg.get("reranker"), "embedder": cfg.get("embedder")}


def retrieval_dev_section(root: Path) -> dict:
    run = load_run(root, DEV_PAPER_RUN)
    cfg = run["config"]
    arms = [{"arm": a, "description": arm_description(d), "metrics": arm_metrics(run, a, RETRIEVAL_METRICS, intervals=False), "latency": latency(run, a)}
            for a, d in cfg["arms"].items()]
    return {"scope": scope_of(run, "DEV", setting=cfg["setting"], n_questions=cfg["scored_questions"], n_papers=cfg["scored_papers"],
                              note="Development split: used to choose among configurations, so these numbers are optimistic by construction. Not a result."),
            "arms": arms}


def setting_comparison_section(root: Path) -> dict:
    paper, corpus = load_run(root, DEV_PAPER_RUN), load_run(root, DEV_CORPUS_RUN)
    keys = ["nDCG@10", "HitRate@5", "DocHit@5"]
    entries = []
    for run, label in ((paper, "paper (the question's paper is given)"), (corpus, "corpus-wide (all papers searched, no filter)")):
        for a in ("A", "D"):
            entries.append({"setting": label, "arm": a, "run_folder": run["rel"], "code_fingerprint": run["config"]["code_fingerprint"],
                            "metrics": {k: {"mean": run["summary"]["arms"][a]["means"][k]} for k in keys}})
    cfg = corpus["config"]
    return {"scope": scope_of(corpus, "DEV", setting="EXTENSION: corpus-wide vs the primary single-paper setting", n_questions=cfg["scored_questions"], n_papers=cfg["scored_papers"],
                              note=f"Same 174 dev questions in both settings; corpus-wide numbers from {DEV_CORPUS_RUN}, single-paper numbers from {DEV_PAPER_RUN}."),
            "entries": entries}


def pick_generation_run(root: Path) -> str:
    runs = sorted(p for p in (Path(root) / "runs" / "gen").glob("*_dev_gen_R*") if (p / "config.json").exists())
    complete = [p for p in runs if json.loads((p / "config.json").read_text(encoding="utf-8")).get("complete") is True]
    if not complete:
        raise BuildError("no complete generation run under runs/gen")
    return f"runs/gen/{complete[-1].name}"


def generation_section(root: Path) -> dict:
    rel = pick_generation_run(root)
    run = load_run(root, rel)
    cfg, s = run["config"], run["summary"]
    n_papers = len({r["paper_id"] for r in run["rows"]})
    keep = lambda d, keys: {k: d[k] for k in keys if k in d}  # noqa: E731
    return {"scope": scope_of(run, "EXPLORATORY", setting=cfg["setting"], n_questions=cfg["sample"]["n_drawn"], n_papers=n_papers,
                              note=cfg["label"]),
            "generator": keep(cfg["generator"], ["model", "provider", "reasoning_effort", "temperature", "top_k"]),
            "sample": {"seed": cfg["sample"]["seed"], "n_drawn": cfg["sample"]["n_drawn"]},
            "short_mode": s["short_mode"], "long_mode": s["long_mode"], "finish_reasons": s["finish_reasons"],
            "metrics_note": "Answer-F1 is the maximum token F1 over the annotators' references (SQuAD normalisation); an abstention scores 0. Intervals: 95% paper-clustered bootstrap."}


def ragas_section(root: Path, gen_rel: str) -> dict | None:
    candidates = []
    for p in sorted((Path(root) / "runs" / "gen").glob("*_gen_ragas*")):
        cfg = json.loads((p / "config.json").read_text(encoding="utf-8"))
        if Path(cfg["generation_run"]["path"]).name == Path(gen_rel).name:
            candidates.append(p)
    if not candidates:
        return None
    run = load_run(root, f"runs/gen/{candidates[-1].name}")
    cfg, s = run["config"], run["summary"]
    scope = scope_of(run, "EXPLORATORY", setting="RAGAS faithfulness and answer relevancy of the long answers, UNCALIBRATED Gemini judge", n_questions=s["n_scored_rows"],
                     n_papers=len({r["paper_id"] for r in run["rows"]}), note=cfg["label"])
    out = {"scope": scope, "complete": bool(cfg["complete"]), "n_scored": s["n_scored_rows"], "n_scorable": s["selection"]["scored_candidates"], "selection": s["selection"],
           "judge": {k: cfg["judge"][k] for k in ("model", "temperature", "thinking_budget", "max_output_tokens", "strictness")}, "ragas_version": cfg["ragas_version"],
           "stopped_reason": s.get("stopped_reason")}
    if cfg["complete"]:
        out["metrics"] = s["metrics"]
    return out


def examples_and_papers(root: Path, gen_rel: str) -> dict:
    corpus_path = Path(root) / CORPUS
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    papers = sorted(({"id": pid, "title": (rec.get("title") or "").strip()} for pid, rec in corpus.items()), key=lambda p: p["id"])
    run = load_run(root, gen_rel)
    titles = {p["id"]: p["title"] for p in papers}
    good = [r for r in run["rows"] if r["mode"] == "long" and not r["abstained"] and r["citations"] and not r["invalid_citations"]]
    chosen, used = [], set()
    for r in sorted(good, key=lambda r: hashlib.sha256(("examples:" + r["question_id"]).encode()).hexdigest()):
        if r["paper_id"] in used:
            continue
        used.add(r["paper_id"])
        chosen.append({"question_id": r["question_id"], "paper_id": r["paper_id"], "paper_title": titles.get(r["paper_id"], ""), "question": r["question"]})
        if len(chosen) == EXAMPLES:
            break
    return {"schema": SCHEMA, "source": {"corpus": CORPUS, "corpus_sha256": sha256_file(corpus_path), "license": "Qasper, CC BY 4.0 (Dasigi et al., 2021)",
                                          "link": "https://allenai.org/data/qasper"},
            "papers": papers,
            "examples": {"run_folder": gen_rel, "code_fingerprint": run["config"]["code_fingerprint"], "items": chosen,
                         "rule": "public dev questions whose long answer in the generation run cited a valid source and did not abstain; one per paper, ordered by sha256('examples:' + question id). "
                                 "Chosen because they ANSWERED, so they are not representative of the abstention or error rate."}}


# ---- the claims log -----------------------------------------------------------------------------------------------------------------

def parse_claims_log(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        if not line.startswith("|") or line.startswith("|--") or line.startswith("| Metric"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5:
            continue
        metric, value, folders, fp, note = cells[0], cells[1], cells[2], cells[3], "|".join(cells[4:])
        rows.append({"metric": metric.replace("**", ""), "value": value, "run_folders": re.findall(r"`(runs/[^`]+)`", folders),
                     "code_fingerprint": (re.findall(r"`([0-9a-f]{64})`", fp) or [""])[0], "scope_note": note})
    return rows


def chunker_findings(rows: list[dict]) -> list[dict]:
    return [r for r in rows if any(f.startswith(("runs/measurements/chunk")) for f in r["run_folders"])]


def crosscheck_claims(test: dict, rows: list[dict]) -> list[str]:
    """Differences between the TEST means computed from the run and the claims-log rows that state them (3 decimals). Missing rows are not an error."""
    prefix = {"D": "TEST frozen configuration D", "A": "TEST arm A", "F": "TEST arm F", "G": "TEST arm G"}
    problems = []
    for arm in test["arms"]:
        for metric, entry in arm["metrics"].items():
            for row in rows:
                if row["metric"].startswith(prefix[arm["arm"]]) and row["metric"].endswith(metric) and row["run_folders"][:1] == [TEST_RUN]:
                    if row["value"].split()[0] != f"{entry['mean']:.3f}":
                        problems.append(f"{arm['arm']} {metric}: claims log says {row['value']!r}, the run gives {entry['mean']:.3f}")
    return problems


# ---- build and validate ---------------------------------------------------------------------------------------------------------------

def build(root: Path) -> tuple[dict, dict]:
    root = Path(root)
    claims = parse_claims_log((root / "docs" / "CLAIMS_LOG.md").read_text(encoding="utf-8"))
    test = retrieval_test_section(root)
    gen = generation_section(root)
    results = {"schema": SCHEMA, "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "sources": {"claims_log_sha256": sha256_file(root / "docs" / "CLAIMS_LOG.md"), "claims_log_rows": len(claims)},
               "retrieval_test": test, "retrieval_dev": retrieval_dev_section(root), "setting_comparison": setting_comparison_section(root),
               "generation": gen, "ragas": ragas_section(root, gen["scope"]["run_folder"]),
               "chunker_findings": chunker_findings(claims), "claims_log": claims}
    papers = examples_and_papers(root, gen["scope"]["run_folder"])
    problems = validate(results, root) + crosscheck_claims(test, claims)
    if problems:
        raise BuildError("; ".join(problems))
    return results, papers


def _numeric_leaves(value: Any) -> Iterable[Any]:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _numeric_leaves(v)
    elif isinstance(value, list):
        for v in value:
            yield from _numeric_leaves(v)


def validate_source(root: Path, run_folder: Any, fingerprint: Any, where: str) -> list[str]:
    """A source is valid if the run folder exists and the fingerprint is a sha256 equal to the run's own ``config.json`` (when it records one)."""
    problems = []
    if not run_folder or not isinstance(run_folder, str) or not (Path(root) / run_folder).is_dir():
        return [f"{where}: no existing source run folder ({run_folder!r})"]
    if not isinstance(fingerprint, str) or not HEX64.match(fingerprint):
        return [f"{where}: no code fingerprint"]
    cfg_path = Path(root) / run_folder / "config.json"
    if cfg_path.exists():
        recorded = json.loads(cfg_path.read_text(encoding="utf-8")).get("code_fingerprint")
        if recorded and recorded != fingerprint:
            problems.append(f"{where}: fingerprint does not match {run_folder}/config.json")
    return problems


def validate(results: dict, root: Path) -> list[str]:
    """Every section needs a ``scope`` with an existing run folder and a matching fingerprint; every number must live inside such a section."""
    problems: list[str] = []
    meta = {"schema", "built_utc", "sources"}
    for key, section in results.items():
        if key in meta or section is None:
            continue
        if key == "claims_log" or key == "chunker_findings":
            for i, row in enumerate(section):
                label = f"{key}[{i}] ({row['metric'][:50]})"
                if not row["run_folders"]:
                    problems.append(f"{label}: no source run folder")
                    continue
                problems += validate_source(root, row["run_folders"][0], row["code_fingerprint"], label)  # the primary source is the run folder
                for extra in row["run_folders"][1:]:  # secondary sources (an analysis file) must exist
                    if not (Path(root) / extra).exists():
                        problems.append(f"{label}: secondary source {extra} does not exist")
            continue
        scope = section.get("scope") if isinstance(section, dict) else None
        if not scope:
            problems.append(f"{key}: no scope (a section without a source cannot hold numbers)")
            continue
        if scope.get("label") not in SCOPE_LABELS:
            problems.append(f"{key}: scope label {scope.get('label')!r} is not one of {SCOPE_LABELS}")
        problems += validate_source(root, scope.get("run_folder"), scope.get("code_fingerprint"), f"{key}.scope")
        for entry in section.get("entries", []):
            problems += validate_source(root, entry.get("run_folder"), entry.get("code_fingerprint"), f"{key}.entries")
    return problems


def strip_volatile(results: dict) -> dict:
    out = dict(results)
    out.pop("built_utc", None)
    return out


def write(root: Path, results: dict, papers: dict) -> tuple[Path, Path]:
    out_dir = Path(root) / "app_data"
    out_dir.mkdir(exist_ok=True)
    r_path, p_path = out_dir / "results.json", out_dir / "papers.json"
    r_path.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    p_path.write_text(json.dumps(papers, indent=1, ensure_ascii=False), encoding="utf-8")
    return r_path, p_path
