"""What the app reads from the committed artifacts (``app_data/results.json``, ``app_data/papers.json``). Pure: no streamlit, no network, never raises on a bad file.

``load_artifacts`` returns an ``Artifacts`` whose ``problems`` list says what is wrong (missing file, bad JSON, a section without a scope); the UI shows a notice and
renders what is valid. The helpers return the few facts other tabs quote (the reranker's latency cost, how much weaker corpus-wide search is), each WITH its scope label,
so a caller cannot quote a number without saying where it comes from. Nothing here computes a result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

SCOPE_LABELS = ("TEST", "DEV", "EXPLORATORY")


@dataclass
class Artifacts:
    results: dict | None = None
    papers: list[dict] = field(default_factory=list)
    examples: list[dict] = field(default_factory=list)
    source: dict = field(default_factory=dict)  # license / link / corpus hash of the paper list
    problems: list[str] = field(default_factory=list)

    @property
    def titles(self) -> dict[str, str]:
        return {p["id"]: p["title"] for p in self.papers}

    def title(self, paper_id: str) -> str:
        return self.titles.get(paper_id) or paper_id

    @property
    def has_results(self) -> bool:
        return bool(self.results)


def _read_json(path: Path, problems: list[str], what: str):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        problems.append(f"{what} not found at {path}. Build it with: python scripts/build_app_artifacts.py")
    except (OSError, ValueError) as exc:
        problems.append(f"{what} could not be read ({type(exc).__name__}). Rebuild it with: python scripts/build_app_artifacts.py")
    return None


def structure_problems(results: dict) -> list[str]:
    """A section that holds numbers must carry a scope (label, run folder, fingerprint). Checked at load time without touching the disk."""
    problems = []
    for key, section in results.items():
        if key in ("schema", "built_utc", "sources", "claims_log", "chunker_findings") or section is None:
            continue
        scope = section.get("scope") if isinstance(section, dict) else None
        if not isinstance(scope, dict) or scope.get("label") not in SCOPE_LABELS or not scope.get("run_folder") or not scope.get("code_fingerprint"):
            problems.append(f"section {key!r} has no valid scope: it is not shown")
    return problems


def load_artifacts(results_path: str | Path, papers_path: str | Path, base: str | Path = ".") -> Artifacts:
    base = Path(base)
    resolve = lambda p: Path(p) if Path(p).is_absolute() else base / p  # noqa: E731
    out = Artifacts()
    results = _read_json(resolve(results_path), out.problems, "results.json")
    if isinstance(results, dict):
        bad = structure_problems(results)
        out.problems += bad
        dropped = {p.split("'")[1] for p in bad}
        out.results = {k: v for k, v in results.items() if k not in dropped}
    papers = _read_json(resolve(papers_path), out.problems, "papers.json")
    if isinstance(papers, dict):
        out.papers = list(papers.get("papers") or [])
        out.examples = list((papers.get("examples") or {}).get("items") or [])
        out.source = papers.get("source") or {}
    return out


# ---- facts quoted by other tabs (each carries its scope) ---------------------------------------------------------------------------

def _arm(section: dict | None, name: str) -> dict | None:
    return next((a for a in (section or {}).get("arms", []) if a["arm"] == name), None)


def rerank_facts(results: dict | None) -> dict | None:
    """The reranker's measured cost and benefit on the TEST split (arm G against the frozen arm D), or None if the artifact has no such section."""
    test = (results or {}).get("retrieval_test")
    d, g = _arm(test, "D"), _arm(test, "G")
    cmp_ = next((c for c in (test or {}).get("comparisons", []) if c["pair"] == "D vs G"), None)
    if not (d and g and cmp_ and "rerank_p50_ms" in g["latency"]):
        return None
    return {"scope": test["scope"], "rerank_p50_ms": g["latency"]["rerank_p50_ms"], "rerank_p95_ms": g["latency"]["rerank_p95_ms"],
            "retrieval_p50_ms_without": d["latency"]["total_p50_ms"], "retrieval_p50_ms_with": g["latency"]["total_p50_ms"],
            "ndcg_without": d["metrics"]["nDCG@10"]["mean"], "ndcg_with": g["metrics"]["nDCG@10"]["mean"],
            "ndcg_gain": -cmp_["diff"], "gain_ci_low": -cmp_["ci_high"], "gain_ci_high": -cmp_["ci_low"], "holm_significant": cmp_["holm_significant"]}


def cross_paper_facts(results: dict | None) -> dict | None:
    """How much weaker searching all papers is than searching one (frozen arm D, DEV split), or None."""
    section = (results or {}).get("setting_comparison")
    if not section:
        return None
    by = {(e["setting"].split(" ")[0], e["arm"]): e["metrics"] for e in section["entries"]}
    paper, corpus = by.get(("paper", "D")), by.get(("corpus-wide", "D"))
    if not (paper and corpus):
        return None
    return {"scope": section["scope"], "paper_hit5": paper["HitRate@5"]["mean"], "corpus_hit5": corpus["HitRate@5"]["mean"],
            "paper_ndcg": paper["nDCG@10"]["mean"], "corpus_ndcg": corpus["nDCG@10"]["mean"], "corpus_doc_hit5": corpus["DocHit@5"]["mean"]}


def fmt_ms(ms: float) -> str:
    return f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms"


def scope_caption(scope: dict) -> str:
    """One line that says where a table comes from: label, split, setting, n, run folder and the first characters of the fingerprint."""
    return (f"{scope['label']} · {scope.get('split') or 'n/a'} split · {scope['setting']} · {scope['n_questions']} questions"
            f"{', ' + str(scope['n_papers']) + ' papers' if scope.get('n_papers') else ''} · run `{scope['run_folder']}` · fingerprint `{scope['code_fingerprint'][:12]}`")
