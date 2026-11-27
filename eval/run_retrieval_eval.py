"""Evaluation v2: RETRIEVAL ONLY. Calls the production Retriever (rag/retriever.py) and nothing else:
no LLM, no generation, no network.

    python eval/run_retrieval_eval.py --split dev --arm all --mode paper      # PRIMARY (Qasper's canonical setting)
    python eval/run_retrieval_eval.py --split dev --arm D --mode corpus --out runs/eval   # EXTENSION: corpus-wide

Arms (paper-mode grid of Stage 2a; the pre-registration lists them):
  A recursive / vector          B recursive / lexical        C recursive / hybrid, naive interleave
  D recursive / hybrid, RRF     E semantic  / vector         F semantic  / hybrid, RRF      (RRF: k=60, weights 1.0 / 0.7)
  G = D + cross-encoder rerank  H = F + rerank               R20_05..R60_10 RRF grid        B2/D2 stopword-filtered lexical leg
``--arm core`` = A-F, ``--arm all`` = all 15.

``--mode paper`` passes doc_ids=[the question's paper] to the Retriever (filter parity is tested elsewhere);
``--mode corpus`` passes none. Only ANSWERABLE questions are scored; unanswerable ones are reserved for the
generation evaluation. Output: runs/eval/<UTC>_<fingerprint8>_<split>_<mode>/ with config.json,
per_question.jsonl, summary.json and SHA256SUMS.

TEST SPLIT GUARD: ``--split test`` needs --i-understand-this-is-final, an existing eval/test_freeze.json whose
arm definitions match the arms being run, and every evaluation is appended to eval/test_ledger.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from eval.gold_spans import GoldSpan, paragraph_index  # noqa: E402
from eval.metrics import is_relevant as old_is_relevant  # noqa: E402  (the OLD token-F1 >= 0.25 metric, kept as a secondary column)
from eval.metrics import MATCH_THRESHOLD as OLD_THRESHOLD  # noqa: E402
from rag.eval_metrics import DEEP_KS, KS, count_relevant_chunks, hit_rate_at_k, question_metrics, reciprocal_rank  # noqa: E402

TOP_K = 10
CANDIDATE_K = 20
LATENCY_LABEL = ("retrieval_latency_ms: time spent inside Retriever.retrieve only (embed, vector, lexical, fuse, hydrate, "
                 "rerank, total). Excludes metric computation, any LLM or generation, and is NOT end-to-end latency.")
# Qasper's questions were written from a paper's title and abstract and the official task is single-document, so
# paper mode is the PRIMARY setting; corpus-wide retrieval is an extension (the questions are often unanswerable
# without knowing the paper).
SETTING_LABELS = {
    "paper": "PRIMARY: Qasper's canonical single-document task (the question's paper is given; doc_ids filter)",
    "corpus": "EXTENSION: corpus-wide retrieval (not Qasper's canonical task; questions were written about one paper)",
    "corpus_titled": "EXTENSION: corpus-wide, title-qualified (user names the paper)",
}
TITLED_QUERY = 'In the paper "{title}": {question}'  # corpus_titled: no doc filter; the title is only in the query text
PREREG_FILE = ROOT / "eval" / "PREREGISTRATION.md"
PREREG_HASH_FILE = ROOT / "eval" / "PREREGISTRATION.sha256"
PREREG_DEVIATIONS_MARKER = "## Deviations"
STAGES = ("total", "route", "embed", "vector", "lexical", "fuse", "hydrate", "rerank")
METRIC_KEYS = (
    [f"HitRate@{k}" for k in KS] + ["MRR", "nDCG@10"] + [f"DocHit@{k}" for k in KS] + [f"EvidenceRecall@{k}" for k in KS]
    + [f"EvidenceF1@{k}" for k in KS]
)
OLD_KEYS = [f"old_HitRate@{k}" for k in KS] + ["old_MRR"]
DEEP_KEYS = [f"Recall@{k}" for k in DEEP_KS] + [f"EvidenceRecall@{k}" for k in DEEP_KS]  # present only in runs made with top_k >= k
FREEZE_FILE = ROOT / "eval" / "test_freeze.json"
LEDGER_FILE = ROOT / "eval" / "test_ledger.jsonl"
DATASET_DIR = ROOT / "eval" / "datasets" / "qasper_dev"


@dataclass(frozen=True)
class Arm:
    name: str
    strategy: str  # recursive | semantic
    mode: str  # vector | lexical | hybrid
    fusion: str | None = None  # naive | rrf (hybrid only)
    rrf_k: int | None = None
    vector_weight: float | None = None
    lexical_weight: float | None = None
    lexical: str = "fts"  # fts | fts_stopword (stopword-filtered OR query, arm B2/D2)
    rerank: bool = False  # cross-encoder over the first ``rerank_k`` fused candidates, then keep top_k
    rerank_k: int | None = None
    backend: str = "minilm"  # minilm | gemini: which embedder produced the vectors of the store (same recursive chunks)
    route_m: int | None = None  # two-stage: route to the top-m papers with a paper-level index, then retrieve inside them


ARMS: dict[str, Arm] = {
    "A": Arm("A", "recursive", "vector"),
    "B": Arm("B", "recursive", "lexical"),
    "C": Arm("C", "recursive", "hybrid", "naive"),
    "D": Arm("D", "recursive", "hybrid", "rrf", 60, 1.0, 0.7),
    "E": Arm("E", "semantic", "vector"),
    "F": Arm("F", "semantic", "hybrid", "rrf", 60, 1.0, 0.7),
    "G": Arm("G", "recursive", "hybrid", "rrf", 60, 1.0, 0.7, rerank=True, rerank_k=20),
    "H": Arm("H", "semantic", "hybrid", "rrf", 60, 1.0, 0.7, rerank=True, rerank_k=20),
    "R20_05": Arm("R20_05", "recursive", "hybrid", "rrf", 20, 1.0, 0.5),
    "R20_07": Arm("R20_07", "recursive", "hybrid", "rrf", 20, 1.0, 0.7),
    "R20_10": Arm("R20_10", "recursive", "hybrid", "rrf", 20, 1.0, 1.0),
    "R60_05": Arm("R60_05", "recursive", "hybrid", "rrf", 60, 1.0, 0.5),
    "R60_10": Arm("R60_10", "recursive", "hybrid", "rrf", 60, 1.0, 1.0),
    "B2": Arm("B2", "recursive", "lexical", lexical="fts_stopword"),
    "D2": Arm("D2", "recursive", "hybrid", "rrf", 60, 1.0, 0.7, lexical="fts_stopword"),
    # ---- Stage 2b (eval/PREREGISTRATION_2b.md) ----
    "GA": Arm("GA", "recursive", "vector", backend="gemini"),
    "GD": Arm("GD", "recursive", "hybrid", "rrf", 60, 1.0, 0.7, backend="gemini"),
    "RT1": Arm("RT1", "recursive", "hybrid", "rrf", 60, 1.0, 0.7, route_m=1),
    "RT3": Arm("RT3", "recursive", "hybrid", "rrf", 60, 1.0, 0.7, route_m=3),
    "RT5": Arm("RT5", "recursive", "hybrid", "rrf", 60, 1.0, 0.7, route_m=5),
    "GRT3": Arm("GRT3", "recursive", "hybrid", "rrf", 60, 1.0, 0.7, backend="gemini", route_m=3),
}
ARM_ALIASES = {"all": list(ARMS)[:15], "core": ["A", "B", "C", "D", "E", "F"],
               "2b_paper": ["A", "D", "GA", "GD"], "2b_corpus": ["D", "RT1", "RT3", "RT5", "GRT3"]}


def component_key(arm: Arm) -> str:
    """Which (strategy, vector backend) store/retriever the arm reads: ``recursive`` or ``recursive@gemini``."""
    return arm.strategy if arm.backend == "minilm" else f"{arm.strategy}@{arm.backend}"


def arm_config(arms: list[Arm], top_k: int = TOP_K, candidate_k: int = CANDIDATE_K) -> dict:
    """The part of the configuration that the test freeze pins."""
    return {"arms": {a.name: asdict(a) for a in arms}, "top_k": top_k, "candidate_k": candidate_k,
            "rerank": any(a.rerank for a in arms)}


def select_arms(spec: str) -> list[Arm]:
    names = ARM_ALIASES[spec] if spec in ARM_ALIASES else [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [n for n in names if n not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s): {unknown}; choose from {list(ARMS)} or 'all'")
    return [ARMS[n] for n in names]


# ---- guard ---------------------------------------------------------------------------------------

class GuardRefused(SystemExit):
    pass


def check_test_guard(*, confirmed: bool, config: dict, freeze_path: Path) -> None:
    if not confirmed:
        raise GuardRefused("REFUSED: --split test is final. Pass --i-understand-this-is-final to evaluate it.")
    if not Path(freeze_path).exists():
        raise GuardRefused(f"REFUSED: no frozen configuration at {freeze_path}. It is created only when the owner asks.")
    frozen = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    for key in ("top_k", "candidate_k", "rerank"):
        if frozen.get(key) != config[key]:
            raise GuardRefused(f"REFUSED: {key} differs from the frozen configuration ({config[key]!r} != {frozen.get(key)!r}).")
    for name, definition in config["arms"].items():
        if frozen.get("arms", {}).get(name) != definition:
            raise GuardRefused(f"REFUSED: arm {name} differs from the frozen configuration.")


def append_ledger(ledger_path: Path, entry: dict) -> None:
    Path(ledger_path).parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


# ---- evaluation ----------------------------------------------------------------------------------

def _retrieve(retriever, arm: Arm, query: str, doc_ids, top_k: int, candidate_k: int, router=None):
    kw = dict(mode=arm.mode, top_k=top_k, candidate_k=candidate_k)
    if arm.mode == "hybrid":
        kw["fusion"] = arm.fusion
        if arm.fusion == "rrf":
            kw.update(rrf_k=arm.rrf_k, vector_weight=arm.vector_weight, lexical_weight=arm.lexical_weight)
    if arm.rerank:
        kw.update(rerank=True, rerank_k=arm.rerank_k)
    if arm.route_m:  # two-stage: the router (query text only) picks the papers; doc_ids is never passed by the caller
        if router is None:
            raise ValueError(f"arm {arm.name} routes but no router was provided")
        return retriever.retrieve_routed(query, router, route_m=arm.route_m, **kw)
    return retriever.retrieve(query, doc_ids=doc_ids, **kw)


def query_for(q: dict, eval_mode: str, titles: dict | None) -> str:
    """The text sent to the Retriever. Only corpus_titled changes it (the title is the user telling us which paper)."""
    if eval_mode == "corpus_titled":
        return TITLED_QUERY.format(title=titles[q["paper_id"]], question=q["question"])
    return q["question"]


def retriever_for(arm: Arm, base, reranker=None):
    """The Retriever for ``arm``: ``base`` itself, or a sibling over the same store, embedder and index with the
    stopword-filtered lexical leg and/or the injected reranker."""
    if arm.lexical == "fts" and not arm.rerank:
        return base
    if arm.rerank and reranker is None:
        raise ValueError(f"arm {arm.name} needs a reranker but none was provided")
    from rag.lexical_query import StopwordFilteredLexical
    from rag.retriever import Retriever

    lexical = StopwordFilteredLexical(base.lexical) if arm.lexical == "fts_stopword" else base.lexical
    return Retriever(base.store, base.embedder, lexical, reranker=reranker if arm.rerank else None, cfg=base.cfg)


def evaluate_arm(arm: Arm, retriever, chunk_index: dict, questions: list[dict], *, eval_mode: str, warmup: int,
                 seed: int, top_k: int = TOP_K, candidate_k: int = CANDIDATE_K, paragraphs: dict | None = None,
                 titles: dict | None = None, router=None, deep_ks: tuple = ()) -> list[dict]:
    """One row per question for ``arm``. Retrieval is timed by the Retriever itself (perf_counter inside
    ``retrieve``); everything else (relevance, metrics) happens after the call, outside the timed section.

    ``paragraphs`` (doc_id -> paragraph index) enables paragraph-level Evidence-F1 for questions that carry
    per-annotator ``references``; otherwise those metrics are None. ``titles`` (paper_id -> title) is needed
    for ``eval_mode == "corpus_titled"``. ``router`` (paper-level index) is needed for routed arms, which are
    defined for the corpus-wide settings only: the router receives the query text and nothing else.
    """
    if arm.route_m and eval_mode == "paper":
        raise ValueError(f"arm {arm.name} routes between papers; it is not defined in paper mode (the paper is given)")
    order = list(questions)
    random.Random(seed).shuffle(order)  # fixed order effect, recorded by the seed
    n_warm = min(warmup, len(order))
    for q in order[:n_warm]:  # warm-up: caches, lazy model load; results discarded
        _retrieve(retriever, arm, query_for(q, eval_mode, titles), [q["paper_id"]] if eval_mode == "paper" else None,
                  top_k, candidate_k, router)

    rows = []
    for q in order:
        doc_ids = [q["paper_id"]] if eval_mode == "paper" else None
        result = _retrieve(retriever, arm, query_for(q, eval_mode, titles), doc_ids, top_k, candidate_k, router)
        spans = [GoldSpan(**s) for s in q["gold_spans"]]
        n_rel = count_relevant_chunks(chunk_index.get(q["paper_id"], ()), spans)
        refs = q.get("references")
        m = question_metrics(
            result.hits, spans, q["paper_id"], n_rel,
            paragraphs=None if paragraphs is None or not refs else paragraphs.get(q["paper_id"]),
            reference_evidence=None if not refs else [set(r["evidence"]) for r in refs],
            deep_ks=deep_ks,
        )
        old = [old_is_relevant(q["gold_evidence"], h.text) for h in result.hits]
        old_metrics = {f"old_HitRate@{k}": hit_rate_at_k(old, k) for k in KS} | {"old_MRR": reciprocal_rank(old)}
        rels = m.pop("relevance")
        rows.append({
            "question_id": q["id"],
            "paper_id": q["paper_id"],
            "arm": arm.name,
            "strategy": arm.strategy,
            "eval_mode": eval_mode,
            "n_retrieved": len(result.hits),
            **({"routed_doc_ids": result.routed_doc_ids, "router_hit": q["paper_id"] in result.routed_doc_ids}
               if result.routed_doc_ids is not None else {}),
            "retrieved": [
                {"rank": h.rank, "chunk_id": h.chunk_id, "doc_id": h.doc_id, "section": h.section,
                 "char_start": h.char_start, "char_end": h.char_end,
                 "legs": {leg: {"rank": li.rank, "score": li.score} for leg, li in sorted(h.legs.items())},
                 "relevant": bool(rels[i]), "old_relevant": bool(old[i])}
                for i, h in enumerate(result.hits)
            ],
            "metrics": {k: (None if v is None else (bool(v) if isinstance(v, bool) else float(v) if not isinstance(v, int) else v))
                        for k, v in m.items()} | old_metrics,
            "retrieval_latency_ms": {k: float(v) for k, v in result.latency_ms.items()},
        })
    rows.sort(key=lambda r: r["question_id"])
    return rows


def _mean(values):
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def summarize(rows: list[dict], arms: list[Arm], *, warmup: int, eval_mode: str | None = None) -> dict:
    out = {"setting": SETTING_LABELS.get(eval_mode), "primary_setting": eval_mode == "paper",
           "latency_label": LATENCY_LABEL, "warmup_requests_discarded_per_arm": warmup,
           "old_metric_note": f"old_* = the previous relevance rule (token-F1 >= {OLD_THRESHOLD} against the evidence text); "
                              "shown only to compare metrics, not as a result", "arms": {}}
    for arm in arms:
        r = [x for x in rows if x["arm"] == arm.name]
        deep = [k for k in DEEP_KEYS if any(k in x["metrics"] for x in r)]
        entry = {"strategy": arm.strategy, "mode": arm.mode, "fusion": arm.fusion, "n_questions": len(r),
                 "n_unreachable_questions": sum(not x["metrics"]["reachable"] for x in r),
                 "means": {k: _mean(x["metrics"].get(k) for x in r) for k in METRIC_KEYS + deep + OLD_KEYS}, "retrieval_latency_ms": {}}
        for stage in STAGES:
            v = np.array([x["retrieval_latency_ms"].get(stage, 0.0) for x in r], dtype=float)
            entry["retrieval_latency_ms"][stage] = (
                {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "max": float(v.max()), "n_measured": int(len(v))}
                if len(v) else None)
        out["arms"][arm.name] = entry
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preregistration_info(path: Path, hash_file: Path) -> dict | None:
    """sha256 of the pre-registration, recorded in every run's config.json.

    ``sha256_original_text`` covers the text BEFORE a ``## Deviations`` section (appending deviations must not change
    it); ``sha256_file`` covers the whole file. If a recorded hash exists and the original text no longer matches it,
    the run is refused: the pre-registration may only be appended to.
    """
    path = Path(path)
    if not path.exists():
        return None
    text = path.read_bytes()
    data = text.replace(b"\r\n", b"\n")  # editors may rewrite line endings; the text is what is pinned
    cut = data.find(("\n" + PREREG_DEVIATIONS_MARKER).encode())
    original = (data if cut < 0 else data[:cut].rstrip() + b"\n").replace(b"\n", b"\r\n")  # canonical CRLF form, as first recorded
    info = {"path": str(path), "sha256_original_text": hashlib.sha256(original).hexdigest(),
            "sha256_file": hashlib.sha256(text).hexdigest(), "has_deviations_section": cut >= 0, "recorded_sha256": None}
    hash_file = Path(hash_file)
    if hash_file.exists():
        info["recorded_sha256"] = hash_file.read_text(encoding="utf-8").split()[0]
        if info["recorded_sha256"] != info["sha256_original_text"]:
            raise SystemExit("REFUSED: the pre-registration text no longer matches its recorded sha256 "
                             "(it may only be appended to, under '## Deviations').")
    return info


def write_run(out_root: Path, config: dict, rows: list[dict], summary: dict, *, started: datetime, split: str, eval_mode: str) -> Path:
    fp8 = (config["code_fingerprint"] or "nofingerprint")[:8]
    base = f"{started.strftime('%Y%m%dT%H%M%SZ')}_{fp8}_{split}_{eval_mode}"
    run_dir, n = Path(out_root) / base, 1
    while run_dir.exists():
        n += 1
        run_dir = Path(out_root) / f"{base}_{n}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(run_dir / "per_question.jsonl", "w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda r: (r["arm"], r["question_id"])):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    names = ("config.json", "per_question.jsonl", "summary.json")
    (run_dir / "SHA256SUMS").write_text("".join(f"{_sha256(run_dir / n_)}  {n_}\n" for n_ in names), encoding="utf-8")
    return run_dir


def verify_checksums(run_dir: Path) -> bool:
    for line in (Path(run_dir) / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if _sha256(Path(run_dir) / name) != digest:
            return False
    return True


# ---- environment description ------------------------------------------------------------------------

def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def _ram_gb() -> float | None:
    try:
        import ctypes

        class MEM(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

        m = MEM()
        m.dwLength = ctypes.sizeof(MEM)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return round(m.ullTotalPhys / 2**30, 1)
    except Exception:
        return None


def hardware_info() -> dict:
    return {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
            "cpu_count_logical": os.cpu_count(), "ram_gb": _ram_gb(), "python": platform.python_version(),
            "sqlite": sqlite3.sqlite_version}


def library_versions() -> dict:
    return {p: _version(p) for p in ("numpy", "torch", "sentence-transformers", "transformers", "chromadb", "scikit-learn",
                                     "langchain-text-splitters", "hypothesis")}


# ---- main ----------------------------------------------------------------------------------------

def build_real_components(keys: set[str]):
    """Real local stores + real embedders. Imported lazily so tests stay light.

    A key is a strategy (``recursive``: MiniLM vectors from the local HuggingFace cache only) or ``recursive@gemini`` (the same
    chunks with Gemini vectors, in the separate store of ``embedding_gemini.chroma_dir`` / ``ADVRAG_GEMINI_CHROMA``; the
    query embeddings come from the Gemini API through its disk cache and need ``GEMINI_API_KEY`` unless fully cached).
    """
    from config import cfg
    from db.store import build_store, resolve_store_path
    from rag.embedders import SentenceTransformerEmbedder
    from rag.lexical import SqliteFtsIndex, lexical_index_path
    from rag.retriever import Retriever

    embedder = SentenceTransformerEmbedder.from_config(cfg)
    comps = {}
    for key in sorted(keys):
        strategy, _, backend = key.partition("@")
        if backend == "gemini":
            from rag.gemini_embedder import GeminiEmbedder

            path = resolve_store_path(os.environ.get("ADVRAG_GEMINI_CHROMA") or cfg.embedding_gemini.chroma_dir)
            vec_embedder = GeminiEmbedder.from_config(cfg, min_interval_s=float(getattr(cfg.embedding_gemini, "min_interval_s", 0.0)))
            store = build_store(cfg, strategy, path=path)
            lex = SqliteFtsIndex(lexical_index_path(cfg, strategy, chroma_path=path))
        else:
            vec_embedder = embedder
            store = build_store(cfg, strategy)
            lex = SqliteFtsIndex(lexical_index_path(cfg, strategy))
        index: dict[str, list] = {}
        for c in store.get_all():
            index.setdefault(c.doc_id, []).append(c)
        comps[key] = {"retriever": Retriever(store, vec_embedder, lex, cfg=cfg), "chunk_index": index,
                      "info": {"vector_collection": store.name, "vector_count": store.count(), "lexical_path": str(lex.path),
                               "lexical_count": lex.count(), "ingest_signature": store.get_signature(),
                               "vector_embedder": getattr(vec_embedder, "name", None), "vector_embedder_revision": getattr(vec_embedder, "revision", None)}}
    return comps, embedder


def main(argv=None, *, components=None, embedder=None, reranker=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--arm", default="all", help="arm name(s) comma-separated, or 'all'")
    ap.add_argument("--mode", choices=list(SETTING_LABELS), default="paper",
                    help="paper = PRIMARY (Qasper's canonical single-document task); corpus = EXTENSION (corpus-wide); "
                         "corpus_titled = EXTENSION (corpus-wide, the query names the paper)")
    ap.add_argument("--top-k", type=int, default=TOP_K, help="hits kept per query (default 10; 20 only for the DocHit@20 diagnostic; 50 adds Recall@50)")
    ap.add_argument("--candidate-k", type=int, default=CANDIDATE_K, help="candidates taken from each leg before fusion (default 20; use >= top-k, e.g. 100, for a 50-deep hybrid list)")
    ap.add_argument("--prereg", default=str(PREREG_FILE), help="pre-registration file whose sha256 is recorded in config.json")
    ap.add_argument("--prereg-hash-file", default=str(PREREG_HASH_FILE))
    ap.add_argument("--prefetch-queries", action="store_true",
                    help="before the timed arms, embed every question once through the Gemini embedder (cold cache = real network "
                         "latency, recorded separately in config.json) so the arms' own latency has no network in it")
    ap.add_argument("--out", default=str(ROOT / "runs" / "eval"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--i-understand-this-is-final", dest="final", action="store_true")
    ap.add_argument("--golden", default=str(DATASET_DIR / "golden_set_v2.json"))
    ap.add_argument("--corpus", default=str(DATASET_DIR / "corpus.json"), help="needed for paragraph-level Evidence-F1")
    ap.add_argument("--split-file", default=str(DATASET_DIR / "split.json"))
    ap.add_argument("--freeze-file", default=str(FREEZE_FILE))
    ap.add_argument("--ledger-file", default=str(LEDGER_FILE))
    args = ap.parse_args(argv)

    arms = select_arms(args.arm)
    deep_ks = tuple(k for k in DEEP_KS if k <= args.top_k)  # Recall@50 only when the lists really are 50 deep
    config_arms = arm_config(arms, top_k=args.top_k, candidate_k=args.candidate_k)
    if args.split == "test":
        check_test_guard(confirmed=args.final, config=config_arms, freeze_path=Path(args.freeze_file))

    golden_path, split_path = Path(args.golden), Path(args.split_file)
    from eval.heldout import refuse_heldout_inputs

    refuse_heldout_inputs([golden_path, split_path, Path(args.corpus)])  # the official test set is opened only through eval.heldout
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("golden_set_sha256") != _sha256(golden_path):
        raise SystemExit("REFUSED: split.json was made for a different golden set (sha256 mismatch).")
    ids = set(split[args.split])
    by_id = {q["id"]: q for q in golden["questions"]}
    questions = [by_id[i] for i in sorted(ids) if by_id[i]["answerable"]]
    n_unanswerable = sum(not by_id[i]["answerable"] for i in ids)

    paragraphs = None
    titles = None
    corpus = None
    corpus_path = Path(args.corpus)
    if args.mode == "corpus_titled" or any(a.route_m for a in arms) or any(q.get("references") for q in questions):
        if not corpus_path.exists():
            raise SystemExit(f"REFUSED: the corpus {corpus_path} is missing (needed for titles, paper routing and Evidence-F1).")
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        if any(q.get("references") for q in questions):
            paragraphs = {pid: paragraph_index(pid, corpus[pid]) for pid in {q["paper_id"] for q in questions}}
        if args.mode == "corpus_titled":
            titles = {pid: corpus[pid]["title"] for pid in {q["paper_id"] for q in questions}}
            if not all(titles.values()):
                raise SystemExit("REFUSED: a scored paper has no title; corpus_titled would be ill-defined.")
    if any(a.route_m for a in arms) and args.mode == "paper":
        raise SystemExit("REFUSED: routed arms (RTm) are defined for the corpus-wide settings only.")

    prereg = preregistration_info(Path(args.prereg), Path(args.prereg_hash_file))

    from utils.fingerprint import code_fingerprint

    fingerprint = code_fingerprint()
    started = datetime.now(timezone.utc)

    if components is None:
        components, embedder = build_real_components({component_key(a) for a in arms})
    if any(a.rerank for a in arms) and reranker is None:
        from config import cfg as _cfg
        from rag.rerankers import CrossEncoderReranker

        reranker = CrossEncoderReranker.from_config(_cfg)  # lazy: loaded at the first (warm-up) call, local cache only
    if args.split == "test":
        append_ledger(Path(args.ledger_file), {"utc": started.isoformat(), "code_fingerprint": fingerprint, "split": "test",
                                               "mode": args.mode, "arms": [a.name for a in arms], "golden_set_sha256": split["golden_set_sha256"]})

    # paper-level routers: one per vector backend, built from titles and abstracts ONLY (no question, no gold information)
    routers, router_info = {}, {}
    if any(a.route_m for a in arms):
        from rag.router_docs import PaperRouter, papers_from_corpus

        papers = papers_from_corpus(corpus)
        for a in arms:
            if a.route_m and a.backend not in routers:
                routers[a.backend] = PaperRouter(papers, components[component_key(a)]["retriever"].embedder)
                router_info[a.backend] = {"papers_indexed": len(routers[a.backend]), "text": "title + abstract of each paper only",
                                          "legs": "exact cosine over paper vectors + SQLite FTS5 (porter, OR)", "fusion": "RRF k=60, 1.0 / 0.7",
                                          "candidates_per_leg": routers[a.backend].candidate_k}
    gemini_embedders = {k: c["retriever"].embedder for k, c in components.items() if k.endswith("@gemini")}
    prefetch_stats = None
    if args.prefetch_queries and gemini_embedders:
        emb = next(iter(gemini_embedders.values()))
        build_stats = emb.stats()  # paper-index embedding (if routed arms exist) happened above
        emb.reset_stats()
        for text in dict.fromkeys(query_for(q, args.mode, titles) for q in questions):
            emb.encode_query(text)
        prefetch_stats = {"router_and_setup": build_stats, "queries": emb.stats()}
        emb.reset_stats()

    rows: list[dict] = []
    for arm in arms:
        comp = components[component_key(arm)]
        print(f"arm {arm.name} ({arm.strategy}/{arm.mode}{'-' + arm.fusion if arm.fusion else ''}"
              f"{'+rerank' if arm.rerank else ''}{' stopword-lexical' if arm.lexical != 'fts' else ''}"
              f"{' vectors=' + arm.backend if arm.backend != 'minilm' else ''}{' route_m=' + str(arm.route_m) if arm.route_m else ''}) "
              f"on {len(questions)} {args.split} questions, mode={args.mode}")
        rows += evaluate_arm(arm, retriever_for(arm, comp["retriever"], reranker), comp["chunk_index"], questions,
                             eval_mode=args.mode, warmup=args.warmup, seed=args.seed, top_k=args.top_k, candidate_k=args.candidate_k,
                             paragraphs=paragraphs, titles=titles, router=routers.get(arm.backend), deep_ks=deep_ks)

    config = {
        "created_utc": started.isoformat(),
        "split": args.split,
        "eval_mode": args.mode,
        "setting": SETTING_LABELS[args.mode],
        "query_template": TITLED_QUERY if args.mode == "corpus_titled" else "the question text, unchanged",
        "preregistration": prereg,
        **config_arms,
        "reranker": reranker.describe() if reranker is not None and any(a.rerank for a in arms) and hasattr(reranker, "describe") else None,
        "seed": args.seed,
        "deep_recall_ks": list(deep_ks),
        "recall_definition": "Recall@k = relevant chunks in the top k / relevant chunks that exist in the gold paper's index (reachable questions only); EvidenceRecall@k = share of evidence paragraphs covered by a top-k chunk",
        "warmup_requests_discarded_per_arm": args.warmup,
        "scored_questions": len(questions),
        "scored_papers": len({q["paper_id"] for q in questions}),
        "unanswerable_questions_not_scored": n_unanswerable,
        "relevance_rule": "same doc_id and section, character overlap >= 0.5 * min(len(span), len(chunk))",
        "evidence_f1": ("paragraph-level: retrieved chunks mapped to the source paragraphs their offsets overlap (>= 1 character); "
                        "F1 of that paragraph-text set against each annotator's evidence paragraphs, max over annotators "
                        "(official Qasper definition); annotators whose evidence is only tables/figures are skipped"
                        if paragraphs is not None else "not computed (questions carry no references)"),
        "corpus": {"path": str(corpus_path), "sha256": _sha256(corpus_path)} if paragraphs is not None else None,
        "latency_label": LATENCY_LABEL,
        "embedder": None if embedder is None else {"class": type(embedder).__name__, "name": embedder.name, "revision": embedder.revision,
                                                   "max_seq_length": embedder.max_seq_length},
        "code_fingerprint": fingerprint,
        "golden_set": {"path": str(golden_path), "sha256": _sha256(golden_path)},
        "split_file": {"path": str(split_path), "sha256": _sha256(split_path)},
        "stores": {k: comp["info"] for k, comp in components.items() if any(component_key(a) == k for a in arms)},
        "routers": router_info or None,
        "gemini": None if not gemini_embedders else {
            "embedder": next(iter(gemini_embedders.values())).describe(),
            "query_prefetch_network_latency": prefetch_stats,
            "stats_during_timed_arms": next(iter(gemini_embedders.values())).stats(),
            "note": "latency of the timed arms excludes the network when --prefetch-queries was used (embeddings come from the disk cache); "
                    "the network latency of a query embedding is in query_prefetch_network_latency.queries.network_latency_ms"},
        "hardware": hardware_info(),
        "libraries": library_versions(),
    }
    summary = summarize(rows, arms, warmup=args.warmup, eval_mode=args.mode)
    run_dir = write_run(Path(args.out), config, rows, summary, started=started, split=args.split, eval_mode=args.mode)
    print(f"wrote {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
