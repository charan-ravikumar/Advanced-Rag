"""Router diagnostics on the DEV split (descriptive, no retrieval, no gold information inside the router).

    python scripts/router_diagnostics.py [--backend minilm|gemini|both]

Builds the paper-level router (title + abstract of every corpus paper) and reports DocHit@m for m = 1, 3, 5, 10: the share of
answerable dev questions whose paper is among the router's top-m. The question text is the only input to ``route``; the gold
paper id is used only afterwards, to score. Saved under runs/measurements/<UTC>_router_diagnostics/ with the code fingerprint
and the Stage 2b pre-registration hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

DATASET_DIR = ROOT / "eval" / "datasets" / "qasper_dev"
MS = (1, 3, 5, 10)


def doc_hit_at_m(router, questions: list[dict], ms=MS) -> dict:
    """DocHit@m for each m; ``router.route`` receives the question text only."""
    top = {q["id"]: router.route(q["question"], max(ms)) for q in questions}
    return {f"DocHit@{m}": sum(q["paper_id"] in top[q["id"]][:m] for q in questions) / len(questions) for m in ms}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backend", choices=["minilm", "gemini", "both"], default="minilm")
    ap.add_argument("--split", default="dev", choices=["dev"])  # the test split is never evaluated here
    ap.add_argument("--out", default=str(ROOT / "runs" / "measurements"))
    ap.add_argument("--prereg", default=str(ROOT / "eval" / "PREREGISTRATION_2b.md"))
    ap.add_argument("--prereg-hash-file", default=str(ROOT / "eval" / "PREREGISTRATION_2b.sha256"))
    args = ap.parse_args(argv)

    from config import cfg
    from eval.run_retrieval_eval import preregistration_info
    from rag.embedders import SentenceTransformerEmbedder
    from rag.router_docs import PaperRouter, papers_from_corpus
    from utils.fingerprint import code_fingerprint

    prereg = preregistration_info(Path(args.prereg), Path(args.prereg_hash_file))
    golden = json.loads((DATASET_DIR / "golden_set_v2.json").read_text(encoding="utf-8"))
    split = json.loads((DATASET_DIR / "split.json").read_text(encoding="utf-8"))
    by_id = {q["id"]: q for q in golden["questions"]}
    questions = [by_id[i] for i in sorted(split[args.split]) if by_id[i]["answerable"]]
    corpus = json.loads((DATASET_DIR / "corpus.json").read_text(encoding="utf-8"))
    papers = papers_from_corpus(corpus)

    embedders = {}
    if args.backend in ("minilm", "both"):
        embedders["minilm"] = SentenceTransformerEmbedder.from_config(cfg)
    if args.backend in ("gemini", "both"):
        from rag.gemini_embedder import GeminiEmbedder

        embedders["gemini"] = GeminiEmbedder.from_config(cfg, min_interval_s=float(getattr(cfg.embedding_gemini, "min_interval_s", 0.0)))

    results = {}
    for name, emb in embedders.items():
        router = PaperRouter(papers, emb)
        results[name] = {"embedder": getattr(emb, "name", name), "revision": getattr(emb, "revision", None), "papers_indexed": len(router),
                         "questions": len(questions), "questions_papers": len({q["paper_id"] for q in questions}), **doc_hit_at_m(router, questions),
                         **({"embedder_stats": emb.stats()} if hasattr(emb, "stats") else {})}
    started = datetime.now(timezone.utc)
    out = Path(args.out) / f"{started.strftime('%Y%m%dT%H%M%SZ')}_router_diagnostics"
    out.mkdir(parents=True)
    payload = {"created_utc": started.isoformat(), "code_fingerprint": code_fingerprint(), "preregistration": prereg,
               "scope": "dev split, answerable questions; development only, not publishable",
               "router": "title + abstract per paper; exact cosine + FTS5, RRF k=60 1.0/0.7, 20 candidates per leg",
               "golden_set_sha256": hashlib.sha256((DATASET_DIR / "golden_set_v2.json").read_bytes()).hexdigest(), "results": results}
    (out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out / "results.json")
    for name, r in results.items():
        print(f"{name:7} " + "  ".join(f"DocHit@{m} {r[f'DocHit@{m}']:.3f}" for m in MS) + f"   (n={r['questions']} questions, {r['papers_indexed']} papers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
