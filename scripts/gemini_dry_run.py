"""Gemini embedding DRY RUN (step 2 of Stage 2b): can the chunks be embedded in a realistic time with the quota of this key?

    python scripts/gemini_dry_run.py [--sustained-seconds 150]

Needs GEMINI_API_KEY in the environment (never printed, never written). Only public Qasper chunk text and dev questions are sent.

1. lists the embedding models available to the key and checks the configured one (id, version);
2. embeds ONE paper's recursive chunks (taken from the local store) in a single batch: wall time, HTTP status, and the token count
   reported by the API's countTokens;
3. times 5 single-question requests (RETRIEVAL_QUERY);
4. burst probe: 3 back-to-back batches of 100 other chunks with retries OFF, to see whether and how a 429 appears
   (per-minute vs per-day, which quota metric, and whether a batch counts as one request or as one per text);
5. sustained phase: batches of 100 with normal retries/backoff for a fixed time, to measure the real throughput;
6. extrapolates the wall time for all 7,255 chunks + the dev questions + the paper index and prints FEASIBLE or HARD STOP
   (more than about 24 hours, a per-day quota, or repeated quota errors).

Everything embedded is cached on disk, so the work is not wasted. The result (no text, no key) is saved under
runs/measurements/<UTC>_gemini_dry_run/results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

HARD_STOP_HOURS = 24.0
TOTAL_CHUNKS = 7255
N_QUERIES, N_PAPERS = 174, 200


def quota_details(data: dict) -> list[dict]:
    """The non-secret parts of a 429 body: which quota, which limit."""
    out = []
    for d in (data.get("error") or {}).get("details", []) or []:
        for v in d.get("violations", []) or []:
            out.append({k: v.get(k) for k in ("quotaMetric", "quotaId", "quotaDimensions") if k in v} | {"quotaValue": v.get("quotaValue")})
        if "retryDelay" in d:
            out.append({"retryDelay": d["retryDelay"]})
    return out


class Recorder:
    """Wraps the real transport and records status codes and 429 details (never headers, bodies or URLs)."""

    def __init__(self, inner):
        self.inner, self.log = inner, []

    def __call__(self, method, url, headers, body, timeout):
        t = time.perf_counter()
        status, data, hdr = self.inner(method, url, headers, body, timeout)
        self.log.append({"t": round(time.time(), 3), "op": url.rsplit(":", 1)[-1].split("?")[0].rsplit("/", 1)[-1], "status": status,
                         "ms": round((time.perf_counter() - t) * 1000, 1), "n_texts": len((body or {}).get("requests", [])) or None,
                         **({"quota": quota_details(data), "message": str((data.get("error") or {}).get("message", ""))[:200]} if status != 200 else {})})
        return status, data, hdr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sustained-seconds", type=float, default=150.0)
    ap.add_argument("--out", default=str(ROOT / "runs" / "measurements"))
    args = ap.parse_args(argv)

    from config import cfg
    from db.store import build_store
    from rag.gemini_embedder import (
        KEY_ENV,
        GeminiEmbedder,
        GeminiError,
        QuotaExhausted,
        urllib_transport,
    )
    from utils.fingerprint import code_fingerprint

    if not os.environ.get(KEY_ENV):
        print(f"{KEY_ENV} is not set in the environment; nothing was sent anywhere.", file=sys.stderr)
        return 3

    g = cfg.embedding_gemini
    rec = Recorder(urllib_transport)
    emb = GeminiEmbedder(g.model, g.version, g.dims, transport=rec, max_retries=0)  # retries OFF: we want to SEE the 429s
    res: dict = {"created_utc": datetime.now(timezone.utc).isoformat(), "code_fingerprint": code_fingerprint(),
                 "configured": {"model": g.model, "version": g.version, "dims": g.dims},
                 "data_sent": "public Qasper chunks and dev questions only"}
    try:
        models = emb.list_embedding_models()
    except GeminiError as exc:
        print(f"could not list models: {exc}", file=sys.stderr)
        return 4
    res["embedding_models_available"] = models
    chosen = next((m for m in models if m["id"] == g.model), None)
    if chosen is None:
        print(f"HARD STOP: the configured model {g.model!r} is not available to this key. Available: {[m['id'] for m in models]}", file=sys.stderr)
        res["verdict"] = "HARD STOP: configured model not available"
        _save(args.out, res)
        return 5
    res["chosen_model"] = chosen
    print(f"model {chosen['id']} version {chosen['version']} inputTokenLimit {chosen['inputTokenLimit']}; "
          f"{len(models)} embedding models available: {[m['id'] for m in models]}")

    store = build_store(cfg, "recursive")  # chroma_v2 (CHROMA_LOCAL_PATH): the SAME chunks the Gemini store will hold
    by_doc: dict[str, list] = {}
    for c in store.get_all():
        by_doc.setdefault(c.doc_id, []).append(c)
    ordered = sorted(by_doc)
    paper = ordered[0]
    texts = [c.text for c in sorted(by_doc[paper], key=lambda c: c.chunk_idx)]
    others = [c.text for d in ordered[1:] for c in sorted(by_doc[d], key=lambda c: c.chunk_idx)]
    res["paper_chunks"] = {"paper_id": paper, "n_chunks": len(texts), "chars": sum(map(len, texts))}
    assert sum(len(v) for v in by_doc.values()) == TOTAL_CHUNKS, "the recursive store should hold 7,255 chunks"

    emb = GeminiEmbedder(g.model, g.version, g.dims, transport=rec, max_retries=0)
    try:
        res["paper_chunks"]["tokens_reported_by_countTokens"] = emb.remote_token_count(texts)
        t = time.perf_counter()
        emb.encode_documents(texts)
        res["one_paper_single_batch"] = {"seconds": round(time.perf_counter() - t, 3), "texts": len(texts), "http_calls": emb.stats()["network_calls"]}
        print(f"one paper: {len(texts)} chunks, {res['paper_chunks']['tokens_reported_by_countTokens']} tokens, "
              f"{res['one_paper_single_batch']['seconds']} s in one batch request")

        lat = []
        for q in ("What dataset do they use?", "How is the model trained?", "Which baselines are compared?", "What is the learning rate?", "Do they use pretrained embeddings?"):
            t = time.perf_counter()
            emb.encode_query(q)
            lat.append(round((time.perf_counter() - t) * 1000, 1))
        res["single_query_latency_ms"] = lat
        print(f"single query request latency ms: {lat}")

        i = 0
        res["burst"] = {"batches_attempted": 0, "texts_per_batch": 100, "stopped_by": None}
        for b in range(3):
            part = others[i:i + 100]
            i += 100
            res["burst"]["batches_attempted"] += 1
            try:
                emb.encode_documents(part)
            except QuotaExhausted as exc:
                res["burst"]["stopped_by"] = str(exc)[:300]
                break
        print(f"burst: {res['burst']}")

        # sustained throughput with the production retry/backoff
        sus = GeminiEmbedder(g.model, g.version, g.dims, transport=rec, max_retries=6)
        t0, done, stopped = time.perf_counter(), 0, None
        while time.perf_counter() - t0 < args.sustained_seconds and i < len(others):
            part = others[i:i + 100]
            i += 100
            try:
                sus.encode_documents(part)
                done += len(part)
            except QuotaExhausted as exc:
                stopped = str(exc)[:300]
                break
        elapsed = time.perf_counter() - t0
        s = sus.stats()
        res["sustained"] = {"seconds": round(elapsed, 1), "chunks_embedded": done, "stopped_by": stopped, "http_429": s["http_429"], "retries": s["retries"],
                            "chunks_per_second": round(done / elapsed, 3) if elapsed else None}
        print(f"sustained: {res['sustained']}")
    except QuotaExhausted as exc:
        res["early_stop"] = str(exc)[:300]
    except GeminiError as exc:
        res["early_stop"] = f"error: {str(exc)[:300]}"
        print(f"error: {exc}", file=sys.stderr)

    res["http_log_summary"] = {"statuses": {str(s): sum(e["status"] == s for e in rec.log) for s in {e["status"] for e in rec.log}},
                               "non_200": [e for e in rec.log if e["status"] != 200][:12], "calls": len(rec.log)}
    cache_n = emb.cache.count()
    res["cached_embeddings_now"] = cache_n
    # extrapolation
    sus_rate = (res.get("sustained") or {}).get("chunks_per_second")
    remaining = max(0, TOTAL_CHUNKS - cache_n)
    per_day = any("perday" in json.dumps(e.get("quota", [])).lower().replace("_", "").replace("-", "") for e in rec.log if e["status"] == 429)
    if per_day:
        verdict, hours = "HARD STOP: a per-day quota was hit", None
    elif not sus_rate:
        verdict, hours = "HARD STOP: no sustained throughput could be measured (quota errors from the start)", None
    else:
        hours = (remaining / sus_rate + (N_QUERIES + N_PAPERS) / max(sus_rate, 1e-9)) / 3600
        verdict = (f"HARD STOP: extrapolated {hours:.1f} h > {HARD_STOP_HOURS:.0f} h" if hours > HARD_STOP_HOURS
                   else f"FEASIBLE: about {hours:.2f} h for the remaining {remaining} chunks (+ questions and paper index)")
        if res["sustained"]["stopped_by"]:
            verdict = "HARD STOP: repeated quota errors in the sustained phase (" + res["sustained"]["stopped_by"][:80] + ")"
    res["extrapolation"] = {"chunks_total": TOTAL_CHUNKS, "already_cached": cache_n, "remaining": remaining, "sustained_chunks_per_s": sus_rate,
                            "estimated_hours": None if hours is None else round(hours, 3), "hard_stop_hours": HARD_STOP_HOURS}
    res["verdict"] = verdict
    path = _save(args.out, res)
    print(f"\nVERDICT: {verdict}\nsaved {path}")
    return 0 if verdict.startswith("FEASIBLE") else 6


def _save(out_root: str, res: dict) -> Path:
    started = datetime.now(timezone.utc)
    out = Path(out_root) / f"{started.strftime('%Y%m%dT%H%M%SZ')}_gemini_dry_run"
    out.mkdir(parents=True)
    (out / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return out / "results.json"


if __name__ == "__main__":
    sys.exit(main())
