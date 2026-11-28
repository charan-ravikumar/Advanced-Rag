"""Open-loop load test of POST /v1/retrieve against a running API (retrieval only: no model provider is called).

    python scripts/load_test.py --label before --rates 5,10,20,30 [--url http://127.0.0.1:8000] [--duration 30] [--slo-ms 300]

Requests are sent on a fixed schedule whether or not earlier ones have finished, and latency is measured from the SCHEDULED send time,
so a server that falls behind cannot hide it. A rate PASSES when the p95 latency of the successful requests is within ``--slo-ms``, at
most ``--max-failed`` of the requests failed (any status other than 200, rejections included), and the generator itself kept up.
``sustained_qps`` is the highest rate that passed with every lower rate passing too.

Each request carries a new X-Session-Id, so the per-client rate limit does not apply. Questions are public Qasper dev questions.
Writes runs/load/<UTC>_<fingerprint8>_<label>/ (config.json, per_request.jsonl, summary.json, SHA256SUMS).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GOLDEN = ROOT / "eval" / "datasets" / "qasper_dev" / "golden_set_v2.json"
SPLIT = ROOT / "eval" / "datasets" / "qasper_dev" / "split.json"
MAX_GENERATOR_LAG_MS = 20.0  # p95 of (actual - scheduled send time); above it the numbers describe the generator, not the server


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile (0 < q <= 100); None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


def summarise_step(rate: float, duration: float, records: list[dict], slo_ms: float, max_failed: float) -> dict:
    ok = [r["latency_ms"] for r in records if r["status"] == 200]
    sent = len(records)
    failed = (sent - len(ok)) / sent if sent else 1.0
    lag = percentile([r["send_lag_ms"] for r in records], 95)
    rejected = [r["latency_ms"] for r in records if r["status"] in (429, 503)]
    p95 = percentile(ok, 95)
    passed = p95 is not None and p95 <= slo_ms and failed <= max_failed and lag is not None and lag <= MAX_GENERATOR_LAG_MS
    return {"offered_qps": rate, "duration_s": duration, "sent": sent, "ok": len(ok), "failed_fraction": failed,
            "status": {str(k): v for k, v in sorted(Counter(r["status"] for r in records).items(), key=lambda kv: str(kv[0]))},
            "ok_qps": len(ok) / duration, "p50_ms": percentile(ok, 50), "p95_ms": p95, "p99_ms": percentile(ok, 99),
            "rejected_p95_ms": percentile(rejected, 95), "generator_lag_p95_ms": lag, "passed": passed}


def sustained_qps(steps: list[dict]) -> float:
    best = 0.0
    for step in sorted(steps, key=lambda s: s["offered_qps"]):
        if not step["passed"]:
            break
        best = step["offered_qps"]
    return best


def dev_questions() -> list[str]:
    ids = set(json.loads(SPLIT.read_text(encoding="utf-8"))["dev"])  # question ids
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))["questions"]
    return [q["question"] for q in golden if q["id"] in ids and q["answerable"]]


async def one(client, url: str, question: str, scheduled: float, out: list[dict]) -> None:
    sent = time.perf_counter()
    try:
        response = await client.post(url, json={"question": question}, headers={"X-Session-Id": uuid.uuid4().hex})
        status: int | str = response.status_code
    except Exception as exc:  # noqa: BLE001: a timeout or a refused connection is a failed request, recorded by its class
        status = type(exc).__name__
    out.append({"status": status, "latency_ms": (time.perf_counter() - scheduled) * 1000, "send_lag_ms": (sent - scheduled) * 1000})


async def step(client, url: str, rate: float, duration: float, questions: list[str], rng: random.Random) -> list[dict]:
    records: list[dict] = []
    tasks = []
    start = time.perf_counter()
    for i in range(int(rate * duration)):
        scheduled = start + i / rate
        delay = scheduled - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(asyncio.create_task(one(client, url, rng.choice(questions), scheduled, records)))
    await asyncio.gather(*tasks)
    return records


async def drain(client, url: str, question: str) -> None:
    """Wait until the server answers a single request quickly again, so one step's backlog does not spill into the next."""
    for _ in range(120):
        records: list[dict] = []
        await one(client, url, question, time.perf_counter(), records)
        if records[0]["status"] == 200 and records[0]["latency_ms"] < 200:
            break
        await asyncio.sleep(1)
    await asyncio.sleep(2)


async def run(args, rates: list[float]) -> tuple[dict, list[dict], list[dict]]:
    import httpx

    questions = dev_questions()
    rng = random.Random(args.seed)
    url = args.url.rstrip("/") + "/v1/retrieve"
    limits = httpx.Limits(max_connections=2000, max_keepalive_connections=200)
    async with httpx.AsyncClient(timeout=args.timeout, limits=limits, trust_env=False) as client:
        meta = (await client.get(args.url.rstrip("/") + "/v1/meta")).raise_for_status().json()
        await step(client, url, args.warmup_rate, args.warmup, questions, rng)  # discarded
        await drain(client, url, questions[0])
        steps, rows = [], []
        for rate in rates:
            records = await step(client, url, rate, args.duration, questions, rng)
            summary = summarise_step(rate, args.duration, records, args.slo_ms, args.max_failed)
            steps.append(summary)
            rows += [{"offered_qps": rate, **r} for r in records]
            print(f"rate={rate:g} sent={summary['sent']} status={summary['status']} p50={summary['p50_ms'] or 0:.0f}ms p95={summary['p95_ms'] or 0:.0f}ms "
                  f"lag_p95={summary['generator_lag_p95_ms']:.1f}ms passed={summary['passed']}", flush=True)
            await drain(client, url, questions[0])
    return {"server_meta": meta, "n_questions": len(questions)}, steps, rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--label", required=True, help="a short name for what is being measured, e.g. before, after")
    ap.add_argument("--rates", required=True, help="offered requests per second, comma separated, ascending")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--warmup", type=float, default=10.0)
    ap.add_argument("--warmup-rate", type=float, default=5.0)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--slo-ms", type=float, default=300.0)
    ap.add_argument("--max-failed", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--note", default="", help="recorded in config.json")
    ap.add_argument("--out-root", default=str(ROOT / "runs" / "load"))
    args = ap.parse_args(argv)
    rates = [float(r) for r in args.rates.split(",")]
    if rates != sorted(set(rates)) or rates[0] <= 0:
        ap.error("--rates must be positive and ascending")

    from utils.fingerprint import code_fingerprint

    started, fingerprint = datetime.now(timezone.utc), code_fingerprint()
    info, steps, rows = asyncio.run(run(args, rates))
    meta = info["server_meta"]
    config = {"created_utc": started.isoformat(), "code_fingerprint": fingerprint, "split": "dev", "label": args.label, "note": args.note,
              "endpoint": "/v1/retrieve", "method": "open loop, latency from the scheduled send time; a new X-Session-Id per request",
              "rates": rates, "duration_s": args.duration, "warmup_s": args.warmup, "warmup_rate": args.warmup_rate, "timeout_s": args.timeout,
              "slo_ms": args.slo_ms, "max_failed_fraction": args.max_failed, "max_generator_lag_ms": MAX_GENERATOR_LAG_MS, "seed": args.seed,
              "n_questions": info["n_questions"], "golden_set_sha256": hashlib.sha256(GOLDEN.read_bytes()).hexdigest(),
              "server": {"version_info": meta.get("version_info"), "reranker_available": meta.get("flags", {}).get("reranker_available")},
              "hardware": {"machine": platform.machine(), "system": platform.system(), "cpus": os.cpu_count(),
                           "python": platform.python_version(), "note": "load generator and server on the same machine"},
              "complete": True}
    summary = {"slo_ms": args.slo_ms, "max_failed_fraction": args.max_failed, "sustained_qps": sustained_qps(steps), "steps": steps}
    run_dir = Path(args.out_root) / f"{started.strftime('%Y%m%dT%H%M%SZ')}_{fingerprint[:8]}_{args.label}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_bytes((json.dumps(config, indent=2) + "\n").encode())
    (run_dir / "summary.json").write_bytes((json.dumps(summary, indent=2) + "\n").encode())
    (run_dir / "per_request.jsonl").write_bytes("".join(json.dumps(r) + "\n" for r in rows).encode())
    names = ("config.json", "per_request.jsonl", "summary.json")
    (run_dir / "SHA256SUMS").write_bytes("".join(f"{hashlib.sha256((run_dir / n).read_bytes()).hexdigest()}  {n}\n" for n in names).encode())
    print(f"sustained_qps={summary['sustained_qps']:g} at p95<={args.slo_ms:g}ms -> {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
