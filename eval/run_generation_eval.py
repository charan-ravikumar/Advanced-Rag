"""Generation evaluation, arm R only (EXPLORATORY: Phase 4 minimal scope, see ``eval/PREREGISTRATION_gen.md`` and its dated Deviations).

    $env:CHROMA_LOCAL_PATH = "$env:LOCALAPPDATA\\advrag\\chroma_v2"
    python eval/run_generation_eval.py --split dev --already-used-tokens 30000

One seeded random sample (``--n 40``, ``--seed 0``) of answerable dev questions that have at least one usable (non-table) reference. For each, the top-5 chunks of
the frozen retrieval configuration D are retrieved ONCE (paper mode) and the same context is given to the generator in SHORT mode and in LONG mode
(``RagService.answer_short`` / ``answer_long``; tracing off; the local prompt templates). Metrics: Answer-F1 (max over annotators), yes/no accuracy, abstention rate,
citation validity (``eval/answer_metrics.py``), with paper-clustered bootstrap intervals.

Quota: a **token governor** counts the tokens of every response's
``usage`` in a ledger (rolling 24 hours); ``--daily-token-budget`` (default 180,000) stops the run cleanly BEFORE a call that would exceed it, and a provider rate-limit
answer also stops it ("quota exhausted, resume later", exit code 3). Every finished generation is cached on disk keyed by (model, reasoning effort, prompt name/version/hash,
arm, mode, question id, context hash, caps, temperature), so running the same command again costs nothing for what is done and continues with the rest. Each session writes a run folder
(config.json, per_question.jsonl, summary.json, SHA256SUMS); ``complete`` says whether every question and mode has an answer.

The test split is REFUSED: it needs ``--i-understand-this-is-final`` AND a generation freeze file (``eval/gen_freeze.json``), which does not exist and is not created here.
Only public Qasper questions and public paper text are sent to Gemini. API keys come from the environment (or .env through ``rag_core.env.load_env``), never printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Sequence

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")  # set before anything could import ragas
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.answer_metrics import score_long, score_short, summarize_long, summarize_short  # noqa: E402
from eval.disk_cache import DiskCache  # noqa: E402
from eval.gen_arms import ContextHit, FixedContextRetriever, context_hash, usable_references  # noqa: E402
from eval.quota_plan import Pacer  # noqa: E402
from rag_core.schemas import AnswerResult  # noqa: E402
from rag_core.claims import MIN_QUOTE_WORDS, SCHEMA_VERSION as CLAIMS_SCHEMA_VERSION  # noqa: E402

LABEL = "EXPLORATORY: n=40 answerable dev questions, one arm (R = RAG, configuration D, top-5), one seeded sample; not a published result"
PREREG_FILE = ROOT / "eval" / "PREREGISTRATION_gen.md"
PREREG_HASH_FILE = ROOT / "eval" / "PREREGISTRATION_gen.sha256"
FREEZE_FILE = ROOT / "eval" / "gen_freeze.json"
DATASET_DIR = ROOT / "eval" / "datasets" / "qasper_dev"
CACHE_SCHEMA = 3  # 3: long answers are structured (claims with quotes, rag_core/claims.py); 2: the citation parser read 【S1】. Old files are left in place and never read
DEFAULT_TOKEN_ESTIMATE = 2500  # used for the governor and the pacer until this session has measured calls
EXIT_QUOTA = 3
# conservative pacing defaults for Gemini, NOT documented limits of gemini-3.5-flash-lite (check the quota of your key in Google AI Studio and adjust)
GEMINI_RPM, GEMINI_TPM = 10, 100_000


class GuardRefused(SystemExit):
    pass


class QuotaExhausted(Exception):
    """Stop the session cleanly: the daily token budget is reached or the provider said rate limit. Everything finished is cached."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def check_test_guard(*, confirmed: bool, freeze_path: Path) -> None:
    if not confirmed:
        raise GuardRefused("REFUSED: --split test is final. Pass --i-understand-this-is-final to evaluate it.")
    if not Path(freeze_path).exists():
        raise GuardRefused(f"REFUSED: no generation freeze file at {freeze_path}. It is created only when the owner asks; the test split is not run in this phase.")


# ---- the sample -------------------------------------------------------------------------------------------------------------------

def eligible_questions(questions: Sequence[dict]) -> list[dict]:
    """Answerable questions with at least one usable (non-table, answerable-typed) reference."""
    return [q for q in questions if q.get("answerable") and usable_references(q.get("references") or [])]


def select_sample(questions: Sequence[dict], n: int, seed: int) -> list[dict]:
    """``random.Random(seed).sample(sorted(eligible ids), n)`` (all of them if fewer than ``n``), in the order drawn."""
    pool = {q["id"]: q for q in eligible_questions(questions)}
    ids = sorted(pool)
    drawn = random.Random(seed).sample(ids, min(n, len(ids)))
    return [pool[i] for i in drawn]


# ---- the disk cache -----------------------------------------------------------------------------------------------------------------

def generation_key(*, model: str, reasoning_effort: str | None, prompt_id: str, arm: str, mode: str, question_id: str, context_hash_: str,
                   max_output_tokens: int, temperature: float, top_k: int) -> str:
    payload = {"schema": CACHE_SCHEMA, "model": model, "effort": reasoning_effort, "prompt": prompt_id, "arm": arm, "mode": mode, "question": question_id,
               "context": context_hash_, "max_output_tokens": max_output_tokens, "temperature": temperature, "top_k": top_k}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


GenerationCache = DiskCache  # one JSON file per key (written atomically); a corrupt file is a miss (see eval/disk_cache.py)


# ---- the token governor ---------------------------------------------------------------------------------------------------------------

class TokenLedger:
    """Append-only JSONL of the tokens each call used (from the response ``usage``); ``used`` sums a trailing window."""

    def __init__(self, path: Path, clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self.path, self._clock = Path(path), clock

    def record(self, tokens: int, label: str = "") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"utc": self._clock().isoformat(), "tokens": int(tokens), "label": label}) + "\n")

    def used(self, window_hours: float = 24.0) -> int:
        if not self.path.exists():
            return 0
        cutoff, total = self._clock() - timedelta(hours=window_hours), 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if datetime.fromisoformat(entry["utc"]) >= cutoff:
                    total += int(entry["tokens"])
            except (ValueError, KeyError, TypeError):
                continue  # a damaged line never blocks the run (it just is not counted)
        return total


class TokenGovernor:
    """``can_spend(estimate)``: the trailing-24-hour ledger total plus ``already_used`` (tokens spent outside the ledger in the same window) plus the
    estimate of the next call must stay within ``budget``."""

    def __init__(self, ledger: TokenLedger, budget: int, already_used: int = 0):
        self.ledger, self.budget, self.already_used = ledger, int(budget), int(already_used)

    def used(self) -> int:
        return self.ledger.used() + self.already_used

    def can_spend(self, estimate: int) -> bool:
        return self.used() + int(estimate) <= self.budget

    def spend(self, tokens: int, label: str = "") -> None:
        self.ledger.record(tokens, label)


# ---- one session -------------------------------------------------------------------------------------------------------------------

def _source_rows(result) -> list[dict]:
    return [{"label": s.label, "chunk_id": s.chunk_id, "doc_id": s.doc_id, "section": s.section, "text": s.text, "rank": s.rank, "score": s.score}
            for s in result.used_sources]


def _row(q: dict, mode: str, result, *, from_cache: bool) -> dict:
    refs = q["references"]
    if mode == "short":
        metrics, best_type = score_short(result.short_answer, result.abstained, refs)
    else:
        metrics, best_type = score_long(result.citations, result.invalid_citations, result.abstained, result.claims), None
    # ``arm`` is per mode ("R_short", "R_long") so that eval/compare.py, which reads rows by (arm, question_id), never sees two rows with one key
    return {"question_id": q["id"], "paper_id": q["paper_id"], "arm": f"R_{mode}", "arm_base": "R", "mode": mode, "answerable": True, "question": q["question"],
            "answer": result.answer,
            "short_answer": result.short_answer, "claims": [c.model_dump() for c in result.claims], "used_sources": _source_rows(result), "citations": result.citations,
            "invalid_citations": result.invalid_citations, "abstained": result.abstained, "provider": result.provider, "model": result.model,
            "reasoning_effort": result.reasoning_effort, "usage": result.usage.model_dump(), "latency_ms": result.latency_ms.model_dump(),
            "finish_reason": result.finish_reason, "errors": [e.code for e in result.errors], "from_cache": from_cache,
            "best_reference_type": best_type, "metrics": metrics}


def run_session(questions: Sequence[dict], modes: Sequence[str], *, retrieve: Callable[[dict], Sequence], service, fixed: FixedContextRetriever,
                cache: GenerationCache, governor: TokenGovernor, pacer: Pacer, cfg, prompt_ids: dict[str, str],
                log: Callable[[str], None] = print, token_estimate: int = DEFAULT_TOKEN_ESTIMATE) -> dict:
    """Generate every (question, mode) that is not cached, until done or the quota stops it. Returns rows, counters and the stop reason.

    ``retrieve(q)`` returns the retrieved hits (the real Retriever in production). The SAME hits are given to both modes. A degraded result (the provider failed)
    is never cached and never turned into a row; a rate-limit failure stops the session as quota exhausted."""
    rows: list[dict] = []
    stats = {"calls_made": 0, "cache_hits": 0, "tokens_used_this_session": 0, "failed": [], "stopped_reason": None}
    measured: dict[str, list[int]] = {m: [] for m in modes}
    pending: list[tuple[str, str]] = []
    try:
        for q in questions:
            hits = list(retrieve(q))
            chash = context_hash(hits)
            for mode in modes:
                key = generation_key(model=service.gateway.primary_model, reasoning_effort=service.gateway.primary_reasoning_effort, prompt_id=prompt_ids[mode],
                                     arm="R", mode=mode, question_id=q["id"], context_hash_=chash,
                                     max_output_tokens=cfg.short_max_output_tokens if mode == "short" else cfg.max_output_tokens,
                                     temperature=cfg.temperature, top_k=cfg.top_k)
                cached = cache.get(key)
                if cached is not None:
                    stats["cache_hits"] += 1
                    rows.append(_row(q, mode, AnswerResult.model_validate(cached["result"]), from_cache=True))
                    continue
                if stats["stopped_reason"]:
                    pending.append((q["id"], mode))
                    continue
                estimate = int(sum(measured[mode]) / len(measured[mode])) if measured[mode] else token_estimate
                if not governor.can_spend(estimate):
                    stats["stopped_reason"] = f"daily token budget reached ({governor.used()} used of {governor.budget}; next call estimated at {estimate})"
                    pending.append((q["id"], mode))
                    continue
                pacer.wait(estimate)
                fixed.set(hits)
                result = (service.answer_short if mode == "short" else service.answer_long)(q["question"], doc_ids=[q["paper_id"]])
                tokens = (result.usage.prompt_tokens or 0) + (result.usage.completion_tokens or 0)
                stats["calls_made"] += 1
                if result.degraded or result.partial or result.errors:
                    if any(a.error_kind == "rate_limit" for a in result.attempts):
                        stats["stopped_reason"] = "provider rate limit (HTTP 429): quota exhausted for now"
                    else:
                        stats["failed"].append({"question_id": q["id"], "mode": mode, "errors": [e.code for e in result.errors]})
                    pending.append((q["id"], mode))
                    continue
                pacer.record(tokens)
                governor.spend(tokens, label=f"R/{mode}/{q['id']}")
                measured[mode].append(tokens or estimate)
                stats["tokens_used_this_session"] += tokens
                cache.set(key, {"result": result.model_dump(mode="json"), "context_hash": chash, "question_id": q["id"], "mode": mode})
                rows.append(_row(q, mode, result, from_cache=False))
                log(f"  {mode:<5} {q['id']}  tokens={tokens}  finish={result.finish_reason}  abstained={result.abstained}")
    except QuotaExhausted as exc:  # a caller may raise it from retrieve or the service wrapper
        stats["stopped_reason"] = exc.reason
    stats["pending"] = pending
    stats["rows"] = rows
    return stats


def summarize(rows: Sequence[dict], session: dict) -> dict:
    short, long_ = [r for r in rows if r["mode"] == "short"], [r for r in rows if r["mode"] == "long"]
    return {"label": LABEL, "arm": "R", "short_mode": summarize_short(short) if short else None, "long_mode": summarize_long(long_) if long_ else None,
            "finish_reasons": {m: _count([r["finish_reason"] for r in rows if r["mode"] == m]) for m in ("short", "long")},
            "usage_tokens": {m: {"prompt": sum((r["usage"].get("prompt_tokens") or 0) for r in rows if r["mode"] == m and not r["from_cache"]),
                                 "completion": sum((r["usage"].get("completion_tokens") or 0) for r in rows if r["mode"] == m and not r["from_cache"]),
                                 "reasoning": sum((r["usage"].get("reasoning_tokens") or 0) for r in rows if r["mode"] == m and not r["from_cache"])}
                             for m in ("short", "long")},
            "usage_note": "token sums cover the generations made in THIS session (cache hits cost nothing and are excluded)",
            "session": session}


def _count(values: Sequence) -> dict:
    out: dict = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return out


# ---- main --------------------------------------------------------------------------------------------------------------------------

def build_retrieve(retriever, cfg) -> Callable[[dict], Sequence]:
    """Paper-mode retrieval with exactly the arguments ``RagService`` uses (a test pins the equality), so the hashed context IS the generated context."""
    def retrieve(q: dict):
        return retriever.retrieve(q["question"], mode=cfg.retrieval_mode, fusion=cfg.fusion, top_k=cfg.top_k, candidate_k=cfg.candidate_k, rerank_k=cfg.rerank_k,
                                  doc_ids=[q["paper_id"]], rerank=False, rrf_k=cfg.rrf_k, vector_weight=cfg.vector_weight, lexical_weight=cfg.lexical_weight).hits

    return retrieve


def main(argv=None, *, retriever=None, gateway=None, index_version: str | None = None, clock: Callable[[], datetime] | None = None,
         sleep: Callable[[float], None] | None = None, log: Callable[[str], None] = print) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--arm", choices=["R"], default="R", help="only arm R exists in this phase (the oracle and title arms were cut)")
    ap.add_argument("--mode", choices=["short", "long", "both"], default="both")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="process only the first LIMIT questions of the sample (a smoke test)")
    ap.add_argument("--out", default=str(ROOT / "runs" / "gen"))
    ap.add_argument("--cache-dir", default=None, help="default: <out>/cache")
    ap.add_argument("--ledger", default=None, help="default: <out>/token_ledger.jsonl")
    ap.add_argument("--daily-token-budget", type=int, default=180000)
    ap.add_argument("--already-used-tokens", type=int, default=0, help="tokens spent outside the ledger in the same 24 h (other runs, probes)")
    ap.add_argument("--token-estimate", type=int, default=DEFAULT_TOKEN_ESTIMATE)
    ap.add_argument("--prereg", default=str(PREREG_FILE))
    ap.add_argument("--prereg-hash-file", default=str(PREREG_HASH_FILE))
    ap.add_argument("--golden", default=str(DATASET_DIR / "golden_set_v2.json"))
    ap.add_argument("--split-file", default=str(DATASET_DIR / "split.json"))
    ap.add_argument("--freeze-file", default=str(FREEZE_FILE))
    ap.add_argument("--i-understand-this-is-final", dest="final", action="store_true")
    args = ap.parse_args(argv)

    if args.split == "test":
        check_test_guard(confirmed=args.final, freeze_path=Path(args.freeze_file))
    from eval.run_retrieval_eval import hardware_info, preregistration_info, verify_checksums, write_run  # lazy: numpy and the retrieval metrics

    prereg = preregistration_info(Path(args.prereg), Path(args.prereg_hash_file))  # refuses if the pre-registered text changed
    golden_path, split_path = Path(args.golden), Path(args.split_file)
    from eval.heldout import refuse_heldout_inputs

    refuse_heldout_inputs([golden_path, split_path])  # the official test set is opened only through eval.heldout
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("golden_set_sha256") != hashlib.sha256(golden_path.read_bytes()).hexdigest():
        raise SystemExit("REFUSED: split.json was made for a different golden set (sha256 mismatch).")
    by_id = {q["id"]: q for q in golden["questions"]}
    pool = [by_id[i] for i in sorted(split[args.split])]
    sample = select_sample(pool, args.n, args.seed)
    todo = sample[: args.limit] if args.limit else sample
    modes = ["short", "long"] if args.mode == "both" else [args.mode]

    from rag_core.config import RagConfig
    from rag_core.prompt_store import LocalPromptStore, describe_prompts
    from rag_core.prompts import PROMPT_NAMES
    from rag_core.service import RagService, index_version_for

    if retriever is None:  # production: real store, real embedder (local cache only), real Gemini gateway
        from config import cfg as project_cfg
        from eval.run_retrieval_eval import build_real_components
        from rag_core.env import load_env, present
        from rag_core.gateway import build_gateway

        load_env()
        if not present("GEMINI_API_KEY")["GEMINI_API_KEY"]:
            print("SKIPPED: GEMINI_API_KEY is not available; nothing was sent anywhere.")
            return 0
        if not os.environ.get("CHROMA_LOCAL_PATH"):
            print("REFUSED: set CHROMA_LOCAL_PATH to %LOCALAPPDATA%\\advrag\\chroma_v2.", file=sys.stderr)
            return 2
        components, embedder = build_real_components({"recursive"})
        retriever = components["recursive"]["retriever"]
        rag_cfg = RagConfig.from_config(project_cfg)
        rag_cfg.max_attempts, rag_cfg.timeout_s = 4, 60.0
        gateway = build_gateway(rag_cfg)
        index_version = index_version_for(retriever.store)
        store_info, embedder_info = components["recursive"]["info"], {"name": embedder.name, "revision": embedder.revision}
    else:
        rag_cfg, store_info, embedder_info = RagConfig(), {"injected": True}, {"injected": True}
        rag_cfg.max_attempts = 4
        index_version = index_version or "test"

    out_root = Path(args.out)
    prompts = LocalPromptStore()
    fixed = FixedContextRetriever("R")
    service = RagService(fixed, gateway, None, rag_cfg, index_version=index_version or "unversioned", prompts=prompts)
    prompt_ids = {m: prompts.resolve(PROMPT_NAMES[m]).cache_id for m in ("short", "long")}
    now = clock or (lambda: datetime.now(timezone.utc))
    governor = TokenGovernor(TokenLedger(Path(args.ledger) if args.ledger else out_root / "token_ledger.jsonl", now), args.daily_token_budget, args.already_used_tokens)
    cache = GenerationCache(Path(args.cache_dir) if args.cache_dir else out_root / "cache")
    # the pacer shares the injected clock (tests advance it from the injected sleep); in production both are real
    pacer = Pacer(rpm=GEMINI_RPM, tpm=GEMINI_TPM, clock=(lambda: now().timestamp()) if clock else None, sleep=sleep)
    used_before = governor.used()
    started = now()
    log(f"{LABEL}\nsample: {len(sample)} questions (seed {args.seed}); this session: {len(todo)} questions x modes {modes}; token budget {args.daily_token_budget}, used in the last 24 h: {used_before}")

    session = run_session(todo, modes, retrieve=build_retrieve(retriever, rag_cfg), service=service, fixed=fixed, cache=cache, governor=governor, pacer=pacer,
                          cfg=rag_cfg, prompt_ids=prompt_ids, log=log, token_estimate=args.token_estimate)
    rows = session.pop("rows")
    done = {(r["question_id"], r["mode"]) for r in rows}
    remaining = [(q["id"], m) for q in todo for m in modes if (q["id"], m) not in done]
    session_done = not remaining
    complete = session_done and len(todo) == len(sample)  # ``complete`` means the WHOLE sample has an answer in every requested mode (a --limit smoke run is not complete)
    session.update({"complete": complete, "remaining": remaining, "tokens_used_in_window_before": used_before, "tokens_used_in_window_after": governor.used(),
                    "daily_token_budget": args.daily_token_budget, "already_used_tokens_declared": args.already_used_tokens})
    from utils.fingerprint import code_fingerprint

    fingerprint = code_fingerprint()
    try:
        from importlib import metadata

        sdk_versions = {p: metadata.version(p) for p in ("openai", "pydantic", "numpy")}
    except Exception:  # noqa: BLE001
        sdk_versions = {}
    config = {
        "created_utc": started.isoformat(), "label": LABEL, "split": args.split, "eval_mode": "paper", "arm": "R", "modes": modes,
        "setting": "PRIMARY setting (paper mode, the question's paper is given); retrieval = frozen configuration D, top-5 chunks to the generator (a design default)",
        "preregistration": prereg, "scope_note": "Phase 4 minimal scope: see the dated Deviations in eval/PREREGISTRATION_gen.md (arms O, T, F and the comparisons were cut)",
        "sample": {"seed": args.seed, "n_requested": args.n, "n_drawn": len(sample), "rule": "random.Random(seed).sample(sorted(ids of answerable dev questions with at least one usable non-table reference), n)",
                   "eligible_questions": len(eligible_questions(pool)), "question_ids": [q["id"] for q in sample], "processed_this_session": [q["id"] for q in todo]},
        "generator": {"model": service.gateway.primary_model, "provider": service.gateway.primary_name, "reasoning_effort": rag_cfg.reasoning_effort,
                      "temperature": rag_cfg.temperature, "max_output_tokens": {"long": rag_cfg.max_output_tokens, "short": rag_cfg.short_max_output_tokens},
                      "top_k": rag_cfg.top_k, "max_attempts": rag_cfg.max_attempts, "tracing": "off (trace_evaluation=false)"},
        "retrieval": rag_cfg.retrieval_settings(False), "prompts": describe_prompts(prompts), "prompt_cache_ids": prompt_ids,
        "long_answer_format": {"kind": "structured output (JSON schema, claims with source ids and verbatim quotes)", "schema_version": CLAIMS_SCHEMA_VERSION,
                               "min_quote_words": MIN_QUOTE_WORDS},
        "token_governor": {"daily_token_budget": args.daily_token_budget, "already_used_tokens_declared": args.already_used_tokens, "ledger": str(governor.ledger.path),
                           "rolling_window_hours": 24, "documented_provider_limits": "none recorded: the pacer uses conservative defaults (GEMINI_RPM / GEMINI_TPM), not documented limits of the model"},
        "cache": {"dir": str(cache.dir), "schema": CACHE_SCHEMA, "key": "model, reasoning effort, prompt name|version|sha256, arm, mode, question id, context hash, caps, temperature, top_k"},
        "interval": "95% paper-clustered bootstrap, 5,000 resamples, seed 0 (eval/compare.py::clustered_bootstrap); n=40: wide by construction",
        "metrics": "Answer-F1 (SQuAD normalisation, max over usable annotator references), yes/no accuracy, abstention rate, citation validity (eval/answer_metrics.py)",
        "ragas": "not used by this runner (scored later, in venv_eval, by eval/run_ragas.py)", "code_fingerprint": fingerprint, "store": store_info, "embedder": embedder_info,
        "golden_set": {"path": str(golden_path), "sha256": hashlib.sha256(golden_path.read_bytes()).hexdigest()}, "split_file": {"path": str(split_path)},
        "libraries": sdk_versions, "hardware": hardware_info(), "complete": complete,
    }
    summary = summarize(rows, session)
    run_dir = write_run(out_root, config, rows, summary, started=started, split=args.split, eval_mode="gen_R")
    log(f"wrote {run_dir}  (checksums ok: {verify_checksums(run_dir)})")
    log(f"SESSION REPORT: tokens used this session {session['tokens_used_this_session']}; calls {session['calls_made']}; cache hits {session['cache_hits']}; "
        f"questions x modes done {len(rows)} of {len(todo) * len(modes)}; remaining {len(remaining)}; "
        f"tokens in the last 24 h {session['tokens_used_in_window_after']} of {args.daily_token_budget}")
    if session["failed"]:
        log(f"FAILED (not cached, will be retried on the next run): {session['failed']}")
    if session["stopped_reason"]:
        log(f"STOPPED: {session['stopped_reason']}. Quota exhausted, resume later: run the same command again (finished generations are cached).")
        return EXIT_QUOTA
    return 0 if session_done else EXIT_QUOTA


if __name__ == "__main__":
    sys.exit(main())
