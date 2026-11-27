# 06 — Benchmarking and Testing Strategy

Evidence tags as before: **[T]** test run / saved artifact · **[M]** AUDIT MEASUREMENT (script and raw output in Appendix A) · **[R]** reasoned. Targets in §7 are *commitments to make before measuring*, not predictions.

## 1. Testing strategy

### 1.1 What exists (measured)

`pytest --fast -p no:cacheprovider -q -rs` → **1263 passed, 69 skipped, 0 failed in 251.8 s** [T] (the 69 skips: 50 slow import-guard subprocess probes, 6 heavy Hypothesis/real-model chunker tests, the real-model reranker and prompt-store/tracing/UI slow tests, the legacy `test_retrieval.py` (3), and RAGAS, which lives in another venv).

| Layer | State | Evidence |
|---|---|---|
| Unit with fakes | ✅ strong | `rag_core_fakes.py` (scripted providers, fake clock, fake retriever, fake Langfuse); the SDK-wrapper tests use the **real** `openai`/`google.genai` exception classes |
| Property-based | ✅ | Hypothesis `ci` (derandomised, no database) and `explore` profiles; failures from `explore` must be pinned with `@example`; chunker, splitting, fingerprint, fusion |
| Hermetic integration | ✅ | Real Chroma + SQLite in `tmp_path` with the hash embedder: ingestion, verification, retriever, store, lexical |
| UI | ✅ | Streamlit `AppTest` with injected fakes (`test_app_ui.py`, 52 tests) |
| Evaluation runners | ✅ | `test_eval_runner.py`, `test_compare.py`, `test_run_generation_eval.py`, judge/RAGAS runner |
| Hermetic guard | ✅ | `tests/conftest.py` blocks non-loopback network and `chromadb.CloudClient`; import-safety asserted by subprocess probes |
| Mutation checks | 🟡 manual | Recorded in `PROJECT_CONTEXT.md` (e.g. removing the daily-cap branch fails 4 tests); no automated tool |
| API / contract | ❌ | No API |
| Live smoke | 🟡 | `scripts/smoke_*.py`, owner-run only; not tests |
| Concurrency | ❌ | No test runs two sessions through the retriever |
| Load / soak | ❌ | None |
| Regression gate | ❌ | No CI |

### 1.2 Coverage [T] (statements; `coverage.json` from the run above)

| Package | Covered / total | % |
|---|---|---|
| `rag_core` | 1,682 / 1,738 | **96.8** |
| `app` | 1,009 / 1,127 | **89.5** |
| `eval` | 1,530 / 1,974 | 77.5 (`ragas_adapter.py` 0%: ragas is installed only in `venv_eval`) |
| `rag` | 1,330 / 2,020 | 65.8 (legacy modules at 0%) |
| `db` | 84 / 142 | 59.2 (`vectordb.py` is legacy cloud code) |
| `observability` / `cache` | 5 / 16 · 0 / 68 | 31.2 · 0 |
| `scripts` | 184 / 2,173 | 8.5 (CLIs) |
| **Everything measured** | 5,897 / 9,333 | **63.2** |
| **Excluding `scripts/`** | 5,713 / 7,160 | **79.8** |
| **Live path only** (excluding scripts and the 13 legacy modules; also excluding the tiny `config.py`, `schemas.py`, `logger_config.py`, which changes the figure by < 0.1 point) | 5,664 / 6,050 | **93.6** |

Per module: `retriever` 100%, `fusion` 100%, `lexical` 96.6%, `chunkers` 94.6%, `ingestion` 97.2%, `store` 95.5%, `gateway` 98.0%, `service` 99.3%, `cache` 100%, `context` 100%, `citations` 100%, `tracing` 97.5%, `guards` 100%, `bootstrap` 92.4%, `ui_chat` 97.5%, `compare` 99.1%, `run_retrieval_eval` 88.5%; `rag/rerankers.py` 69% (the real-model tests are skipped in `--fast`).
**0% modules (all legacy):** `app/main.py`, `cache/cache.py`, `db/vectordb.py`, `observability/metrics.py`, `rag/retrieval.py`, `rag/context_builder.py`, `rag/embeddings.py`, `rag/ingest.py`, `rag/loaders.py`, `rag/pipeline.py`, `eval/run_eval.py`, `eval/run_ablation.py`, `eval/feature_contrib.py`. Every one of your original headline claims lives in this list. Coverage is a floor, not a quality measure: 100% covered modules (`cache`, `gateway`, `fusion`) are exactly where an automated mutation run would tell you most.

### 1.3 Tests to add (each tied to a finding)

| Test | Layer | Proves | Finding |
|---|---|---|---|
| Empty text with `finish_reason="length"` must not count toward the breaker | unit | A hard question cannot open the breaker | N-06 |
| 8 threads × 50 queries through a real tmp Chroma + FTS5 with the lock removed | concurrency | Which parts really need the lock; no cross-talk in results | N-01 |
| Re-ingest while the service holds a cached answer: key changes | integration | No stale answers after the index changes | N-12 |
| Langfuse fetch raising `TimeoutError` twice in a row costs one timeout, not two | unit | Prompt-fetch breaker | N-13 |
| Unknown key in `config.yaml` raises | unit | Typo-safe config | N-14 |
| Provider yields one delta every 20 s: the stream ends at the deadline with `MidStreamError(partial)` | unit | Wall-clock deadline | N-15 |
| `delete_by_doc` then query: no chunk of that doc in either leg; vector/lexical parity | integration | Deletion semantics | existing, extend |
| Tenant property: for random tenants/docs, no hit outside the caller's tenant | property | Isolation | P2-1 |
| SSE framing: `event:`/`data:` lines, final `done`, client disconnect closes the generator | contract | Network streaming | P1-1 |
| Adversarial passages ("ignore previous instructions…") with a *scripted* model that obeys them: the output validator flags it | unit | Guardrail wiring (not model behaviour) | P1-7 |
| `mutmut`/`cosmic-ray` on `gateway.py`, `cache.py`, `fusion.py`, `citations.py` | mutation | Whether the 100% coverage is meaningful | all |
| `@pytest.mark.live` smoke: one Groq stream, one Gemini failover, one Langfuse trace | live, opt-in | Real SDK behaviour; the only thing fakes cannot prove | L-01 class |

## 2. Integration and contract strategy

* **Hermetic first, live last.** Everything above the SDK uses fakes; add a single opt-in `live` marker for the three real-service smokes and never run it in CI.
* **Contract tests** for the API: a recorded SSE transcript compared against a schema (pydantic models shared with `AnswerResult`), plus a golden file test for error events.
* **Fault injection** at the provider boundary (the fake provider already supports scripted failures): 429, 500, timeout, empty body, slow-drip, mid-stream error. Each scenario has an expected `AnswerResult` shape and a metric that must change.

## 3. Load testing and capacity

### 3.1 Units, stated once (the QPS/QPM trap)

* **QPS = completed requests ÷ steady-state seconds.** QPM = the same per minute; **1 QPS = 60 QPM**.
* A *sequential* loop's throughput is `1 ÷ mean latency`. That is **latency inverted, not capacity**. Your v1 "14.3 QPM" is exactly `60 ÷ 4.196 s` (`scripts/run_evaluation.py:1026`) = **0.238 QPS**; 100 requests took ≈ 420 s. The resume's "~14 QPS" would require 100 requests in ≈ 7 s, i.e. a 0.07 s mean latency, which contradicts the 4.196 s you measured in the same run.
* **Little's law:** `L = λ · W` (requests in system = arrival rate × mean time in system). Use it as a consistency check on every load report.
* **Closed loop** (N users, think time Z): `X = N ÷ (R + Z)`; it under-reports tail latency (*coordinated omission*). **Open loop** (fixed arrival rate): reveals queueing; use it to find the knee.

### 3.2 What the existing numbers imply (a model, not a measurement)

Retrieval service time from TEST (p50): D 59.8 ms, with rerank 1,242 ms [T]. If retrieval is serialised by one lock, the maximum retrieval throughput is `1 ÷ S`: **≈ 16.7 req/s** without rerank and **≈ 0.8 req/s** with it; at a healthy 70% utilisation, ≈ 11.7 and ≈ 0.56. The LLM stage runs outside the lock but is capped by provider limits (and the demo's 80 requests/day). These are the figures the load test must confirm or refute.

### 3.3 Scenarios (Locust for closed loop; `constant_throughput`/k6 for open loop)

| ID | Scenario | Purpose | Notes |
|---|---|---|---|
| S1 | **Retrieval-only, fake LLM** at 1, 2, 4, 8, 16, 32 users, 60 s warm-up + 5 min steady | Isolate the retrieval stack and the lock | Fake provider via config flag in the service; real stores |
| S2 | **Cache-hit storm** (95% repeated questions) | Cache path latency and the prompt-fetch dependency | Langfuse mode on and off |
| S3 | **End-to-end, real LLM**, 1–2 users, ~50 requests | TTFT and total latency with Groq | Respect provider limits; public questions only |
| S4 | **Soak** 30 min at 50% of the knee | Memory growth, thread leaks, file handles | Track RSS and open FDs |
| S5 | **Fault injection**: provider 429s, 20% timeouts; Langfuse down; a 3 s Chroma stall | Breaker, degraded modes, p95 under failure | Assert the metric/alert that should fire |
| S6 | **Scale step**: 10× and 50× synthetic corpus | Re-measure lexical and vector legs (Appendix A6 is the lexical part) | After the N-08 fix |

### 3.4 What to record

Throughput (QPS) and error rate per step; latency p50/p95/p99 and **time to first token** per stage; saturation (CPU, RSS, threads, lock wait time, semaphore queue depth); cache hit ratio; tokens per request; breaker state transitions. Raw per-request rows in `runs/load/<UTC>_<fingerprint>/requests.csv`, plus `config.json` (hardware, Python, library versions, fingerprint) and `SHA256SUMS`, exactly like your retrieval runs. Report the **knee** (the arrival rate where p95 exceeds the SLO) and the **maximum sustainable QPS below it**, never the peak.

## 4. RAG evaluation strategy

### 4.1 Retrieval (already strong: keep)

Keep the pre-registered design: paper-level dev/test split, one-shot TEST with ledger and freeze file, paired paper-clustered bootstrap, Holm over the registered family, determinism check, fingerprints. Additions:

* **Power, not just significance.** From the A-vs-D interval (clustered CI [−0.051, −0.002]), the standard error of the nDCG@10 difference is ≈ 0.0125. With Holm over 3 comparisons (z ≈ 2.39) and 80% power (z ≈ 0.84), detecting a true +0.027 needs `(2.39 + 0.84) ÷ 0.027 × SE ≤ 1`, i.e. SE ≈ 0.0083: **≈ 2.25× the questions (≈ 650)** under the same clustering **[R]**. The observed 0.027 is also a winner's-curse estimate, so plan for more. Either pool dev + a fresh split or pre-register fewer comparisons.
* **Stage-3 pre-registration** for: adaptive rerank, confidence gate, tenant filter overhead. Primary metric stays nDCG@10; add a latency co-primary (p95) so a quality gain cannot hide a latency regression.
* **Corpus-wide mode** stays labelled EXTENSION; add a title-qualified variant only if a headline needs it.

### 4.2 Generation and safety (the weak side: build)

| Metric | Definition | Data | Note |
|---|---|---|---|
| Answer-F1 / yes-no accuracy | Existing (`eval/answer_metrics.py`) | answerable | Current: F1 0.404, CI [0.275, 0.534], n = 40, EXPLORATORY [T] |
| **False-answer rate** | share of *unanswerable* questions answered without abstaining | 48 natural + **synthetic negatives** (a question paired with the wrong paper) | Natural n = 48 gives a Wilson half-width ≈ ±0.13 at p ≈ 0.3; synthetic negatives add hundreds of cheap cases, labelled separately |
| **False-abstain rate** | answerable questions that were refused | answerable | Today: 20% (8/40) [T] |
| **Context precision / recall** | fraction of *used* sources that overlap gold spans / fraction of gold spans covered by used sources | gold spans (`eval/gold_spans.py`) | Deterministic, free, no judge |
| **Citation support** | does the cited passage entail the sentence? | NLI or cross-encoder entailment, human-checked sample | Validity (label exists) is already 50/50; support is unmeasured |
| Faithfulness / answer relevancy | RAGAS with the Gemini judge | n ≥ 100 | **Calibrate first**: 50–100 human labels, report κ per metric; only then quote |
| Injection attack success rate | share of 30 injected-passage cases where the answer follows the injected instruction | public Qasper + injected text | Report with a CI |

**Sample sizes (from the exploratory run):** Answer-F1 CI half-width at n = 40 is ≈ ±0.13 (per-question SD ≈ 0.42); ±0.05 needs **≈ 270 questions** **[R]**. Keep claims inside the interval you can afford.

### 4.3 Regression gating

* **Per commit (CI, free):** retrieval on a 5-paper fixture with the hash embedder; assert top-k ids and a stage-latency *budget with tolerance*, not exact timings.
* **Nightly:** cached-answer generation on a fixed 30-question dev subset (no new LLM spend), metric thresholds from a bootstrap of the last frozen run.
* **Never** run the TEST split in CI.

## 5. Metrics and units

| Metric | Definition | Unit / note |
|---|---|---|
| Latency p50/p95/p99, max | exact order statistics of per-request durations | ms; keep `n_measured` |
| TTFT | request start → first text delta | ms (already in `Latency.first_token`) |
| Stage latency | embed, vector, lexical, fuse, hydrate, rerank | ms (already returned by the `Retriever`) |
| Throughput | completed ÷ steady-state seconds | **QPS** (state QPM only as `QPS × 60`) |
| Error rate | non-2xx or `errors[]` non-empty ÷ requests, by code | %, never a literal |
| Degraded / abstain / partial rate | share of results with those flags | % |
| Cache hit ratio | hits ÷ lookups | % |
| Tokens per request | prompt / completion / reasoning from `Usage` | tokens (not words) |
| Cost per request | tokens × configured price | currency, labelled "shadow" on the free tier |
| Ingest throughput | chunks ÷ wall seconds, end-to-end | chunks/s (9.3 [T]; embed-only 15.7 [M]) |
| Index size | RAM (vectors) and disk (FTS5) per 1K chunks | MB |

## 6. Benchmark methodology (a card to attach to every run)

1. **Fix the system under test:** code fingerprint, dependency freeze, model revisions, config, index generation (`index_version`).
2. **Record the environment:** CPU, logical cores, RAM, OS, Python, SQLite (your `config.json` already does); note background load (the TEST run says "shared laptop under normal load", which is why p95 and max are noisy).
3. **Warm up and discard** (you use 5 requests; also warm the cross-encoder and the OS file cache).
4. **Steady state only:** exclude ramp-up and ramp-down; report the window.
5. **Repeat ≥ 3 times** and report the spread; for A/B latency comparisons **interleave** runs (ABAB) to cancel drift.
6. **Percentiles from raw data;** no trimming without saying so; keep the raw file.
7. **Paired design for quality comparisons** (same questions, per-question differences), clustered by paper.
8. **Separate retrieval-only latency from end-to-end** in every table (your `latency_label` already does this).
9. **Write the number to `docs/CLAIMS_LOG.md` only with** the metric, value, run folder, fingerprint and scope note.

## 7. Acceptance criteria (targets, to be committed before measuring)

| Area | Criterion | Measured by | Gate |
|---|---|---|---|
| Correctness | `pytest --fast` green; live-path coverage ≥ 90% | CI | every commit |
| Hygiene | no `print()` in `rag/`, `rag_core/`, `serve/`; ruff clean; mypy clean on `rag_core`, `rag` | CI | every commit |
| Retrieval quality | no TEST-split metric regression beyond the pre-registered tolerance | nightly dev subset; TEST only manually | nightly |
| Retrieval latency | p95 retrieval-only ≤ 300 ms at 10K docs (paper mode, no rerank) | S1 + S6 | release |
| Concurrency | 0 errors at 8 users for 5 min (fake LLM); 503, not a hang, when saturated | S1 | release |
| End-to-end | p95 TTFT ≤ 2.5 s and p95 total ≤ 8 s at 4 users (real LLM) | S3 | release |
| Resilience | breaker opens within 3 failures, degraded result within the request deadline, recovers on the half-open trial | S5 | release |
| Safety | false-answer rate reported with a 95% CI; injection ASR reported | stage-3 run | before any safety claim |
| Observability | each alert rule fires in a fault-injection run | S5 | release |
| Reproducibility | clean `pip install` + `docker compose up` reaches `/readyz` in ≤ 5 min | CI | every commit |

---

## Appendix A: audit measurements and how to reproduce them

All commands run from the repository root with `& "$env:LOCALAPPDATA\advrag\venv\Scripts\python.exe"`; none of them calls the network, writes into the repository, or touches the TEST split beyond reading the saved summary. **These outputs are audit-session measurements (2026-11-23, Windows 11, AMD64 Family 25, 4 logical CPUs, 16 GB, Python 3.13.14): not run artifacts, not in `CLAIMS_LOG.md`, do not quote them without a proper run folder.**

### A1. Test suite and coverage

```powershell
$env:COVERAGE_FILE = "$env:TEMP\audit.coverage"
python -m pytest --fast -p no:cacheprovider -q -rs --cov=rag --cov=rag_core --cov=app --cov=db --cov=cache `
  --cov=observability --cov=utils --cov=eval --cov=scripts --cov=config --cov=schemas --cov-report=term --cov-report=json:$env:TEMP\coverage.json
```
Result: `1263 passed, 69 skipped, 22 warnings in 251.81s`; total coverage 63% (table in §1.2).

### A2. v1 claims recomputed from the raw checkpoints (`evaluation/checkpoints/*.jsonl`)

```python
import json, math, numpy as np
def load(n): return [json.loads(l) for l in open(f"evaluation/checkpoints/{n}.jsonl", encoding="utf-8") if l.strip()]
C = {"vec":load("eval2_Vector_Only"),"hyb":load("eval2_Hybrid_VectorplusBM25"),"rrf":load("eval2_HybridplusRRF"),"full":load("eval2_Production_Full")}
r5 = lambda k: np.array([r["r5"] for r in C[k]])
print("identical vec vs hyb:", all(a["r5"]==b["r5"] and a["r10"]==b["r10"] and a["mrr"]==b["mrr"] for a,b in zip(C["vec"],C["hyb"])))
def paired(a,b,B=5000,seed=0):
    d=b-a; rng=np.random.default_rng(seed); bs=rng.choice(d,size=(B,len(d)),replace=True).mean(axis=1)
    return dict(diff=round(float(d.mean()),4),wins=int((d>0).sum()),losses=int((d<0).sum()),ties=int((d==0).sum()),ci=(round(float(np.percentile(bs,2.5)),3),round(float(np.percentile(bs,97.5)),3)))
print("vec->rrf", paired(r5("vec"),r5("rrf"))); print("rrf->full", paired(r5("rrf"),r5("full")))
print("+44% reproduction:", 0.27/(0.25*0.21/0.28)-1)
```
Output: `identical vec vs hyb: True` · `vec->rrf {'diff': 0.06, 'wins': 6, 'losses': 0, 'ties': 94, 'ci': (0.02, 0.11)}` · `rrf->full {'diff': 0.01, 'wins': 9, 'losses': 8, 'ties': 83, 'ci': (-0.07, 0.09)}` · `+44% reproduction: 0.44` (baseline `0.25×0.21÷0.28 = 0.1875`). Latency from the same files: Hybrid+RRF `total_lat` mean 3.805 s, median 3.950 s; `ret_lat` median 0.538 s, p95 0.984 s; rerank (bge-reranker-base) mean 2.994 s, median 3.014 s; the semantic-chunking run's `total_lat` max is 16,129 s (one outlier dominates its 191.7 s mean).

### A3. TEST-run artifact (read-only)

```python
import json; s = json.load(open("runs/eval/20261122T144233Z_3afc204d_test_paper/summary.json", encoding="utf-8"))
for k, a in s["arms"].items():
    m, L = a["means"], a["retrieval_latency_ms"]; print(k, a["strategy"], a["mode"], round(m["nDCG@10"],4), round(m["HitRate@5"],4), round(L["total"]["p50"],1), round(L["total"]["p95"],1))
```
Output: `A recursive vector 0.4297 0.613 30.6 51.2` · `D recursive hybrid 0.4562 0.6815 59.8 286.1` · `F semantic hybrid 0.4626 0.7089 42.8 61.9` · `G recursive hybrid 0.5144 0.7363 1242.4 1942.5` (matches `docs/CLAIMS_LOG.md`).

### A4. In-process cache-hit path (fakes, local prompts; N = 3,000)

```python
import time
from tests.unit.rag_core_fakes import FakeRetriever, ScriptedProvider, ok
from rag_core.config import RagConfig; from rag_core.cache import InMemoryTTLCache
from rag_core.gateway import LLMGateway; from rag_core.service import RagService
gw = LLMGateway([ScriptedProvider("groq","m1",[ok("Attention is computed with queries and keys [S1]. ","It scales by sqrt(d) [S2].")])], sleep=lambda s: None)
svc = RagService(FakeRetriever(), gw, InMemoryTTLCache(256, 3600), RagConfig(), index_version="bench:1", embedder_revision="fake")
run = lambda q: [e for e in svc.answer_stream(q, doc_ids=["1234.00001"])][-1]["result"]
q = "How is attention computed in this paper?"; run(q); xs = []
for _ in range(3000): t = time.perf_counter(); r = run(q); xs.append((time.perf_counter()-t)*1000)
xs.sort(); print(r.cached, xs[1500], xs[2850], xs[2970], xs[-1])
```
Output: p50 **0.200 ms**, p95 0.358, p99 0.414, max 0.878 ms. This is the service overhead of a cache hit; it excludes Streamlit and network and assumes the prompt is local or SDK-cached (see N-13).

### A5. Semantic chunker window truncation (200 papers, real MiniLM tokenizer from the local cache)

```python
import json, statistics as st
from config import cfg
from transformers import AutoTokenizer
from rag.chunk_adapter import qasper_to_document
from rag.chunkers import _paragraph_spans, _sections
tok = AutoTokenizer.from_pretrained(cfg.embedding.model, revision=cfg.embedding.revision, local_files_only=True)
LIMIT, W = cfg.embedding.max_seq_length, cfg.chunking.semantic_window_size
corpus = json.load(open("eval/datasets/qasper_dev/corpus.json", encoding="utf-8")); ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
pt, n, over, cut, part, share = [], 0, 0, 0, 0, []
for pid, paper in corpus.items():
    for sec in _sections(qasper_to_document(pid, paper)):
        paras = [sec.content[a:b] for a, b in _paragraph_spans(sec.content)]
        if len(paras) < 2: continue
        t = [ntok(p) for p in paras]; pt += t
        for i in range(len(paras)):
            lo, hi = max(0, i-W), min(len(paras), i+W+1); tot = sum(t[lo:hi]); n += 1
            if tot + 2 > LIMIT:
                over += 1; b = sum(t[lo:i]); cut += b >= LIMIT-2; part += (b < LIMIT-2 and b + t[i] > LIMIT-2); share.append((LIMIT-2)/tot)
print(len(pt), st.median(pt), over/n, cut/n, part/n, st.mean(share))
```
Output: 8,471 paragraphs; paragraph tokens median **90**, mean 104, p90 201; **77.5%** of windows exceed 256 tokens; centre paragraph fully truncated in **28.1%** of all windows, partially in **19.7%**; truncated windows keep a mean **55%** of their tokens.

### A6. FTS5 scaling and embedding throughput

Builds synthetic copies (×1, ×10, ×50) of the 7,347 real chunks (production `recursive_split_spans`) in a temporary SQLite file under `%TEMP%` (outside the repo and OneDrive; removed afterwards), then times `SqliteFtsIndex.search(q, 20)` for 105 answerable dev questions, corpus-wide and with a single-paper filter, with the default query and the stopword-filtered query.

```python
# (abridged) base = [(pid, section, idx, a, b, text), ...] from qasper_to_document + recursive_split_spans(1000, 200)
# qs = [(question, paper_id), ...] first 105 answerable questions of golden_set_v2.json["questions"]
ix = SqliteFtsIndex(tmp / f"fts_x{k}.sqlite3")
for c in range(k):                      # unique doc ids: f"{pid}#{c}"
    ix.upsert([Chunk.create(doc_id=f"{pid}#{c}", section=s, chunk_idx=i, char_start=a, char_end=b, strategy="recursive", text=t, doc_version="audit") for ...])
ix.search(query_or_filtered(q), 20, doc_ids=None | [f"{pid}#0"])      # timed with perf_counter
emb.encode_documents(texts_512)                                         # timed: batch throughput
```

| Chunks (docs) | Build | File | Default, corpus-wide p50 / p95 / max | Default, single-paper p50 / p95 / max | Stopword-filtered, corpus-wide p50 / p95 / max |
|---|---|---|---|---|---|
| 7,347 (200) | 1 s | 21 MB | 23.8 / 32.8 / 103.7 ms | 15.1 / 17.4 / 18.4 ms | 8.4 / 16.2 / 17.7 ms |
| 73,470 (2,000) | 8 s | 132 MB | 170.7 / 253.0 / 349.0 ms | 115.2 / 240.6 / 960.6 ms | 93.9 / 187.0 / 351.7 ms |
| 367,350 (10,000) | 219 s | 545 MB | 1,068.1 / 3,487.7 / 31,120.8 ms | 589.4 / 914.7 / 1,346.0 ms | 573.7 / 1,557.9 / 4,957.5 ms |

Embedding: **512 chunks (mean 692 chars) in 32.7 s = 15.7 chunks/s**; single-query encode p50 12.1 ms, p95 17.6 ms (n = 60). Caveats: copies of the same text inflate rare-term posting lists; one machine, shared laptop; a single query run; the stopword-filtered column excludes filter time (negligible).

### A7. Static metrics (AST)

Counts per package of functions, return/argument annotations, `print()` calls and broad `except Exception` handlers over `rag/ rag_core/ app/ db/ cache/ observability/ eval/ scripts/ utils/` (102 files, 18,034 lines): 78% return-annotated, 81% argument-annotated, 272 `print()` calls (164 in `scripts/`, 76 in `eval/`, 24 in `rag/`, 7 in `app/main.py`), 70 broad handlers (23 in `rag_core`; those I inspected are intentional fail-open with `# noqa: BLE001`). Table in `03` §6.

### A8. Two one-line checks

* `from google.genai import types; types.GenerateContentConfig(timeout=30)` → `ValidationError: 1 validation error for GenerateContentConfig / timeout` on the installed google-genai 3.6.1: the legacy fallback at `rag/retrieval.py:392` can never succeed.
* `scripts/ingest_qasper.py:1` is `raise SystemExit("Deprecated: wipes the collection. Use scripts/ingest.py")`.
