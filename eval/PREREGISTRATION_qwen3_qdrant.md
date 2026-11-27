# Pre-registration: Stage 3 (Qwen3-Embedding-0.6B in Qdrant against the MiniLM + Chroma baseline; DEV split, then ONE final TEST run)

Written 2026-11-26, BEFORE any Qwen3 or Qdrant arm was built or run, and before the baseline run of section 3. Same rules as Stage 2a
(`eval/PREREGISTRATION.md`): this file must not be edited after its sha256 was recorded (`eval/PREREGISTRATION_qwen3_qdrant.sha256`); a
deviation is appended as a dated `## Deviations` section and the text above it is never rewritten. Every run of this stage passes
`--prereg eval/PREREGISTRATION_qwen3_qdrant.md --prereg-hash-file eval/PREREGISTRATION_qwen3_qdrant.sha256` and records the hash in its
`config.json`; the runner refuses to run if the original text no longer matches.

Scope: retrieval only (no LLM, no network at run time), `CHROMA_LOCAL_PATH=%LOCALAPPDATA%\advrag\chroma_v2` for the existing store. Dev numbers
of this stage are for development only: they go into run folders and `eval/baseline_dev.json`, never into `README.md` or `docs/CLAIMS_LOG.md`.
The TEST split is run exactly once, at the end, as section 6 says, and only when the owner asks. `eval/test_freeze.json` (Stage 2a) is not touched.
Everything not stated here (seed 0, 5 warm-up requests per arm, question order, span-based relevance with overlap >= 0.5 x the shorter of span and
chunk, 95% paired paper-clustered bootstrap with 5,000 resamples and seed 0 from `eval/compare.py`, Holm correction) is as in Stage 2a.

## 1. The question and the arms

Question: does a different embedding model, served from a different vector store, retrieve better than the system that is served today?
The comparison is between two COMPLETE vector legs (embedder AND store together), because that is the decision the owner faces. Two diagnostic
arms separate the two factors. The lexical leg (SQLite FTS5 over the same text) and the fusion code (weighted RRF, k=60, 1.0 / 0.7) are IDENTICAL
in every hybrid arm; the chunks are IDENTICAL (same text, same ids: section 7 verifies it).

| Arm | Chunks | Vectors | Vector store | Retrieval | Role |
|---|---|---|---|---|---|
| A | recursive (`chroma_v2`) | MiniLM (`all-MiniLM-L6-v2`, pinned revision) | Chroma (HNSW) | vector only | BASELINE, existing arm |
| D | same | MiniLM | Chroma | hybrid RRF k=60, vector 1.0, lexical 0.7 | BASELINE, the served configuration, existing arm |
| QA | same | Qwen3-Embedding-0.6B | Qdrant (local mode) | vector only | candidate |
| QD | same | Qwen3-Embedding-0.6B | Qdrant (local mode) | hybrid RRF, same parameters as D | candidate |
| QAc | same | Qwen3-Embedding-0.6B | Chroma (HNSW) | vector only | diagnostic: same vectors as QA in the store of A |

Nothing else is tried in this stage. In particular: no other Qwen3 size, no truncated (MRL) dimensions, no quantization, no Qdrant sparse or BM25
vectors, no reranker, no instruction wording other than the one in section 7, no change of chunking. Adding any of them needs a NEW pre-registration.

## 2. Setting, depth and metrics

- Unit of analysis: the 174 answerable dev questions from 71 papers (`eval/datasets/qasper_dev`, `split.json`).
- Depth, identical for EVERY arm of this stage including the baseline: `top_k=50`, `candidate_k=100` (each leg returns 100 candidates before fusion).
  This differs from Stage 2a (`top_k=10`, `candidate_k=20`) because Recall@50 needs 50-deep lists; a hybrid list built from 20 candidates per leg cannot
  reach 50. The baseline is therefore RE-RUN at this depth (section 3); the Stage 2a dev numbers are not reused as the baseline.
- PRIMARY metrics (two, each in its own setting; see the reason below):
  1. **nDCG@10**, PAPER mode (Qasper's canonical single-document task: the question's paper is given), binary gain, span-based relevance.
  2. **Recall@50**, CORPUS mode (EXTENSION: corpus-wide retrieval, no paper filter). Recall@k = relevant chunks in the top k / relevant chunks that exist
     in the gold paper's index, averaged over REACHABLE questions (those with at least one relevant chunk; the count is reported). Defined in
     `rag/eval_metrics.py::recall_at_k`.
- Why Recall@50 is primary in corpus mode only: the corpus holds 7,255 chunks in 200 papers, and the dev papers have between 11 and 79 chunks (median 35;
  64 of the 71 dev papers have 50 or fewer; measured 2026-11-26 on `chroma_v2`). In paper mode a top-50 list therefore usually contains the
  whole paper, and Recall@50 is close to 1.0 for every arm by construction; it would not discriminate. It is still computed and reported in paper mode, as
  a descriptive column that no decision uses.
- Secondary metrics (descriptive; exploratory families in section 4): nDCG@10 in corpus mode, MRR, HitRate@5, EvidenceRecall@10, EvidenceRecall@50,
  EvidenceF1@5 (paragraph level, max over annotators), DocHit@1/5/10 (corpus mode), and retrieval latency (p50 and p95, CPU only, per stage, with the
  `embed` stage reported separately for every arm). Latency decides nothing.
- Two modes are run for every arm: `paper` and `corpus`. `corpus_titled` is not run in this stage.

## 3. Baseline freeze (made before any candidate exists)

The current setup is run through the dev evaluation FIRST and its result is frozen; every later number of this stage is compared with it.

- Arms A and D, split dev, modes paper and corpus, `top_k=50`, `candidate_k=100`, two repeats per mode (determinism, `eval/determinism.py`).
- Each run folder records the code fingerprint (`python -m utils.fingerprint`) and this file's sha256. The four folders, their `SHA256SUMS`, the fingerprint
  and the summary means are written into `eval/baseline_dev.json`, which is not edited afterwards (a later correction is a new file).
- The first repeat of each mode is THE frozen baseline; the second repeat is used only for the determinism report.
- `compare.py` pairs arms inside ONE run folder, so every candidate run of this stage includes A and D again (same code, same stores) next to the
  candidate arms. A baseline that does not reproduce the frozen one up to the known HNSW non-determinism (retrieved lists and metrics, as `determinism.py`
  reports them) is reported before any candidate result is read, and the candidate comparison uses the in-run baseline rows.
- Chroma HNSW is known not to be bit-reproducible across processes; Qdrant local mode is exact (brute force), which is a stated confound (section 8).

## 4. Dev comparisons, their count and Holm correction

All differences are first minus second. Holm is applied across the comparisons within ONE `compare.py` invocation; each family is its own invocation.

- Family D1 (DECISION, section 5): D:QD on nDCG@10 (paper mode) and D:QD on Recall@50 (corpus mode). They live in two run folders (one per mode), so they are
  two `compare.py` invocations; the correction across the two primary metrics is a Bonferroni split applied by the decision rule itself: a primary comparison
  counts as significant only if its clustered 95% CI excludes 0 AND its clustered bootstrap p-value is below 0.025.
- Family D2 (vector-only and factor separation, paper, nDCG@10): A:QA, QD:QA, A:QAc, QAc:QA -> m = 4. The same four pairs on Recall@50 in corpus
  mode -> m = 4 (separate invocation).
  - A:QAc is the EMBEDDER effect (same store, same chunks, different vectors); QAc:QA is the STORE effect (same vectors, different store); QD:QA asks
    whether the lexical leg helps the new embedder as D:A did for MiniLM.
- Family D3 (exploratory, secondary metrics): the six pairs D:QD, A:QA, QD:QA, A:QAc, QAc:QA, D:A on nDCG@10 (corpus), MRR, HitRate@5, EvidenceRecall@10,
  EvidenceRecall@50, EvidenceF1@5, per mode -> reported with their Holm flags, never used for a decision.
- The Holm flag, not the raw interval, decides "significant"; an interval without its counts (n, papers, W/L/T, p, m) is never reported.

## 5. Decision rule (applied to the dev runs; it recommends, the owner freezes)

Let QD be the candidate and D the baseline (the served configuration).

1. QD is SIGNIFICANTLY BETTER than D on a primary metric if (QD - D) is positive, its clustered 95% CI excludes 0 AND its clustered bootstrap p-value is
   below 0.025 (family D1: the 0.05 level split over the two primary metrics).
2. QD is SIGNIFICANTLY WORSE on a primary metric by the mirror condition.
3. Outcome ADOPT-CANDIDATE: QD is significantly better than D on at least one primary metric and significantly worse on neither.
4. Outcome KEEP-BASELINE: every other result, including "no significant difference" and "better on one, worse on the other". A tie goes to the existing,
   already served system (fewer moving parts: one model, one store, no new dependency).
5. The factor-separation results (family D2) are reported to say WHY, and change no outcome.
6. Latency, memory, index build time and embedding time are reported next to the outcome and are not part of it. If ADOPT-CANDIDATE is the outcome, the
   report also states what it costs (model size on disk, ingest time, query embedding time) so that the owner can decide whether to freeze it.
7. Nothing is frozen by this rule; the owner freezes.

## 6. The final TEST run (exactly one)

- The test split is run ONCE for this stage, when the owner asks, with `--i-understand-this-is-final`, after the owner has frozen the configuration chosen by
  section 5 and created `eval/test_freeze_qwen3_qdrant.json` (passed as `--freeze-file`; it pins the arms, depth, model id and revision, and store
  signatures; it is never edited or recreated). Entries go to a ledger of their own (`--ledger-file eval/test_ledger_qwen3_qdrant.jsonl`).
- "One run" means one pair of invocations (paper and corpus mode) made together with the same code fingerprint, arms D, A, QD and QA, `top_k=50`,
  `candidate_k=100`. If the pair fails before any result is written, it may be repeated with the identical configuration; once a result has been read it is
  never repeated, whatever it shows. Nothing is tuned on test.
- Test comparisons (exactly these; Holm over m = 4, one family): D:QD and A:QA on nDCG@10 (paper mode), and D:QD and A:QA on Recall@50 (corpus mode).
  Reported as one table with counts, flags and the primary-metric definitions of section 2.
- Disclosure: the TEST split has already been evaluated once for the MiniLM arms (Stage 2a, depth 10; `eval/test_ledger.jsonl`). The Stage 2a numbers stay
  as logged and are not replaced. This stage's test run re-evaluates the MiniLM arms at depth 50 (Recall@50 did not exist then) in the same invocation as the
  candidate. The candidate arms were chosen from the owner's plan, not from any test result, and no test number of any kind has been used to choose or
  change anything in this stage, but the MiniLM test results from Stage 2a are known to the author, so the test is not a fresh look for the baseline.

## 7. How the candidate is built (fixed now; none of it is tuned)

- Embedding model: `Qwen/Qwen3-Embedding-0.6B` (Apache-2.0), revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` (the Hugging Face model sha read
  2026-11-26), loaded with `sentence-transformers` from the local cache only. It is downloaded ONCE, with the owner's approval, before any candidate arm is
  built; if the download is blocked, the candidate arms are not run and this is reported as a Deviation. The revision is recorded in every run's `config.json`.
- Encoding: documents (chunks) WITHOUT an instruction; queries with the instruction prefix
  `Instruct: Given a question about a scientific paper, retrieve passages from the paper that answer the question\nQuery: <question>`
  (the model's documented query format; this one string is used for every query and is not varied). Last-token pooling and L2 normalisation as in the model's
  sentence-transformers configuration; native output size (no truncated dimensions); `float32` on CPU; `max_seq_length=1024` for chunks and queries. The
  number of chunks or queries that would still be truncated is counted and recorded and is expected to be 0; a non-zero count is reported, not fixed silently.
- Qdrant: `qdrant-client` installed ONLY from the pip index already configured on this machine (version recorded in `config.json`), in EMBEDDED LOCAL mode:
  `QdrantClient(path=...)` under `%LOCALAPPDATA%\advrag\qdrant_qwen3` (outside OneDrive). No server, no Qdrant Cloud, no network. One collection, cosine
  distance, no quantization, no payload-index tuning; point id = UUIDv5 of the chunk id, payload = `chunk_id`, `doc_id` (= paper id), `section`,
  `char_start`, `char_end`. Search: exact nearest neighbours as local mode does, with the paper filter on `doc_id` for paper mode.
- QAc: the SAME Qwen3 vectors written to a separate Chroma collection (default HNSW settings, as for MiniLM). The embedding is computed once and reused for
  both stores, so QA and QAc differ only in the store.
- Verification before any candidate arm is evaluated: every chunk id of `chroma_v2` (recursive, 7,255 chunks) is present exactly once in each candidate
  store with identical text, `doc_id`, section and offsets (a read-only check, counts recorded; the same field comparison `scripts/export_chunks.py` makes
  for the export of step 3). The lexical index is the existing one (same ids). Any mismatch stops the stage.
- Candidate runs include A and D again (section 3) and are made in one run folder per mode.

## 8. Known confounds and limits (stated now so that they cannot be discovered later)

- QA vs A changes the embedder AND the store (and Qdrant local search is exact while Chroma's HNSW is approximate). This is why QAc exists: A:QAc is the
  embedder effect, QAc:QA the store effect. Neither diagnostic changes the decision.
- Qwen3 is a much larger model than MiniLM (about 0.6 billion parameters against about 22 million); any gain has a cost in model size, embedding time and
  memory, which section 5 item 6 reports.
- Only 71 dev papers, 174 questions: wide intervals; "no significant difference" is the likely outcome for small effects and is a valid result.
- Recall@50 uses chunk-level relevance, which depends on the chunking; every arm here uses the same chunks, so the comparison is fair, but Recall@50 values
  are not comparable with a system that uses different chunks.
- Dev-set conclusions say nothing about other corpora; one dataset (Qasper), English scientific text.
- The Stage 2a baseline numbers (depth 10) and this stage's baseline (depth 50) are different measurements and are never mixed in one table.

## 9. Other commitments

- No arm, grid point, metric, threshold, instruction string, depth or decision rule is added or changed after results have been seen. Anything that has to
  change is a dated Deviation, and a new arm is a new pre-registration.
- Every run of this stage records this file's sha256, the code fingerprint and, for candidate runs, the model revision and the `qdrant-client` version.
- A stage that stops (blocked download, a failed verification, a missing package) is reported as stopped; it is not worked around.
