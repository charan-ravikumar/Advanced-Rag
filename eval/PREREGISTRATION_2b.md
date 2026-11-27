# Pre-registration: Stage 2b (Gemini embedding arm and two-stage paper routing, DEV split only)

Written 2026-11-21, BEFORE any Stage 2b configuration was run (no Gemini call, no routing run, no new arm evaluated). Same rules as
Stage 2a (`eval/PREREGISTRATION.md`, original-text sha256 `90dbf0e1625ba6dfcc399c93257b1ef042e1ca791199f2d4b3bfc225a1a13531`): this file must not
be edited after its sha256 was recorded (`eval/PREREGISTRATION_2b.sha256`); a deviation is appended as a dated `## Deviations` section and the
text above it is never rewritten. Every Stage 2b run passes `--prereg eval/PREREGISTRATION_2b.md --prereg-hash-file eval/PREREGISTRATION_2b.sha256`
and records the hash in its `config.json`; the runner refuses to run if the original text no longer matches.

Scope: retrieval only, dev split only, `CHROMA_LOCAL_PATH=%LOCALAPPDATA%\advrag\chroma_v2` for the existing stores. The test split is not run, and
`eval/test_freeze.json`, `docs/CLAIMS_LOG.md` and `README.md` are not touched. Nothing measured on dev here is publishable.
Everything not stated here (top_k=10, candidate_k=20, seed 0, 5 warm-up requests per arm, 95% paired paper-clustered bootstrap with 5,000
resamples and seed 0, span-based relevance, Holm correction) is as in Stage 2a.

## 1. Levers and arms

### Lever 1: a Gemini embedding arm on the SAME recursive chunks (paper mode, primary setting)
The Gemini model is the current generally-available Gemini embedding model on the owner's tier (expected `gemini-embedding-001`); its id and
version are recorded at the dry run (section 4) before any arm runs, and not changed afterwards. 768 output dimensions, L2-normalised
client-side; `RETRIEVAL_DOCUMENT` for chunks and paper index entries, `RETRIEVAL_QUERY` for questions. The chunks are the identical recursive chunks
of `chroma_v2` (same chunk ids, verified); only the vectors differ. The lexical (FTS5) index is the same text.

| Arm | Description |
|---|---|
| A | recursive, vector-only, MiniLM (existing arm) |
| D | recursive, hybrid RRF (k=60, vector 1.0, lexical 0.7), MiniLM vectors (existing arm) |
| GA | recursive, vector-only, Gemini vectors |
| GD | D's configuration with Gemini vectors (RRF k=60, 1.0 / 0.7, same FTS index) |

### Lever 2: two-stage paper routing (corpus mode with plain questions, no title)
Stage 1 (the router, `rag/router_docs.py`): a paper-level index over `title + abstract` of each paper, built ONLY from paper metadata (no question
text and no gold information). Vector leg: exact brute-force cosine over the paper vectors with the arm's embedder (deterministic, no ANN);
lexical leg: SQLite FTS5 (porter, OR of the words) over the same text; hybrid RRF (k=60, vector 1.0, lexical 0.7, 20 candidates per leg); returns
the top-m paper ids (ties by doc id). Stage 2: arm D's configuration restricted to those papers (`doc_ids = routed`), top_k=10, candidate_k=20.
MiniLM truncates the title+abstract text at 256 tokens at embed time (a known limitation).

| Arm | Description |
|---|---|
| D | plain corpus-wide D (existing arm; the baseline) |
| RT1 | route to the top 1 paper, then D |
| RT3 | route to the top 3 papers, then D |
| RT5 | route to the top 5 papers, then D |
| GRT3 | (only if Gemini arms exist) route to the top 3 with a Gemini router, then GD |

Router diagnostics (descriptive, no comparison): router DocHit@m for m = 1, 3, 5, 10 on the dev questions, for the MiniLM router and, if it exists,
the Gemini router.

## 2. Runs

| Run | Setting | Arms | Repeats |
|---|---|---|---|
| P2b | paper mode (PRIMARY) | A, D, GA, GD | 2 |
| C2b | corpus mode, plain questions (EXTENSION: the deployed path) | D, RT1, RT3, RT5, GRT3 | 2 |

Gemini query-embedding latency (a network call) is recorded separately from the local stages: the embedder records per-call network time, cache hits and
misses, retries and HTTP 429 counts. Because every Gemini embedding is cached on disk, the first repeat of a run measures the network cost
(cold) and the second one mostly hits the cache; both are reported and labelled. Retrieval latency excludes metric computation.

## 3. Metrics, comparisons, Holm, decision rule

Paper mode: primary nDCG@10; secondary MRR, HitRate@5, EvidenceRecall@10, EvidenceF1@5.
Corpus mode: primary **HitRate@5**; secondary DocHit@5 and nDCG@10.

Comparisons (difference = first arm minus second; clustered 95% CI primary; one Holm family per `compare.py` invocation, i.e. "Holm over all
comparisons in each setting"):

- Paper family: pairs A:GA and D:GD, on the 5 paper metrics -> m = 10.
- Corpus family: pairs D:RT1, D:RT3, D:RT5 and (if GRT3 exists) GRT3:RT3, on HitRate@5, DocHit@5, nDCG@10 -> m = 12 (9 without GRT3).
- Decision-rule families (as in 2a, separate invocations): BEST:X on the primary metric of the setting -> m = 3 (paper) and m = 4 (corpus; 3 without GRT3).
- Total: 10 + 12 + 3 + 4 = 29 comparisons in four Holm families (fewer if the Gemini arms do not exist).

Decision rule (unchanged from 2a section 3, applied per setting on the arms of that setting's run, reported WITHOUT freezing): BEST = highest mean of the
setting's primary metric (ties: lower complexity, smaller m, name); every other arm X is compared with BEST; X is indistinguishable if the
clustered CI of the difference includes 0; the recommendation is the least complex candidate (ties: higher mean, then smaller m, then name). Complexity
points, one each: vector leg, lexical leg, fusion step; two for an external-API embedder (network dependency, cost, data leaves the machine); two for
two-stage routing (a second index and stage). A: 1. D: 3. GA: 3. GD: 5. RT1, RT3, RT5: 5. GRT3: 7. As in 2a the Holm-sensitivity outcome is reported too,
with the reasoning of 2a Deviation 7 (non-transitivity) applying if the two differ.

## 4. Gemini dry run and contingencies

- The API key is read from the environment variable `GEMINI_API_KEY` only; it is never printed, logged or written to any file or artifact.
- Before any Gemini arm is built: embed ONE paper's recursive chunks, record the model id and version, tokens, whether a batch counts as one request
  or one per text (if visible), wall time and any HTTP 429; extrapolate to the 7,255 chunks plus the dev questions (and the 200 paper-index
  entries). If the extrapolated wall time exceeds about 24 hours, or repeated quota errors appear, STOP and report; no Gemini arm (GA, GD, GRT3) is then run.
- Only public Qasper text (paper chunks, titles, abstracts) and dev questions are sent to Google.
- Embeddings are cached on disk by (model, version, task, dims, sha256(text)); ingestion is resumable and exits cleanly when the quota is exhausted.
  Network failures or quota exhaustion mid-run are not silently skipped: the run fails.
- If the Gemini arms cannot be completed, that is a recorded Deviation and their comparisons are dropped from the counts above.

## 5. The frozen configuration and the test-split comparisons

The frozen configuration is a PAIR: the paper-mode configuration (the winner of the paper-mode rule over {A, D, GA, GD}, with 2a's reranker, grid and
stopword arms not re-opened) and the corpus-mode configuration (the winner of the corpus-mode rule over {D, RT1, RT3, RT5, GRT3}). The final retrieval
configuration may combine the two, because the deployed application searches all papers (the corpus path) while paper mode is the benchmark: a routing arm
with the paper-mode winner as its second stage. Only combinations that were actually tested are allowed (routing was tested with stage 2 = D, and
GRT3 with GD); anything else needs a new pre-registration. The owner freezes it; nothing is frozen in this stage.

The test split is evaluated ONCE per setting, with `--i-understand-this-is-final`, and the ledger is appended. Comparisons (a self-comparison is dropped;
one Holm family per setting): 
- Paper mode, nDCG@10 (the 2a list): A vs FROZEN_PAPER; D vs FROZEN_PAPER; F vs FROZEN_PAPER; FROZEN_PAPER vs FROZEN_PAPER + reranker (only if FROZEN_PAPER is D or F);
  and the 2b additions: A vs GA; D vs GD.
- Corpus mode, HitRate@5: D vs the best routing arm (the routing arm with the highest dev HitRate@5 among RT1, RT3, RT5, GRT3).
Secondary metrics are reported descriptively in a separate invocation and are exploratory.

## 6. Other commitments

- Every Stage 2b run records this file's sha256 (`config.json`, `preregistration.sha256_original_text`).
- Each configuration is run twice and run-to-run agreement is reported with `eval/determinism.py` (metrics identical, retrieved lists identical, timing only).
- No arm, metric, router parameter, m or decision rule is added or changed after results have been seen.

## Deviations

### 2026-11-22: NOT EXECUTED: scope cut by the owner

Stage 2b was cut by the owner before any of its arms ran. Nothing in this file was executed: no Gemini embedding call was made (no key was ever
available to the process, and none was used), no Gemini store was built, and none of the arms GA, GD, RT1, RT3, RT5 or GRT3 was evaluated, so no
comparison, decision-rule outcome or test-split comparison of this file exists or may be quoted. The frozen configuration (D) and the single test run
follow `eval/PREREGISTRATION.md` only; this file played no part in them. The code written for it was kept and not deleted; it is dormant and is not
imported by the generation path (`rag_core`). The only run that touched anything of it is a descriptive MiniLM router diagnostic saved under
`runs/measurements/`; it informs no decision. The original text above was not edited.
