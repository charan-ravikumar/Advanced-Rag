# Pre-registration: Stage 2a (diagnostics and local-only retrieval arms, DEV split only)

Written 2026-11-21, BEFORE any configuration of this stage was run. This file must not be edited after its sha256 was recorded
(`eval/PREREGISTRATION.sha256`). If something has to deviate, a dated `## Deviations` section is appended at the end; the text above it
is never rewritten. The hash recorded in every run's `config.json` is computed over the text BEFORE the first line that starts with
`## Deviations` (so appending deviations does not change it); the hash of the whole file is recorded next to it.

Scope: retrieval only (no LLM, no network at run time), dev split only, `CHROMA_LOCAL_PATH=%LOCALAPPDATA%\advrag\chroma_v2`.
The test split is not run in this stage. `eval/test_freeze.json`, `docs/CLAIMS_LOG.md` and `README.md` are not touched.
Nothing measured on dev in this stage is a publishable number.

## 1. Setting and metrics

- Primary setting: PAPER mode (`doc_ids = [the question's paper]`; Qasper's canonical single-document task).
- Primary metric: **nDCG@10** (binary gain, span-based relevance as in `rag/eval_metrics.py`).
- Secondary metrics: MRR, HitRate@5, EvidenceRecall@10, EvidenceF1@5 (paragraph-level, max over annotators).
- Unit of analysis: the 174 answerable dev questions from 71 papers. Interval: 95% paired **paper-clustered** bootstrap
  (5,000 resamples, seed 0) from `eval/compare.py`; the per-question interval is a secondary column only.
- Common settings for every arm: `top_k=10`, `candidate_k=20` (each leg returns 20), question order seed 0, 5 warm-up requests per arm
  discarded from latency, recursive store unless the arm says semantic.

## 2. Dev configurations (the complete list; nothing else is tried in this stage)

Paper-mode grid, 15 arms, evaluated in ONE run folder so that every pair is paired on the same rows:

| Arm | Store | Retrieval | Detail |
|---|---|---|---|
| A | recursive | vector | |
| B | recursive | lexical (FTS5) | OR of all query words |
| C | recursive | hybrid, naive interleave | |
| D | recursive | hybrid, RRF | k=60, vector 1.0, lexical 0.7 (the current default) |
| E | semantic | vector | |
| F | semantic | hybrid, RRF | k=60, 1.0 / 0.7 |
| G | recursive | D + rerank | cross-encoder, `rerank_k=20` fused candidates, keep top 10 |
| H | semantic | F + rerank | same reranker |
| R20_05 | recursive | hybrid, RRF | k=20, 1.0 / 0.5 |
| R20_07 | recursive | hybrid, RRF | k=20, 1.0 / 0.7 |
| R20_10 | recursive | hybrid, RRF | k=20, 1.0 / 1.0 |
| R60_05 | recursive | hybrid, RRF | k=60, 1.0 / 0.5 |
| R60_10 | recursive | hybrid, RRF | k=60, 1.0 / 1.0 |
| B2 | recursive | lexical, stopword-filtered | OR-joined FTS5 query after removing English stopwords (fixed list in `rag/lexical_query.py`); if every word is a stopword the unfiltered query is used |
| D2 | recursive | hybrid, RRF over vector + B2 | k=60, 1.0 / 0.7 |

(The RRF grid is rrf_k in {20, 60} x lexical_weight in {0.5, 0.7, 1.0}, vector weight 1.0: six configs, of which `R60_07` is arm D.)

Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`, downloaded once from HuggingFace, revision pinned and recorded in `config.yaml` and in each
run's `config.json`, loaded lazily and local-only afterwards; scores are raw cross-encoder outputs on `(query, chunk text)`, ties broken
by chunk_id. Reranker latency is recorded separately (`rerank` stage). If the download is blocked, arms G and H are not run and this is
reported (a Deviation), and the stage stops.

Other runs of this stage:

| Run | Mode | Arms | Repeats | Purpose |
|---|---|---|---|---|
| P | paper | the 15 above | 2 | main table; determinism |
| T | `corpus_titled` (query = `In the paper "<title>": <question>`, no doc filter) | A-F | 2 | title-qualified corpus extension |
| K | corpus (no filter), `top_k=20` | A-F | 1 | only to measure DocHit@20 (the existing corpus runs stop at 10); its metrics at k <= 10 are expected to match the existing corpus run up to HNSW noise |

Existing dev run reused for diagnostics (no new retrieval): `runs/eval/20261121T121338Z_b45a6048_dev_corpus` (corpus mode, arms A-F).
The error analysis uses `runs/eval/20261121T121057Z_b45a6048_dev_paper` (arm D). Error-analysis categories are PROPOSED only and are not used
to add or change any arm in this stage.

## 3. Decision rule (applied to the paper-mode run P, reported WITHOUT freezing anything)

1. BEST = the arm with the highest mean nDCG@10 among the 15 (ties: lower complexity, then name).
2. For every other arm X compute `compare.py` BEST:X on nDCG@10 with the paper-clustered CI.
3. X is *indistinguishable from BEST* if its CI for the difference includes 0. Candidates = BEST plus all indistinguishable arms.
4. The recommended frozen configuration = the candidate with the LOWEST COMPLEXITY; ties: higher mean nDCG@10, then name.
5. Complexity = number of points, one point each for: vector leg; lexical leg; fusion step (naive or RRF); semantic chunker;
   stopword-filtered lexical query; non-default RRF parameters (anything other than k=60, lexical weight 0.7); two points for a reranker.

| Arm | A | B | E | B2 | C | D | F | D2 | R* | G | H |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Complexity | 1 | 1 | 2 | 2 | 3 | 3 | 4 | 4 | 4 | 5 | 6 |

6. The rule uses the unadjusted 95% clustered CIs, as specified. As a sensitivity report only, the same rule is also applied with
   "indistinguishable" meaning "not significant after Holm over the 14 BEST:X comparisons"; if the two outcomes differ, both are reported
   and the more complex of the two is NOT preferred automatically.
7. Nothing is frozen in this stage: the outcome is stated only.

## 4. Planned TEST-split comparisons (exactly these, run once at the freeze)

With FROZEN = the configuration recommended by section 3 (after the owner freezes it):
1. A vs FROZEN
2. D vs FROZEN
3. F vs FROZEN
4. FROZEN vs FROZEN + reranker (only if the reranker arm of the frozen configuration is in the grid above: i.e. FROZEN is D or F)

A comparison of an arm with itself is dropped (so m is 3 or 4). Metric: nDCG@10, paper mode, paper-clustered CI. Holm correction over the
m comparisons of this family. Secondary metrics are reported descriptively in a separate invocation and are exploratory.
The test split is evaluated once, with `--i-understand-this-is-final`, and the run is appended to `eval/test_ledger.jsonl`.

## 5. Dev comparisons in this stage, their count and Holm correction

All differences are first minus second; Holm is applied across all comparisons within ONE `compare.py` invocation; the families are
separate invocations.

- Family P1 (paper, primary metric nDCG@10): 15 pairs: A:C, A:D, C:D, D:B, A:E, D:F, D:G, F:H, B:B2, D:D2, D:R20_05, D:R20_07, D:R20_10,
  D:R60_05, D:R60_10 -> m = 15.
- Family P2 (paper, secondary metrics MRR, HitRate@5, EvidenceRecall@10, EvidenceF1@5): the same 15 pairs -> m = 60, exploratory.
- Family P3 (decision rule): BEST:X for the 14 other arms, nDCG@10 -> m = 14.
- Family C (existing corpus run) and family T (corpus_titled run P2): pairs A:D and D:F on nDCG@10, MRR, HitRate@5, EvidenceRecall@10,
  EvidenceF1@5 -> m = 10 per mode; corpus-wide results are an EXTENSION, not the primary setting.
- Total: 15 + 60 + 14 + 10 + 10 = 109 comparisons in 5 Holm families. The Holm flag, not the raw interval, decides "significant";
  an interval without its counts and flag is never reported.

## 6. Other commitments

- Every run of this stage records this file's sha256 in `config.json`.
- Each new configuration is run twice (K once, as stated) and run-to-run agreement is reported with `eval/determinism.py`
  (metrics identical, retrieved lists identical, timing only). Chroma HNSW is known not to be bit-reproducible across processes.
- Diagnostics (not comparisons): DocHit@k for k in 1, 5, 10, 20 per arm from corpus runs; HitRate@5 conditional on DocHit@5 = 1 with n,
  compared with paper-mode HitRate@5 on the same questions.
- No arm, grid point, metric, threshold or decision rule is added or changed after results have been seen.

## Deviations (appended 2026-11-21, after the runs of this stage; the text above is unchanged)

None of these changes an arm, a grid point, a metric, a threshold or the decision rule.

1. `eval/compare.py` was changed BEFORE any Stage 2a run so that a metric that is undefined for a question (Evidence-F1 is `None` when every
   annotator's evidence is a table or figure) is left out for both arms instead of crashing. This only affects the secondary metric
   EvidenceF1@5 (family P2 and the corpus families); the primary metric nDCG@10 is defined for all 174 questions. Test added in
   `tests/unit/test_compare.py`.
2. DocHit@k diagnostics were also computed for the `corpus_titled` run T (section 6 named only the corpus runs); descriptive only.
3. The paper-mode table, families P1/P2/P3 and the diagnostics use the first repeat of run P (`20261121T125301Z_0d717273_dev_paper`); the second repeat
   is used only for the determinism check. The error analysis sampled 25 of the 26 arm-D misses (sample size as specified; the population is only 26).
4. Family C uses the existing corpus run `20261121T121338Z_b45a6048_dev_corpus`, as stated in section 2; it predates this file, so its
   `config.json` has no pre-registration hash.
5. Run K (corpus, `top_k=20`) matched the existing corpus run exactly at k <= 10 (0 of 1,044 rows differ in HitRate@10, nDCG@10, DocHit@10 or in the top-10 list).
6. Tooling, not content: after this file was appended to, the hash check in `eval/run_retrieval_eval.py::preregistration_info` was made
   line-ending-insensitive (the file was created, and its hash recorded, with CRLF line endings; the first version of the check
   mis-handled the final line ending). The original text still hashes to the recorded `90dbf0e1...3531`; the runs of this stage
   recorded that same value in their `config.json` (`sha256_original_text`).
7. Interpretation note (appended 2026-11-21 23:38, Stage 2b housekeeping; explains, does not change, the rule). Section 3 step 6 reports two
   outcomes when they differ and says the more complex one is not preferred automatically. The observed outcomes were D (the rule as stated:
   unadjusted clustered CIs) and A (the Holm-sensitivity version). D is the one carried forward, for this reason: "indistinguishable from the
   best" is not transitive. A was not distinguishable from the best arm H after Holm over 14 comparisons (an under-powered test), but in the
   direct comparison A vs D on nDCG@10 (family P1, `runs/analysis/20261121T_stage2a_compare/P1_paper_primary_nDCG10.txt`) A was itself
   Holm-significantly WORSE than D (diff -0.047, clustered CI [-0.077, -0.017], W/L/T 47/77/50), while D was indistinguishable from H.
   So the Holm version picks an arm that is demonstrably beaten by a candidate that is also within noise of the best; failing to reject
   "A = H" is weak evidence compared with rejecting "A = D". The stated (primary) rule's outcome D is therefore preferred. This is a
   statement of reasoning only: no threshold, arm or rule was changed, and nothing was frozen.
