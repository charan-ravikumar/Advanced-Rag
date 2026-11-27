# Pre-registration: Phase 4 (generation evaluation: Answer-F1, citations, RAGAS, baselines; DEV split only)

Written 2026-11-22, BEFORE any generation run of this phase (the quota dry run of step 3 makes a few real calls AFTER this file is hashed, and
nothing it shows changes anything below except through a dated `## Deviations` entry). This file must not be edited after its sha256 was
recorded (`eval/PREREGISTRATION_gen.sha256`). If something has to deviate, a dated `## Deviations` section is appended at the end; the text above it
is never rewritten. The hash recorded in every run's `config.json` is computed over the text BEFORE the first line that starts with
`## Deviations` (so appending deviations does not change it); the hash of the whole file is recorded next to it. Same enforcement as
`eval/PREREGISTRATION.md` (Stage 2a): `eval/run_generation_eval.py` and `eval/run_ragas.py` refuse to run if the original text no longer matches.

Scope: generation evaluation on the DEV split only. The test split is NOT run in this phase: `eval/run_generation_eval.py --split test` is
refused unless a generation freeze file (`eval/gen_freeze.json`) exists, and that file is not created here. `eval/test_freeze.json`,
`docs/CLAIMS_LOG.md` and `README.md` are not touched. Nothing measured on dev in this phase is a publishable number.

## 1. Setting

- Primary setting: PAPER mode (the question's paper is given, `doc_ids = [paper]`; Qasper's canonical single-document task).
- Retrieval: the frozen configuration D unchanged: recursive store, hybrid RRF (k=60, vector weight 1.0, lexical weight 0.7), no reranker,
  `candidate_k=20`, store `%LOCALAPPDATA%\advrag\chroma_v2`.
- LLM context: the top **5** retrieved chunks (`generation.top_k=5`: a design default, NOT tuned), built by `rag_core/context.py` (relevance order, the
  conservative token estimate, an oversize first chunk is truncated).
- Generator: Groq `openai/gpt-oss-120b`, `reasoning_effort="low"` (a design default, not tuned), temperature 0, `max_output_tokens` 1024 (long
  answers) and 512 (short answers). Reasoning tokens count toward these caps. A reply that stops with `finish_reason` other than `stop` is scored as
  produced (an empty or truncated answer scores what it scores) and counted in the report; it is never silently re-run with other settings.
- Prompts: pinned to the LOCAL templates in `rag_core/prompts.py` (`PROMPT_VERSION` 2026-11-22.1), never the moving Langfuse `production` label:
  `rag-answer-long` sha256 `5d99e3a0c8aa6b9f2256d48dda6361812a9b4b55bae0ddcd53ad97de399376e4`, `rag-answer-short` sha256
  `f1d1d1cf61f87bebe8368f4d1049a7fe4611f41a754ea7ff5788ca295cca4a4d`. Both are recorded in every run's `config.json`. Tracing is off
  (`trace_evaluation=false`).
- Judge (RAGAS scoring only, never in the answer path): Gemini `gemini-2.5-flash`, temperature 0, `thinking_budget=1024`, `max_output_tokens=4096`,
  API version `v1beta`; RAGAS 0.4.4 (collections API) in the separate `venv_eval`; answer-relevancy `strictness=1`; RAGAS analytics off
  (`RAGAS_DO_NOT_TRACK=true`). Embeddings for answer relevancy: the local MiniLM at its pinned revision.
- Units: the dev split of `eval/datasets/qasper_dev/golden_set_v2.json`: 174 answerable questions from 71 papers, plus 19 unanswerable
  questions (used only for the abstention check on arms R and T).

## 2. Arms (what each isolates)

| Arm | Context given to the generator | Isolates |
|---|---|---|
| R | RAG: the top-5 retrieved chunks of configuration D, restricted to the question's paper | the system as built |
| O | ORACLE: the gold evidence paragraphs (the union over annotators of their verbatim evidence paragraphs; table/figure evidence is not retrievable and is excluded), in document order, **up to 5** (the first five in document order if there are more); each paragraph is one source. Never any retrieved chunk. Undefined for a question with no gold paragraph (excluded from O and from every R-vs-O comparison) and for unanswerable questions | the generator's ceiling when retrieval is perfect |
| T | TITLE + ABSTRACT only (one source: the title and the abstract of the question's paper) | what the model does without retrieval (questions were written from title and abstract) |
| F | the FULL paper in context | run ONLY if the measured limits allow it: the 95th-percentile full-paper prompt must fit under the per-minute token limit AND the daily token budget must cover the arm inside the plan of step 3. Otherwise the report states "not run: token limit" |

Short mode runs for R, O, T (and F if run); long mode runs for R and O. Unanswerable questions: arms R and T only.

## 3. Metrics

**Primary: Answer-F1 in SHORT-answer mode**, over the answerable questions that have at least one usable reference.
- SQuAD-style normalisation (lowercase, remove punctuation, remove the articles a/an/the, collapse whitespace; punctuation includes Unicode punctuation,
  applied identically to prediction and reference), token-level F1, **max over the usable references** (official Qasper definition).
- A reference is usable if its type is not `unanswerable` and it is not table-only (no verbatim evidence listed and at least one evidence item not retrievable).
  A question with no usable reference has no Answer-F1 for any arm (excluded everywhere).
- Extractive answers are stored joined with a space; after normalisation this equals the official ", " join (punctuation is removed).
- An abstention on an answerable question scores F1 = 0 (it is a miss, as in the official evaluation of "unanswerable" predictions).

**Secondary** (reported, not decisive):
- Answer-F1 by answer type, grouped by the type of the best-matching reference (ties: the first in reference order): extractive, abstractive. Yes/no accuracy: over
  questions with at least one usable yes/no reference, the predicted yes or no token equals a reference. Both are descriptive (small cells).
- Abstention rate on answerable questions (a miss) and on unanswerable questions (a hit), arms R and T.
- Citation validity rate (long mode): valid cited labels divided by cited labels; the share of answers with no citation is reported separately. A
  valid marker does not mean the claim is supported (that is what faithfulness is for).
- RAGAS (long mode, arms R and O, a seeded subsample, sized after step 3): faithfulness and answer relevancy (non-abstained answers only; an abstention is
  noncommittal by construction), context precision and context recall with the first usable annotator reference as the reference.
- **Validation study** (RAGAS context metrics against exact span metrics, on the same questions, arm R): Spearman rank correlation of RAGAS context recall and
  of context precision with the span-based evidence recall of the context actually sent; and a binary comparison on "gold evidence present in the context"
  (span rule: a sent chunk overlaps a gold span by at least half of the shorter of the two, the Stage 2a relevance rule) against RAGAS context recall >= 0.5: the
  fraction of questions on which they disagree, with the disagreeing questions listed for inspection. The threshold 0.5 is fixed here.
- **Judge calibration**: a seeded sample of 40 long-mode R answers, human labels filled by the owner later (no human step in this phase); agreement
  (percent, Cohen's kappa, bootstrap interval) between the human faithful/not-faithful label and RAGAS faithfulness binarised at **>= 0.5 means faithful**. The threshold is fixed
  here, before any label exists.
- Noise floor: arm R, short mode, run twice on a seeded subsample of 30 answerable questions (seed 0): the share of identical short answers and the Answer-F1 difference.

## 4. Comparisons and the number of tests

- Interval: 95% paired **paper-clustered** bootstrap (5,000 resamples, seed 0) from `eval/compare.py` (the per-question interval is a secondary column);
  **Holm** correction over all comparisons in one invocation.
- Invocation 1 (primary, m = 2): `R vs T` and `R vs O` on Answer-F1 (short mode). `R vs T` uses every answerable question with a usable reference;
  `R vs O` only the questions where O is defined (a `--common` option of `compare.py` restricts both arms to the questions defined for both and prints
  how many were dropped).
- Invocation 2 (secondary, descriptive, its own Holm family): the same pairs on yes/no accuracy, and `R vs O` on RAGAS faithfulness and answer relevancy
  over the scored subsample. Results of invocation 2 are labelled exploratory.
- The same comparisons will be run ONCE on the test split after review and after a generation freeze file is created by the owner. No arm is added or changed in this
  phase except through a new pre-registration.
- No configuration is selected in this phase and nothing is frozen: these arms are measurements (what retrieval adds over the abstract, what perfect retrieval would add), not candidates.

## 5. Sample sizes and quota

Set AFTER the quota dry run (step 3) and recorded as a dated `## Deviations` entry together with the measured limits. The default intent is the full dev
answerable set (174 questions) for Answer-F1 and a fixed seeded subsample for the RAGAS metrics (the same question ids for every metric and both arms:
questions ordered by `sha256("<seed>:<question id>")`, the first n taken among those where O is defined). Generations are cached on disk (keyed by model, reasoning
effort, prompt name, version and hash, arm, question id and context hash) and every judge call is cached (keyed by judge model, thinking budget, metric, ragas version and the hash of all inputs), so runs are resumable and a
quota stop ("quota exhausted, resume later") loses nothing. If the plan exceeds about 24 hours of wall time at the free limits, the phase stops and reports before any scored run.

## 6. What the numbers can and cannot say

- R vs T is confounded by how the questions were written (from the title and the abstract), so T may be strong; R vs O is an upper bound on what better retrieval can add,
  not a prediction, because annotators' evidence is sufficient for their own answer but is not the only sufficient context.
- LLM output is not deterministic even at temperature 0; the noise floor is reported next to every difference.
- Answer-F1 rewards word overlap: a correct abstractive answer in other words scores low. RAGAS scores are judge opinions until calibrated against the owner's labels;
  the calibration result is PENDING until the owner fills the sheet.

## Deviations

### 2026-11-22: quota dry run (step 3): the full plan does not fit in 24 hours; STOPPED before any scored generation run

Evidence: `runs/gen_dryrun/20261122T172032Z_quota_dry_run/dry_run.json` (code fingerprint and this file's hash recorded in it). The dry run made 25 real Groq calls
(5 seeded dev questions x arms R, O, T in short mode and R, O in long mode) to measure tokens per call; the answers were not analysed or reused.

- Groq limits for `openai/gpt-oss-120b` (docs `https://console.groq.com/docs/rate-limits.md`, matching the live response headers): 30 requests/minute, 1,000 requests/day,
  8,000 tokens/minute, **200,000 tokens/day**. The binding limit of the full plan is the daily token limit (`projection.full_plan.binding_limit = "tpd"` in the artifact);
  the artifact holds the projected calls, tokens, hours per limit and the number of daily allowances. The projected wall time exceeds the 24-hour stop in step 3.
  No reduction of arms, modes or sample sizes has been applied: sample sizes are NOT set by this entry.
- Counts the plan would run (artifact `counts`): the dev set has 174 answerable questions (all with a usable reference and an oracle context; 3 of them have more than 5 gold paragraphs and
  are truncated to 5) and 19 unanswerable ones.
- Arm F (full paper): the artifact records the estimated full-paper prompt sizes (95th percentile and maximum) next to the per-minute token limit; by the criterion in section 2 the arm is **not run: token limit**.
- Gemini judge (`gemini-2.5-flash`): the burst probe in the artifact (`gemini_rpm_probe`) received HTTP 429 naming the **Free Tier** limit of 5 requests per minute for this model;
  the per-day limit is not exposed by the API or the documentation (account specific, AI Studio only) and was not measured. The RAGAS call count per answer is structural (faithfulness 2, answer relevancy 1 with
  `strictness=1`, context precision one per context, context recall 1).
- Environment fact recorded here: RAGAS 0.4.4 runs in `%LOCALAPPDATA%\advrag\venv_eval` (see `requirements-eval.txt`, `docs/ragas_notes.md`).
- Nothing in the original text above was changed. The reduced plan (sample sizes, modes, whether to accept a multi-day resumable run) awaits the owner's decision and will be
  recorded in a further dated Deviations entry before any scored run.

### 2026-11-22: the owner REPLACED Phase 4 with a minimal version (recorded BEFORE any scored run)

Decision of the owner after the quota stop above. The pre-registered arms, comparisons, sample sizes and secondary analyses in sections 2 to 5 are **not run in this phase**; what runs instead:

- **Sample:** ONE seeded random sample of **40 answerable dev questions that have at least one usable (non-table) reference**: `random.Random(0).sample(sorted(eligible question ids), 40)`. The seed and the
  40 question ids are recorded in every run's `config.json`. Dev split only; the test split is never run (`eval/run_generation_eval.py --split test` is refused: no generation freeze file exists and none is created).
- **Arm:** **R only** (retrieved top-5 of configuration D, paper mode, exactly as in section 1), in **short mode and long mode** on the 40. The generator, `reasoning_effort="low"`, temperature 0, output caps, local prompt templates
  (name, version, sha256 recorded) and tracing-off are as in section 1.
- **Token governor:** a daily token budget, default **180,000 tokens**, from the project's OWN accounting of `usage` in the responses (Groq's headers give no tokens per day), over a rolling 24 hours; the tokens used outside the ledger
  in the same window are passed with `--already-used-tokens`. When the next call would exceed it, or Groq answers HTTP 429 (rate limit), the run exits cleanly ("quota exhausted, resume later"). Generations are cached on disk, so re-running costs nothing for what is done.
- **Metrics (all EXPLORATORY: n=40, one arm):** Answer-F1 in short mode (SQuAD normalisation, max over the usable annotator references), Answer-F1 by answer type of the best-matching reference, yes/no accuracy, abstention rate on answerable questions, and
  citation validity rate in long mode (valid cited labels over cited labels, and the share of answers with no citation). 95% intervals are paper-clustered bootstrap (5,000 resamples, seed 0; `eval/compare.py::clustered_bootstrap`).
- **RAGAS (EXPLORATORY: n=40, an UNCALIBRATED judge):** faithfulness and answer relevancy on the long answers only (non-abstained answers), Gemini `gemini-2.5-flash` at temperature 0 (`thinking_budget=1024`, `max_output_tokens=4096`, `v1beta`), ragas 0.4.4 in `venv_eval`,
  answer-relevancy `strictness=1`, local MiniLM embeddings. Every judge call is cached on disk (judge model, thinking budget, metric, ragas version, hash of the inputs) and the run is resumable. Context precision and recall are not scored.
- **Cut from this phase (not run, not claimed):** arms O, T and F; the R-vs-T and R-vs-O comparisons; the noise-floor repeat; the RAGAS-versus-span validation study; the judge calibration sheet and the 0.5 faithfulness threshold's use; the generation run on the test split.
  Any later use of them needs its own dated entry.
- Nothing in the original text above was changed.

### 2026-11-27: generator, judge and long-answer format changed (recorded at the owner's request; no scored run has used the new setup)

Three changes made AFTER the exploratory generation run of this phase. That run used the original setup, and nothing below changes or re-labels it.

- **Generator: Groq `openai/gpt-oss-120b` was replaced by Gemini `gemini-3.5-flash-lite`** (`generation.providers: ["gemini"]`, `generation.gemini_model`). This happened before today
  and was not recorded here at the time. It is recorded now, after the fact. Groq has been removed from the answer path. `reasoning_effort` is no longer sent (Gemini has no such
  parameter) and is recorded as `null`. The Groq limits in the 2026-11-22 entries do not apply to the new generator. The 180,000-token daily budget is kept as the project's own
  budget; it is NOT a documented Gemini limit. The pacing defaults in `eval/run_generation_eval.py` (`GEMINI_RPM`, `GEMINI_TPM`) are unverified conservative placeholders.
- **Judge: Gemini `gemini-2.5-flash` was replaced by Gemini `gemini-3.8-flash`** (`generation.judge.model`), with the same temperature, thinking budget and output limit. The owner
  chose a model that differs from the generator but is from the SAME family (Gemini). Family-level self-preference bias therefore remains possible, and every scoring run records
  this in `config.json` (`judge.family_note`). The judge is still uncalibrated. The model id was confirmed in the live model list on 2026-11-27. A one-call settings check
  (`scripts/check_model_ids.py`) failed twice with HTTP 503/504 (server side), so those settings have NOT yet been verified on this model.
- **Long-answer format: free text with `[S#]` markers was replaced by schema-constrained structured output.** The generator is given a JSON response schema (`rag_core/claims.py`,
  schema version `claims-1`). The answer is a list of claims, each citing one or more sources by id (an enum of exactly the ids that were sent) with a word-for-word quote from each.
  The long answer abstains through a `status` field instead of the `INSUFFICIENT_CONTEXT` sentinel. The short answer is unchanged.
  - Citation validity keeps its meaning: valid cited ids over cited ids, now read from the structured claims instead of a regex over the text.
  - A new EXPLORATORY metric, `QuoteVerified`, is recorded: the share of quotes found word for word in the source they cite. It was not pre-registered, it needs at least three
    words per quote (a design default), and a found quote does not show that the source supports the claim.
  - The long prompt changed (`PROMPT_VERSION` `2026-11-27.1`) and the generation cache schema moved to 3. Earlier cached answers are kept on disk and are never served for the
    new format.
- Any run with the new setup is a NEW exploratory run. Its numbers are not comparable with the earlier run's (different generator, judge and answer format), and no comparison
  between them is pre-registered.
- Nothing in the original text above was changed.
