# TODO

## Baseline generation evaluation (hallucination and abstention): deferred until paid model quota is available

Deferred by the owner on 2026-11-28. **Return point:** commit `cbf0897` (main when it was deferred:
`git checkout cbf0897` to see that state; run on the then-current main unless there is a reason not to).

**Task.** Measure the current generator for baselines B0, B1, B4: Answer-F1, abstention recall on the unanswerable
questions, false-abstention rate on answerable questions, citation label validity (cited labels exist in the
context that was sent). LLM judge. One run folder with config, metrics, bootstrap confidence intervals and n; claims-log
rows labelled DEV/EXPLORATORY; run it twice to check determinism.

| Baseline | Meaning |
|---|---|
| B0 | no retrieval: the question only |
| B1 | the frozen retrieval configuration D with the current generator |
| B4 | oracle context: the gold evidence paragraphs |

**Decided by the owner**
- Judge: `gemini-3.8-flash` (same Gemini family as the generator, so the row's "different family" requirement is
  NOT met: label every result accordingly).
- Unanswerable questions: all 48 (19 dev + 29 from the old test split of `split.json`), **for this generation
  evaluation only**. This does not allow any other use of the test split. The official held-out set is not involved.
- Do not run it on free-tier quota.

**Still to build or settle**
- B0 does not exist. `eval/gen_arms.py` has R (= B1) and O (= B4). Without sources the structured long-answer
  schema cannot be used (`answer_schema` needs source labels), so B0 needs its own prompt.
- `eval/run_generation_eval.py` samples only answerable dev questions and refuses the test split: it needs the
  unanswerable set, abstention recall and false abstention, and a recorded exception for the 29.
- The original row described B1 with free-text answers. Long answers are structured claims since 2026-11-27;
  say which one is measured.
- Append a dated Deviation to `eval/PREREGISTRATION_gen.md` before the run.
- Determinism: the generation cache would make a repeat identical by construction. The second run needs its own
  empty cache, which doubles the cost.
- Verify the judge settings live (`scripts/check_model_ids.py`: failed with HTTP 503/504 on 2026-11-27) and replace
  the unverified pacing values in `run_generation_eval.py` with the paid quota's limits.
- Build the index on the machine that runs it (this Mac has the Docker volume only, no local `chroma_v2`).
- Do a pilot of about 10 questions first and show the owner the projected cost.
