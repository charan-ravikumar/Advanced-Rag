# RAGAS notes (read before writing or changing `eval/run_ragas.py`)

Facts below come from the RAGAS 0.4.4 documentation (https://docs.ragas.io/en/stable/) and from reading the installed package, and were checked
by small live probes with fixed public text. No performance number appears here: token costs are measured by the scoring runs and recorded in
their `config.json` / `summary.json` (see "Token cost per metric").

## Version and environment

- **ragas 0.4.4** (latest stable on the configured index; 0.4.0 to 0.4.4 all exist). The legacy metric classes (`ragas.metrics.Faithfulness`,
  `SingleTurnSample`, `LangchainLLMWrapper`) are deprecated; use the **collections API**: `from ragas.metrics.collections import ...`.
- **Second venv.** `pip install --dry-run ragas==0.4.4` into the main venv would downgrade pinned packages (openai 3.57.0 to 1.109.1, rich 15.0.0 to
  13.9.4, tenacity 9.1.4 to 8.5.0), and the Groq provider needs the pinned openai. So RAGAS lives in `%LOCALAPPDATA%\advrag\venv_eval`
  (the exact main freeze, plus ragas). The full pinned list is `requirements-eval.txt` (never merged into the deploy requirements). Versions that differ from
  the main venv: jiter, openai (2.54.0), rich, tenacity. Extra installs beyond `ragas`: `instructor==1.17.0` (ragas leaves it unpinned and the resolver
  picked 1.3.2, which has no `from_genai`), `openai>=1.109.1,<3` (needed by `langchain-openai`), `jsonref` (needed by instructor for Gemini). `pip check` is clean.
- **Generation runs in the main venv; RAGAS scoring runs in `venv_eval`.** They share only files: the generation run folder is the input of the scoring run.
- **Analytics opt-out: `RAGAS_DO_NOT_TRACK=true`** (read in `ragas/_analytics.py`; without it RAGAS posts usage events to `t.explodinggradients.com`).
  It must be set BEFORE `import ragas`; every eval entry point sets it first (`eval/run_ragas.py`, `eval/run_generation_eval.py`, the scripts) and
  `CLAUDE.md` says so.

## Metrics (collections API): inputs and how they are computed

| Metric | Class | `ascore(...)` inputs | LLM calls per scored answer | Embeddings |
|---|---|---|---|---|
| Faithfulness | `Faithfulness(llm)` | `user_input`, `response`, `retrieved_contexts: list[str]` | 2 (split the answer into statements; verdict per statement against the contexts) | no |
| Answer relevancy | `AnswerRelevancy(llm, embeddings, strictness=3)` | `user_input`, `response` | `strictness` (one generated question each) | yes: cosine of the question and the generated questions |
| Context precision | `ContextPrecision(llm)` | `user_input`, `reference`, `retrieved_contexts` | one per retrieved context (is it useful for arriving at the reference) | no |
| Context recall | `ContextRecall(llm)` | `user_input`, `retrieved_contexts`, `reference` | 1 (is each statement of the reference attributable to the contexts) | no |

- Every `ascore` returns a `MetricResult` (`.value`, `.reason`). `.score()` is the synchronous form.
- **Context precision** is the mean of precision@k over the ranks that are relevant (order-sensitive); **context recall** is the share of the REFERENCE's
  claims that the contexts support. Both need a `reference`; this project uses the first usable annotator reference (deterministic, recorded), not a max over annotators.
- **Answer relevancy at temperature 0**: the `strictness` generated questions come from the same prompt, so they are near-identical and the metric
  collapses to one question. This project sets `strictness=1` (one call instead of three) and says so. An abstention ("INSUFFICIENT_CONTEXT") is
  noncommittal and scores 0 by construction, so answer relevancy is reported only over non-abstained answers.
- The collections metrics REQUIRE an `InstructorBaseRagasLLM` (and, for answer relevancy, a `BaseRagasEmbedding`); the legacy `LangchainLLMWrapper` is rejected.

## How the judge LLM and the embeddings are supplied

- **Documented path for Gemini.** `llm_factory("gemini-2.5-flash", provider="google", client=genai.Client(...))` selects the instructor adapter
  (`instructor.from_genai`) and works (checked live with a trivial structured call). It was NOT used for scoring because it cannot do what the judge
  needs: it forwards only `temperature`, `max_tokens`, `top_p`, `top_k` to Gemini, so a **thinking budget cannot be set** (probe: an invalid
  `thinking_budget=-5` placed in `generation_config` was accepted silently, i.e. it never reached the API), it returns only the parsed object (no token usage,
  no finish reason), and its optional cache is keyed on the call arguments alone (no metric name, no judge settings, no ragas version).
- **What is used instead.** `InstructorBaseRagasLLM` is the documented extension point for custom LLMs (two methods: `generate` and `agenerate` of
  `(prompt, response_model)`). The judge implements it on top of `google-genai` with `response_schema=<the pydantic model RAGAS passes>`,
  `temperature=0`, `thinking_config`, `max_output_tokens`, the `v1beta` API version (the SDK's default version rejects `thinking_config` with HTTP 400),
  records `usage_metadata` (prompt, output, thinking tokens) and the finish reason, retries 429/5xx with backoff, and caches every call on disk.
- **Embeddings.** `BaseRagasEmbedding` (abstract `embed_text` / `aembed_text`; batch methods are derived) wrapped around the local MiniLM embedder
  (`all-MiniLM-L6-v2` at its pinned revision, local cache only). No Gemini embedding quota is used.

## Default prompts (RAGAS's own, not modified)

Faithfulness: `StatementGeneratorPrompt` ("break down each sentence into one or more fully understandable statements ... no pronouns") and
`NLIStatementPrompt` ("judge the faithfulness of a series of statements based on a given context ... verdict 1 if the statement can be directly inferred").
Answer relevancy: `AnswerRelevancePrompt` ("generate a question for the given answer and identify if the answer is noncommittal"). Context precision:
`ContextPrecisionPrompt` ("given question, answer and context verify if the context was useful in arriving at the given answer"). Context recall:
`ContextRecallPrompt` ("analyze each statement in the answer and classify if the statement can be attributed to the given context"). The prompts are
English, written for general text; they have not been validated on scientific papers, which is what the calibration sheet is for.

## Token cost per metric

Measured, not estimated: each scoring run writes the judge's prompt, output and thinking tokens per metric into its `summary.json`. The calls per
answer above are structural (from the code). Until a scoring run exists no token number is stated here.

## Things that bit others (kept here)

- Forgetting `RAGAS_DO_NOT_TRACK` sends analytics. Set it first.
- ragas leaves `instructor` unpinned; a resolver backtrack gives an old release without `from_genai`.
- Gemini's `thinking_config` needs `api_version="v1beta"`; thinking tokens count toward `max_output_tokens`, so a small output limit can return EMPTY text.
- A judge that returns a score for an answer it was never shown (e.g. empty contexts) is not evidence: scoring skips rows with no contexts.
