"""The ONLY place that touches ragas and google-genai (all imports are inside functions; run it in ``venv_eval``). EXPLORATORY: Phase 4 minimal scope.

* ``build_gemini_raw``: the judge's network call, directly on ``google-genai`` with ``response_schema`` (the pydantic model RAGAS passes), temperature 0, a thinking budget,
  an output limit and API version ``v1beta`` (the SDK's default version rejects ``thinking_config``). RAGAS's own Google path cannot set a thinking budget or report usage
  (docs/ragas_notes.md), which is why this adapter exists. The raw call is wrapped by ``eval.judge_core.CachingJudge`` (disk cache, quota handling, usage).
* ``make_ragas_llm``: a two-method ``InstructorBaseRagasLLM`` (RAGAS's documented extension point) over a ``CachingJudge``; the collections metrics accept it.
* ``MiniLMEmbedding``: the local pinned MiniLM as a ``BaseRagasEmbedding`` (no Gemini embedding quota).
* ``RagasScorer``: Faithfulness and AnswerRelevancy (strictness 1) from ``ragas.metrics.collections``, one CachingJudge PER METRIC.
"""

from __future__ import annotations

import asyncio
import os

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")  # before any ragas import (ragas posts usage analytics otherwise)

from eval.judge_core import CachingJudge, JudgeError, JudgeSettings, JudgeStats, RawJudgeResult, classify_http_error  # noqa: E402


def package_version(name: str) -> str | None:
    from importlib import metadata

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def build_gemini_raw(settings: JudgeSettings, api_key_env: str = "GEMINI_API_KEY"):
    """``async raw(prompt, response_model) -> RawJudgeResult``. The key is read from the environment here and kept nowhere else."""
    from google import genai
    from google.genai import types

    key = os.environ.get(api_key_env)
    if not key:
        raise JudgeError(f"{api_key_env} is not set")
    client = genai.Client(api_key=key, http_options=types.HttpOptions(api_version=settings.api_version, timeout=120000,
                                                                     retry_options=types.HttpRetryOptions(attempts=1)))

    async def raw(prompt: str, response_model) -> RawJudgeResult:
        config = types.GenerateContentConfig(temperature=settings.temperature, max_output_tokens=settings.max_output_tokens,
                                             thinking_config=types.ThinkingConfig(thinking_budget=settings.thinking_budget),
                                             response_mime_type="application/json", response_schema=response_model)
        try:
            reply = await client.aio.models.generate_content(model=settings.model, contents=prompt, config=config)
        except Exception as exc:  # noqa: BLE001: translated; the text is the API's own public message (no key is ever in it)
            raise classify_http_error(getattr(exc, "code", None), str(exc)) from None
        meta = getattr(reply, "usage_metadata", None)
        finish = reply.candidates[0].finish_reason if getattr(reply, "candidates", None) else None
        return RawJudgeResult(text=reply.text or "", prompt_tokens=getattr(meta, "prompt_token_count", None), output_tokens=getattr(meta, "candidates_token_count", None),
                              thinking_tokens=getattr(meta, "thoughts_token_count", None), finish_reason=None if finish is None else str(finish))

    return raw


def make_ragas_llm(judge: CachingJudge):
    """A RAGAS-compatible LLM (the collections metrics require an ``InstructorBaseRagasLLM``) that delegates to the cached judge."""
    from ragas.llms.base import InstructorBaseRagasLLM

    class CachedJudgeLLM(InstructorBaseRagasLLM):
        def __init__(self, inner: CachingJudge):
            self.judge = inner

        def generate(self, prompt, response_model):
            return self.judge.generate(prompt, response_model)

        async def agenerate(self, prompt, response_model):
            return await self.judge.agenerate(prompt, response_model)

    return CachedJudgeLLM(judge)


def make_minilm_embedding(embedder):
    """The local sentence embedder as a ``BaseRagasEmbedding`` (``embed_text`` / ``aembed_text``; the batch methods are derived by RAGAS)."""
    from ragas.embeddings.base import BaseRagasEmbedding

    class MiniLMEmbedding(BaseRagasEmbedding):
        def __init__(self, inner):
            super().__init__(cache=None)
            self.inner = inner

        def embed_text(self, text: str, **kwargs):
            return [float(x) for x in self.inner.encode_query(text)]

        async def aembed_text(self, text: str, **kwargs):
            return await asyncio.to_thread(self.embed_text, text)

    return MiniLMEmbedding(embedder)


class RagasScorer:
    """``score(question, answer, contexts) -> {"Faithfulness": x, "AnswerRelevancy": y}``. One persistent event loop (the Google async client binds to a loop)."""

    def __init__(self, raw, cache, settings: JudgeSettings, stats: JudgeStats, embeddings, *, version: str, pacer=None, sleep=None, strictness: int = 1):
        from ragas.metrics.collections import AnswerRelevancy, Faithfulness

        kw = {"pacer": pacer, **({"sleep": sleep} if sleep else {})}
        self.faithfulness = Faithfulness(llm=make_ragas_llm(CachingJudge(raw, cache, settings, "faithfulness", version, stats, **kw)))
        self.relevancy = AnswerRelevancy(llm=make_ragas_llm(CachingJudge(raw, cache, settings, "answer_relevancy", version, stats, **kw)), embeddings=embeddings,
                                         strictness=strictness)
        self._loop = asyncio.new_event_loop()

    def score(self, question: str, answer: str, contexts: list[str]) -> dict:
        async def both():
            f = await self.faithfulness.ascore(user_input=question, response=answer, retrieved_contexts=contexts)
            r = await self.relevancy.ascore(user_input=question, response=answer)
            return f, r

        f, r = self._loop.run_until_complete(both())
        return {"Faithfulness": float(f.value), "AnswerRelevancy": float(r.value)}

    def close(self) -> None:
        self._loop.close()


def build_real_scorer(settings: JudgeSettings, cache, stats: JudgeStats, pacer, sleep) -> RagasScorer:
    """Production wiring: Gemini judge, local MiniLM embeddings from the pinned revision (local HuggingFace cache only)."""
    from config import cfg
    from rag.embedders import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder.from_config(cfg)
    return RagasScorer(build_gemini_raw(settings), cache, settings, stats, make_minilm_embedding(embedder), version=package_version("ragas") or "unknown",
                       pacer=pacer, sleep=sleep, strictness=1)
