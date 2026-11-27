"""RagService: the in-process entry point (no HTTP) for a UI or an evaluation.

    service = RagService(retriever, gateway, cache, RagConfig(), prompts=..., tracer=...)
    for event in service.answer_stream("What dataset do they use?", doc_ids=["1601.02403"], session_id="abc"):
        ...   # {"type": "sources"|"token"|"error"|"done", ...}
    result = service.answer_short("Do they use BERT?")   # non-streaming, short-answer prompt (evaluation)

Events (plain dicts):
  {"type": "sources", "sources": [Source, ...], "context": {...}}   the passages sent to the model (S1..Sn), before any token
  {"type": "token", "text": str}                                    a piece of the answer (never the abstention sentinel)
  {"type": "error", "error": ErrorInfo}                             a typed failure; ``done`` always follows
  {"type": "done", "result": AnswerResult}                          exactly one, always last

Failure behaviour: a bad question raises ``InvalidQuestion`` before anything runs; a retrieval failure, every provider failing,
or a stream that breaks halfway is reported as an error event followed by ``done`` (degraded or partial result), never as a crash.

Prompts: the prompt is resolved BEFORE the cache lookup (``prompts``: a ``PromptStore``; default the local templates). The cache key carries the
prompt name, version and the sha256 of its text. ``AnswerResult.prompt`` says which prompt was used and where it came from.

Tracing: optional and fail open. Span tree of one request (at most about 8 observations, one billable unit each on Langfuse):
  rag.request -> route -> [prompt.fallback event] -> cache.hit | cache.miss (event) -> retrieve -> context -> generation
A cache hit has no retrieve / context / generation. What may leave the process is decided by ``tracing.Capture`` (masked by default).
"""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Sequence

from rag.router import route_query
from rag_core.cache import CacheBackend, config_hash, is_cacheable, make_cache_key
from rag_core.citations import could_become_abstention, is_abstention
from rag_core.claims import SCHEMA_VERSION, ParsedAnswer, StreamRenderer, answer_schema, parse_answer, render_answer
from rag_core.config import RagConfig, TracingConfig
from rag_core.context import build_context
from rag_core.errors import AllProvidersFailed, AnswerTooLong, InvalidQuestion, MidStreamError
from rag_core.gateway import Delta, LLMGateway, StreamDone
from rag_core.prompt_store import LocalPromptStore, PromptStore, ResolvedPrompt, describe_prompts
from rag_core.prompts import PROMPT_NAMES, compile_messages
from rag_core.schemas import AnswerResult, ErrorInfo, Latency, Source, Usage
from rag_core.tracing import Capture, NoopTrace, SafeTrace, SafeTracer, cost_details

DIRECT_REPLY = ("Hello! I answer questions about the research papers in this collection. "
                "Ask me something about a paper: its method, its data, its results.")
ABSTAIN_MESSAGE = "The retrieved passages do not contain enough information to answer this question."
NO_RESULTS_MESSAGE = "No relevant passages were found in the indexed papers for this question."
DEGRADED_MESSAGE = ("The answer service is unavailable right now, so no answer could be generated. "
                    "These are the most relevant passages that were found; you can read them directly.")
INVALID_OUTPUT_MESSAGE = ("The model's reply did not follow the required answer format, so no answer is shown. "
                          "These are the most relevant passages that were found; you can read them directly.")
TOO_LONG_MESSAGE = ("The model used its whole output budget before it wrote an answer, so no answer was generated. "
                    "Try a more specific question. These are the most relevant passages that were found; you can read them directly.")
RETRIEVAL_FAILED_MESSAGE = "Search is unavailable right now, so no passages could be retrieved. Please try again later."
BUDGET_MESSAGE = ("The daily answer budget of this demo is used up, so no answer was generated. "
                  "These are the most relevant passages that were found; you can read them directly.")


def index_version_for(store, manifest_path: str | Path | None = None) -> str:
    """``<chunk count>:<manifest sha256[:16]>`` for the cache key. Without a manifest file the store's ingest signature is used."""
    if manifest_path is not None and Path(manifest_path).exists():
        part = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()[:16]
    else:
        part = (store.get_signature() or "no-signature")[:16]
    return f"{store.count()}:{part}"


class RefreshingIndexVersion:
    """A callable index version that is recomputed at most once per ``ttl_s``. A re-ingest while the app runs changes the chunk count and the
    manifest, so cached answers made against the old index stop matching within one TTL. A failing recompute keeps the last good value (and waits
    one TTL before trying again): a busy store must never break a request."""

    def __init__(self, compute: Callable[[], str], *, ttl_s: float = 30.0, clock: Callable[[], float] = time.monotonic):
        self._compute, self._ttl_s, self._clock = compute, ttl_s, clock
        self._lock = threading.Lock()
        self._value = compute()  # a failure here is a startup failure, like before
        self._at = clock()

    def __call__(self) -> str:
        with self._lock:
            if self._clock() - self._at >= self._ttl_s:
                self._at = self._clock()
                try:
                    self._value = self._compute()
                except Exception:  # noqa: BLE001: keep serving with the last known version
                    pass
            return self._value


def clean_short_answer(text: str) -> str:
    """First line, trimmed, without wrapping quotes or backticks."""
    line = text.strip().splitlines()[0].strip() if text.strip() else ""
    return line.strip("`\"' ").strip()


def clean_question(question, max_chars: int) -> str:
    """The cleaned question, or ``InvalidQuestion`` (typed). A module function so that a remote client applies exactly the same rule as the service."""
    if not isinstance(question, str):
        raise InvalidQuestion("invalid_type", "The question must be text.")
    cleaned = "".join(ch for ch in question if ch in "\n\t" or ch.isprintable()).strip()
    if not cleaned:
        raise InvalidQuestion("empty", "Please type a question.")
    if len(cleaned) > max_chars:
        raise InvalidQuestion("too_long", f"The question is too long ({len(cleaned)} characters; the limit is {max_chars}).")
    return cleaned


def to_sources(hits) -> list[Source]:
    return [Source(chunk_id=h.chunk_id, doc_id=h.doc_id, section=h.section or "", text=h.text, rank=h.rank, score=float(h.score), label=f"S{i}")
            for i, h in enumerate(hits, 1)]


class RagService:
    def __init__(self, retriever, gateway: LLMGateway, cache: CacheBackend | None, cfg: RagConfig, *, index_version: str | Callable[[], str] = "unversioned",
                 embedder_revision: str | None = None, router: Callable[[str], str] = route_query,
                 perf: Callable[[], float] = time.perf_counter, prompts: PromptStore | None = None, tracer=None,
                 tracing: TracingConfig | None = None):
        self.retriever, self.gateway, self.cache, self.cfg = retriever, gateway, cache, cfg
        self._index_version = index_version  # a string, or a callable (``RefreshingIndexVersion``) that follows re-ingests
        self.embedder_revision = embedder_revision or str(getattr(getattr(retriever, "embedder", None), "revision", "unknown"))
        self._router, self._perf = router, perf
        self.prompts: PromptStore = prompts or LocalPromptStore()
        self.tracing_cfg = tracing or TracingConfig()
        self.tracer = SafeTracer(tracer)  # always wrapped: an observability bug never changes or breaks an answer
        self.capture = Capture(self.tracing_cfg.capture)

    # ---- input guard --------------------------------------------------------------------------------------------
    @property
    def index_version(self) -> str:
        return self._index_version() if callable(self._index_version) else self._index_version
    def validate_question(self, question) -> str:
        """The cleaned question, or ``InvalidQuestion`` (typed, raised before any retrieval or LLM call)."""
        return clean_question(question, self.cfg.max_question_chars)

    # ---- public API ---------------------------------------------------------------------------------------------
    def answer_stream(self, question: str, doc_ids: Sequence[str] | None = None, rerank: bool = False, session_id: str | None = None) -> Iterator[dict]:
        try:
            cleaned = self.validate_question(question)  # eager: raises here, not on the first next()
        except InvalidQuestion as exc:
            self._trace_rejected(question, exc, session_id, "long")
            raise
        return self._run(cleaned, doc_ids, rerank, "long", session_id, traced=True)

    def answer_short(self, question: str, doc_ids: Sequence[str] | None = None, rerank: bool = False) -> AnswerResult:
        """Non-streaming, short-answer prompt (evaluation). Never raises for provider or retrieval failures: the result says so.
        Not traced unless ``tracing.trace_evaluation`` is true."""
        return self._answer_blocking("short", question, doc_ids, rerank)

    def answer_long(self, question: str, doc_ids: Sequence[str] | None = None, rerank: bool = False) -> AnswerResult:
        """Non-streaming long-answer prompt with citations (evaluation): the same pipeline as ``answer_stream`` consumed to its final result.
        Not traced unless ``tracing.trace_evaluation`` is true."""
        return self._answer_blocking("long", question, doc_ids, rerank)

    def _answer_blocking(self, mode: str, question: str, doc_ids, rerank: bool) -> AnswerResult:
        traced = self.tracing_cfg.trace_evaluation
        try:
            cleaned = self.validate_question(question)
        except InvalidQuestion as exc:
            if traced:
                self._trace_rejected(question, exc, None, mode)
            raise
        result = None
        for event in self._run(cleaned, doc_ids, rerank, mode, None, traced=traced):
            if event["type"] == "done":
                result = event["result"]
        return result

    def _retrieval_kwargs(self, doc_ids, rerank: bool) -> dict:
        """The retrieval arguments of every request (the frozen configuration): ONE definition for the pipeline and ``passages_only``."""
        cfg = self.cfg
        return dict(mode=cfg.retrieval_mode, fusion=cfg.fusion, top_k=cfg.top_k, candidate_k=cfg.candidate_k, rerank_k=cfg.rerank_k,
                    doc_ids=None if doc_ids is None else list(doc_ids), rerank=rerank, rrf_k=cfg.rrf_k, vector_weight=cfg.vector_weight, lexical_weight=cfg.lexical_weight)

    def passages_only(self, question: str, doc_ids: Sequence[str] | None = None, rerank: bool = False, *, code: str = "daily_budget_reached",
                      message: str = BUDGET_MESSAGE) -> AnswerResult:
        """Retrieval WITHOUT any LLM call (and without cache or tracing): a degraded result carrying the retrieved passages and a typed notice. Used when the demo's
        daily answer budget is used up. Raises ``InvalidQuestion`` for a bad question, like ``answer_stream``; a retrieval failure is a typed error in the result."""
        cleaned = self.validate_question(question)
        t0 = self._perf()
        result = AnswerResult(mode="long", degraded=True, answer=message)
        try:
            retrieved = self.retriever.retrieve(cleaned, **self._retrieval_kwargs(doc_ids, rerank))
        except Exception as exc:  # noqa: BLE001: reported as a typed error, never a crash
            result.answer = RETRIEVAL_FAILED_MESSAGE
            result.errors.append(ErrorInfo(code="retrieval_failed", message=f"{type(exc).__name__} while retrieving passages"))
            result.latency_ms = Latency(total=(self._perf() - t0) * 1000.0)
            return result
        result.retrieved_sources = to_sources(retrieved.hits)
        result.errors.append(ErrorInfo(code=code, message=message, recoverable=False))
        elapsed = (self._perf() - t0) * 1000.0
        result.latency_ms = Latency(retrieve=elapsed, total=elapsed)
        return result

    def retrieve_sources(self, question: str, doc_ids: Sequence[str] | None = None, rerank: bool = False) -> tuple[list[Source], float]:
        """Retrieval only (no LLM, no cache, no tracing): ``(passages labelled S1.., milliseconds)``. Raises ``InvalidQuestion`` for a bad question and lets a retrieval
        failure propagate (the caller decides how to report it)."""
        cleaned = self.validate_question(question)
        t0 = self._perf()
        retrieved = self.retriever.retrieve(cleaned, **self._retrieval_kwargs(doc_ids, rerank))
        return to_sources(retrieved.hits), (self._perf() - t0) * 1000.0

    def flush_traces(self, timeout_s: float | None = None) -> None:
        """Flush buffered traces (bounded; never raises). The UI calls it after a request; scripts before they exit."""
        self.tracer.flush(timeout_s)

    def describe_prompts(self) -> dict:
        """name/version/label/source/sha256 of both prompts as resolved right now: what an evaluation run records in its config.json."""
        return describe_prompts(self.prompts)

    # ---- helpers --------------------------------------------------------------------------------------------------------
    def _resolve_prompt(self, mode: str) -> ResolvedPrompt:
        name = PROMPT_NAMES[mode]
        try:
            return self.prompts.resolve(name)
        except Exception as exc:  # noqa: BLE001: never block an answer on the prompt store
            fallback = LocalPromptStore().resolve(name)
            fallback.source, fallback.fallback_kind, fallback.fallback_reason = "local_fallback", "fetch_error", f"resolve_failed:{type(exc).__name__}"
            return fallback

    def _cache_key(self, question: str, doc_ids, rerank: bool, mode: str, resolved: ResolvedPrompt) -> str:
        prompt_id = f"{resolved.cache_id}|{SCHEMA_VERSION}" if mode == "long" else resolved.cache_id  # a schema change must not serve old answers
        return make_cache_key(question=question, doc_ids=doc_ids, retrieval_config_hash=config_hash(self.cfg.retrieval_settings(rerank)),
                              index_version=self.index_version, embedder_revision=self.embedder_revision, prompt=prompt_id,
                              model_id=self.gateway.primary_model, top_k=self.cfg.top_k, mode=mode,
                              reasoning_effort=self.gateway.primary_reasoning_effort)

    def _trace_rejected(self, question, exc: InvalidQuestion, session_id, mode: str) -> None:
        trace = self.tracer.start_request("rag.request", session_id=session_id,
                                          input=self.capture.question(question) if isinstance(question, str) else None,
                                          metadata=self.capture.meta({"mode": mode, "rejected": True}))
        trace.set_error(exc.code, exc.message)
        trace.end(metadata={"error_code": exc.code})

    def _open_trace(self, question: str, doc_ids, rerank: bool, mode: str, session_id) -> SafeTrace:
        return self.tracer.start_request(
            "rag.request", session_id=session_id, input=self.capture.question(question),
            metadata=self.capture.meta({"mode": mode, "doc_filter": None if doc_ids is None else list(doc_ids), "rerank": rerank, "top_k": self.cfg.top_k,
                                        "index_version": self.index_version, "embedder_revision": self.embedder_revision, "capture": self.capture.mode,
                                        "reasoning_effort": self.gateway.primary_reasoning_effort}))

    def _close_trace(self, trace: SafeTrace, result: AnswerResult | None) -> None:
        if result is None:  # the consumer stopped iterating, or the generator was closed
            trace.end(metadata={"aborted": True})
            return
        if result.errors:
            # WARNING = degraded (retrieved passages served instead of an answer); ERROR = a failed request. The dashboard's error and degraded
            # rates are the counts of root observations at each level (the Metrics API cannot filter on the status message).
            trace.set_error(result.errors[0].code, result.errors[0].message, "WARNING" if result.degraded else "ERROR")
        trace.end(output=self.capture.answer(result.answer), metadata=self.capture.meta({
            "cache_hit": result.cached, "mode": result.mode, "provider": result.provider, "model": result.model, "finish_reason": result.finish_reason,
            "reasoning_effort": result.reasoning_effort,
            "degraded": result.degraded, "abstained": result.abstained, "partial": result.partial, "citations": len(result.citations), "claims": len(result.claims),
            "invalid_citations": len(result.invalid_citations), "error_codes": [e.code for e in result.errors],
            "prompt": None if result.prompt is None else result.prompt.model_dump()}))

    @staticmethod
    def _apply_claims(result: AnswerResult, parsed: ParsedAnswer) -> bool:
        """Fill the long-answer fields of ``result`` from a parsed structured answer; True if the model abstained."""
        abstained = parsed.abstained
        result.abstained = abstained
        result.claims = [] if abstained else parsed.claims
        result.answer = ABSTAIN_MESSAGE if abstained else render_answer(parsed.claims)
        result.citations, result.invalid_citations = ([], []) if abstained else (parsed.cited, parsed.invalid)
        return abstained

    # ---- the pipeline -------------------------------------------------------------------------------------------
    def _run(self, question: str, doc_ids, rerank: bool, mode: str, session_id, traced: bool) -> Iterator[dict]:
        trace = self._open_trace(question, doc_ids, rerank, mode, session_id) if traced else SafeTrace(NoopTrace(), self.tracer.stats)
        final = None
        try:
            for event in self._pipeline(question, doc_ids, rerank, mode, trace):
                if event["type"] == "done":
                    final = event["result"]
                yield event
        finally:
            self._close_trace(trace, final)

    def _pipeline(self, question: str, doc_ids, rerank: bool, mode: str, trace: SafeTrace) -> Iterator[dict]:
        t0 = self._perf()

        def elapsed() -> float:
            return (self._perf() - t0) * 1000.0

        result = AnswerResult(mode=mode, trace_id=trace.trace_id, reasoning_effort=self.gateway.primary_reasoning_effort)

        # 1. route: small talk gets a fixed reply, no retrieval, no LLM, not cached
        with trace.span("route", input=self.capture.question(question)) as span:
            decision = self._router(question)
            span.update(output=decision)
        if decision == "direct":
            result.mode, result.answer, result.provider, result.model, result.finish_reason = "direct", DIRECT_REPLY, "rules", "canned", "stop"
            result.latency_ms = Latency(total=elapsed())
            yield {"type": "sources", "sources": [], "context": {}}
            yield {"type": "token", "text": DIRECT_REPLY}
            yield {"type": "done", "result": result}
            return

        # 2. resolve the prompt BEFORE the cache lookup (its name, version and text hash are part of the key)
        resolved = self._resolve_prompt(mode)
        result.prompt, result.prompt_version = resolved.info(), str(resolved.version)
        if resolved.source == "local_fallback":
            trace.event("prompt.fallback", level="ERROR" if resolved.fallback_kind == "invalid" else "WARNING", status_message=resolved.fallback_reason,
                        metadata=self.capture.meta({"prompt": result.prompt.model_dump(), "kind": resolved.fallback_kind}))

        # 3. cache
        key = self._cache_key(question, doc_ids, rerank, mode, resolved) if self.cache is not None else None
        if key is not None:
            hit = self.cache.get(key)
            trace.event("cache.hit" if hit is not None else "cache.miss")
            if hit is not None:
                hit.cached, hit.trace_id, hit.request_id = True, trace.trace_id, uuid.uuid4().hex  # feedback belongs to THIS request, not to the cached one
                hit.latency_ms = Latency(retrieve=0.0, first_token=elapsed(), total=elapsed())
                yield {"type": "sources", "sources": hit.used_sources, "context": {"from_cache": True}}
                yield {"type": "token", "text": hit.answer}
                hit.latency_ms.total = elapsed()
                yield {"type": "done", "result": hit}
                return

        # 4. retrieve
        t = self._perf()
        failure: Exception | None = None
        retrieved = None
        with trace.span("retrieve", as_type="retriever", input=self.capture.question(question)) as span:
            try:
                retrieved = self.retriever.retrieve(question, **self._retrieval_kwargs(doc_ids, rerank))
            except Exception as exc:  # noqa: BLE001: reported as a typed error, never a crash
                failure = exc
                span.update(level="ERROR", status_message=f"{type(exc).__name__} while retrieving passages")
            else:
                span.update(output={"chunk_ids": [h.chunk_id for h in retrieved.hits]},
                            stage_latency_ms={k: round(float(v), 3) for k, v in (getattr(retrieved, "latency_ms", None) or {}).items()})
        if failure is not None:
            result.degraded, result.answer = True, RETRIEVAL_FAILED_MESSAGE
            result.errors.append(ErrorInfo(code="retrieval_failed", message=f"{type(failure).__name__} while retrieving passages"))
            result.latency_ms = Latency(retrieve=(self._perf() - t) * 1000.0, total=elapsed())
            yield {"type": "error", "error": result.errors[-1]}
            yield {"type": "done", "result": result}
            return
        retrieve_ms = (self._perf() - t) * 1000.0
        hits = list(retrieved.hits)
        result.retrieved_sources = [Source(chunk_id=h.chunk_id, doc_id=h.doc_id, section=h.section or "", text=h.text, rank=h.rank,
                                           score=float(h.score)) for h in hits]

        # 5. context
        with trace.span("context") as span:
            ctx = build_context(hits, self.cfg.max_context_tokens, max_chunks=self.cfg.top_k)
            span.update(output={"used_chunk_ids": ctx.used_chunk_ids, "truncated_first": ctx.truncated_first, "dropped": ctx.dropped,
                                "estimated_tokens": ctx.tokens, "sources": self.capture.sources(ctx.sources)})
        if not ctx.sources:  # nothing to ground an answer on: do not ask the model
            result.abstained, result.answer = True, NO_RESULTS_MESSAGE
            result.latency_ms = Latency(retrieve=retrieve_ms, total=elapsed())
            yield {"type": "sources", "sources": [], "context": {}}
            yield {"type": "done", "result": result}
            return
        yield {"type": "sources", "sources": ctx.sources,
               "context": {"truncated_first": ctx.truncated_first, "dropped": ctx.dropped, "estimated_tokens": ctx.tokens}}

        # 6. generate (streaming), inside ONE generation observation
        messages = compile_messages(resolved.messages, ctx.sources, question)
        max_out = self.cfg.short_max_output_tokens if mode == "short" else self.cfg.max_output_tokens
        structured = mode == "long"  # the long answer is structured output (claims with quotes); the short answer is plain text
        schema = answer_schema([s.label for s in ctx.sources]) if structured else None
        renderer = StreamRenderer() if structured else None
        parts: list[str] = []
        released, suppressed, first_token_ms, first_token_wall, done = False, False, None, None, None
        generation = trace.generation(
            "generation", model=self.gateway.primary_model, provider=self.gateway.primary_name,
            input=self.capture.generation_input(messages, question, ctx.sources), prompt=resolved.client if resolved.source == "langfuse" else None,
            model_parameters={"temperature": self.cfg.temperature, "max_tokens": max_out, "reasoning_effort": self.gateway.primary_reasoning_effort,
                              "response_format": SCHEMA_VERSION if structured else "text"},
            prompt_name=resolved.name, prompt_version=str(resolved.version), prompt_source=resolved.source, prompt_sha256=resolved.sha256)
        gen = generation.__enter__()

        def attempts_meta(attempts) -> list[dict]:
            return [{"provider": a.provider, "attempt": a.attempt, "outcome": a.outcome, "error_kind": a.error_kind, "latency_ms": round(a.latency_ms, 1)} for a in attempts]

        try:
            try:
                for event in self.gateway.stream(messages, max_output_tokens=max_out, temperature=self.cfg.temperature,
                                                 **({"response_schema": schema} if structured else {})):
                    if isinstance(event, Delta):
                        if first_token_ms is None:
                            first_token_ms, first_token_wall = elapsed(), datetime.now(timezone.utc)
                        parts.append(event.text)
                        if renderer is not None:  # structured: show the claims' text as it arrives (never the JSON, never an abstention)
                            shown = renderer.feed(event.text)
                            if shown:
                                released = True
                                yield {"type": "token", "text": shown}
                            continue
                        if suppressed:
                            continue
                        if not released:
                            buffered = "".join(parts)
                            if is_abstention(buffered):  # the UI shows its own message; the sentinel is never streamed
                                suppressed = True
                                continue
                            if could_become_abstention(buffered):
                                continue
                            released = True
                            yield {"type": "token", "text": buffered}
                        else:
                            yield {"type": "token", "text": event.text}
                    elif isinstance(event, StreamDone):
                        done = event
            except MidStreamError as exc:
                text = "".join(parts)
                result.partial, result.finish_reason = True, "error"
                result.provider, result.model, result.attempts = exc.provider, exc.model, exc.attempts
                result.used_sources = list(ctx.sources)
                if structured:  # keep the claims that arrived (the JSON is parsed as far as it goes)
                    self._apply_claims(result, parse_answer(text, ctx.sources))
                    tail = renderer.finish(result.answer)
                    if tail:
                        yield {"type": "token", "text": tail}
                    released = True
                else:
                    result.answer = text
                result.errors.append(ErrorInfo(code="mid_stream", provider=exc.provider, recoverable=False,
                                               message=f"{exc.provider} stopped after {len(text)} characters ({exc.cause.kind}); the answer is incomplete."))
                result.latency_ms = Latency(retrieve=retrieve_ms, first_token=first_token_ms, total=elapsed())
                gen.update(level="ERROR", status_message="mid_stream", output=self.capture.answer(result.answer), completion_start_time=first_token_wall,
                           provider=exc.provider, model=exc.model, finish_reason="error", attempts=attempts_meta(exc.attempts))
                if not released and text and not suppressed:
                    yield {"type": "token", "text": text}
                yield {"type": "error", "error": result.errors[-1]}
                yield {"type": "done", "result": result}
                return
            except AnswerTooLong as exc:  # a question-level outcome: no breaker was touched, the passages are still shown
                result.degraded, result.answer, result.attempts = True, TOO_LONG_MESSAGE, exc.attempts
                result.errors.append(ErrorInfo(code="answer_too_long", message="The model ran out of output budget before it wrote any text."))
                result.latency_ms = Latency(retrieve=retrieve_ms, total=elapsed())
                gen.update(level="WARNING", status_message="answer_too_long", attempts=attempts_meta(exc.attempts))
                yield {"type": "error", "error": result.errors[-1]}
                yield {"type": "done", "result": result}
                return
            except AllProvidersFailed as exc:
                result.degraded, result.answer, result.attempts = True, DEGRADED_MESSAGE, exc.attempts
                result.errors.append(ErrorInfo(code="all_providers_failed", message="Every language-model provider failed or was unavailable."))
                result.latency_ms = Latency(retrieve=retrieve_ms, total=elapsed())
                gen.update(level="ERROR", status_message="all_providers_failed", attempts=attempts_meta(exc.attempts))
                yield {"type": "error", "error": result.errors[-1]}
                yield {"type": "done", "result": result}
                return
            except Exception as exc:  # noqa: BLE001: a bug must not crash the UI; it is reported with its class name only
                result.degraded, result.answer = True, DEGRADED_MESSAGE
                result.errors.append(ErrorInfo(code="internal", message=f"{type(exc).__name__} in the generation path"))
                result.latency_ms = Latency(retrieve=retrieve_ms, first_token=first_token_ms, total=elapsed())
                gen.update(level="ERROR", status_message=f"internal:{type(exc).__name__}")
                yield {"type": "error", "error": result.errors[-1]}
                yield {"type": "done", "result": result}
                return

            # 7. validate and assemble
            text = "".join(parts)
            result.used_sources = list(ctx.sources)  # exactly what the model saw
            result.provider, result.model, result.finish_reason = done.provider, done.model, done.finish_reason
            parsed: ParsedAnswer | None = None
            if structured:
                parsed = parse_answer(text, ctx.sources)
                abstained = self._apply_claims(result, parsed)
                if not abstained:
                    tail = renderer.finish(result.answer)
                    if tail:
                        yield {"type": "token", "text": tail}
                if parsed.status is None and not parsed.claims:  # not the required format at all: no answer is shown, the passages are
                    result.degraded, result.answer, result.claims = True, INVALID_OUTPUT_MESSAGE, []
                    result.errors.append(ErrorInfo(code="invalid_output", message="The model's reply was not a structured answer."))
                    yield {"type": "error", "error": result.errors[-1]}
                elif not parsed.complete and not abstained:  # the JSON was cut off (output limit): the claims that arrived are kept and the answer is marked incomplete
                    result.partial = True
                    result.errors.append(ErrorInfo(code="truncated", recoverable=False,
                                                   message="The model stopped before it finished the answer (output limit); the answer is incomplete."))
                    yield {"type": "error", "error": result.errors[-1]}
            else:
                if not released and not suppressed and text:
                    yield {"type": "token", "text": text}  # a short answer that stayed held back until the stream ended
                abstained = is_abstention(text)
                result.abstained = abstained
                result.answer = ABSTAIN_MESSAGE if abstained else text.strip()
                result.short_answer = None if abstained else clean_short_answer(text)
            result.usage = Usage(prompt_tokens=done.prompt_tokens, completion_tokens=done.completion_tokens, reasoning_tokens=done.reasoning_tokens)
            result.attempts = done.attempts
            result.latency_ms = Latency(retrieve=retrieve_ms, first_token=first_token_ms, total=elapsed())
            update = {"output": self.capture.answer(result.answer), "provider": done.provider, "model": done.model, "finish_reason": done.finish_reason,
                      "reasoning_tokens": done.reasoning_tokens,
                      "completion_start_time": first_token_wall, "attempts": attempts_meta(done.attempts), "abstained": result.abstained,
                      "citations": len(result.citations), "invalid_citations": len(result.invalid_citations)}
            if parsed is not None:
                verified, total = parsed.quotes
                update.update(claims=len(parsed.claims), quotes_verified=verified, quotes=total, json_complete=parsed.complete,
                              malformed_items=parsed.malformed_items)
            if result.errors:
                update.update(level="WARNING" if result.partial else "ERROR", status_message=result.errors[-1].code)
            if done.prompt_tokens is not None or done.completion_tokens is not None:
                update["usage"] = {"prompt_tokens": done.prompt_tokens, "completion_tokens": done.completion_tokens}
            cost = cost_details(self.cfg.prices, done.model, done.prompt_tokens, done.completion_tokens)
            if cost is not None:
                update["cost_details"] = cost
            gen.update(**update)
            if key is not None and is_cacheable(result, expected_model=self.gateway.primary_model):
                self.cache.set(key, result)
            yield {"type": "done", "result": result}
        finally:
            generation.__exit__(None, None, None)
