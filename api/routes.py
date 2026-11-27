"""The endpoints. Every handler is ``async``; the blocking work (retrieval, the model call, SQLite, Langfuse) runs in the thread pool, so the event loop is never blocked and
many streams can be open at once.

    GET  /v1/health         liveness (no auth)
    GET  /v1/ready          readiness + the self-check
    GET  /v1/meta           what is running, limits, daily budget
    POST /v1/ask            the answer as server-sent events (sources, token..., done)
    POST /v1/answer         the same pipeline, one JSON body (``AnswerResult``)
    POST /v1/retrieve       retrieval only (no model call)
    POST /v1/feedback       a thumb for an answer
    GET  /v1/observability  the shared Langfuse metrics snapshot

Order of checks for a query: authentication -> readiness -> per-key rate limit (429; ``slowapi``, see ``api.limits``) -> question validation (422) -> daily budget (when it is used up the
response is still 200 and carries the retrieved passages and a ``daily_budget_reached`` error, never an LLM call).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import anyio
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from api.deps import ApiException, ApiState, authenticate, ready_state
from api.limits import FEEDBACK_LIMIT, limiter, query_limits
from api.schemas import (
    AskRequest,
    Budget,
    CheckModel,
    ErrorEvent,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    Flags,
    HealthResponse,
    Limits,
    MetaResponse,
    ObservabilityResponse,
    ReadyResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from api.sse import OPEN_COMMENT, encode_event, encode_service_event
from rag_core.errors import InvalidQuestion
from rag_core.schemas import AnswerResult, ErrorInfo

router = APIRouter(prefix="/v1")

STREAM_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}  # no proxy buffering: tokens must reach the client as they are written
COMMON_ERRORS: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorResponse, "description": "Missing or wrong bearer token (only when the server has ADVRAG_API_TOKEN set)."},
    422: {"model": ErrorResponse, "description": "The request body is invalid, or the question was rejected (`code`: invalid_type, empty, too_long, invalid_request)."},
    429: {"model": ErrorResponse, "description": "Per-client rate limit or session cap (`code`: rate_limited, session_cap). `Retry-After` says when to retry."},
    503: {"model": ErrorResponse, "description": "The service is not ready (failed self-check): see `GET /v1/ready`."},
}

SSE_DESCRIPTION = """Streams the answer as [server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html) (`text/event-stream`).

| event | data (JSON) | when |
|---|---|---|
| `sources` | `SourcesEvent` | once, before any token: the passages sent to the model (labelled S1..Sn) |
| `token` | `TokenEvent` | many: a piece of the answer text |
| `error` | `ErrorEvent` | a typed failure; a `done` event follows (except `internal`) |
| `done` | `DoneEvent` | exactly once, last: the full `AnswerResult` |

Lines starting with `:` are comments (stream open / keep-alive) and carry no data. A question over the daily budget is still answered with HTTP 200: the stream carries a
`daily_budget_reached` error and a `done` event with the retrieved passages (no model call was made).

```
curl -N -X POST http://127.0.0.1:8000/v1/ask -H 'Content-Type: application/json' -d '{"question": "What dataset do they use?"}'
```"""


def _is_no_llm(result: AnswerResult) -> bool:
    """True if the request made no LLM call (cache hit, small talk, a failure before the model): its unit of the daily budget is given back."""
    return result.cached or result.provider in (None, "rules")


def _validated(state: ApiState, body) -> str:
    try:
        return state.resources.service.validate_question(body.question)
    except InvalidQuestion as exc:
        raise ApiException(422, exc.code, exc.message) from None


def _budget_events(service, question: str, doc_ids, rerank: bool) -> Iterator[dict]:
    """The daily budget is used up: the retrieved passages and the typed notice, in the same event shapes as a real answer."""
    result = service.passages_only(question, doc_ids, rerank)
    if result.errors:
        yield {"type": "error", "error": result.errors[-1]}
    yield {"type": "done", "result": result}


def _finish_request(state: ApiState, events: Iterator[dict] | None, acquired: bool, result: AnswerResult | None) -> None:
    """Blocking clean-up, run in the thread pool: close the generator (this releases the provider's connection), give back an unused unit of the budget, flush traces."""
    close = getattr(events, "close", None)
    if close is not None:
        close()
    if acquired and result is not None and _is_no_llm(result):
        state.resources.daily.release()
    try:
        state.resources.service.flush_traces()
    except Exception:  # noqa: BLE001: tracing must never break a request
        pass


def _run_to_result(state: ApiState, events: Iterator[dict], acquired: bool) -> AnswerResult | None:
    result = None
    try:
        for event in events:
            if event["type"] == "done":
                result = event["result"]
    finally:
        _finish_request(state, events, acquired, result)
    return result


def _open_events(state: ApiState, body, cleaned: str) -> tuple[Iterator[dict], bool]:
    """``(events, acquired)``: the real pipeline when the daily budget allows an LLM call, else the passages-only events."""
    service = state.resources.service
    if not state.resources.daily.try_acquire():
        return _budget_events(service, cleaned, body.doc_ids, body.rerank), False
    try:
        return service.answer_stream(cleaned, doc_ids=body.doc_ids, rerank=body.rerank, session_id=body.session_id), True
    except Exception:
        state.resources.daily.release()
        raise


async def _sse(state: ApiState, events: Iterator[dict], acquired: bool):
    result = None
    try:
        yield OPEN_COMMENT
        async for event in iterate_in_threadpool(events):
            if event["type"] == "done":
                result = event["result"]
            yield encode_service_event(event)
    except Exception:  # noqa: BLE001: the pipeline reports its failures as events; this is the last resort and leaks no detail
        yield encode_event("error", ErrorEvent(error=ErrorInfo(code="internal", message="The answer stream failed unexpectedly.", recoverable=False)))
    finally:
        with anyio.CancelScope(shield=True):  # also when the client disconnected: the generator is closed and the budget settled
            await run_in_threadpool(_finish_request, state, events, acquired, result)


# ---- query endpoints -----------------------------------------------------------------------------------------------------------------

@router.post("/ask", tags=["query"], summary="Ask a question: the answer as a stream of server-sent events", description=SSE_DESCRIPTION, response_class=StreamingResponse,
             responses={**COMMON_ERRORS, 200: {"description": "A `text/event-stream`: sources, token..., done.", "content": {"text/event-stream": {"schema": {"type": "string"}}}}})
@limiter.shared_limit(query_limits, scope="query")
async def ask(body: AskRequest, request: Request, state: ApiState = Depends(ready_state)):
    cleaned = _validated(state, body)
    events, acquired = _open_events(state, body, cleaned)
    return StreamingResponse(_sse(state, events, acquired), media_type="text/event-stream", headers=STREAM_HEADERS)


@router.post("/answer", tags=["query"], summary="Ask a question: the complete answer in one JSON body", response_model=AnswerResult, responses=COMMON_ERRORS,
             description="The same pipeline as `POST /v1/ask`, consumed to its end on the server. For scripts and tests that do not need the stream.")
@limiter.shared_limit(query_limits, scope="query")
async def answer(body: AskRequest, request: Request, state: ApiState = Depends(ready_state)):
    cleaned = _validated(state, body)
    events, acquired = _open_events(state, body, cleaned)
    result = await run_in_threadpool(_run_to_result, state, events, acquired)
    if result is None:
        raise ApiException(500, "internal", "The answer ended without a result.")
    return result


@router.post("/retrieve", tags=["query"], summary="Retrieval only: the best passages, no language model", response_model=RetrieveResponse,
             responses={**COMMON_ERRORS, 503: {"model": ErrorResponse, "description": "Not ready, or retrieval failed (`code`: not_ready, retrieval_failed)."}})
@limiter.shared_limit(query_limits, scope="query")
async def retrieve(body: RetrieveRequest, request: Request, state: ApiState = Depends(ready_state)):
    service = state.resources.service
    cleaned = _validated(state, body)
    try:
        sources, latency_ms = await run_in_threadpool(service.retrieve_sources, cleaned, body.doc_ids, body.rerank)
    except Exception as exc:  # noqa: BLE001
        raise ApiException(503, "retrieval_failed", f"{type(exc).__name__} while retrieving passages") from None
    return RetrieveResponse(sources=sources, latency_ms=latency_ms, index_version=service.index_version)


@router.post("/feedback", tags=["query"], summary="Rate an answer (thumbs up or down)", response_model=FeedbackResponse, responses=COMMON_ERRORS,
             description="Saved to the local feedback store first (so it survives a missing trace id or a Langfuse outage), then forwarded to Langfuse when `trace_id` is given. "
                         "A second vote for the same `request_id` replaces the first.")
@limiter.limit(FEEDBACK_LIMIT)
async def feedback(body: FeedbackRequest, request: Request, state: ApiState = Depends(ready_state)):
    record = state.resources.record_feedback
    if record is None:
        raise ApiException(503, "not_ready", "Feedback is not available.")
    outcome = await run_in_threadpool(record, body.request_id, body.trace_id, body.value, body.comment)
    return FeedbackResponse(saved=outcome.saved, forwarded=outcome.forwarded)


# ---- operations ------------------------------------------------------------------------------------------------------------------------

@router.get("/health", tags=["operations"], summary="Liveness: the process is up", response_model=HealthResponse)
async def health():
    return HealthResponse()


@router.get("/ready", tags=["operations"], summary="Readiness: the self-check (index, models, keys)", response_model=ReadyResponse, responses={
    401: COMMON_ERRORS[401], 503: {"model": ReadyResponse, "description": "A fatal check failed; the body lists every check."}})
async def ready(state: ApiState = Depends(authenticate)):
    resources = state.resources
    checks = [CheckModel(name=c.name, ok=c.ok, detail=c.detail, fatal=c.fatal) for c in (resources.selfcheck.checks if resources is not None else [])]
    body = ReadyResponse(ready=state.ready, checks=checks)
    return JSONResponse(status_code=200 if body.ready else 503, content=body.model_dump())


@router.get("/meta", tags=["operations"], summary="What is running, the limits and the daily budget", response_model=MetaResponse, responses={401: COMMON_ERRORS[401], 503: COMMON_ERRORS[503]})
async def meta(state: ApiState = Depends(ready_state)):
    res = state.resources
    daily = res.daily
    version_info = {k: v for k, v in res.version_info.items() if k != "index_dir"}  # a server path is nobody else's business
    return MetaResponse(
        version_info=version_info, limits=Limits(top_k=res.service.cfg.top_k, max_question_chars=res.service.cfg.max_question_chars),
        budget=Budget(cap=daily.cap, used=daily.used, remaining=daily.remaining),
        flags=Flags(gemini_key_present=res.gemini_key_present, langfuse_keys_present=res.langfuse_keys_present, tracing_on=res.tracing_on,
                    prices_configured=res.prices_configured, reranker_available=res.reranker_available),
        warmup_ms=res.warmup_ms)


@router.get("/observability", tags=["operations"], summary="The shared Langfuse metrics snapshot", response_model=ObservabilityResponse,
            responses={401: COMMON_ERRORS[401], 503: COMMON_ERRORS[503]},
            description="Read through the server's cached reader: one Metrics API request per TTL for all clients, never one per call. Each panel carries a `status` "
                        "(`ok`, `no_data`, `unavailable`, `no_keys`) instead of failing.")
async def observability(days: int | None = Query(None, ge=1, le=90, description="Window in days (default: the reader's window)."), state: ApiState = Depends(authenticate)):
    reader = state.resources.reader if state.resources is not None else None
    if reader is None:
        raise ApiException(503, "not_ready", "The metrics reader is not available.")
    snapshot = await run_in_threadpool(reader.snapshot, days)
    return ObservabilityResponse(window_days=days or getattr(reader, "window_days", None), panels={name: panel.as_dict() for name, panel in snapshot.items()})
