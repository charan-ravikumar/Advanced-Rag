"""Adapters that let the UI use the API through the SAME interfaces it used in-process: ``res.service`` (``answer_stream``, ``validate_question``, ``cfg``, ``flush_traces``),
``res.daily`` (the daily budget), ``res.reader`` (the metrics snapshot) and ``res.record_feedback``. The chat tab is therefore unchanged: it consumes the same event dicts.

Pure: no streamlit, no environment, no HTTP library (the HTTP is in ``api.client``). A failure of the API never raises into the page: it becomes the same typed ``error`` and
``done`` events a failing model produces (a degraded result with a notice), so the UI shows a warning, not a stack trace.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from types import SimpleNamespace

from api.client import ApiClient, ApiError, ApiUnavailable
from api.schemas import AskRequest, FeedbackRequest, Limits
from rag_core.errors import InvalidQuestion
from rag_core.schemas import AnswerResult, ErrorInfo
from rag_core.scores import FeedbackResult
from rag_core.service import clean_question

UNAVAILABLE_MESSAGE = ("The answer service cannot be reached right now, so no answer could be generated. Please try again in a moment.")


def _failure_events(code: str, message: str) -> list[dict]:
    error = ErrorInfo(code=code, message=message, recoverable=True)
    return [{"type": "error", "error": error}, {"type": "done", "result": AnswerResult(degraded=True, answer=message, errors=[error])}]


class RemoteService:
    """``RagService`` as the UI sees it, backed by ``POST /v1/ask``."""

    def __init__(self, client: ApiClient, limits: Limits):
        self._client = client
        self.cfg = SimpleNamespace(top_k=limits.top_k, max_question_chars=limits.max_question_chars)

    def validate_question(self, question) -> str:
        """The same rule as the server's (``rag_core.service.clean_question``), applied before any request is made."""
        return clean_question(question, self.cfg.max_question_chars)

    def answer_stream(self, question: str, doc_ids: Sequence[str] | None = None, rerank: bool = False, session_id: str | None = None) -> Iterator[dict]:
        cleaned = self.validate_question(question)  # eager, like RagService.answer_stream: raises InvalidQuestion at the call
        request = AskRequest(question=cleaned, doc_ids=list(doc_ids) if doc_ids else None, rerank=rerank, session_id=session_id)
        try:
            stream = self._client.ask_stream(request)
        except ApiError as exc:
            if exc.status == 422:
                raise InvalidQuestion(exc.code, exc.message) from None
            return iter(_failure_events(exc.code, exc.message))  # 429 (the message says how long to wait), 503, 401 ...
        except ApiUnavailable:
            return iter(_failure_events("api_unavailable", UNAVAILABLE_MESSAGE))
        return self._guarded(stream)

    @staticmethod
    def _guarded(stream: Iterator[dict]) -> Iterator[dict]:
        try:
            yield from stream
        except ApiUnavailable:  # the connection broke mid-stream: the UI keeps what it received and reports the end as unexpected
            yield {"type": "error", "error": ErrorInfo(code="api_unavailable", message=UNAVAILABLE_MESSAGE, recoverable=True)}

    def flush_traces(self, timeout_s: float | None = None) -> None:
        """The server flushes its own traces at the end of every request."""


class RemoteBudget:
    """The daily budget lives on the server, which enforces it (a question over the budget gets the retrieved passages and a notice, no LLM call). This object only
    reports what is left (cached for a few seconds: the page asks on every rerun) and never refuses."""

    def __init__(self, client: ApiClient, *, ttl_s: float = 5.0, clock: Callable[[], float] = time.monotonic):
        self._client, self._ttl, self._clock = client, ttl_s, clock
        self._at: float | None = None
        self._remaining = 0

    def try_acquire(self) -> bool:
        return True

    def release(self) -> None:
        pass

    @property
    def remaining(self) -> int:
        now = self._clock()
        if self._at is None or now - self._at >= self._ttl:
            try:
                self._remaining = self._client.meta().budget.remaining
            except (ApiError, ApiUnavailable):
                pass  # keep the last known value
            self._at = now
        return self._remaining


class _Panel:
    def __init__(self, data: dict):
        self._data = data

    def as_dict(self) -> dict:
        return self._data


class RemoteReader:
    """``ObservabilityReader.snapshot`` through ``GET /v1/observability``. A failure raises (``ui_observability`` turns it into an 'unavailable' panel and does not cache it)."""

    def __init__(self, client: ApiClient, window_days: int = 7):
        self._client, self.window_days = client, window_days

    def snapshot(self, days: int | None = None) -> dict[str, _Panel]:
        response = self._client.observability(days)
        self.window_days = response.window_days or self.window_days
        return {name: _Panel(panel) for name, panel in response.panels.items()}


def make_record_feedback(client: ApiClient) -> Callable[..., FeedbackResult]:
    def record(request_id: str, trace_id: str | None, value: int, comment: str | None = None) -> FeedbackResult:
        try:
            outcome = client.feedback(FeedbackRequest(request_id=request_id, trace_id=trace_id, value=int(value), comment=comment))
        except Exception:  # noqa: BLE001: a vote must never break the page
            return FeedbackResult(False, False)
        return FeedbackResult(outcome.saved, outcome.forwarded)

    return record
