"""A small synchronous client of the API (httpx, imported lazily: importing this module never imports an HTTP client).

    client = ApiClient("http://127.0.0.1:8000", token=None)
    for event in client.ask_stream(AskRequest(question="What dataset do they use?")):   # the same event dicts as RagService.answer_stream
        ...

``ask_stream`` opens the request EAGERLY (so an HTTP error is raised at the call, not at the first ``next``) and returns a generator that closes the connection when it is
exhausted, closed or garbage collected. Errors are typed: ``ApiError`` (the server answered with an error: ``status``, ``code``, ``message``) and ``ApiUnavailable`` (no usable
answer at all: connection refused, timeout, a dropped stream). Nothing here prints or logs; the token is only sent as a header.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from api.schemas import (
    AskRequest,
    FeedbackRequest,
    FeedbackResponse,
    MetaResponse,
    ObservabilityResponse,
    ReadyResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from api.sse import decode_service_event, iter_sse
from rag_core.schemas import AnswerResult

SESSION_HEADER = "X-Session-Id"  # the rate-limit key on the server (api.limits.SESSION_HEADER; repeated here so the client does not import fastapi or slowapi)


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, retry_after_s: float | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.retry_after_s = status, code, message, retry_after_s


class ApiUnavailable(Exception):
    """The server could not be reached, or the connection broke. ``reason`` is the exception class name only (never a URL with credentials or a response body)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _raise_for_error(response) -> None:
    if response.status_code < 400:
        return
    code, message = "http_error", f"The service answered HTTP {response.status_code}."
    try:
        error = response.json()["error"]
        code, message = str(error["code"]), str(error["message"])
    except Exception:  # noqa: BLE001: a proxy's HTML page or an empty body
        pass
    retry = None
    try:
        retry = float(response.headers.get("Retry-After")) if response.headers.get("Retry-After") else None
    except ValueError:
        pass
    raise ApiError(response.status_code, code, message, retry)


class ApiClient:
    def __init__(self, base_url: str, *, token: str | None = None, timeout_s: float = 10.0, stream_timeout_s: float = 120.0, http: Any = None):
        self.base_url = base_url.rstrip("/")
        self._token, self.timeout_s, self.stream_timeout_s, self._http = (token or "").strip() or None, timeout_s, stream_timeout_s, http

    def _client(self):
        if self._http is None:
            import httpx

            headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
            self._http = httpx.Client(base_url=self.base_url, headers=headers, timeout=self.timeout_s)
        return self._http

    def _call(self, method: str, path: str, *, json: Any = None, params: dict | None = None, timeout: float | None = None, session_id: str | None = None):
        headers = {SESSION_HEADER: session_id} if session_id else None
        try:
            return self._client().request(method, path, json=json, params=params, headers=headers, timeout=timeout or self.timeout_s)
        except Exception as exc:  # noqa: BLE001: httpx.TransportError and friends
            raise ApiUnavailable(type(exc).__name__) from None

    # ---- operations ------------------------------------------------------------------------------------------------------------------
    def ready(self) -> ReadyResponse:
        """The self-check. A 503 with a body is a normal answer (not ready, with the failed checks), not an error."""
        response = self._call("GET", "/v1/ready")
        if response.status_code in (200, 503):
            try:
                return ReadyResponse.model_validate(response.json())
            except Exception:  # noqa: BLE001: a 503 from a proxy has no such body
                pass
        _raise_for_error(response)
        raise ApiUnavailable("UnexpectedResponse")

    def meta(self) -> MetaResponse:
        response = self._call("GET", "/v1/meta")
        _raise_for_error(response)
        return MetaResponse.model_validate(response.json())

    def observability(self, days: int | None = None) -> ObservabilityResponse:
        response = self._call("GET", "/v1/observability", params={"days": days} if days else None, timeout=self.stream_timeout_s)
        _raise_for_error(response)
        return ObservabilityResponse.model_validate(response.json())

    # ---- queries ---------------------------------------------------------------------------------------------------------------------
    def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        response = self._call("POST", "/v1/retrieve", json=request.model_dump(exclude_none=True), session_id=request.session_id)
        _raise_for_error(response)
        return RetrieveResponse.model_validate(response.json())

    def answer(self, request: AskRequest) -> AnswerResult:
        response = self._call("POST", "/v1/answer", json=request.model_dump(exclude_none=True), timeout=self.stream_timeout_s, session_id=request.session_id)
        _raise_for_error(response)
        return AnswerResult.model_validate(response.json())

    def feedback(self, request: FeedbackRequest) -> FeedbackResponse:
        response = self._call("POST", "/v1/feedback", json=request.model_dump(exclude_none=True))
        _raise_for_error(response)
        return FeedbackResponse.model_validate(response.json())

    def ask_stream(self, request: AskRequest) -> Iterator[dict]:
        http = self._client()
        headers = {"Accept": "text/event-stream", **({SESSION_HEADER: request.session_id} if request.session_id else {})}
        built = http.build_request("POST", "/v1/ask", json=request.model_dump(exclude_none=True), headers=headers, timeout=self.stream_timeout_s)
        try:
            response = http.send(built, stream=True)
        except Exception as exc:  # noqa: BLE001
            raise ApiUnavailable(type(exc).__name__) from None
        if response.status_code != 200:
            try:
                response.read()
                _raise_for_error(response)
            finally:
                response.close()
            raise ApiUnavailable("UnexpectedResponse")
        return self._events(response)

    @staticmethod
    def _events(response) -> Iterator[dict]:
        try:
            for name, data in iter_sse(response.iter_lines()):
                event = decode_service_event(name, data)
                if event is not None:
                    yield event
        except (ApiError, ApiUnavailable):
            raise
        except Exception as exc:  # noqa: BLE001: a dropped connection or a malformed event
            raise ApiUnavailable(type(exc).__name__) from None
        finally:
            response.close()
