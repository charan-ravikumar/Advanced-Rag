"""The application factory.

    python -m uvicorn api.main:app --host 127.0.0.1 --port 8000        # then open http://127.0.0.1:8000/docs

``create_app`` builds the FastAPI object and nothing else: importing this module opens no index, loads no model and reads no file. The resources (index, embedder, gateway,
tracer, feedback store) are built ONCE at startup, in the thread pool, by ``app.bootstrap.build_resources`` (the same composition and self-check the Streamlit app used to
run in-process). A failed check does not stop the server: it answers ``503 not_ready`` on the query endpoints and lists the failed checks on ``GET /v1/ready``.

Environment: the index directory (``ADVRAG_INDEX_DIR`` or ``CHROMA_LOCAL_PATH``), ``GEMINI_API_KEY`` and the Langfuse keys, as before (``rag_core.env.load_env`` loads the
allowlisted ones from ``.env`` at startup, never overriding). ``ADVRAG_API_TOKEN``: when set, every endpoint except ``/v1/health`` needs ``Authorization: Bearer <token>``.
Bind to 127.0.0.1 (the default of the command above) unless a token is set.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from starlette.concurrency import run_in_threadpool

from api.deps import ApiException, ApiState
from api.limits import configure as configure_limits
from api.limits import limiter
from api.routes import router
from api.schemas import API_VERSION, MAX_BODY_BYTES, SSE_EVENT_MODELS, ErrorBody, ErrorResponse

TITLE = "Advanced RAG API"
DESCRIPTION = """Hybrid retrieval (vector + SQLite FTS5, weighted RRF) and cited, streamed answers over the Qasper papers.

* **`POST /v1/ask`**: the answer as server-sent events (`sources`, `token`..., `done`).
* **`POST /v1/answer`**: the same pipeline as one JSON body.
* **`POST /v1/retrieve`**: retrieval only (no model call).
* **`POST /v1/feedback`**, **`GET /v1/meta`**, **`GET /v1/observability`**, **`GET /v1/ready`**, **`GET /v1/health`**.

Errors are `{"error": {"code", "message"}}`. Every answer is an `AnswerResult`: citations are parsed and validated against the passages that were sent to the model.
Questions are independent (no conversation memory). Only public research text belongs here."""
TAGS = [{"name": "query", "description": "Ask, retrieve, rate."}, {"name": "operations", "description": "Health, readiness, metadata, metrics."}]


def default_resources():
    """Production resources. Imports are local so that importing ``api.main`` stays free of heavy modules."""
    from app.bootstrap import build_resources
    from config import cfg
    from rag_core.env import load_env

    load_env()
    return build_resources(cfg)


def _build_safely(factory: Callable[[], Any]):
    try:
        return factory()
    except Exception as exc:  # noqa: BLE001: build_resources does not raise; this keeps a broken factory from killing the server
        from app.bootstrap import Check, Resources, SelfCheck
        from app.guards import AppConfig

        return Resources(settings=AppConfig(), selfcheck=SelfCheck([Check("Startup", False, f"the resources could not be built ({type(exc).__name__})")]))


def _error(status: int, code: str, message: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content=ErrorResponse(error=ErrorBody(code=code, message=message)).model_dump(), headers=headers)


def _with_event_schemas(app: FastAPI) -> dict:
    """The OpenAPI document plus the four SSE event models (they are not response models of any route, so FastAPI would not list them)."""
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes, tags=TAGS)
    components = schema.setdefault("components", {}).setdefault("schemas", {})
    for model in SSE_EVENT_MODELS.values():
        js = model.model_json_schema(ref_template="#/components/schemas/{model}")
        for name, definition in js.pop("$defs", {}).items():
            components.setdefault(name, definition)
        components.setdefault(model.__name__, js)
    app.openapi_schema = schema
    return schema


class BodyLimitMiddleware:
    """Refuses a request body that is too large BEFORE it is read: 413 above ``max_bytes``, 411 when a POST does not declare its length (h11 enforces the declared length, so a
    declared size cannot be exceeded). Pure ASGI, so it also protects the streaming endpoint."""

    def __init__(self, app, max_bytes: int = MAX_BODY_BYTES):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] in ("POST", "PUT", "PATCH"):
            declared = dict(scope["headers"]).get(b"content-length")
            if declared is None:
                return await _error(411, "length_required", "Content-Length is required.")(scope, receive, send)
            try:
                too_big = int(declared) > self.max_bytes
            except ValueError:
                return await _error(400, "invalid_request", "Content-Length is not a number.")(scope, receive, send)
            if too_big:
                return await _error(413, "payload_too_large", f"The request body is larger than {self.max_bytes} bytes.")(scope, receive, send)
        await self.app(scope, receive, send)


def create_app(resources_factory: Callable[[], Any] | None = None, *, token: str | None = None) -> FastAPI:
    """``resources_factory``: returns an ``app.bootstrap.Resources`` (tests inject a fake one). ``token``: overrides ``ADVRAG_API_TOKEN``."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resources = await run_in_threadpool(_build_safely, resources_factory or default_resources)
        s = resources.settings
        configured = token if token is not None else os.environ.get("ADVRAG_API_TOKEN", "")
        app.state.api = ApiState(resources=resources, token=(configured or "").strip() or None)
        configure_limits(s.rate_limit_requests, s.rate_limit_window_seconds, s.session_request_cap)
        yield
        service = getattr(resources, "service", None)
        if service is not None:
            await run_in_threadpool(service.flush_traces)

    app = FastAPI(title=TITLE, version=API_VERSION, description=DESCRIPTION, openapi_tags=TAGS, lifespan=lifespan)
    app.state.api = ApiState()  # not ready until the lifespan has built the resources
    app.state.limiter = limiter
    app.include_router(router)
    app.add_middleware(BodyLimitMiddleware)

    @app.exception_handler(RateLimitExceeded)
    async def rate_limited(_: Request, exc: RateLimitExceeded):
        window = int(exc.limit.limit.get_expiry())  # the window of the limit that was hit (a day for the per-key daily cap)
        daily = window >= 86400
        message = ("This client reached its daily limit of questions." if daily else f"Too many requests. Please wait up to {window} seconds before asking again.")
        return _error(429, "session_cap" if daily else "rate_limited", message, {"Retry-After": str(window)})

    @app.exception_handler(ApiException)
    async def api_exception(_: Request, exc: ApiException):
        return _error(exc.status, exc.code, exc.message, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        return _error(422, "invalid_request", f"{where}: {first.get('msg', 'invalid request')}" if where else str(first.get("msg", "invalid request")))

    @app.exception_handler(Exception)
    async def unexpected(_: Request, __: Exception):
        return _error(500, "internal", "An unexpected error occurred.")  # no detail: it could carry a path, a key or a prompt

    app.openapi = lambda: _with_event_schemas(app)  # type: ignore[method-assign]
    return app


app = create_app()
