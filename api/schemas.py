"""Wire schemas of the HTTP API (pydantic v2). They are the single source of truth for the request and response bodies, the SSE events and the OpenAPI document.

Import-safe and FastAPI-free on purpose: ``api.client`` (used by the Streamlit app) imports this module to parse what the server sends.
The answer itself is ``rag_core.schemas.AnswerResult``, unchanged: the API adds no second representation of an answer.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag_core.schemas import AnswerResult, ErrorInfo, Source

API_VERSION = "1.0.0"
ID_PATTERN = r"^[A-Za-z0-9_\-]{1,64}$"  # session ids, request ids, trace ids: opaque tokens, nothing else
QUESTION_PAYLOAD_LIMIT = 5000  # a guard on the body size; the real limit (``generation.max_question_chars``) is applied by the service
MAX_BODY_BYTES = 16 * 1024  # request bodies are tiny: a question, a few ids, a comment. Larger bodies are refused (413) before they are read.
DOC_ID_PATTERN = r"^[A-Za-z0-9._\-]{1,64}$"  # Qasper paper ids such as 1601.02403; nothing that could look like a path, a filter expression or markup
DocId = Annotated[str, Field(pattern=DOC_ID_PATTERN, max_length=64)]


class StrictModel(BaseModel):
    """Request models refuse unknown fields: a typo or a probing client gets a 422, not silent acceptance."""

    model_config = ConfigDict(extra="forbid")


class QueryRequest(StrictModel):
    """What every query endpoint takes."""

    question: str = Field(..., max_length=QUESTION_PAYLOAD_LIMIT, description="The question. It is cleaned (control characters removed) and limited to "
                          "``generation.max_question_chars`` characters by the service; a violation is a 422 with a machine-readable code.",
                          examples=["What dataset do the authors evaluate on?"])
    doc_ids: list[DocId] | None = Field(None, max_length=50, description="Restrict the search to these paper ids (Qasper ids: letters, digits, `.`, `_`, `-`; at most 64 characters each). "
                                        "Omit to search all papers.", examples=[["1601.02403"]])
    rerank: bool = Field(False, description="Re-score the best passages with the cross-encoder before choosing the ones the model sees (slower).")
    session_id: str | None = Field(None, pattern=ID_PATTERN, description="An opaque id chosen by the client (letters, digits, `_`, `-`; at most 64 characters), used to group traces. "
                                   "The rate limit is keyed by the `X-Session-Id` HEADER (same id), else by the client address.")

    @field_validator("doc_ids")
    @classmethod
    def _unique(cls, value: list[str] | None) -> list[str] | None:
        return None if value is None else list(dict.fromkeys(value))


class AskRequest(QueryRequest):
    """``POST /v1/ask`` and ``POST /v1/answer``."""


class RetrieveRequest(QueryRequest):
    """``POST /v1/retrieve``: retrieval only, no language model."""


class RetrieveResponse(BaseModel):
    sources: list[Source]
    latency_ms: float
    index_version: str


class FeedbackRequest(StrictModel):
    request_id: str = Field(..., pattern=ID_PATTERN, description="``AnswerResult.request_id`` of the answer being rated.")
    trace_id: str | None = Field(None, pattern=ID_PATTERN, description="``AnswerResult.trace_id`` when the answer was traced; the vote is then also forwarded to Langfuse.")
    value: Literal[0, 1] = Field(..., description="1 = helpful, 0 = not helpful.")
    comment: str | None = Field(None, max_length=300, description="Optional. Masked (links, e-mail addresses, long numbers) before it is stored or forwarded.")


class FeedbackResponse(BaseModel):
    saved: bool = Field(..., description="Written to the local feedback store.")
    forwarded: bool = Field(..., description="Handed to Langfuse (needs a trace id).")


# ---- server-sent events of POST /v1/ask --------------------------------------------------------------------------------------------------

class SourcesEvent(BaseModel):
    """``event: sources``: the passages sent to the model (S1..Sn), before any token."""

    sources: list[Source]
    context: dict[str, Any] = Field(default_factory=dict)


class TokenEvent(BaseModel):
    """``event: token``: a piece of the answer text."""

    text: str


class ErrorEvent(BaseModel):
    """``event: error``: a typed failure. A ``done`` event follows unless ``error.code`` is ``internal`` (then the stream just ends)."""

    error: ErrorInfo


class DoneEvent(BaseModel):
    """``event: done``: exactly one, last: the complete ``AnswerResult`` (citations parsed, usage, latencies, trace id, request id)."""

    result: AnswerResult


SSE_EVENT_MODELS: dict[str, type[BaseModel]] = {"sources": SourcesEvent, "token": TokenEvent, "error": ErrorEvent, "done": DoneEvent}


# ---- health, readiness, metadata ----------------------------------------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str = API_VERSION


class CheckModel(BaseModel):
    name: str
    ok: bool
    detail: str
    fatal: bool = True


class ReadyResponse(BaseModel):
    ready: bool = Field(..., description="True when every fatal check passed and the service can answer.")
    checks: list[CheckModel]


class Limits(BaseModel):
    top_k: int = Field(..., description="Passages sent to the model per question.")
    max_question_chars: int


class Budget(BaseModel):
    cap: int = Field(..., description="LLM-backed answers per UTC day, all clients together; beyond it only the retrieved passages are returned.")
    used: int
    remaining: int


class Flags(BaseModel):
    gemini_key_present: bool
    langfuse_keys_present: bool
    tracing_on: bool
    prices_configured: bool
    reranker_available: bool


class MetaResponse(BaseModel):
    api_version: str = API_VERSION
    version_info: dict[str, Any] = Field(..., description="What is running: index, embedder, reranker, generator, prompt, retrieval settings.")
    limits: Limits
    budget: Budget
    flags: Flags
    warmup_ms: float | None = None


class ObservabilityResponse(BaseModel):
    window_days: int | None = None
    panels: dict[str, dict[str, Any]] = Field(..., description="Panel name -> {status, data, as_of, message, sample}; shared and cached on the server (one Metrics API read per TTL).")


class ErrorBody(BaseModel):
    code: str = Field(..., description="invalid_type | empty | too_long | rate_limited | session_cap | unauthorized | not_ready | retrieval_failed | internal")
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody
