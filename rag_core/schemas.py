"""The typed result of a question (pydantic v2). Everything a UI, an evaluation or a log needs, in one object."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field


class Source(BaseModel):
    """One passage. ``label`` ("S1", "S2", ...) is the id the model is told to cite; ``chunk_id`` is the stable store id."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    section: str = ""
    text: str
    rank: int
    score: float
    label: str = ""


class Usage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None  # as the API reports it (includes the reasoning tokens)
    reasoning_tokens: int | None = None  # the share of completion_tokens spent on reasoning, when reported separately


class Latency(BaseModel):
    """Milliseconds from ``perf_counter``. ``first_token`` is measured from the start of the request to the first text delta
    the provider yielded (None if none was); ``retrieve`` is the retrieval call only; ``total`` is the whole request."""

    retrieve: float | None = None
    first_token: float | None = None
    total: float | None = None


class AttemptRecord(BaseModel):
    """One attempt at one provider: what was tried, how long it took and how it ended."""

    provider: str
    model: str
    attempt: int  # 1-based within the provider
    latency_ms: float
    outcome: str  # "ok" | "error" | "skipped_circuit_open" | "skipped_deadline"
    error_class: str | None = None
    error_kind: str | None = None
    status: int | None = None
    first_token_seen: bool = False


class ErrorInfo(BaseModel):
    code: str  # invalid_question | retrieval_failed | all_providers_failed | answer_too_long | mid_stream | internal
    message: str
    provider: str | None = None
    recoverable: bool = True


class PromptInfo(BaseModel):
    """Which prompt produced an answer: enough to reproduce it and to see a silent edit (the hash changes)."""

    name: str
    version: str | None = None  # Langfuse version number, or the local PROMPT_VERSION
    label: str | None = None  # "production" for the Langfuse path, None for pinned versions and local templates
    source: str  # "langfuse" | "local" | "local_fallback"
    sha256: str  # of the template text
    fallback_reason: str | None = None


class SpanCitation(BaseModel):
    """One citation of a claim: the source label, the words the model copied from it, and where they are in ``Source.text``.
    ``verified`` means the quote was found word for word in that source (``rag_core.claims.locate_quote``); it does NOT mean the source supports the claim."""

    source: str
    quote: str
    start: int | None = None  # character offsets into the cited Source.text (None when not verified)
    end: int | None = None
    verified: bool = False


class Claim(BaseModel):
    """One statement of a long answer and its citations (structured output: the model fills a JSON schema, nothing is parsed out of prose)."""

    text: str
    citations: list[SpanCitation] = Field(default_factory=list)


class AnswerResult(BaseModel):
    answer: str = ""  # long mode: rendered by the code from ``claims`` ("claim [S1] claim [S2][S3]"); short mode: the model's text
    short_answer: str | None = None  # only for answer_short(); None when the model abstained
    claims: list[Claim] = Field(default_factory=list)  # long mode: the structured answer (empty for short mode, abstentions and failures)
    citations: list[str] = Field(default_factory=list)  # labels ("S1") that the claims cite AND that were provided
    invalid_citations: list[str] = Field(default_factory=list)  # labels that name a source that was never provided (the schema forbids them; still checked)
    abstained: bool = False  # the model replied INSUFFICIENT_CONTEXT, or nothing relevant was retrieved
    used_sources: list[Source] = Field(default_factory=list)  # EXACTLY the passages sent to the LLM (label, id, text)
    retrieved_sources: list[Source] = Field(default_factory=list)  # everything retrieved; shown when degraded
    provider: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None  # the effort the generator was asked for (None: the parameter was not sent)
    finish_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)
    latency_ms: Latency = Field(default_factory=Latency)
    degraded: bool = False  # no provider could answer: ``answer`` is a notice and ``retrieved_sources`` are the passages
    partial: bool = False  # the stream broke mid-way: ``answer`` is what was received
    cached: bool = False
    errors: list[ErrorInfo] = Field(default_factory=list)
    attempts: list[AttemptRecord] = Field(default_factory=list)
    prompt_version: str | None = None
    prompt: PromptInfo | None = None  # name, version, label, source, sha256 of the template that was used
    trace_id: str | None = None  # None if the request was not traced (no keys, disabled, sampled out, over the daily cap)
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)  # one per answer (a cache hit gets a new one): the key of the user's feedback
    mode: str = "long"  # long | short | direct (small talk, no retrieval, no LLM)
