"""Typed errors of the generation path. Messages never contain an API key (providers scrub them)."""

from __future__ import annotations

# Kinds of provider failure. ``retryable`` ones are retried on the SAME provider with backoff; the others are not
# (retrying an invalid key or a malformed request cannot help) but may still fail over to the next provider.
RETRYABLE_KINDS = frozenset({"rate_limit", "server", "timeout", "connection"})
# Failures that say something about the PROVIDER's health and therefore count towards its circuit breaker. A 400/422
# ("bad_request") is about this one request (for example an over-long prompt), so a user cannot open a breaker with it.
BREAKER_KINDS = frozenset({"rate_limit", "server", "timeout", "connection", "auth", "not_found", "other"})


class RagError(Exception):
    """Base class of every error raised on purpose by rag_core."""


class InvalidQuestion(RagError):
    """The question was rejected by the input guard before any retrieval or LLM call. ``code`` is machine readable."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class ProviderError(RagError):
    """One failed call to one provider. ``kind`` is one of the kinds above, ``status`` the HTTP status if there was one."""

    def __init__(self, kind: str, message: str = "", *, status: int | None = None, retry_after_s: float | None = None,
                 error_class: str | None = None):
        super().__init__(message or kind)
        self.kind = kind
        self.status = status
        self.retry_after_s = retry_after_s
        self.error_class = error_class or type(self).__name__

    @property
    def retryable(self) -> bool:
        return self.kind in RETRYABLE_KINDS


class MidStreamError(RagError):
    """A provider failed AFTER it had already yielded text. The gateway does not retry or fail over (splicing a second
    model's answer onto a partial one would produce text nobody wrote); the caller gets what was streamed so far."""

    def __init__(self, provider: str, model: str, partial_text: str, cause: ProviderError, attempts: list):
        super().__init__(f"{provider} failed after {len(partial_text)} characters had been streamed: {cause.kind}")
        self.provider, self.model, self.partial_text, self.cause, self.attempts = provider, model, partial_text, cause, attempts


class AllProvidersFailed(RagError):
    """Every provider failed (or its breaker was open) before any text was yielded. ``attempts`` holds one record each."""

    def __init__(self, attempts: list):
        super().__init__("all providers failed before producing any text")
        self.attempts = attempts


class AnswerTooLong(RagError):
    """Every provider stopped with ``finish_reason="length"`` before it wrote any visible text (a reasoning model spent its whole output
    budget thinking). This says something about the question, not about a provider's health, so it never counts towards a breaker."""

    def __init__(self, attempts: list):
        super().__init__("the output budget was spent before any text was written")
        self.attempts = attempts


class RetrievalFailed(RagError):
    """The retriever raised; the service reports it as a typed error event, never as a crash."""
