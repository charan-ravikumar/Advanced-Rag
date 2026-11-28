"""Shared pieces of the API: the typed error, the application state and the FastAPI dependencies (authentication, readiness)."""

from __future__ import annotations

import hmac
from dataclasses import dataclass, field
from typing import Any

from fastapi import Depends, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.guards import AppConfig
from rag.concurrency import AdmissionGate

bearer_scheme = HTTPBearer(auto_error=False, description="Optional shared token. The server requires it only when ADVRAG_API_TOKEN is set in its environment.")


class ApiException(Exception):
    """An error with an HTTP status and a machine-readable ``code``; rendered as ``{"error": {"code", "message"}}`` (``api.schemas.ErrorResponse``)."""

    def __init__(self, status: int, code: str, message: str, headers: dict[str, str] | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.headers = status, code, message, headers or {}


@dataclass
class ApiState:
    resources: Any = None  # app.bootstrap.Resources (service, daily budget, reader, feedback recorder, self-check ...)
    token: str | None = None
    gate: AdmissionGate = field(default_factory=lambda: AdmissionGate(AppConfig().retrieval_max_in_flight))  # the queue limit of retrieval

    @property
    def ready(self) -> bool:
        return self.resources is not None and bool(self.resources.ready)


def get_state(request: Request) -> ApiState:
    return request.app.state.api


def authenticate(state: ApiState = Depends(get_state), credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme)) -> ApiState:
    if state.token is None:
        return state
    supplied = credentials.credentials if credentials is not None and credentials.scheme.lower() == "bearer" else ""
    if not hmac.compare_digest(supplied.encode("utf-8"), state.token.encode("utf-8")):
        raise ApiException(401, "unauthorized", "A valid bearer token is required.", {"WWW-Authenticate": "Bearer"})
    return state


def ready_state(state: ApiState = Depends(authenticate)) -> ApiState:
    if not state.ready:
        raise ApiException(503, "not_ready", "The service is not ready: see GET /v1/ready for the failed checks.")
    return state
