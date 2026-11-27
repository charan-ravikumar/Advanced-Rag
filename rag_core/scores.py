"""Score schema, defined up front: one constant and one validated entry per score name, and a fail-open ``submit_feedback``.

Names are fixed here so that the UI, the offline evaluation (a later phase) and the Langfuse dashboards agree on them:

* ``user_feedback``            BOOLEAN 1 (helpful) / 0 (not helpful): written now, from the thumbs buttons.
* ``ragas_faithfulness``       NUMERIC 0..1   reserved for the offline evaluation (claims supported by the used sources).
* ``ragas_answer_relevancy``   NUMERIC 0..1   reserved.
* ``answer_f1``                NUMERIC 0..1   reserved (token F1 against a reference answer).
* ``judge_agreement``          NUMERIC 0..1   reserved (agreement between the LLM judge and a human label).

A score is one billable Langfuse unit, so ``submit_feedback`` uses a deterministic ``score_id`` (a user who changes their mind UPDATES the score
instead of adding a second one). Import-safe: the SDK is only reached through ``langfuse_client`` (lazily). Comments are masked like any other
free text before they leave the process.

``record_feedback`` is what the app calls: it saves the vote in a local SQLite file FIRST (keyed by request id, comment masked), then forwards it with
``submit_feedback`` when the request has a trace id. So a vote is never silently dropped because the request was not traced or Langfuse was down.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_core.tracing import mask_text

USER_FEEDBACK = "user_feedback"
RAGAS_FAITHFULNESS = "ragas_faithfulness"
RAGAS_ANSWER_RELEVANCY = "ragas_answer_relevancy"
ANSWER_F1 = "answer_f1"
JUDGE_AGREEMENT = "judge_agreement"

COMMENT_MAX_CHARS = 500


@dataclass(frozen=True)
class ScoreSpec:
    name: str
    data_type: str  # BOOLEAN | NUMERIC (Langfuse data types)
    minimum: float
    maximum: float
    description: str
    reserved: bool  # True: defined for a later phase, not written by the app yet


SCORES: dict[str, ScoreSpec] = {s.name: s for s in (
    ScoreSpec(USER_FEEDBACK, "BOOLEAN", 0, 1, "1 = the user found the answer helpful, 0 = not helpful", reserved=False),
    ScoreSpec(RAGAS_FAITHFULNESS, "NUMERIC", 0, 1, "share of the answer's claims supported by the sources that were sent", reserved=True),
    ScoreSpec(RAGAS_ANSWER_RELEVANCY, "NUMERIC", 0, 1, "how well the answer addresses the question", reserved=True),
    ScoreSpec(ANSWER_F1, "NUMERIC", 0, 1, "token F1 of the short answer against the reference answers (max over annotators)", reserved=True),
    ScoreSpec(JUDGE_AGREEMENT, "NUMERIC", 0, 1, "agreement between the LLM judge and a human label", reserved=True),
)}


def validate_score(name: str, value: Any) -> tuple[float, str]:
    """``(value as float, Langfuse data type)`` or ``ValueError`` for an unknown name, a non-number, or a value outside the score's range
    (BOOLEAN scores accept only 0 and 1)."""
    spec = SCORES.get(name)
    if spec is None:
        raise ValueError(f"unknown score {name!r}; defined scores: {sorted(SCORES)}")
    if isinstance(value, bool):
        value = int(value)
    if not isinstance(value, (int, float)) or value != value:  # NaN fails the equality
        raise ValueError(f"score {name!r} needs a number, got {value!r}")
    if not spec.minimum <= value <= spec.maximum:
        raise ValueError(f"score {name!r} must be within [{spec.minimum}, {spec.maximum}], got {value!r}")
    if spec.data_type == "BOOLEAN" and value not in (0, 1):
        raise ValueError(f"score {name!r} is boolean: 0 or 1, got {value!r}")
    return float(value), spec.data_type


def submit_score(trace_id: str | None, name: str, value: Any, comment: str | None = None, *, client_factory: Callable[[], Any] | None = None) -> bool:
    """Send one validated score. Returns True if it was handed to the SDK; False if there is no trace id (the request was not traced), no keys,
    an invalid name/value, or the SDK failed. NEVER raises: scoring must not break the UI."""
    try:
        if not trace_id:
            return False
        number, data_type = validate_score(name, value)
        if client_factory is not None:
            client = client_factory()
        else:
            from rag_core.langfuse_client import get_langfuse_client

            client = get_langfuse_client()
        if client is None:
            return False
        kwargs: dict[str, Any] = {"name": name, "value": number, "trace_id": trace_id, "data_type": data_type, "score_id": f"{trace_id}-{name}"}
        if comment:
            kwargs["comment"] = mask_text(comment, COMMENT_MAX_CHARS)
        client.create_score(**kwargs)
        return True
    except Exception:  # noqa: BLE001: fail open
        return False


def submit_feedback(trace_id: str | None, value: int | bool, comment: str | None = None, *, client_factory: Callable[[], Any] | None = None) -> bool:
    """The thumbs buttons: ``value`` 1 or 0. Fail-open (see ``submit_score``); the comment is masked and truncated."""
    return submit_score(trace_id, USER_FEEDBACK, value, comment, client_factory=client_factory)


# ---- local-first feedback ---------------------------------------------------------------------------------------------------------------

class FeedbackStore:
    """A small SQLite file of the thumbs, keyed by (request id, score name). The file and its folder are created on the first save (building the app writes
    nothing). One connection per call, so any thread can use it. Only the MASKED comment is ever stored; the question and the answer never are."""

    _SCHEMA = ("CREATE TABLE IF NOT EXISTS feedback (request_id TEXT NOT NULL, name TEXT NOT NULL, value INTEGER NOT NULL, comment TEXT, trace_id TEXT, "
               "created_at TEXT NOT NULL, updated_at TEXT NOT NULL, forwarded INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (request_id, name))")

    def __init__(self, path: str | Path, *, now: Callable[[], datetime] | None = None):
        self.path = Path(path)
        self._now = now or (lambda: datetime.now(timezone.utc))

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.execute(self._SCHEMA)
        return conn

    def save(self, request_id: str, name: str, value: int, comment: str | None, trace_id: str | None) -> None:
        stamp = self._now().isoformat(timespec="seconds")
        with closing(self._connect()) as conn, conn:
            conn.execute("INSERT INTO feedback (request_id, name, value, comment, trace_id, created_at, updated_at, forwarded) VALUES (?, ?, ?, ?, ?, ?, ?, 0) "
                         "ON CONFLICT (request_id, name) DO UPDATE SET value = excluded.value, comment = excluded.comment, trace_id = excluded.trace_id, "
                         "updated_at = excluded.updated_at, forwarded = 0", (request_id, name, int(value), comment, trace_id, stamp, stamp))

    def mark_forwarded(self, request_id: str, name: str) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("UPDATE feedback SET forwarded = 1 WHERE request_id = ? AND name = ?", (request_id, name))


@dataclass(frozen=True)
class FeedbackResult:
    saved: bool  # written to the local store
    forwarded: bool  # handed to Langfuse (needs a trace id)

    @property
    def recorded(self) -> bool:
        return self.saved or self.forwarded


def record_feedback(store: FeedbackStore | None, request_id: str | None, trace_id: str | None, value: Any, comment: str | None = None, *,
                    client_factory: Callable[[], Any] | None = None) -> FeedbackResult:
    """Persist the vote locally FIRST (so it survives a missing trace id, an untraced request or a Langfuse outage), then forward it to Langfuse when the
    request has a trace id. A vote for the same request id replaces the earlier one. The comment is masked and truncated before it touches the disk.
    NEVER raises; an invalid value is neither saved nor sent."""
    try:
        validate_score(USER_FEEDBACK, value)
    except ValueError:
        return FeedbackResult(False, False)
    saved = False
    try:
        if store is not None and request_id:
            store.save(request_id, USER_FEEDBACK, int(value), mask_text(comment, COMMENT_MAX_CHARS) if comment else None, trace_id or None)
            saved = True
    except Exception:  # noqa: BLE001: a broken local store must not stop the forwarding
        saved = False
    forwarded = submit_feedback(trace_id, value, comment, client_factory=client_factory) if trace_id else False
    if forwarded and saved and store is not None and request_id:  # saved implies both; stated for the type checker
        try:
            store.mark_forwarded(request_id, USER_FEEDBACK)
        except Exception:  # noqa: BLE001
            pass
    return FeedbackResult(saved, forwarded)
