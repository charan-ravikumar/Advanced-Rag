"""Server-sent events: the wire format shared by the server and the client. Pure (no FastAPI, no httpx).

One event is ``event: <name>\\ndata: <one line of JSON>\\n\\n``. Comment lines (``: ...``) are ignored by every SSE parser and are used as a keep-alive / stream-open mark.
"""

from __future__ import annotations

import json
from typing import Iterable, Iterator

from pydantic import BaseModel

from api.schemas import SSE_EVENT_MODELS

OPEN_COMMENT = ": stream open\n\n"
KEEPALIVE_COMMENT = ": keep-alive\n\n"


_LINE_BREAKS = {ord(c): f"\\u{ord(c):04x}" for c in "\x85\u2028\u2029"}  # JSON leaves these raw, but several line splitters treat them as line ends; they only occur inside strings


def encode_event(name: str, model: BaseModel) -> str:
    # model_dump_json escapes \n, \r and the other control characters, and the translation above covers the rest: one data line is always enough
    return f"event: {name}\ndata: {model.model_dump_json().translate(_LINE_BREAKS)}\n\n"


def encode_service_event(event: dict) -> str:
    """A ``RagService`` event dict (``{"type": ..., ...}``) as one SSE event."""
    kind = event["type"]
    model = SSE_EVENT_MODELS[kind]
    payload = {k: v for k, v in event.items() if k != "type"}
    return encode_event(kind, model.model_validate(payload))


def iter_sse(lines: Iterable[str]) -> Iterator[tuple[str, str]]:
    """``(event name, data)`` pairs from the text lines of an SSE stream (``\\n`` already removed). Multi-line data is joined with ``\\n``; comments and unknown
    fields are skipped; an event without ``data`` is not dispatched; a trailing event without its blank line is dispatched at the end of the stream."""
    name, data = "message", []
    for raw in lines:
        line = raw.rstrip("\r")
        if line == "":
            if data:
                yield name, "\n".join(data)
            name, data = "message", []
        elif line.startswith(":"):
            continue
        else:
            field, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if field == "event":
                name = value
            elif field == "data":
                data.append(value)
    if data:
        yield name, "\n".join(data)


def decode_service_event(name: str, data: str) -> dict | None:
    """The inverse of ``encode_service_event``: the ``RagService`` event dict (with typed ``Source``/``ErrorInfo``/``AnswerResult`` objects), or None for an unknown event name
    (a newer server may send events this client does not know: they are skipped, not fatal)."""
    model = SSE_EVENT_MODELS.get(name)
    if model is None:
        return None
    parsed = model.model_validate(json.loads(data))
    return {"type": name, **{field: getattr(parsed, field) for field in type(parsed).model_fields}}
