"""Structured long answers: the response schema, the parser (also for a stream that is still arriving), quote verification, rendering. Pure.

The long answer is a JSON object constrained by a PER-REQUEST response schema (Gemini ``response_json_schema``)::

    {"status": "answered" | "insufficient_context",
     "claims": [{"text": "one statement", "citations": [{"source": "S1", "quote": "words copied from S1"}]}]}

``source`` is an enum of exactly the labels that were sent, so a constrained model cannot name a source that does not exist (the code still
checks, because a provider may ignore the schema). Each quote is looked up in its source's text: the prompt escaped ``&``, ``<`` and ``>``, so
the quote is unescaped first, and runs of whitespace count as one space. Found -> ``start``/``end`` character offsets into ``Source.text`` and
``verified=True``; not found (paraphrased, stitched together, invented) or shorter than ``MIN_QUOTE_WORDS`` -> ``verified=False``.

What this CANNOT do: a verified quote proves that the words are in the passage, not that they support the claim. That is what a faithfulness
evaluation is for.

Failure handling: invalid JSON at the end of a stream (the output limit cut it, or the provider ignored the schema) is parsed as far as it goes
(``complete=False``); text that is not a JSON object at all gives no claims and ``complete=False``. Malformed items (a claim without text, a
citation without a source) are skipped and counted, never guessed.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic_core import from_json

from rag_core.prompts import STATUS_ANSWERED, STATUS_INSUFFICIENT
from rag_core.schemas import Claim, SpanCitation

SCHEMA_VERSION = "claims-1"  # bump when the response schema changes (part of the cache key; recorded in evaluation runs)
MIN_QUOTE_WORDS = 3  # a design default, not tuned: a one- or two-word "quote" ("BERT") is found almost anywhere and proves nothing
MAX_CLAIMS = 10  # hard bound in the schema; the prompt asks for at most five


def answer_schema(labels: Sequence[str]) -> dict:
    """The JSON schema of one long answer for the sources ``labels`` (non-empty: the service never asks the model without sources)."""
    if not labels:
        raise ValueError("a structured answer needs at least one source label")
    citation = {"type": "object", "properties": {"source": {"type": "string", "enum": list(labels)}, "quote": {"type": "string"}},
                "required": ["source", "quote"]}
    claim = {"type": "object", "properties": {"text": {"type": "string"}, "citations": {"type": "array", "items": citation, "minItems": 1}},
             "required": ["text", "citations"]}
    return {"type": "object",
            "properties": {"status": {"type": "string", "enum": [STATUS_ANSWERED, STATUS_INSUFFICIENT]},
                           "claims": {"type": "array", "items": claim, "maxItems": MAX_CLAIMS}},
            "required": ["status", "claims"]}


# ---- quote location --------------------------------------------------------------------------------------------------------------
def _collapse(text: str) -> tuple[str, list[int]]:
    """``text`` with every run of whitespace replaced by one space (leading/trailing runs dropped), and for each kept character its index in ``text``."""
    out: list[str] = []
    index: list[int] = []
    pending_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            pending_space = bool(out)
            continue
        if pending_space:
            out.append(" ")
            index.append(i - 1)
            pending_space = False
        out.append(ch)
        index.append(i)
    return "".join(out), index


def locate_quote(quote: str, source_text: str) -> tuple[int, int] | None:
    """``(start, end)`` of ``quote`` in ``source_text`` (``source_text[start:end]`` is the matched passage), or None. Case-sensitive; see the module doc."""
    needle = " ".join(html.unescape(quote or "").split())
    if len(needle.split(" ")) < MIN_QUOTE_WORDS:
        return None
    hay, index = _collapse(source_text or "")
    pos = hay.find(needle)
    if pos < 0:
        return None
    return index[pos], index[pos + len(needle) - 1] + 1


# ---- parsing -----------------------------------------------------------------------------------------------------------------------
@dataclass
class ParsedAnswer:
    status: str | None = None
    claims: list[Claim] = field(default_factory=list)
    cited: list[str] = field(default_factory=list)  # labels that were provided, first appearance order, unique
    invalid: list[str] = field(default_factory=list)  # labels that were NOT provided (only possible if the provider ignored the schema)
    complete: bool = False  # the whole text was one valid JSON object
    malformed_items: int = 0

    @property
    def abstained(self) -> bool:
        """The model said the sources are insufficient, or it said "answered" but gave no claim (nothing to show)."""
        return self.status == STATUS_INSUFFICIENT or (self.status == STATUS_ANSWERED and not self.claims)

    @property
    def quotes(self) -> tuple[int, int]:
        """``(verified, total)`` over all citations."""
        spans = [c for claim in self.claims for c in claim.citations]
        return sum(1 for c in spans if c.verified), len(spans)


def _load(text: str) -> tuple[Any, bool]:
    try:
        return from_json(text), True
    except ValueError:
        pass
    try:
        return from_json(text, allow_partial="trailing-strings"), False
    except ValueError:
        return None, False


def parse_answer(text: str, sources: Sequence) -> ParsedAnswer:
    """The structured answer in ``text`` checked against ``sources`` (objects with ``label`` and ``text``). Never raises."""
    by_label = {s.label: s for s in sources}
    obj, complete = _load(text.strip())
    out = ParsedAnswer(complete=complete and isinstance(obj, dict))
    if not isinstance(obj, dict):
        return out
    status = obj.get("status")
    out.status = status if status in (STATUS_ANSWERED, STATUS_INSUFFICIENT) else None
    raw_claims = obj.get("claims")
    for raw in raw_claims if isinstance(raw_claims, list) else []:
        claim_text = raw.get("text") if isinstance(raw, dict) else None
        if not isinstance(claim_text, str) or not claim_text.strip():
            out.malformed_items += 1
            continue
        spans: list[SpanCitation] = []
        raw_cites = raw.get("citations")
        for cite in raw_cites if isinstance(raw_cites, list) else []:
            label = cite.get("source") if isinstance(cite, dict) else None
            quote = cite.get("quote") if isinstance(cite, dict) else None
            if not isinstance(label, str) or not isinstance(quote, str):
                out.malformed_items += 1
                continue
            source = by_label.get(label)
            found = locate_quote(quote, source.text) if source is not None else None
            spans.append(SpanCitation(source=label, quote=quote, start=found[0] if found else None, end=found[1] if found else None,
                                      verified=found is not None))
            bucket = out.cited if source is not None else out.invalid
            if label not in bucket:
                bucket.append(label)
        out.claims.append(Claim(text=claim_text.strip(), citations=spans))
    if out.status == STATUS_INSUFFICIENT:  # an abstention carries no claims, whatever else the model wrote
        out.claims, out.cited, out.invalid = [], [], []
    return out


# ---- rendering ---------------------------------------------------------------------------------------------------------------------
def _markers(labels: Sequence[str]) -> str:
    seen: list[str] = []
    for label in labels:
        if label not in seen:
            seen.append(label)
    return "".join(f"[{label}]" for label in seen)


def render_answer(claims: Sequence[Claim]) -> str:
    """The answer as one text: each claim followed by its source markers, ``"Claim one [S1]. Claim two [S2][S3]"``. Generated by the code, so the
    markers are always well formed."""
    return _join([(c.text, [x.source for x in c.citations], True) for c in claims])


def _join(parts: Sequence[tuple[str, Sequence[str], bool]]) -> str:
    pieces = []
    for text, labels, closed in parts:
        text = text.strip()
        marks = _markers(labels) if closed else ""
        pieces.append(f"{text} {marks}" if text and marks else (text or marks))
    return " ".join(p for p in pieces if p)


class StreamRenderer:
    """Turns the JSON deltas of a structured answer into display deltas while it streams.

    Nothing is shown until ``status`` is ``answered`` (an abstention is never shown). Each claim's text is shown as it arrives; its markers are
    added once the claim is closed (the next claim began, or the stream ended). The display only ever GROWS: if a re-parse would change text
    that was already shown, nothing more is shown until the end, and the final result (``render_answer``) is what the UI keeps."""

    def __init__(self) -> None:
        self.buffer = ""
        self.shown = ""
        self._stalled = False

    def feed(self, delta: str) -> str:
        self.buffer += delta
        if self._stalled:
            return ""
        try:
            obj = from_json(self.buffer, allow_partial="trailing-strings")
        except ValueError:
            return ""
        if not isinstance(obj, dict) or obj.get("status") != STATUS_ANSWERED:
            return ""
        raw = [c for c in (obj.get("claims") or []) if isinstance(c, dict)] if isinstance(obj.get("claims"), list) else []
        parts: list[tuple[str, list[str], bool]] = []
        for i, claim in enumerate(raw):
            raw_text, raw_cites = claim.get("text"), claim.get("citations")
            text = raw_text if isinstance(raw_text, str) else ""
            cites: list = raw_cites if isinstance(raw_cites, list) else []
            labels = [c["source"] for c in cites if isinstance(c, dict) and isinstance(c.get("source"), str)]
            parts.append((text, labels, i < len(raw) - 1))
        display = _join(parts)
        if not display.startswith(self.shown):
            self._stalled = True
            return ""
        new, self.shown = display[len(self.shown):], display
        return new

    def finish(self, final_answer: str) -> str:
        """What is still missing from the display once the final answer is known ("" if the display cannot be extended to it)."""
        if final_answer.startswith(self.shown):
            new, self.shown = final_answer[len(self.shown):], final_answer
            return new
        return ""
