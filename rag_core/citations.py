"""Citation validation and abstention detection. Pure functions.

What this does: finds ``[S#]`` markers in the answer and checks each against the labels that were actually sent to the model.
What it CANNOT do: a valid marker only says that the model pointed at a source that exists; it does not say that the source
supports the claim. Whether the claim is supported is what a faithfulness evaluation is for.

Parsing rules (all covered by tests): a marker is ``[S<digits>]``, case-sensitive, optionally several ids in one bracket
(``[S1, S2]``); ``[S1][S2]`` is two markers. The lenticular ``【S1】`` and full-width ``［S1］`` brackets count as brackets too (the
generator was observed to answer with ``【S1】`` at low reasoning effort, which the ASCII-only parser silently read as "no citations").
Malformed forms (``[S]``, ``[s1]``, ``[S 1]``, ``S1``, an unclosed ``[S1``) are not
markers. Markers inside fenced or inline code are ignored (they are code, such as ``a[S1]``, not citations). ``[S0]``/``[S9]``
are markers but name no provided source, so they are reported as invalid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from rag_core.prompts import INSUFFICIENT_CONTEXT

_FENCE = re.compile(r"```.*?```", re.DOTALL)
_OPEN_FENCE = re.compile(r"```.*\Z", re.DOTALL)  # an unterminated fence runs to the end
_INLINE = re.compile(r"`[^`\n]*`")
_MARKER = re.compile(r"[\[\u3010\uff3b]\s*(S\d+(?:\s*,\s*S\d+)*)\s*[\]\u3011\uff3d]")  # [S1], and the lenticular 【S1】 / full-width ［S1］ some models emit
_LEADING_NOISE = re.compile(r"^[\s`*_\"'(\[>]+")


@dataclass
class CitationParse:
    cited: list[str] = field(default_factory=list)  # valid labels, first appearance order, unique
    invalid: list[str] = field(default_factory=list)  # labels that were not provided, first appearance order, unique
    markers: int = 0  # number of marker occurrences outside code


def strip_code(text: str) -> str:
    return _INLINE.sub(" ", _OPEN_FENCE.sub(" ", _FENCE.sub(" ", text)))


def parse_citations(text: str, valid_labels: Iterable[str]) -> CitationParse:
    valid = set(valid_labels)
    out = CitationParse()
    for match in _MARKER.finditer(strip_code(text)):
        for label in re.split(r"\s*,\s*", match.group(1)):
            out.markers += 1
            bucket = out.cited if label in valid else out.invalid
            if label not in bucket:
                bucket.append(label)
    return out


def is_abstention(text: str) -> bool:
    """The model replied with the sentinel: exactly, or with surrounding quotes, markdown or punctuation, or followed by an
    explanation (it still signalled that it could not answer). A sentinel buried mid-sentence is NOT an abstention."""
    return _LEADING_NOISE.sub("", text).startswith(INSUFFICIENT_CONTEXT)


def could_become_abstention(prefix: str) -> bool:
    """True while the text received so far is still a proper prefix of the sentinel (or empty): the streaming service holds
    such text back so the UI never flashes ``INSUFFICIENT_CONT`` before it knows what the model is saying."""
    body = _LEADING_NOISE.sub("", prefix)
    return len(body) < len(INSUFFICIENT_CONTEXT) and INSUFFICIENT_CONTEXT.startswith(body)
