"""Abstention detection for the SHORT (plain-text) answer. Pure functions.

The long answer no longer has its citations parsed out of prose: it is structured output (claims with source ids and verbatim quotes, see
``rag_core.claims``), and it abstains through its ``status`` field. The regex ``[S#]`` marker parser that used to live here was removed with
that change. The short answer asks for no citations and abstains with the ``INSUFFICIENT_CONTEXT`` sentinel, which is what this module detects.
"""

from __future__ import annotations

import re

from rag_core.prompts import INSUFFICIENT_CONTEXT

_LEADING_NOISE = re.compile(r"^[\s`*_\"'(\[>]+")


def is_abstention(text: str) -> bool:
    """The model replied with the sentinel: exactly, or with surrounding quotes, markdown or punctuation, or followed by an
    explanation (it still signalled that it could not answer). A sentinel buried mid-sentence is NOT an abstention."""
    return _LEADING_NOISE.sub("", text).startswith(INSUFFICIENT_CONTEXT)


def could_become_abstention(prefix: str) -> bool:
    """True while the text received so far is still a proper prefix of the sentinel (or empty): the streaming service holds
    such text back so the UI never flashes ``INSUFFICIENT_CONT`` before it knows what the model is saying."""
    body = _LEADING_NOISE.sub("", prefix)
    return len(body) < len(INSUFFICIENT_CONTEXT) and INSUFFICIENT_CONTEXT.startswith(body)
