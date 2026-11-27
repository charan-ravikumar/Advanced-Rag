"""Query router, rule part only. Pure: no LLM, no clients, no logging, no I/O.

``classify_by_rules`` returns ``"direct"`` ONLY when the whole message is an allowlisted conversational phrase ("hi", "thanks",
"good morning", ...). Everything else is a research query. The v1 rule "at most 3 words and no question mark means small talk"
was removed: it sent keyword queries such as "attention mechanism" or "BERT" to the small-talk path, and the v1 prefix patterns
also matched "Hello, how does BERT work?" as small talk. Wrongly skipping retrieval is the costly error, so the allowlist is
conservative: a message must BE a phrase (case, punctuation and extra spaces ignored), not merely start with one.
"""

from __future__ import annotations

import re

CONVERSATIONAL_PHRASES = frozenset({
    "hi", "hello", "hey", "hiya", "howdy", "yo", "hi there", "hello there", "hey there",
    "good morning", "good afternoon", "good evening", "good day", "good night",
    "how are you", "how are you doing", "how is it going", "how's it going", "what's up", "whats up", "sup",
    "thanks", "thank you", "thanks a lot", "thank you very much", "many thanks", "thx", "ty", "cheers",
    "bye", "goodbye", "see you", "see you later", "see ya", "cya", "take care",
    "who are you", "what are you", "what can you do", "what do you do",
    "help", "help me",
    "ok", "okay", "sure", "yes", "no", "yep", "nope", "cool", "great", "nice", "got it", "i see",
})

_NOT_WORD = re.compile(r"[^a-z0-9' ]+")


def normalise_phrase(text: str) -> str:
    """Lower case, curly apostrophes straightened, punctuation removed, whitespace collapsed."""
    return " ".join(_NOT_WORD.sub(" ", text.lower().replace("\u2019", "'")).split())


def classify_by_rules(query: str) -> str | None:
    """``"direct"`` if the whole message is an allowlisted conversational phrase, otherwise ``None`` (undecided)."""
    return "direct" if normalise_phrase(query) in CONVERSATIONAL_PHRASES else None


def route_query(query: str) -> str:
    """``"direct"`` or ``"rag"`` using the rules only; undecided queries are retrieved."""
    return classify_by_rules(query) or "rag"
