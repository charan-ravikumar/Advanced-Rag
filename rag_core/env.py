"""Environment loading for ENTRY-POINT scripts only (smoke scripts, seeding, the Streamlit app's main).

``load_env()`` reads a ``.env`` file if there is one and puts the ALLOWLISTED variables (``APP_VARIABLES``) into ``os.environ`` WITHOUT
overriding variables that are already set, and returns the NAMES it loaded. It never returns, prints or logs a value. Anything else
in the file (for example the old Chroma Cloud and Redis credentials) is deliberately not loaded into the process.

The ``rag_core`` library modules read ``os.environ`` themselves and never call this function; importing this module does nothing
(``python-dotenv`` is imported inside the function), so the package stays import-safe.

Secrets rule: nobody opens, prints or echoes ``.env``. To check that a key is available use ``present(...)``, which returns booleans.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]

APP_VARIABLES = (
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_BASE_URL",  # the host variable of the current Langfuse SDK (see docs/langfuse_notes.md)
    "LANGFUSE_HOST",  # the name older SDK versions read
)


def load_env(path: str | Path | None = None, names: Iterable[str] = APP_VARIABLES) -> list[str]:
    """Load the variables in ``names`` from ``path`` (default ``<repo>/.env``) without overriding existing ones.

    Returns the sorted NAMES that were found in the file with a non-empty value (empty list if there is no file). Values stay
    inside ``os.environ``; they are never returned."""
    target = Path(path) if path is not None else ROOT / ".env"
    if not target.is_file():
        return []
    from dotenv import dotenv_values

    wanted = set(names)
    found = {k: v for k, v in dotenv_values(target).items() if k in wanted and v}
    for key, value in found.items():
        os.environ.setdefault(key, value)  # never overrides a variable that is already set
    return sorted(found)


def present(*names: str) -> dict[str, bool]:
    """``{name: True/False}``: whether each variable is set to a non-empty value. Booleans only."""
    return {name: bool(os.environ.get(name)) for name in names}
