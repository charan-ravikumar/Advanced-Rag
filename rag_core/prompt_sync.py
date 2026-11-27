"""Compare the prompts in Langfuse with the templates in code, and seed Langfuse from them. Used by scripts/seed_prompts.py and
scripts/check_prompt_drift.py; the logic lives here so it is tested with a fake client.

States per prompt name: ``missing`` (no production version in Langfuse), ``identical`` (same template text, by sha256 over role and content),
``differs``, ``unreadable`` (the fetched prompt is not a list of plain messages), ``error`` (the fetch failed: network, auth). Seeding creates a
NEW version (label ``production``) only for ``missing``, and for ``differs`` only when the caller passes ``update_differing=True``; it never
edits or deletes an existing version, and running it twice creates nothing the second time (idempotent by hash).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_core.prompts import PROMPT_NAMES, local_template, template_sha256

COMMIT_MESSAGE = "seeded from rag_core/prompts.py (PROMPT_VERSION {version})"


@dataclass
class PromptStatus:
    name: str
    state: str  # missing | identical | differs | unreadable | error
    local_sha256: str
    remote_sha256: str | None = None
    remote_version: int | None = None
    detail: str | None = None
    action: str | None = None  # "created" | "skipped" | None (nothing done)


def _is_not_found(exc: Exception) -> bool:
    return type(exc).__name__ in {"NotFoundError"} or getattr(exc, "status_code", None) == 404


def compare_prompt(client: Any, name: str, *, label: str = "production") -> PromptStatus:
    mode = next(m for m, n in PROMPT_NAMES.items() if n == name)
    local = local_template(mode)
    status = PromptStatus(name=name, state="error", local_sha256=template_sha256(local))
    try:
        remote = client.get_prompt(name, type="chat", label=label, cache_ttl_seconds=0, max_retries=0, fetch_timeout_seconds=10)
    except Exception as exc:  # noqa: BLE001
        if _is_not_found(exc):
            status.state = "missing"
        else:
            status.detail = f"fetch failed ({type(exc).__name__})"
        return status
    status.remote_version = getattr(remote, "version", None)
    if getattr(remote, "is_fallback", False):
        status.detail = "the SDK returned its fallback, not a real prompt"
        return status
    try:
        plain = [{"role": m["role"], "content": m["content"]} for m in remote.prompt]
        status.remote_sha256 = template_sha256(plain)
    except Exception:  # noqa: BLE001
        status.state, status.detail = "unreadable", "the production prompt is not a list of plain messages"
        return status
    status.state = "identical" if status.remote_sha256 == status.local_sha256 else "differs"
    return status


def compare_all(client: Any) -> list[PromptStatus]:
    return [compare_prompt(client, name) for name in PROMPT_NAMES.values()]


def seed(client: Any, *, apply: bool, update_differing: bool = False) -> list[PromptStatus]:
    """Dry run unless ``apply``. See the module docstring for what is created."""
    from rag_core.prompts import PROMPT_VERSION

    results = compare_all(client)
    for status in results:
        wants_create = status.state == "missing" or (status.state == "differs" and update_differing)
        if not wants_create:
            status.action = "skipped" if status.state in {"differs", "unreadable", "error"} else None
            continue
        if not apply:
            continue
        mode = next(m for m, n in PROMPT_NAMES.items() if n == status.name)
        client.create_prompt(name=status.name, prompt=[dict(m) for m in local_template(mode)], labels=["production"], type="chat",
                             commit_message=COMMIT_MESSAGE.format(version=PROMPT_VERSION))
        status.action = "created"
    return results
