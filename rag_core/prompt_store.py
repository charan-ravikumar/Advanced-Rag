"""Prompt resolution: Langfuse Prompt Management with a safe local fallback.

``resolve(name)`` returns a ``ResolvedPrompt`` and NEVER raises or blocks answering:

* ``LocalPromptStore``: the templates in ``rag_core/prompts.py`` (the source of truth; tests and evaluation use it). ``source="local"``.
* ``LangfusePromptStore``: fetches the chat prompt (label ``production``, or an exact pinned version) with a short timeout and the SDK's TTL cache,
  VALIDATES it (a prompt edited in the Langfuse UI bypasses code review and tests: exact variables, the abstention and citation instructions
  present, valid roles, no variables in the system message, a length cap) and returns it with ``source="langfuse"``. A fetch error, a missing key
  or an invalid prompt returns the local template with ``source="local_fallback"`` and a ``fallback_reason``.

Escaping of source and question text stays in code (``prompts.compile_messages``) and happens before the variables are filled in. Import-safe:
the Langfuse SDK is imported lazily and only by ``langfuse_client.get_langfuse_client``; ``mode="local"`` never imports it.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from rag_core.config import PromptsConfig
from rag_core.prompts import PROMPT_NAMES, PROMPT_VERSION, local_template, template_sha256, validate_template
from rag_core.schemas import PromptInfo

NAME_TO_MODE = {name: mode for mode, name in PROMPT_NAMES.items()}


@dataclass
class ResolvedPrompt:
    messages: list[dict]  # the TEMPLATE (variables not yet filled in): [{"role", "content"}, ...]
    name: str
    version: int | str | None  # Langfuse version number, or PROMPT_VERSION for the local template
    label: str | None
    source: str  # "langfuse" | "local" | "local_fallback"
    sha256: str  # of the template text
    fallback_reason: str | None = None
    fallback_kind: str | None = None  # "unavailable" | "fetch_error" | "invalid" (only for local_fallback)
    client: Any = field(default=None, repr=False, compare=False)  # the SDK prompt object, to link a generation to it (only for source="langfuse")

    def info(self) -> PromptInfo:
        return PromptInfo(name=self.name, version=None if self.version is None else str(self.version), label=self.label, source=self.source,
                          sha256=self.sha256, fallback_reason=self.fallback_reason)

    @property
    def cache_id(self) -> str:
        """What the answer cache keys on: name, version and the hash of the template text (so a silent UI edit changes the key)."""
        return f"{self.name}|{self.version}|{self.sha256}"


class PromptStore(Protocol):
    def resolve(self, name: str) -> ResolvedPrompt: ...


class LocalPromptStore:
    """The templates in code. ``templates`` can override some of them (tests)."""

    def __init__(self, templates: dict[str, list[dict]] | None = None):
        self._templates = dict(templates or {})

    def resolve(self, name: str) -> ResolvedPrompt:
        if name not in NAME_TO_MODE:
            raise ValueError(f"unknown prompt name {name!r}; choose from {sorted(NAME_TO_MODE)}")
        messages = [dict(m) for m in self._templates.get(name, local_template(NAME_TO_MODE[name]))]
        return ResolvedPrompt(messages=messages, name=name, version=PROMPT_VERSION, label=None, source="local", sha256=template_sha256(messages))


class LangfusePromptStore:
    def __init__(self, cfg: PromptsConfig | None = None, *, client_factory: Callable[[], Any] | None = None, local: LocalPromptStore | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.cfg = cfg or PromptsConfig(mode="langfuse")
        self._factory, self._local, self._clock = client_factory, local or LocalPromptStore(), clock
        self._failed_until: dict[str, tuple[float, str]] = {}  # prompt name -> (monotonic time until which not to ask again, fallback reason)
        self._lock = threading.Lock()
        for name, version in self.cfg.pin.items():
            if name not in NAME_TO_MODE or not isinstance(version, int) or isinstance(version, bool) or version < 1:
                raise ValueError(f"prompts.pin must map a known prompt name to a positive integer version; got {name!r}: {version!r}")

    def _client(self):
        if self._factory is not None:
            return self._factory()
        from rag_core.langfuse_client import get_langfuse_client

        return get_langfuse_client()

    def _fallback(self, name: str, kind: str, reason: str) -> ResolvedPrompt:
        base = self._local.resolve(name)
        base.source, base.fallback_kind, base.fallback_reason = "local_fallback", kind, reason[:400]
        return base

    def _remember_failure(self, name: str, reason: str) -> ResolvedPrompt:
        if self.cfg.failure_cache_seconds > 0:
            with self._lock:
                self._failed_until[name] = (self._clock() + self.cfg.failure_cache_seconds, reason)
        return self._fallback(name, "fetch_error", reason)

    def resolve(self, name: str) -> ResolvedPrompt:
        if name not in NAME_TO_MODE:  # a programming error, not an outage: raise
            raise ValueError(f"unknown prompt name {name!r}; choose from {sorted(NAME_TO_MODE)}")
        try:
            return self._resolve(name)
        except Exception as exc:  # noqa: BLE001: fail open, whatever happens
            return self._fallback(name, "fetch_error", f"internal:{type(exc).__name__}")

    def _resolve(self, name: str) -> ResolvedPrompt:
        try:
            client = self._client()
        except Exception as exc:  # noqa: BLE001
            return self._fallback(name, "unavailable", f"client_unavailable:{type(exc).__name__}")
        if client is None:
            return self._fallback(name, "unavailable", "no_keys")
        pinned = self.cfg.pin.get(name)
        kwargs = {"version": pinned} if pinned is not None else {"label": "production"}
        with self._lock:
            remembered = self._failed_until.get(name)
        if remembered is not None:
            if self._clock() < remembered[0]:  # a recent fetch failed: do not make this request wait for the timeout again
                return self._fallback(name, "fetch_error", remembered[1])
            with self._lock:
                self._failed_until.pop(name, None)
        try:
            prompt = client.get_prompt(name, type="chat", cache_ttl_seconds=int(self.cfg.cache_ttl_seconds),
                                       fetch_timeout_seconds=max(1, math.ceil(self.cfg.fetch_timeout_seconds)), max_retries=0, **kwargs)
        except Exception as exc:  # noqa: BLE001: a slow or unreachable Langfuse must never block an answer
            return self._remember_failure(name, f"fetch_failed:{type(exc).__name__}")
        if getattr(prompt, "is_fallback", False):
            return self._remember_failure(name, "fetch_failed:sdk_fallback")
        try:
            messages = [dict(m) for m in prompt.prompt]
        except Exception:  # noqa: BLE001
            return self._fallback(name, "invalid", "invalid:the prompt could not be read as a list of messages")
        problems = validate_template(name, messages, max_chars=self.cfg.max_chars)
        if problems:
            return self._fallback(name, "invalid", "invalid:" + "; ".join(problems))
        plain = [{"role": m["role"], "content": m["content"]} for m in messages]
        return ResolvedPrompt(messages=plain, name=name, version=getattr(prompt, "version", None), label=None if pinned is not None else "production",
                              source="langfuse", sha256=template_sha256(plain), client=prompt)


def build_prompt_store(cfg: PromptsConfig, *, client_factory: Callable[[], Any] | None = None) -> PromptStore:
    """``mode="local"`` (tests, evaluation) -> ``LocalPromptStore`` (no SDK import); ``mode="langfuse"`` (the app) -> ``LangfusePromptStore``."""
    if cfg.mode == "local":
        return LocalPromptStore()
    if cfg.mode == "langfuse":
        return LangfusePromptStore(cfg, client_factory=client_factory)
    raise ValueError(f"prompts.mode must be 'langfuse' or 'local', got {cfg.mode!r}")


def describe_prompts(store: PromptStore) -> dict:
    """``{name: {name, version, label, source, sha256, fallback_reason}}`` for both prompts: what an evaluation run records in its config.json."""
    return {name: store.resolve(name).info().model_dump() for name in PROMPT_NAMES.values()}
