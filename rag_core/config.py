"""Settings of the generation path. A plain dataclass with defaults; ``from_config`` reads the ``generation`` section of config.yaml
(passed in, never imported here, so importing rag_core reads no file)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from types import SimpleNamespace

from rag_core.context import DEFAULT_MAX_CHUNKS, DEFAULT_MAX_TOKENS


@dataclass
class RagConfig:
    # --- context ---
    top_k: int = DEFAULT_MAX_CHUNKS  # chunks sent to the LLM. DESIGN DEFAULT, NOT TUNED (see rag_core/context.py)
    max_context_tokens: int = DEFAULT_MAX_TOKENS
    # --- retrieval (the frozen configuration D: hybrid RRF k=60, weights 1.0 / 0.7, no reranker by default) ---
    retrieval_mode: str = "hybrid"
    fusion: str = "rrf"
    candidate_k: int = 20
    rerank_k: int = 20  # used only when a call asks for rerank=True
    rrf_k: int = 60
    vector_weight: float = 1.0
    lexical_weight: float = 0.7
    # --- generation ---
    max_question_chars: int = 500
    temperature: float = 0.0
    # Reasoning tokens count toward the cap (they are reported inside completion_tokens), so the caps leave room for reasoning plus the answer.
    # Design defaults, not tuned. A reply that still stops with finish_reason "length" is reported as such, never silently re-run.
    max_output_tokens: int = 1024
    short_max_output_tokens: int = 512
    # --- gateway ---
    timeout_s: float = 30.0
    max_attempts: int = 3
    breaker_threshold: int = 3
    breaker_cooldown_s: float = 30.0
    # --- cache ---
    cache_max_entries: int = 256
    cache_ttl_s: float = 3600.0
    # provider order and model ids (the first is the primary). Gemini only: no generation failover.
    providers: list[str] = field(default_factory=lambda: ["gemini"])
    gemini_model: str = "gemini-3.5-flash-lite"
    # Gemini thinking budget in tokens for the answer path (0 = off, -1 = automatic, None = do not send, the model default). Thinking tokens count
    # toward max_output_tokens and are reported inside completion_tokens. A design default, not tuned.
    gemini_thinking_budget: int | None = None
    # Kept only so results, cache keys and the UI keep one schema: Gemini has no reasoning-effort parameter, so it stays None.
    reasoning_effort: str | None = None
    # the OFFLINE evaluation judge (a later phase): never used on the answer path
    judge: dict = field(default_factory=lambda: {"provider": "gemini", "model": "gemini-3.8-flash", "temperature": 0.0, "thinking_budget": 1024,
                                                 "max_output_tokens": 4096})
    # per MILLION tokens, {"<model id>": {"input": x, "output": y}}. Empty means no cost is ever computed (never invent a price).
    prices: dict = field(default_factory=dict)

    def retrieval_settings(self, rerank: bool = False) -> dict:
        """What affects WHICH passages the model sees (part of the cache key)."""
        return {"mode": self.retrieval_mode, "fusion": self.fusion, "candidate_k": self.candidate_k, "rerank": rerank,
                "rerank_k": self.rerank_k if rerank else None, "rrf_k": self.rrf_k, "vector_weight": self.vector_weight,
                "lexical_weight": self.lexical_weight, "top_k": self.top_k, "max_context_tokens": self.max_context_tokens}

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_config(cls, cfg) -> RagConfig:
        """Overlay ``cfg.generation`` (a SimpleNamespace from config.py) on the defaults, then read the retrieval knobs from ``cfg.pipeline``
        (their ONE home: a retrieval knob under ``generation`` is an error). Unknown keys are an error."""
        generation = getattr(cfg, "generation", None)
        misplaced = sorted(set(vars(generation)) & _RETRIEVAL_FIELD_NAMES) if generation is not None else []
        if misplaced:
            raise ValueError(f"config section 'generation': {misplaced} are retrieval settings; set them under 'pipeline' (one source per knob)")
        rc = _overlay(cls(), generation, "generation")
        pipeline = getattr(cfg, "pipeline", None)
        for key, field_name in PIPELINE_KEYS.items():
            if pipeline is not None and hasattr(pipeline, key):
                setattr(rc, field_name, getattr(pipeline, key))
        return rc


# config.yaml ``pipeline`` key -> RagConfig field (and, by the same keys, the defaults of rag/retriever.py::Retriever)
PIPELINE_KEYS = {"retrieval": "retrieval_mode", "fusion": "fusion", "top_k": "top_k", "candidate_k": "candidate_k", "rerank_k": "rerank_k", "rrf_k": "rrf_k",
                 "vector_weight": "vector_weight", "bm25_weight": "lexical_weight"}
# the names a ``generation`` section must NOT carry: the RagConfig field names and the pipeline key names of the same knobs
_RETRIEVAL_FIELD_NAMES = set(PIPELINE_KEYS) | set(PIPELINE_KEYS.values())


def _plain(value):
    """SimpleNamespace trees (what config.py builds from YAML) back to dicts/lists."""
    if isinstance(value, SimpleNamespace):
        return {k: _plain(v) for k, v in vars(value).items()}
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def _overlay(base, section, name: str = "config"):
    """Set the fields of the dataclass ``base`` from the namespace ``section``. A key that is not a field is an error (a typo must not silently
    leave the default in place)."""
    if section is None:
        return base
    known = set(asdict(base))
    unknown = sorted(set(vars(section)) - known)
    if unknown:
        raise ValueError(f"config section {name!r}: unknown key(s) {unknown}; allowed keys are {sorted(known)}")
    for key in known:
        if hasattr(section, key):
            setattr(base, key, _plain(getattr(section, key)))
    return base


@dataclass
class PromptsConfig:
    """``prompts`` section of config.yaml (rag_core/prompt_store.py)."""

    mode: str = "local"  # "langfuse" (the app sets it) | "local" (tests and evaluation)
    pin: dict = field(default_factory=dict)  # {prompt name: exact Langfuse version}
    cache_ttl_seconds: int = 60
    fetch_timeout_seconds: float = 2.0
    failure_cache_seconds: float = 60.0  # after a failed fetch, serve the local template for this long without calling Langfuse again (0 = off)
    max_chars: int = 20000

    @classmethod
    def from_config(cls, cfg) -> PromptsConfig:
        return _overlay(cls(), getattr(cfg, "prompts", None), "prompts")


@dataclass
class TracingConfig:
    """``tracing`` section of config.yaml (rag_core/tracing.py). The default is OFF so that nothing traces unless asked."""

    enabled: bool = False
    sample_rate: float = 1.0
    daily_request_cap: int = 150
    capture: str = "masked"  # masked | full | none
    trace_evaluation: bool = False
    flush_timeout_seconds: float = 3.0
    # where the OpenTelemetry spans go (rag_core/otel_tracing.py): "langfuse" (needs both Langfuse keys) and/or "otlp" (any OTLP/HTTP backend,
    # configured by the standard OTEL_EXPORTER_OTLP_* variables, which the OpenTelemetry SDK reads itself)
    exporters: list = field(default_factory=lambda: ["langfuse"])

    @classmethod
    def from_config(cls, cfg) -> TracingConfig:
        return _overlay(cls(), getattr(cfg, "tracing", None), "tracing")
