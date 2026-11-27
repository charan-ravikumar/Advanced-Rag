"""Fake providers, clock and retriever shared by the rag_core tests. No network, no real keys, no SDK."""

from __future__ import annotations

import json
from types import SimpleNamespace

from rag_core.errors import ProviderError
from rag_core.gateway import Delta, Final


def ok(*texts: str, finish: str | None = "stop", prompt_tokens: int | None = 11, completion_tokens: int | None = 7) -> list:
    """A successful scripted step: the deltas, then the final usage."""
    return [*texts, Final(finish, prompt_tokens, completion_tokens)]


def default_quote(label: str) -> str:
    """Words that ARE in the default passage of ``make_hit`` (so the quote is verified)."""
    return f"Passage number {label[1:]} says something"


def answer_json(*claims, status: str = "answered") -> str:
    """A structured long answer as the model would write it. Each claim is ``(text, cites)``; a cite is a label ("S1", quoted with
    ``default_quote``) or a ``(label, quote)`` pair."""
    rows = []
    for text, cites in claims:
        rows.append({"text": text, "citations": [{"source": c, "quote": default_quote(c)} if isinstance(c, str) else {"source": c[0], "quote": c[1]}
                                                 for c in cites]})
    return json.dumps({"status": status, "claims": rows})


ABSTAIN_JSON = json.dumps({"status": "insufficient_context", "claims": []})


def fail(kind: str, status: int | None = None, retry_after: float | None = None, name: str | None = None) -> ProviderError:
    return ProviderError(kind, f"fake {kind}", status=status, retry_after_s=retry_after, error_class=name or f"Fake{kind.title()}Error")


class ScriptedProvider:
    """Each call of ``stream`` consumes one step of ``script``. A step is a list whose items are: a str (a delta), an Exception
    (raised at that point), or a ``Final``. When the script is exhausted the last step repeats."""

    def __init__(self, name: str, model: str, script: list[list]):
        self.name, self.model, self.script = name, model, list(script)
        self.calls: list[dict] = []

    def stream(self, messages, *, timeout_s, max_output_tokens, temperature, response_schema=None):
        self.calls.append({"messages": messages, "timeout_s": timeout_s, "max_output_tokens": max_output_tokens, "temperature": temperature,
                           "response_schema": response_schema})
        step = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        for item in step:
            if isinstance(item, BaseException):
                raise item
            if isinstance(item, Final):
                yield item
            else:
                yield Delta(item)


class FakeClock:
    """Monotonic clock whose ``sleep`` advances it, so backoff and cooldowns are instant and exact."""

    def __init__(self):
        self.t = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


def make_hit(i: int, *, doc_id: str = "1234.00001", text: str | None = None, section: str = "Methods"):
    return SimpleNamespace(chunk_id=f"c{i}", doc_id=doc_id, section=section, text=text if text is not None else f"Passage number {i} says something about attention.",
                           rank=i, score=1.0 / i)


class FakeRetriever:
    def __init__(self, hits=None, *, error: Exception | None = None):
        self.hits = list(hits if hits is not None else [make_hit(i) for i in range(1, 9)])
        self.error = error
        self.calls: list[dict] = []
        self.embedder = SimpleNamespace(revision="fake-embedder-rev")

    def retrieve(self, query, **kw):
        self.calls.append({"query": query, **kw})
        if self.error is not None:
            raise self.error
        return SimpleNamespace(hits=self.hits[: kw.get("top_k", 10)] if kw.get("top_k") else self.hits)


class TickingPerf:
    """A perf_counter stand-in that advances 1 ms per reading: every measured latency is positive and ordered, with no sleeping."""

    def __init__(self, step: float = 0.001):
        self.t, self.step = 0.0, step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


# ---- Langfuse fakes (no SDK, no network) ---------------------------------------------------------------------------------------

class FakePrompt:
    """What ``client.get_prompt(...)`` returns for a chat prompt (the attributes rag_core reads)."""

    def __init__(self, messages, version=3, labels=("production",), is_fallback=False):
        self.prompt, self.version, self.labels, self.is_fallback = messages, version, list(labels), is_fallback


def chat_messages(system: str, user: str = "{{sources}}\n<question>\n{{question}}\n</question>") -> list[dict]:
    """A chat prompt as the SDK returns it: typed messages."""
    return [{"type": "message", "role": "system", "content": system}, {"type": "message", "role": "user", "content": user}]


class FakePromptClient:
    """``get_prompt`` records its keyword arguments and returns ``prompt`` or raises ``error``."""

    def __init__(self, prompt: FakePrompt | None = None, error: Exception | None = None):
        self.prompt, self.error, self.calls = prompt, error, []

    def get_prompt(self, name, **kw):
        self.calls.append({"name": name, **kw})
        if self.error is not None:
            raise self.error
        return self.prompt


class FakeObservation:
    """One Langfuse observation: records its constructor arguments, updates, children and whether it ended."""

    def __init__(self, name, as_type, kwargs, parent=None, root=None):
        self.name, self.as_type, self.kwargs, self.parent = name, as_type, kwargs, parent
        self.updates, self.children, self.ended = [], [], False
        self.trace_id = "lf-trace-1" if root is None else root.trace_id
        self._root = root or self

    def start_observation(self, *, name, as_type="span", **kw):
        child = FakeObservation(name, as_type, kw, parent=self, root=self._root)
        self._root.all.append(child) if hasattr(self._root, "all") else None
        self.children.append(child)
        return child

    def update(self, **kw):
        self.updates.append(kw)

    def end(self, **kw):
        self.ended = True


class FakeLangfuse:
    def __init__(self):
        self.roots, self.scores, self.flushes = [], [], 0
        self.flush_blocks = 0.0

    def start_observation(self, *, name, as_type="span", **kw):
        root = FakeObservation(name, as_type, kw)
        root.all = [root]
        self.roots.append(root)
        return root

    def create_score(self, **kw):
        self.scores.append(kw)

    def flush(self):
        import time as _time

        _time.sleep(self.flush_blocks)
        self.flushes += 1


class FakeHttp:
    """Scripted HTTP layer for the observability reader: ``responses`` maps a substring of the query's ``view`` to (status, body)."""

    def __init__(self, responder):
        self.responder, self.calls = responder, []

    def __call__(self, url, params, auth, timeout_s):
        import json as _json

        query = _json.loads(params["query"])
        self.calls.append({"url": url, "query": query, "auth": auth, "timeout": timeout_s})
        out = self.responder(query)
        if isinstance(out, Exception):
            raise out
        return out
