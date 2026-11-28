"""Concurrency of retrieval: bounded worker pools and the queue limit.

* ``RetrievalExecutors``: three thread pools with a FIXED number of workers: ``embed`` and ``rerank`` (CPU-bound model work) and ``db``
  (SQLite and Chroma reads). A task runs in a copy of the submitter's ``contextvars`` context, so the OpenTelemetry context (and any other
  context variable) is the same inside the task as where the work was requested.
* ``run_chain``: stages that must run one after the other, each on its own pool, as one future (the vector leg: embed, then search). A
  stage is submitted only when the one before it finishes, so no worker ever waits for another worker (no deadlock at any pool size).
* ``AdmissionGate``: the queue limit. At most ``limit`` requests may be in retrieval (running or waiting for a worker) at once; the next
  one is refused at once (the API answers 503). This is what bounds the pools' queues.

Import-safe: nothing is created at import.
"""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

POOLS = ("embed", "db", "rerank")
Stage = tuple[str, Callable[[Any], Any]]  # (pool name, function of the previous stage's result; the first stage gets None)


class RetrievalExecutors:
    def __init__(self, *, embed_workers: int = 1, db_workers: int = 4, rerank_workers: int = 1):
        sizes = {"embed": embed_workers, "db": db_workers, "rerank": rerank_workers}
        if min(sizes.values()) < 1:
            raise ValueError(f"every pool needs at least one worker: {sizes}")
        self.workers = sizes
        self._pools = {name: ThreadPoolExecutor(max_workers=n, thread_name_prefix=f"rag-{name}") for name, n in sizes.items()}

    def submit(self, pool: str, fn: Callable[..., Any], *args: Any, context: contextvars.Context | None = None) -> Future:
        ctx = contextvars.copy_context() if context is None else context.copy()  # one copy per task: a Context cannot be entered twice at once
        return self._pools[pool].submit(ctx.run, fn, *args)

    def shutdown(self) -> None:
        for pool in self._pools.values():
            pool.shutdown(wait=True, cancel_futures=True)


def run_chain(executors: RetrievalExecutors, stages: Sequence[Stage]) -> Future:
    """One future for ``stages`` run in order; its result is the last stage's. The first failure ends the chain with that exception."""
    context = contextvars.copy_context()  # the caller's context: later stages are submitted from worker threads
    done: Future = Future()
    done.set_running_or_notify_cancel()  # it cannot be cancelled, so setting its result can never fail

    def start(i: int, value: Any) -> None:
        pool, fn = stages[i]
        try:
            step = executors.submit(pool, fn, value, context=context)
        except RuntimeError as exc:  # the pools were shut down
            done.set_exception(exc)
            return
        step.add_done_callback(lambda finished: advance(i, finished))

    def advance(i: int, finished: Future) -> None:
        if finished.cancelled():
            done.set_exception(RuntimeError("retrieval was cancelled: the server is shutting down"))
        elif finished.exception() is not None:
            done.set_exception(finished.exception())
        elif i + 1 < len(stages):
            start(i + 1, finished.result())
        else:
            done.set_result(finished.result())

    start(0, None)
    return done


def set_torch_threads(threads: int) -> None:
    """The threads PyTorch uses inside one model call. PROCESS-WIDE (embedder and reranker alike); 0 keeps PyTorch's default."""
    if threads > 0:
        import torch

        torch.set_num_threads(threads)


class Ticket:
    """One admitted request. ``release`` gives the place back; calling it again does nothing."""

    def __init__(self, gate: AdmissionGate):
        self._gate: AdmissionGate | None = gate

    def release(self) -> None:
        gate, self._gate = self._gate, None
        if gate is not None:
            gate._release()


class AdmissionGate:
    def __init__(self, limit: int):
        if limit < 1:
            raise ValueError("the admission limit must be at least 1")
        self.limit, self._active, self._refused, self._lock = int(limit), 0, 0, threading.Lock()

    def try_acquire(self) -> Ticket | None:
        with self._lock:
            if self._active >= self.limit:
                self._refused += 1
                return None
            self._active += 1
        return Ticket(self)

    def _release(self) -> None:
        with self._lock:
            self._active -= 1

    @property
    def active(self) -> int:
        return self._active

    @property
    def refused(self) -> int:
        return self._refused
