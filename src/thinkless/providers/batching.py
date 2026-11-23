"""Cross-request batching: concurrent callers share a local model's forward passes."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from typing import Any

from ..errors import ConfigurationError
from ..logs import get_logger
from ..questions import Question
from .base import BatchItem, DecisionProvider, ProviderResult, State

__all__ = ["Batched"]

logger = get_logger("providers.batched")

_STOP = object()


class Batched(DecisionProvider):
    """Collects the calls of concurrent requests and answers them as one batch.

    A decision server answers each request on its own thread. Without this
    wrapper, a local model sees one state per call and a GPU spends most of
    each forward pass idle. ``Batched`` puts every call on a queue; one worker
    thread takes what is waiting, up to ``max_batch`` calls, and asks the
    wrapped provider once through
    :meth:`~DecisionProvider.answer_batch`. Each caller gets its own result.

    With ``max_wait_ms=0`` (the default) a call never waits for company: a
    lone request goes straight through, and calls that arrive while the
    model is busy go together in the next batch. That adds no latency when
    traffic is light and raises throughput when it is not. A positive
    ``max_wait_ms`` holds the first call of a batch that long for others.

    Use it for local models that batch (GLiNER, Laya, ``HFClassifier``,
    ``HFZeroShot``). Hosted providers gain nothing: their calls are already
    concurrent.

    Args:
        provider: The provider to wrap. Its name, plane, kinds and settings
            are kept, so thresholds and traces are unchanged.
        max_batch: Most calls per batch.
        max_wait_ms: How long the first call of a batch waits for others.
        timeout_s: How long a caller waits for its result before giving up.

    Example:
        >>> engine = Engine([Rules(), Batched(GLiNER()), LLMDecider(llm)], llm=llm)
    """

    def __init__(
        self,
        provider: DecisionProvider,
        *,
        max_batch: int = 32,
        max_wait_ms: float = 0.0,
        timeout_s: float = 120.0,
    ) -> None:
        if isinstance(provider, Batched):
            raise ConfigurationError("the provider is already batched")
        if max_batch < 1:
            raise ConfigurationError("max_batch must be at least 1")
        if max_wait_ms < 0 or timeout_s <= 0:
            raise ConfigurationError("max_wait_ms must be >= 0 and timeout_s > 0")
        self.inner = provider
        self.name = provider.name
        self.price_key = provider.price_key
        # Instance attributes shadow the class defaults with the wrapped provider's.
        self.plane = provider.plane  # type: ignore[misc]
        self.kinds = provider.kinds  # type: ignore[misc]
        self.calibrated = provider.calibrated  # type: ignore[misc]
        self.max_batch = max_batch
        self.max_wait_ms = max_wait_ms
        self.timeout_s = timeout_s
        self.batches_run = 0
        self.calls_batched = 0
        self._queue: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self._worker: threading.Thread | None = None
        self._lock = threading.Lock()
        self._closed = False

    # The engine passes settled decisions only to providers that take them,
    # and a batch has no single context to pass.
    accepts_context = False

    @property
    def batches(self) -> bool:
        return True

    def supports(self, question: Question) -> bool:
        return self.inner.supports(question)

    def warmup(self) -> None:
        self.inner.warmup()

    def answer(self, state: State, questions: Mapping[str, Question]) -> ProviderResult:
        future: Future[ProviderResult] = Future()
        with self._lock:
            if self._closed:
                raise RuntimeError(f"{self.name} is closed")
            self._start()
            self._queue.put((state, questions, future))
        return future.result(timeout=self.timeout_s)

    def answer_batch(self, items: Sequence[BatchItem]) -> list[ProviderResult]:
        # Already a batch (Engine.decide_batch): no queue needed.
        return self.inner.answer_batch(items)

    def _start(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(
                target=self._run, name=f"thinkless-batch-{self.name}", daemon=True
            )
            self._worker.start()

    def _collect(self, first: Any) -> tuple[list[Any], bool]:
        """The first call plus what else is waiting, and whether a stop was seen."""
        pending = [first]
        deadline = time.perf_counter() + self.max_wait_ms / 1000.0
        while len(pending) < self.max_batch:
            remaining = deadline - time.perf_counter()
            try:
                item = (
                    self._queue.get(timeout=remaining)
                    if remaining > 0
                    else self._queue.get_nowait()
                )
            except queue.Empty:
                break
            if item is _STOP:
                return pending, True
            pending.append(item)
        return pending, False

    def _run(self) -> None:
        stopping = False
        while not stopping:
            first = self._queue.get()
            if first is _STOP:
                break
            pending, stopping = self._collect(first)
            live = [p for p in pending if p[2].set_running_or_notify_cancel()]
            if not live:
                continue
            try:
                if len(live) == 1 or not self.inner.batches:
                    for state, questions, future in live:
                        try:
                            future.set_result(self.inner.answer(state, questions))
                        except Exception as exc:
                            future.set_exception(exc)
                else:
                    results = self.inner.answer_batch([(s, q) for s, q, _ in live])
                    if len(results) != len(live):
                        raise RuntimeError(
                            f"{self.name}.answer_batch returned {len(results)} results "
                            f"for {len(live)} calls"
                        )
                    for (_, _, future), result in zip(live, results, strict=True):
                        future.set_result(result)
                    self.batches_run += 1
                    self.calls_batched += len(live)
            except BaseException as exc:  # the worker must never die with callers waiting
                for _, _, future in live:
                    if not future.done():
                        future.set_exception(exc)
                if not isinstance(exc, Exception):
                    raise
        # Calls queued behind the stop are failed rather than left waiting.
        self._fail_waiting(RuntimeError(f"{self.name} is closed"))

    def _fail_waiting(self, error: Exception) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is not _STOP and item[2].set_running_or_notify_cancel():
                item[2].set_exception(error)

    def stop(self, timeout_s: float = 10.0) -> None:
        """Finish the batch in progress, fail queued calls and end the worker thread.

        The wrapped provider stays loaded; :meth:`close` also closes it.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            worker = self._worker
            self._queue.put(_STOP)
        if worker is not None:
            worker.join(timeout_s)
            if worker.is_alive():
                logger.warning(
                    "%s: the batch worker did not stop within %.0f s", self.name, timeout_s
                )
        self._fail_waiting(RuntimeError(f"{self.name} is closed"))

    def close(self) -> None:
        self.stop()
        self.inner.close()

    def __repr__(self) -> str:
        return f"Batched({self.inner!r}, max_batch={self.max_batch})"


def batch_local_providers(
    providers: Sequence[DecisionProvider], **options: Any
) -> list[DecisionProvider]:
    """``providers`` with every local provider that batches wrapped in :class:`Batched`."""
    from ..pricing import LOCAL_PROVIDERS

    return [
        Batched(p, **options)
        if p.batches and p.price_key in LOCAL_PROVIDERS and not isinstance(p, Batched)
        else p
        for p in providers
    ]
