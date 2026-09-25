"""Shadow mode: run a ThinkLess engine next to an existing system without changing what it returns."""

from __future__ import annotations

import contextvars
import functools
import hashlib
import inspect
import json
import logging
import random
import threading
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from .._json import jsonable
from ..decision import Decision
from ..engine import Engine
from ..questions import Kind, Question
from ..tracing.tracer import REDACTED
from .compare import agree

__all__ = ["SCHEMA_VERSION", "Shadow", "ShadowLog", "ShadowStats", "side_from_decision"]

logger = logging.getLogger("thinkless.shadow")

SCHEMA_VERSION = 1

F = TypeVar("F", bound=Callable[..., Any])
CostArg = float | Callable[[Any], float | None] | None


class ShadowStats(BaseModel):
    """Counters for one :class:`Shadow`.

    Attributes:
        seen: Calls that reached the shadow.
        sampled: Calls picked for a shadow run.
        recorded: Shadow runs written to the log.
        skipped: Calls left out by sampling.
        dropped: Calls dropped because ``max_pending`` runs were already queued.
        capped: Calls left out because the shadow spend reached ``max_cost_usd``.
        errors: Shadow runs that raised. They are logged, never re-raised.
        spent_usd: What the shadow engine has cost so far.
    """

    seen: int = 0
    sampled: int = 0
    recorded: int = 0
    skipped: int = 0
    dropped: int = 0
    capped: int = 0
    errors: int = 0
    spent_usd: float = 0.0


class ShadowLog:
    """Appends shadow records to a JSONL file, one per line, safe across threads."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._file = self.path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.close()


def side_from_decision(decision: Decision, *, source: str = "engine") -> dict[str, Any]:
    """The part of a shadow record that describes one ThinkLess decision."""
    return {
        "source": source,
        "value": jsonable(decision.value),
        "level": decision.level,
        "status": decision.status.value,
        "plane": None if decision.plane is None else decision.plane.value,
        "provider": decision.provider,
        "model": decision.model,
        "confidence": decision.confidence,
        "latency_ms": round(decision.latency_ms, 3),
        "cost_usd": decision.cost_usd,
        "tokens": decision.usage.total_tokens,
        "attempts": [
            {
                "provider": a.provider,
                "plane": a.plane.value,
                "value": jsonable(a.value),
                "confidence": a.confidence,
                "accepted": a.accepted,
                "reason": a.reason,
            }
            for a in decision.attempts
        ],
    }


def _comparable_value(question: Question, side: dict[str, Any]) -> Any:
    if question.kind is Kind.SCORE and side.get("level") is not None:
        return side["level"]
    return side.get("value")


def _fingerprint(state: Any) -> str:
    payload = json.dumps(jsonable(state), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _cost(cost_usd: CostArg, result: Any) -> float | None:
    if cost_usd is None:
        return None
    if callable(cost_usd):
        value = cost_usd(result)
        return None if value is None else float(value)
    return float(cost_usd)


class Shadow:
    """Runs a candidate engine on the same inputs as an existing system.

    The existing system (the *primary*) keeps answering: its result is
    returned unchanged, its exceptions propagate, and by default the candidate
    runs on background threads so it adds no latency. Each shadow run appends
    one record to a JSONL log with both answers, whether they agree, and what
    each cost. ``thinkless shadow report`` turns the log into agreement,
    projected savings and a per-question verdict.

    The primary can be anything: a function that calls an LLM, a rules
    engine, a human queue, or another ThinkLess engine. The same class also
    audits a ThinkLess engine in production: make ThinkLess the primary and
    an LLM-only engine the shadow, sampled at a few percent.

    Args:
        engine: The candidate. Give it its own tracer (or none): shadow runs
            are separate traces, marked ``shadow=True``, so they never add to
            the cost of the request they shadow.
        log: A path for the JSONL log, or a :class:`ShadowLog`.
        sample: Share of calls to shadow, between 0 and 1. Shadowing an LLM
            costs what the LLM costs, so sample high-volume traffic.
        background: Run the candidate on worker threads. ``False`` runs it
            inline, which is simpler in tests and batch jobs.
        workers: Worker threads for background runs.
        max_pending: Queued runs beyond this are dropped and counted, so a
            slow candidate can never build an unbounded backlog.
        max_cost_usd: Stop shadowing once the candidate has cost this much.
        capture_content: Store the input in the log. Defaults to the
            candidate tracer's setting. When off, only a fingerprint of the
            input is stored, which still lets you join records to your own
            data.
        seed: Seed for the sampling, for reproducible tests.
        name: Name of the root span of each shadow trace.

    Example:
        >>> shadow = Shadow(candidate, log="shadow/intent.jsonl", sample=0.2)
        >>> @shadow.watch(INTENT)
        ... def classify(ticket: str) -> str:
        ...     return call_existing_llm(ticket)
    """

    def __init__(
        self,
        engine: Engine,
        *,
        log: str | Path | ShadowLog,
        sample: float = 1.0,
        background: bool = True,
        workers: int = 2,
        max_pending: int = 256,
        max_cost_usd: float | None = None,
        capture_content: bool | None = None,
        seed: int | None = None,
        name: str = "shadow",
    ) -> None:
        if not 0.0 <= sample <= 1.0:
            raise ValueError("sample must be between 0 and 1")
        if workers < 1 or max_pending < 1:
            raise ValueError("workers and max_pending must be at least 1")
        self.engine = engine
        self.log = log if isinstance(log, ShadowLog) else ShadowLog(log)
        self.sample = sample
        self.background = background
        self.max_pending = max_pending
        self.max_cost_usd = max_cost_usd
        self.capture_content = (
            engine.tracer.capture_content if capture_content is None else capture_content
        )
        self.name = name
        self._random = random.Random(seed)
        self._lock = threading.Lock()
        self._stats = ShadowStats()
        self._pending: set[Future[None]] = set()
        self._warned_drop = False
        self._executor = (
            ThreadPoolExecutor(max_workers=workers, thread_name_prefix="thinkless-shadow")
            if background
            else None
        )

    # ------------------------------------------------------------ entry points

    def compare(
        self,
        state: Any,
        question: Question,
        primary: Callable[[], Any] | Engine,
        *,
        name: str | None = None,
        cost_usd: CostArg = None,
        to_value: Callable[[Any], Any] | None = None,
    ) -> Any:
        """Run ``primary``, shadow it, and return the primary's result.

        Args:
            state: The input both systems see.
            question: The question the primary answers.
            primary: A zero-argument callable, or an engine (its
                :class:`Decision` is returned).
            name: Question key. Defaults to ``question.key``.
            cost_usd: What one primary call costs, or a function of its
                result. Leave it out when unknown; the report then shows the
                candidate's cost without a saving.
            to_value: Turns the primary's result into an answer comparable
                with the question, for example ``lambda r: r["intent"]``.
        """
        key = name or question.key
        if isinstance(primary, Engine):
            decision = primary.decide(state, question, name=key)
            self._submit(state, question, key, side_from_decision(decision))
            return decision
        started = time.perf_counter()
        result = primary()
        latency = (time.perf_counter() - started) * 1000
        self._submit(state, question, key, self._side(result, latency, cost_usd, to_value))
        return result

    async def acompare(
        self,
        state: Any,
        question: Question,
        primary: Callable[[], Awaitable[Any]] | Engine,
        *,
        name: str | None = None,
        cost_usd: CostArg = None,
        to_value: Callable[[Any], Any] | None = None,
    ) -> Any:
        """Async form of :meth:`compare` for a coroutine-returning primary."""
        key = name or question.key
        if isinstance(primary, Engine):
            decision = await primary.adecide(state, question, name=key)
            self._submit(state, question, key, side_from_decision(decision))
            return decision
        started = time.perf_counter()
        result = await primary()
        latency = (time.perf_counter() - started) * 1000
        self._submit(state, question, key, self._side(result, latency, cost_usd, to_value))
        return result

    def observe(
        self,
        state: Any,
        question: Question,
        value: Any,
        *,
        name: str | None = None,
        cost_usd: float | None = None,
        latency_ms: float | None = None,
    ) -> None:
        """Shadow an answer the primary already gave.

        Use it where the existing decision happens somewhere you cannot wrap,
        or when replaying logged decisions: pass what was decided, and what it
        cost if you know.
        """
        side: dict[str, Any]
        if isinstance(value, Decision):
            side = side_from_decision(value)
        else:
            side = {
                "source": "observed",
                "value": jsonable(value),
                "latency_ms": latency_ms,
                "cost_usd": cost_usd,
            }
        self._submit(state, question, name or question.key, side)

    def watch(
        self,
        question: Question,
        *,
        state: Callable[..., Any] | None = None,
        name: str | None = None,
        cost_usd: CostArg = None,
        to_value: Callable[[Any], Any] | None = None,
    ) -> Callable[[F], F]:
        """Decorate an existing decision function so every call is shadowed.

        Args:
            question: What the function decides.
            state: Builds the shadow input from the call's arguments.
                Defaults to the first positional argument.
            name: Question key. Defaults to ``question.key``.
            cost_usd: What one call costs, or a function of its result.
            to_value: Turns the function's result into a comparable answer.

        Works on plain and async functions. The function's result and
        exceptions are passed through untouched.
        """

        def pick(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            if state is not None:
                return state(*args, **kwargs)
            if args:
                return args[0]
            if len(kwargs) == 1:
                return next(iter(kwargs.values()))
            raise TypeError("pass state= to watch() for functions called without arguments")

        key = name or question.key

        def decorate(func: F) -> F:
            if inspect.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    started = time.perf_counter()
                    result = await func(*args, **kwargs)
                    latency = (time.perf_counter() - started) * 1000
                    self._submit_safely(
                        pick, args, kwargs, question, key, result, latency, cost_usd, to_value
                    )
                    return result

                return async_wrapper  # type: ignore[return-value]

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                started = time.perf_counter()
                result = func(*args, **kwargs)
                latency = (time.perf_counter() - started) * 1000
                self._submit_safely(
                    pick, args, kwargs, question, key, result, latency, cost_usd, to_value
                )
                return result

            return wrapper  # type: ignore[return-value]

        return decorate

    # -------------------------------------------------------------- lifecycle

    @property
    def stats(self) -> ShadowStats:
        with self._lock:
            return self._stats.model_copy()

    def flush(self, timeout: float | None = None) -> None:
        """Wait for queued shadow runs to finish."""
        with self._lock:
            pending = list(self._pending)
        if pending:
            wait_futures(pending, timeout=timeout)

    def close(self, timeout: float | None = None) -> None:
        """Finish queued runs, stop the workers and close the log."""
        self.flush(timeout)
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        self.log.close()

    def __enter__(self) -> Shadow:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -------------------------------------------------------------- internals

    def _side(
        self,
        result: Any,
        latency_ms: float,
        cost_usd: CostArg,
        to_value: Callable[[Any], Any] | None,
    ) -> dict[str, Any]:
        if isinstance(result, Decision) and to_value is None:
            return side_from_decision(result, source="callable")
        value = to_value(result) if to_value is not None else result
        return {
            "source": "callable",
            "value": jsonable(value),
            "latency_ms": round(latency_ms, 3),
            "cost_usd": _cost(cost_usd, result),
        }

    def _submit_safely(
        self,
        pick: Callable[[tuple[Any, ...], dict[str, Any]], Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        question: Question,
        key: str,
        result: Any,
        latency_ms: float,
        cost_usd: CostArg,
        to_value: Callable[[Any], Any] | None,
    ) -> None:
        # A decorated function must behave exactly as before, so nothing the
        # shadow does on the caller's thread may raise.
        try:
            side = self._side(result, latency_ms, cost_usd, to_value)
            self._submit(pick(args, kwargs), question, key, side)
        except Exception:
            logger.exception("shadow could not record a call to %s", key)
            with self._lock:
                self._stats.errors += 1

    def _submit(self, state: Any, question: Question, key: str, primary: dict[str, Any]) -> None:
        future: Future[None] | None = None
        warn = False
        with self._lock:
            self._stats.seen += 1
            if self.sample < 1.0 and self._random.random() >= self.sample:
                self._stats.skipped += 1
                return
            if self.max_cost_usd is not None and self._stats.spent_usd >= self.max_cost_usd:
                self._stats.capped += 1
                return
            if self._executor is not None and len(self._pending) >= self.max_pending:
                self._stats.dropped += 1
                warn, self._warned_drop = not self._warned_drop, True
            else:
                self._stats.sampled += 1
                if self._executor is not None:
                    # A fresh context keeps the shadow run out of the caller's
                    # trace: no parent span, so it never adds to its cost.
                    future = self._executor.submit(
                        contextvars.Context().run, self._run, state, question, key, primary
                    )
                    self._pending.add(future)
        if warn:
            logger.warning(
                "shadow queue is full (%d pending); dropping calls until it drains",
                self.max_pending,
            )
        if future is not None:
            # Outside the lock: a finished future runs the callback right here.
            future.add_done_callback(self._done)
        elif self._executor is None:
            contextvars.Context().run(self._run, state, question, key, primary)

    def _done(self, future: Future[None]) -> None:
        with self._lock:
            self._pending.discard(future)

    def _run(self, state: Any, question: Question, key: str, primary: dict[str, Any]) -> None:
        record: dict[str, Any] = {
            "v": SCHEMA_VERSION,
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "question": key,
            "kind": question.kind.value,
            "spec": question.spec(),
            "state": jsonable(state) if self.capture_content else REDACTED,
            "state_sha": _fingerprint(state),
            "primary": primary,
        }
        try:
            with self.engine.run(self.name, shadow=True, question=key):
                decision = self.engine.decide(state, question, name=key)
        except Exception as exc:
            logger.warning("shadow run for %s failed: %s", key, exc)
            record["shadow"] = {"source": "engine", "error": f"{type(exc).__name__}: {exc}"}
            record["agree"] = None
            with self._lock:
                self._stats.errors += 1
            self._write(record)
            return
        side = side_from_decision(decision)
        agreed, fields = agree(
            question, _comparable_value(question, primary), _comparable_value(question, side)
        )
        if not self.capture_content and question.kind is Kind.EXTRACT:
            # Extracted values are user content; keep only whether they agree.
            primary["value"] = REDACTED if primary.get("value") is not None else None
            side["value"] = REDACTED if side.get("value") is not None else None
            for attempt in side["attempts"]:
                attempt["value"] = REDACTED if attempt["value"] is not None else None
        record["shadow"] = side
        record["agree"] = agreed
        if fields is not None:
            record["fields"] = fields
        with self._lock:
            self._stats.spent_usd += decision.cost_usd
            self._stats.recorded += 1
        self._write(record)

    def _write(self, record: dict[str, Any]) -> None:
        try:
            self.log.write(record)
        except Exception:
            logger.exception("shadow log write failed")
