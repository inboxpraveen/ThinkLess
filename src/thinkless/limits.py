"""Spend limits: stop paying for decisions and generations past a budget."""

from __future__ import annotations

import threading
import time
from collections import deque
from contextvars import ContextVar

from .errors import ThinkLessError

__all__ = ["SpendLimit", "SpendLimitError"]


class SpendLimitError(ThinkLessError):
    """A generation was refused because a spend limit is used up."""


class SpendLimit:
    """A dollar budget for an engine, over its lifetime or a rolling window.

    Once the budget is used up, the engine stops asking paid providers: a
    decision that would have reached a hosted LLM comes back ``uncertain``
    with the best answer the free providers gave, and ``generate`` raises
    :class:`SpendLimitError`. Rules and local models keep answering.

    Args:
        max_usd: The budget.
        window_s: Rolling window in seconds, for example ``3600`` for an
            hourly budget. ``None`` counts everything since the limit was
            created.

    The check happens before each paid call, so the last call before the
    limit can overshoot it by the cost of one call.

    Example:
        >>> engine = Engine(providers, spend_limit=SpendLimit(20.0, window_s=86400))
    """

    def __init__(self, max_usd: float, *, window_s: float | None = None) -> None:
        if max_usd < 0:
            raise ValueError("max_usd must not be negative")
        if window_s is not None and window_s <= 0:
            raise ValueError("window_s must be positive")
        self.max_usd = max_usd
        self.window_s = window_s
        self._lock = threading.Lock()
        self._total = 0.0
        self._events: deque[tuple[float, float]] = deque()

    def _expire(self, now: float) -> None:
        if self.window_s is None:
            return
        horizon = now - self.window_s
        while self._events and self._events[0][0] < horizon:
            self._total -= self._events.popleft()[1]

    def add(self, cost_usd: float) -> None:
        if cost_usd <= 0:
            return
        now = time.monotonic()
        with self._lock:
            self._expire(now)
            self._total += cost_usd
            if self.window_s is not None:
                self._events.append((now, cost_usd))

    @property
    def spent_usd(self) -> float:
        with self._lock:
            self._expire(time.monotonic())
            return max(self._total, 0.0)

    @property
    def remaining_usd(self) -> float:
        return max(self.max_usd - self.spent_usd, 0.0)

    @property
    def exhausted(self) -> bool:
        return self.spent_usd >= self.max_usd

    def __repr__(self) -> str:
        window = "lifetime" if self.window_s is None else f"{self.window_s:g}s window"
        return f"SpendLimit(${self.spent_usd:.4f} of ${self.max_usd:.4f}, {window})"


# The limit of the run that is open on this thread or task, if it has one.
_run_limit: ContextVar[SpendLimit | None] = ContextVar("thinkless_run_limit", default=None)
