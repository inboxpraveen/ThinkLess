"""Span records and the context that links them."""

from __future__ import annotations

import os
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .._json import jsonable

if TYPE_CHECKING:
    from .tracer import Tracer

__all__ = ["Span", "current_span", "current_tracer", "new_span_id", "new_trace_id"]


def new_trace_id() -> str:
    """128-bit hex id, compatible with W3C Trace Context and OpenTelemetry."""
    return os.urandom(16).hex()


def new_span_id() -> str:
    """64-bit hex id, compatible with W3C Trace Context and OpenTelemetry."""
    return os.urandom(8).hex()


@dataclass
class Span:
    """A timed unit of work.

    ``kind`` says what the span represents: ``run`` (a root), ``step`` (an
    application-defined phase), ``decide`` (a batch of questions), ``attempt``
    (one provider call inside a decision), ``llm`` (a generation), ``tool`` (an
    action) or ``rule`` (a deterministic check recorded by application code).

    ``plane`` says where the work ran and drives cost and latency roll-ups.
    """

    trace_id: str
    span_id: str
    parent_id: str | None
    kind: str
    name: str
    plane: str | None = None
    start_ms: float = field(default_factory=lambda: time.time() * 1000.0)
    duration_ms: float | None = None
    status: str = "ok"
    error: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    _t0: float = field(default_factory=time.perf_counter, repr=False)

    def set(self, **attributes: Any) -> Span:
        """Attach attributes. Later values overwrite earlier ones."""
        self.attributes.update({k: jsonable(v) for k, v in attributes.items()})
        return self

    def fail(self, error: BaseException | str) -> None:
        self.status = "error"
        self.error = error if isinstance(error, str) else f"{type(error).__name__}: {error}"

    def finish(self) -> None:
        if self.duration_ms is None:
            self.duration_ms = (time.perf_counter() - self._t0) * 1000.0

    @property
    def is_root(self) -> bool:
        return self.parent_id is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "kind": self.kind,
            "name": self.name,
            "plane": self.plane,
            "start_ms": round(self.start_ms, 3),
            "duration_ms": None if self.duration_ms is None else round(self.duration_ms, 3),
            "status": self.status,
            "error": self.error,
            "attributes": self.attributes,
        }


_current_span: ContextVar[Span | None] = ContextVar("thinkless_span", default=None)
_current_tracer: ContextVar[Tracer | None] = ContextVar("thinkless_tracer", default=None)


def current_span() -> Span | None:
    """The innermost open span in this context, if any."""
    return _current_span.get()


def current_tracer() -> Tracer | None:
    """The tracer that owns the innermost open span, if any."""
    return _current_tracer.get()
