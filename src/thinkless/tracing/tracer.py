"""The tracer: opens spans, links them and hands them to sinks."""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, TypeVar, overload

from .._json import jsonable
from .span import Span, _current_span, _current_tracer, current_tracer, new_span_id, new_trace_id

__all__ = ["REDACTED", "Tracer", "jsonable", "tool"]

logger = logging.getLogger("thinkless.tracing")

REDACTED = "[redacted]"

F = TypeVar("F", bound=Callable[..., Any])


class Tracer:
    """Creates spans and forwards them to sinks.

    Args:
        sinks: Where finished spans go. See :mod:`thinkless.tracing.sinks`.
        capture_content: Record inputs, outputs, prompts and tool arguments.
            Turn it off where traces must not hold user content; structure,
            timings, confidences and token counts are still recorded.

    Sink failures are logged and never propagate into application code.
    """

    def __init__(self, sinks: Sequence[Any] | None = None, *, capture_content: bool = True) -> None:
        self.sinks: list[Any] = list(sinks or [])
        self.capture_content = capture_content

    def add_sink(self, sink: Any) -> None:
        self.sinks.append(sink)

    def content(self, value: Any) -> Any:
        """Return ``value`` as JSON-safe data, or a marker when content capture is off."""
        return jsonable(value) if self.capture_content else REDACTED

    @contextmanager
    def span(
        self, kind: str, name: str, *, plane: str | None = None, **attributes: Any
    ) -> Iterator[Span]:
        parent = _current_span.get()
        span = Span(
            trace_id=parent.trace_id if parent else new_trace_id(),
            span_id=new_span_id(),
            parent_id=parent.span_id if parent else None,
            kind=kind,
            name=name,
            plane=plane,
            attributes={k: jsonable(v) for k, v in attributes.items()},
        )
        span_token = _current_span.set(span)
        tracer_token = _current_tracer.set(self)
        self._emit("on_start", span)
        try:
            yield span
        except BaseException as exc:
            span.fail(exc)
            raise
        finally:
            span.finish()
            _current_span.reset(span_token)
            _current_tracer.reset(tracer_token)
            self._emit("on_end", span)

    def flush(self) -> None:
        for sink in self.sinks:
            flush = getattr(sink, "flush", None)
            if flush is not None:
                self._call(flush)

    def close(self) -> None:
        for sink in self.sinks:
            close = getattr(sink, "close", None)
            if close is not None:
                self._call(close)

    def _emit(self, hook: str, span: Span) -> None:
        for sink in tuple(self.sinks):
            method = getattr(sink, hook, None)
            if method is not None:
                self._call(method, span)

    @staticmethod
    def _call(fn: Callable[..., Any], *args: Any) -> None:
        try:
            fn(*args)
        except Exception:
            logger.exception("trace sink %r failed", fn)


@overload
def tool(fn: F, /) -> F: ...
@overload
def tool(*, name: str | None = None) -> Callable[[F], F]: ...


def tool(fn: F | None = None, /, *, name: str | None = None) -> F | Callable[[F], F]:
    """Record calls to a function as ``tool`` spans.

    Outside a traced run the function runs untouched, so decorated tools stay
    usable in tests and scripts.

    Example:
        >>> @tool
        ... def refund_payment(payment_id: str, amount: float) -> dict: ...
    """

    def decorate(func: F) -> F:
        span_name = name or func.__name__
        signature = inspect.signature(func)

        def arguments(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
            try:
                bound = signature.bind_partial(*args, **kwargs)
            except TypeError:
                return {"args": args, "kwargs": kwargs}
            return {k: v for k, v in bound.arguments.items() if k not in ("self", "cls")}

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = current_tracer()
                if tracer is None:
                    return await func(*args, **kwargs)
                with tracer.span("tool", span_name, plane="tool") as span:
                    span.set(arguments=tracer.content(arguments(args, kwargs)))
                    result = await func(*args, **kwargs)
                    span.set(result=tracer.content(result))
                    return result

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = current_tracer()
            if tracer is None:
                return func(*args, **kwargs)
            with tracer.span("tool", span_name, plane="tool") as span:
                span.set(arguments=tracer.content(arguments(args, kwargs)))
                result = func(*args, **kwargs)
                span.set(result=tracer.content(result))
                return result

        return wrapper  # type: ignore[return-value]

    if fn is not None:
        return decorate(fn)
    return decorate
