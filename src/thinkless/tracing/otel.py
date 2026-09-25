"""OpenTelemetry export.

Mirrors ThinkLess spans into an OpenTelemetry tracer so runs show up in any
OTLP backend (Jaeger, Grafana Tempo, Honeycomb, Langfuse, Arize Phoenix and
others). Generation spans carry the GenAI semantic convention attributes
(``gen_ai.request.model``, ``gen_ai.usage.input_tokens`` and so on); every span
carries its ThinkLess fields under the ``thinkless.`` prefix.

Requires the ``otel`` extra::

    pip install "thinkless[otel]"

Example:
    >>> from opentelemetry.sdk.trace import TracerProvider
    >>> from opentelemetry.sdk.trace.export import BatchSpanProcessor
    >>> from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    >>> provider = TracerProvider()
    >>> provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    >>> tracer = Tracer(sinks=[OTelSink(provider)])
"""

from __future__ import annotations

import json
import threading
from typing import Any

from .span import Span

__all__ = ["CONTENT_KEYS", "OTelSink"]

# Attributes that may hold user content. They are dropped unless the sink is
# created with include_content=True, independent of the tracer setting.
CONTENT_KEYS = frozenset(
    {
        "input",
        "output",
        "state",
        "prompt",
        "messages",
        "system",
        "completion",
        "text",
        "arguments",
        "result",
        "reply",
        "values",
    }
)


def _attr_value(value: Any) -> Any:
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)) and all(
        isinstance(v, (bool, int, float, str)) for v in value
    ):
        return list(value)
    return json.dumps(value, default=str, ensure_ascii=False)


class OTelSink:
    """Forward spans to an OpenTelemetry ``TracerProvider``.

    Args:
        tracer_provider: The provider to export through. Defaults to the global
            provider configured by the application.
        include_content: Export attributes that may contain user content.
    """

    def __init__(
        self, tracer_provider: Any | None = None, *, include_content: bool = False
    ) -> None:
        try:
            from opentelemetry import trace
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise ImportError(
                'OTelSink needs OpenTelemetry. Install it with: pip install "thinkless[otel]"'
            ) from exc
        self._trace = trace
        provider = tracer_provider or trace.get_tracer_provider()
        self._tracer = provider.get_tracer("thinkless")
        self._include_content = include_content
        self._open: dict[str, Any] = {}
        self._lock = threading.Lock()

    def on_start(self, span: Span) -> None:
        with self._lock:
            parent = self._open.get(span.parent_id or "")
        context = self._trace.set_span_in_context(parent) if parent is not None else None
        otel_span = self._tracer.start_span(
            name=f"{span.kind} {span.name}",
            context=context,
            start_time=int(span.start_ms * 1_000_000),
        )
        with self._lock:
            self._open[span.span_id] = otel_span

    def on_end(self, span: Span) -> None:
        with self._lock:
            otel_span = self._open.pop(span.span_id, None)
        if otel_span is None:
            return
        for key, value in self._attributes(span).items():
            otel_span.set_attribute(key, value)
        if span.status == "error":
            from opentelemetry.trace import Status, StatusCode

            otel_span.set_status(Status(StatusCode.ERROR, span.error or "error"))
        end_ms = span.start_ms + (span.duration_ms or 0.0)
        otel_span.end(end_time=int(end_ms * 1_000_000))

    def _attributes(self, span: Span) -> dict[str, Any]:
        attrs: dict[str, Any] = {
            "thinkless.kind": span.kind,
            "thinkless.span_id": span.span_id,
            "thinkless.trace_id": span.trace_id,
        }
        if span.plane:
            attrs["thinkless.plane"] = span.plane
        for key, value in span.attributes.items():
            if key in CONTENT_KEYS and not self._include_content:
                continue
            if value is None:
                continue
            attrs[f"thinkless.{key}"] = _attr_value(value)

        usage = span.attributes.get("usage") or {}
        is_generative = span.kind == "llm" or (span.kind == "attempt" and span.plane == "llm")
        if is_generative:
            attrs["gen_ai.operation.name"] = "chat"
        if span.attributes.get("model"):
            attrs["gen_ai.request.model"] = str(span.attributes["model"])
        if span.attributes.get("provider"):
            attrs["gen_ai.provider.name"] = str(span.attributes["provider"])
        if usage:
            attrs["gen_ai.usage.input_tokens"] = int(usage.get("input_tokens") or 0)
            attrs["gen_ai.usage.output_tokens"] = int(usage.get("output_tokens") or 0)
        return attrs
