"""Destinations for finished spans.

A sink is any object with optional ``on_start(span)``, ``on_end(span)``,
``flush()`` and ``close()`` methods. The built-in sinks cover local audit
files, in-memory collection and terminal output; :mod:`thinkless.tracing.otel`
adds OpenTelemetry export.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from .span import Span

__all__ = ["ConsoleSink", "JSONLSink", "MemorySink", "iter_trace_files", "read_trace"]


class MemorySink:
    """Keeps finished spans in memory. Used by tests and benchmarks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.spans: list[Span] = []

    def on_end(self, span: Span) -> None:
        with self._lock:
            self.spans.append(span)

    def traces(self) -> dict[str, list[dict[str, Any]]]:
        """Finished spans grouped by trace id, as dictionaries."""
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        with self._lock:
            for span in self.spans:
                grouped[span.trace_id].append(span.to_dict())
        return dict(grouped)

    def trace(self, trace_id: str) -> list[dict[str, Any]]:
        return self.traces().get(trace_id, [])

    def clear(self) -> None:
        with self._lock:
            self.spans.clear()


_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


class JSONLSink:
    """Writes one JSON Lines file per trace.

    Files are named ``<UTC timestamp>_<root name>_<trace id prefix>.jsonl`` so a
    directory listing reads as a run log. Each line is one span; the root span
    is always the last line, which makes a truncated file easy to detect.

    Args:
        directory: Output directory, created on first write.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self._lock = threading.Lock()
        self._paths: dict[str, Path] = {}

    def on_start(self, span: Span) -> None:
        if span.is_root:
            self._path_for(span)

    def on_end(self, span: Span) -> None:
        line = json.dumps(span.to_dict(), ensure_ascii=False, default=str)
        with self._lock:
            path = self._path_for(span)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            if span.is_root:
                self._paths.pop(span.trace_id, None)

    def path(self, trace_id: str) -> Path | None:
        """Where a trace is being written, while its root span is open."""
        return self._paths.get(trace_id)

    def _path_for(self, span: Span) -> Path:
        path = self._paths.get(span.trace_id)
        if path is None:
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(span.start_ms / 1000.0))
            label = _UNSAFE.sub("-", span.name).strip("-")[:40] or "trace"
            path = self.directory / f"{stamp}_{label}_{span.trace_id[:8]}.jsonl"
            self._paths[span.trace_id] = path
        return path


class ConsoleSink:
    """Prints each finished trace as a tree, when its root span closes.

    Args:
        render: Callable that receives the finished span dictionaries of one
            trace. Defaults to :func:`thinkless.tracing.console.print_trace`.
    """

    def __init__(self, render: Callable[[list[dict[str, Any]]], None] | None = None) -> None:
        self._lock = threading.Lock()
        self._pending: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._render = render

    def on_end(self, span: Span) -> None:
        with self._lock:
            self._pending[span.trace_id].append(span.to_dict())
            if not span.is_root:
                return
            spans = self._pending.pop(span.trace_id)
        render = self._render
        if render is None:
            from .console import print_trace

            render = print_trace
        render(spans)


def read_trace(path: str | Path) -> list[dict[str, Any]]:
    """Load the spans of one trace file."""
    spans = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                spans.append(json.loads(line))
    return spans


def iter_trace_files(directory: str | Path) -> Iterator[Path]:
    """Trace files under ``directory``, newest first."""
    root = Path(directory)
    if not root.exists():
        return iter(())
    return iter(sorted(root.rglob("*.jsonl"), key=lambda p: p.name, reverse=True))
