"""A self-contained HTML viewer for traces.

The output is one HTML file with the trace data embedded and no external
requests, so it opens offline and can be attached to an issue or a pull
request as-is.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from importlib import resources
from pathlib import Path
from typing import Any

from .sinks import iter_trace_files, read_trace
from .summary import summarize

__all__ = ["build_viewer", "collect_traces", "write_viewer"]

PLACEHOLDER = "__THINKLESS_DATA__"


def collect_traces(
    paths: Iterable[str | Path], *, limit: int | None = None
) -> list[dict[str, Any]]:
    """Load trace files, expanding directories, newest first."""
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        files.extend(iter_trace_files(path) if path.is_dir() else [path])
    traces = []
    for file in files[:limit] if limit else files:
        spans = read_trace(file)
        if not spans:
            continue
        traces.append({"file": file.name, "spans": spans})
    return traces


def build_viewer(
    traces: Sequence[dict[str, Any]] | Sequence[Sequence[dict[str, Any]]], *, title: str = "Traces"
) -> str:
    """Render traces into the viewer HTML.

    Args:
        traces: Either ``{"file", "spans"}`` records from :func:`collect_traces`
            or plain lists of span dictionaries.
        title: Shown in the header.
    """
    records = []
    for item in traces:
        if isinstance(item, dict):
            file, spans = item.get("file"), list(item["spans"])
        else:
            file, spans = None, list(item)
        if not spans:
            continue
        records.append(
            {"file": file, "spans": spans, "summary": summarize(spans).model_dump(mode="json")}
        )
    payload = {
        "title": title,
        "generated_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "traces": records,
    }
    # Keep the JSON inert inside the script tag.
    embedded = json.dumps(payload, ensure_ascii=False, default=str).replace("</", "<\\/")
    template = resources.files("thinkless").joinpath("data/viewer.html").read_text(encoding="utf-8")
    return template.replace(PLACEHOLDER, embedded)


def write_viewer(
    paths: Iterable[str | Path],
    output: str | Path,
    *,
    title: str = "Traces",
    limit: int | None = None,
) -> Path:
    """Collect traces from ``paths`` and write the viewer to ``output``."""
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        build_viewer(collect_traces(paths, limit=limit), title=title), encoding="utf-8"
    )
    return target
