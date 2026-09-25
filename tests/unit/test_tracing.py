from __future__ import annotations

import asyncio
import json

import pytest
from rich.console import Console

from thinkless import JSONLSink, MemorySink, Tracer, tool
from thinkless.tracing import read_trace, summarize
from thinkless.tracing.console import print_trace
from thinkless.tracing.sinks import ConsoleSink, iter_trace_files


@tool
def lookup(order_id: str) -> dict:
    return {"order_id": order_id, "status": "shipped"}


@tool(name="async_lookup")
async def alookup(order_id: str) -> dict:
    return {"order_id": order_id}


def test_tool_outside_a_trace_is_plain() -> None:
    assert lookup("1") == {"order_id": "1", "status": "shipped"}


def test_tool_spans_capture_arguments_and_result() -> None:
    memory = MemorySink()
    tracer = Tracer([memory])
    with tracer.span("run", "demo"):
        lookup("4471")
        asyncio.run(alookup("9"))
    tools = [s for s in memory.spans if s.kind == "tool"]
    assert [s.name for s in tools] == ["lookup", "async_lookup"]
    assert tools[0].attributes["arguments"] == {"order_id": "4471"}
    assert tools[0].attributes["result"]["status"] == "shipped"
    assert tools[0].plane == "tool"


def test_errors_are_recorded_and_reraised() -> None:
    memory = MemorySink()
    tracer = Tracer([memory])
    with pytest.raises(KeyError), tracer.span("run", "demo"):
        raise KeyError("missing")
    assert memory.spans[0].status == "error"
    assert "KeyError" in memory.spans[0].error


def test_jsonl_sink_writes_one_file_per_trace(tmp_path) -> None:
    sink = JSONLSink(tmp_path)
    tracer = Tracer([sink])
    for name in ("first run", "second"):
        with tracer.span("run", name), tracer.span("step", "inner"):
            pass
    files = list(iter_trace_files(tmp_path))
    assert len(files) == 2
    spans = read_trace(files[0])
    assert spans[-1]["kind"] == "run"
    assert spans[-1]["parent_id"] is None
    assert all(json.dumps(s) for s in spans)
    assert any("first-run" in f.name for f in files)


def test_sink_failures_do_not_break_the_app() -> None:
    class Broken:
        def on_end(self, span):
            raise RuntimeError("disk full")

    tracer = Tracer([Broken()])
    with tracer.span("run", "demo"):
        pass


def test_console_sink_renders_tree() -> None:
    rendered = []
    tracer = Tracer([ConsoleSink(render=rendered.append)])
    with tracer.span("run", "demo"), tracer.span("tool", "lookup", plane="tool"):
        pass
    assert len(rendered) == 1
    console = Console(record=True, width=120)
    print_trace(rendered[0], console=console)
    text = console.export_text()
    assert "demo" in text
    assert "tool lookup" in text


def test_summary_requires_spans() -> None:
    with pytest.raises(ValueError):
        summarize([])
