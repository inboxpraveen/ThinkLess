"""Terminal rendering of traces with Rich."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from rich.console import Console, Group
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from .summary import TraceSummary, summarize

__all__ = ["PLANE_STYLES", "print_trace", "render_summary", "render_tree"]

PLANE_STYLES = {
    "rule": "cyan",
    "model": "blue",
    "llm": "magenta",
    "tool": "green",
    "code": "white",
}


def _ms(value: float | None) -> str:
    if value is None:
        return "?"
    if value >= 1000:
        return f"{value / 1000:.2f} s"
    return f"{value:.1f} ms"


def _short(value: Any, limit: int = 60) -> str:
    text = value if isinstance(value, str) else repr(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _decision_line(decision: Mapping[str, Any]) -> Text:
    plane = decision.get("plane")
    style = PLANE_STYLES.get(plane or "", "red")
    conf = decision.get("confidence")
    conf_text = "n/a" if conf is None else f"{conf:.2f}"
    status = decision.get("status")
    line = Text("  ")
    line.append(f"{decision.get('name')}", style="bold")
    line.append(" = ")
    value = decision.get("level") or decision.get("value")
    line.append(_short(value, 40), style=style)
    line.append(f"  conf {conf_text} / {decision.get('threshold')}", style="dim")
    line.append(f"  via {decision.get('provider') or 'none'}", style=style)
    attempts = decision.get("attempts") or []
    if len(attempts) > 1:
        line.append(f"  (tried {' > '.join(attempts)})", style="yellow")
    if status != "accepted":
        line.append(f"  {status}", style="bold yellow")
    return line


def _label(span: Mapping[str, Any]) -> Text:
    kind = span.get("kind", "")
    plane = span.get("plane")
    attrs = span.get("attributes") or {}
    style = PLANE_STYLES.get(plane or "", "white")
    text = Text()
    if kind == "run":
        text.append(span.get("name", ""), style="bold")
        mode = attrs.get("mode")
        if mode:
            text.append(f"  mode={mode}", style="dim")
    elif kind == "step":
        text.append(span.get("name", ""), style="bold white")
    elif kind == "decide":
        text.append("decide ", style="dim")
        text.append(span.get("name", ""))
    elif kind == "attempt":
        text.append(f"{plane or '?'} ", style=f"bold {style}")
        text.append(attrs.get("provider", span.get("name", "")), style=style)
        model = attrs.get("model")
        if model:
            text.append(f" [{model}]", style="dim")
        questions = attrs.get("questions") or []
        accepted = attrs.get("accepted") or []
        text.append(f"  {len(accepted)}/{len(questions)} accepted", style="dim")
    elif kind == "llm":
        text.append("llm ", style=f"bold {style}")
        text.append(span.get("name", ""), style=style)
        model = attrs.get("model")
        if model:
            text.append(f" [{model}]", style="dim")
        usage = attrs.get("usage") or {}
        text.append(
            f"  {usage.get('input_tokens', 0)} in / {usage.get('output_tokens', 0)} out",
            style="dim",
        )
    elif kind == "tool":
        text.append("tool ", style=f"bold {style}")
        text.append(span.get("name", ""), style=style)
    elif kind == "rule":
        text.append("rule ", style=f"bold {style}")
        text.append(f"{span.get('name')} = {_short(attrs.get('value'), 40)}", style=style)
    else:
        text.append(f"{kind} {span.get('name', '')}")
    text.append(f"  {_ms(span.get('duration_ms'))}", style="dim")
    cost = attrs.get("cost_usd")
    if cost:
        text.append(f"  ${cost:.6f}", style="dim")
    if span.get("status") == "error":
        text.append(f"  ERROR {span.get('error')}", style="bold red")
    return text


def render_tree(spans: Sequence[Mapping[str, Any]]) -> Tree:
    """Build a Rich tree for one trace."""
    children: dict[str | None, list[Mapping[str, Any]]] = defaultdict(list)
    ids = {s["span_id"] for s in spans}
    for span in spans:
        parent = span.get("parent_id")
        children[parent if parent in ids else None].append(span)
    for group in children.values():
        group.sort(key=lambda s: s.get("start_ms") or 0.0)

    roots = children[None]
    tree = Tree(_label(roots[0]) if len(roots) == 1 else Text("trace"))

    def add(node: Tree, span: Mapping[str, Any]) -> None:
        for child in children.get(span["span_id"], []):
            branch = node.add(_label(child))
            if child.get("kind") == "decide":
                for decision in (child.get("attributes") or {}).get("decisions") or []:
                    branch.add(_decision_line(decision))
            add(branch, child)

    if len(roots) == 1:
        add(tree, roots[0])
    else:
        for root in roots:
            add(tree.add(_label(root)), root)
    return tree


def render_summary(summary: TraceSummary) -> Table:
    """A compact table of the numbers that matter for one run."""
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim")
    table.add_column()
    planes = summary.decisions_by_plane
    table.add_row("total time", _ms(summary.duration_ms))
    table.add_row(
        "decisions",
        f"{summary.decisions}  (rule {planes.get('rule', 0)}, model {planes.get('model', 0)}, "
        f"llm {planes.get('llm', 0)}, unresolved {planes.get('unresolved', 0)})",
    )
    table.add_row("escalations", f"{summary.escalations}  uncertain {summary.uncertain}")
    table.add_row("llm calls", f"{summary.llm_calls}  ({summary.generation_calls} generation)")
    table.add_row("llm tokens", f"{summary.llm_input_tokens} in / {summary.llm_output_tokens} out")
    table.add_row("tool calls", str(summary.tool_calls))
    table.add_row("cost", f"${summary.cost_usd:.6f}")
    times = ", ".join(f"{k} {_ms(v)}" for k, v in sorted(summary.time_by_plane_ms.items()))
    table.add_row("time by plane", times or "n/a")
    return table


def print_trace(spans: Sequence[Mapping[str, Any]], console: Console | None = None) -> None:
    """Print the tree and summary of one trace."""
    console = console or Console(stderr=True)
    console.print(Group(render_tree(spans), Text(""), render_summary(summarize(spans))))
