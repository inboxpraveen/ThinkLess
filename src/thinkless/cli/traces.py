"""``thinkless trace export`` and ``thinkless trace drift``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

console = Console()

PathsArg = Annotated[list[Path], typer.Argument(help="Trace files or directories.")]


def trace_export(
    paths: PathsArg,
    out: Annotated[Path, typer.Option(help="Where to write the labeling rows.")],
    question: Annotated[str | None, typer.Option(help="Only this question.")] = None,
    status: Annotated[
        list[str] | None,
        typer.Option(help="Only these statuses (accepted, uncertain, abstained). Repeatable."),
    ] = None,
    plane: Annotated[
        list[str] | None, typer.Option(help="Only decisions settled by this plane. Repeatable.")
    ] = None,
    limit: Annotated[int | None, typer.Option(help="At most this many rows, sampled.")] = None,
) -> None:
    """Turn traced decisions into rows to label and calibrate on."""
    from ..tracing.export import export_trace_labels

    written = export_trace_labels(
        paths, out, question=question, statuses=status, planes=plane, limit=limit
    )
    console.print(f"Wrote {written} rows to {out}.")
    if written == 0:
        console.print(
            "Nothing to export. Traces written with capture_content=False hold no input to label."
        )


def trace_drift(
    baseline: Annotated[
        list[Path], typer.Option(help="Traces from the reference period. Repeatable.")
    ],
    current: Annotated[
        list[Path], typer.Option(help="Traces from the period to check. Repeatable.")
    ],
    min_decisions: Annotated[
        int, typer.Option(help="Decisions needed in both periods before flagging.")
    ] = 30,
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
) -> None:
    """Compare two periods of traces and flag questions whose decisions moved.

    Exits with status 1 when any question is flagged, so it can run in a
    scheduled job.
    """
    from ..tracing.export import drift_report

    report = drift_report(baseline, current, min_decisions=min_decisions)
    if as_json:
        typer.echo(json.dumps(report.model_dump(mode="json"), indent=2))
    else:
        table = Table(title="Decision drift")
        for column in (
            "question",
            "decisions",
            "LLM share",
            "not accepted",
            "answer shift",
            "flags",
        ):
            table.add_column(column)
        for q in report.questions:
            table.add_row(
                q.question,
                f"{q.baseline} to {q.current}",
                f"{q.llm_share[0] * 100:.0f}% to {q.llm_share[1] * 100:.0f}%",
                f"{q.uncertain_share[0] * 100:.0f}% to {q.uncertain_share[1] * 100:.0f}%",
                f"{q.answer_shift:.2f}",
                "; ".join(q.flags) or "none",
                style="yellow" if q.flags else None,
            )
        console.print(table)
    if report.drifted:
        raise typer.Exit(code=1)
