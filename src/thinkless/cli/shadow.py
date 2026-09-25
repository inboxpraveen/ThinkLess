"""``thinkless shadow``: read shadow logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

shadow_app = typer.Typer(
    help="Measure a candidate engine against the system you run today.", no_args_is_help=True
)

console = Console()


def _pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def _usd(value: float | None) -> str:
    if value is None:
        return "n/a"
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}" if abs(value) >= 1 else f"{sign}${abs(value):.4f}"


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f} ms"


@shadow_app.command("report")
def shadow_report(
    logs: Annotated[list[Path], typer.Argument(help="Shadow log files (JSONL).")],
    target: Annotated[
        float, typer.Option(help="Agreement the shadow must reach, as a 95% lower bound.")
    ] = 0.95,
    min_calls: Annotated[int, typer.Option(help="Compared calls needed for a verdict.")] = 100,
    volume: Annotated[
        int | None, typer.Option(help="Decisions a month, to project the monthly saving.")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
) -> None:
    """Agreement, projected savings and a verdict for each question in a shadow log."""
    from ..shadow import build_report, read_log

    records = [record for path in logs for record in read_log(path)]
    if not records:
        raise typer.BadParameter("the logs hold no shadow records")
    report = build_report(
        records,
        source=", ".join(str(p) for p in logs),
        target=target,
        min_calls=min_calls,
        monthly_volume=volume,
    )
    if as_json:
        typer.echo(json.dumps(report.model_dump(mode="json"), indent=2))
        return

    table = Table(title=f"Shadow report: {report.records} records")
    for column, justify in (
        ("question", "left"),
        ("compared", "right"),
        ("agreement (95% CI)", "right"),
        ("without an LLM", "right"),
        ("agreement there", "right"),
        ("cost / 1k: now", "right"),
        ("shadow", "right"),
        ("p50: now", "right"),
        ("shadow", "right"),
        ("verdict", "left"),
    ):
        table.add_column(column, justify=justify)  # type: ignore[arg-type]
    for q in report.questions:
        interval = (
            "n/a"
            if q.agreement is None
            else f"{_pct(q.agreement)} ({_pct(q.agreement_low)}-{_pct(q.agreement_high)})"
        )
        table.add_row(
            q.question,
            str(q.compared),
            interval,
            _pct(q.shadow.fast_share, 0),
            _pct(q.fast_agreement),
            _usd(q.primary.cost_per_1k_usd),
            _usd(q.shadow.cost_per_1k_usd),
            _ms(q.primary.latency_p50_ms),
            _ms(q.shadow.latency_p50_ms),
            q.verdict,
            style="green" if q.ready else None,
        )
    console.print(table)

    for q in report.questions:
        if q.primary_by_plane:
            parts = ", ".join(
                f"{plane} {_pct(a.agreement)} of {a.calls} (at least {_pct(a.low)})"
                for plane, a in q.primary_by_plane.items()
            )
            console.print(f"{q.question}: agreement by the plane that answered today: {parts}.")
        if q.errors:
            console.print(f"[yellow]{q.question}: {q.errors} shadow runs failed.[/]")
        for advice in q.thresholds:
            if advice.recommended is None:
                console.print(
                    f"{q.question}@{advice.provider}: no threshold reaches "
                    f"{report.target * 100:.0f}% agreement on {advice.calls} calls."
                )
            else:
                console.print(
                    f"{q.question}@{advice.provider}: threshold {advice.recommended:.2f} would "
                    f"settle {_pct(advice.coverage)} of the {advice.calls} calls it saw at "
                    f"{_pct(advice.agreement)} agreement."
                )

    if report.savings_per_1k_usd is None:
        console.print(
            "\nNo saving is projected: some primary calls have no cost. Pass cost_usd= to "
            "Shadow.watch, compare or observe to include them."
        )
        return
    direction = "less" if report.savings_per_1k_usd >= 0 else "more"
    line = (
        f"\nThe shadow costs {_usd(report.shadow_cost_per_1k_usd)} per 1,000 decisions against "
        f"{_usd(report.primary_cost_per_1k_usd)} now: {_usd(abs(report.savings_per_1k_usd))} "
        f"{direction} per 1,000"
    )
    if report.savings_share is not None:
        line += f" ({_pct(abs(report.savings_share))})"
    line += "."
    if report.monthly_savings_usd is not None and report.monthly_volume:
        line += (
            f" At {report.monthly_volume:,} decisions a month that is "
            f"{_usd(abs(report.monthly_savings_usd))} {direction} a month."
        )
    console.print(line)
    console.print(
        "Savings count only questions that were shadowed; switch a question over only once "
        "its verdict is ready."
    )


@shadow_app.command("export")
def shadow_export(
    logs: Annotated[list[Path], typer.Argument(help="Shadow log files (JSONL).")],
    out: Annotated[Path, typer.Option(help="Where to write the labeling rows.")],
    question: Annotated[str | None, typer.Option(help="Only this question.")] = None,
    all_calls: Annotated[
        bool, typer.Option("--all", help="Export every call, not only disagreements.")
    ] = False,
    label_from: Annotated[
        str, typer.Option(help="Pre-fill labels from primary, shadow or none.")
    ] = "primary",
) -> None:
    """Write shadow calls as rows to label, in the format ``thinkless calibrate`` reads."""
    from ..shadow import export_labels, read_log

    records = [record for path in logs for record in read_log(path)]
    written = export_labels(
        records,
        out,
        question=question,
        disagreements_only=not all_calls,
        label_from=label_from,
    )
    console.print(f"Wrote {written} rows to {out}.")
    if written == 0:
        console.print(
            "Nothing to export. Records logged with capture_content=False hold no input to label."
        )
