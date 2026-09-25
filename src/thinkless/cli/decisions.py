"""``thinkless bench decisions``: the neutral decision-model benchmark."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

decisions_app = typer.Typer(
    help="The neutral decision benchmark: eight public tasks, verifiable results.",
    no_args_is_help=True,
)
console = Console()


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


@decisions_app.command("tasks")
def tasks_command() -> None:
    """List the tasks, their question kinds, sizes and licenses."""
    from ..bench.decisions import benchmark_version, load_tasks

    table = Table(title=f"Decision benchmark {benchmark_version()}")
    for column in ("task", "kind", "domain", "test", "calibration", "license"):
        table.add_column(column)
    for task in load_tasks():
        table.add_row(
            task.name,
            task.kind,
            task.domain,
            str(task.files["test"].rows),
            str(task.files["calibration"].rows),
            task.license,
        )
    console.print(table)


@decisions_app.command("run")
def run_command(
    provider: Annotated[
        str,
        typer.Option(
            help="gliner, laya, llm, jev, systemone:<url>, or module:attribute for your own provider."
        ),
    ],
    name: Annotated[str | None, typer.Option(help="Name on the leaderboard.")] = None,
    llm: Annotated[
        str,
        typer.Option(
            help="Model spec when --provider llm, for example openrouter:qwen/qwen3.7-flash."
        ),
    ] = "local",
    reasoning: Annotated[
        str, typer.Option(help="Reasoning setting for --provider llm.")
    ] = "default",
    device: Annotated[str, typer.Option(help="Device for local models.")] = "auto",
    task: Annotated[list[str] | None, typer.Option(help="Only this task. Repeatable.")] = None,
    out: Annotated[Path | None, typer.Option(help="Result file. Default: ./<name>.json")] = None,
    max_cost_usd: Annotated[float, typer.Option(help="Spend cap for the whole run.")] = 2.0,
    concurrency: Annotated[int, typer.Option(help="Parallel requests, for hosted providers.")] = 1,
    limit: Annotated[
        int | None,
        typer.Option(help="Only the first N rows per split, for a smoke test (never ranked)."),
    ] = None,
    submitted_by: Annotated[str | None, typer.Option(help="Your name or handle.")] = None,
    notes: Annotated[
        str | None, typer.Option(help="Settings or hardware a reader should know.")
    ] = None,
    trained_on_task_data: Annotated[
        bool,
        typer.Option(
            "--trained-on-task-data",
            help="The model was trained on these datasets' train splits (marked on the leaderboard).",
        ),
    ] = False,
    data_dir: Annotated[Path | None, typer.Option(help="Local task files.")] = None,
) -> None:
    """Run one provider on the benchmark and write a result file."""
    from ..bench.decisions import build_submission_provider, load_tasks, run_benchmark

    built = build_submission_provider(
        provider, llm=llm, device=device, reasoning=None if reasoning == "default" else reasoning
    )
    label = name or (llm if provider == "llm" else provider)
    spec = f"llm:{llm}" if provider == "llm" else provider
    if provider == "llm" and reasoning != "default":
        spec += f" --reasoning {reasoning}"
    total = {t.name: t for t in load_tasks(task)}
    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as bar:
        handles: dict[str, TaskID] = {}

        def progress(task_name: str, split: str, done: int, count: int) -> None:
            key = f"{task_name}/{split}"
            if key not in handles:
                handles[key] = bar.add_task(key, total=count)
            bar.update(handles[key], completed=done)

        result = run_benchmark(
            built,
            name=label,
            spec=spec,
            tasks=list(total) if task else None,
            data_dir=data_dir,
            max_cost_usd=max_cost_usd,
            concurrency=concurrency,
            limit=limit,
            submitted_by=submitted_by,
            notes=notes,
            trained_on_task_data=trained_on_task_data,
            progress=progress,
        )
    path = out or Path(re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-") + ".json")
    result.write(path)

    table = Table(title=f"{label} on decision benchmark {result.benchmark_version}")
    for column in ("task", "accuracy", "coverage at 95%", "ECE", "p50", "cost per 1k"):
        table.add_column(column, justify="right" if column != "task" else "left")
    for task_name, entry in result.tasks.items():
        m = entry.metrics
        if m is None:
            table.add_row(task_name, "n/a", "", "", "", "")
            continue
        table.add_row(
            task_name,
            _pct(m.accuracy),
            _pct(m.coverage),
            "n/a" if m.ece is None else f"{m.ece:.3f}",
            "n/a" if m.latency_p50_ms is None else f"{m.latency_p50_ms:.0f} ms",
            f"${m.cost_per_1k_usd:.4f}",
        )
    console.print(table)
    if result.partial:
        console.print("[yellow]Partial run (--limit or the spend cap): it will not be ranked.[/]")
    console.print(f"Result: {path}")


@decisions_app.command("verify")
def verify_command(
    results: Annotated[list[Path], typer.Argument(help="Result files.")],
    data_dir: Annotated[Path | None, typer.Option(help="Local task files.")] = None,
) -> None:
    """Check result files against the published tasks and recompute their metrics.

    Exits with status 1 when any file fails.
    """
    from ..bench.decisions import verify_result

    failed = False
    for path in results:
        check = verify_result(path, data_dir=data_dir)
        status = "[green]ok[/]" if check.ok else "[red]failed[/]"
        rank = "ranked" if check.rankable else "not ranked"
        console.print(f"{status}  {path}  ({check.name}, {rank})")
        for problem in check.problems:
            console.print(f"   [red]{problem}[/]")
        for note in check.notes:
            console.print(f"   {note}")
        failed = failed or not check.ok
    if failed:
        raise typer.Exit(code=1)


@decisions_app.command("leaderboard")
def leaderboard_command(
    paths: Annotated[list[Path], typer.Argument(help="Result files or directories.")],
    out: Annotated[Path | None, typer.Option(help="Write the markdown here.")] = None,
    data_dir: Annotated[Path | None, typer.Option(help="Local task files.")] = None,
) -> None:
    """Verify results and write the leaderboard as markdown."""
    from ..bench.decisions import collect, leaderboard_markdown

    text = leaderboard_markdown(collect(paths, data_dir=data_dir))
    if out is None:
        typer.echo(text)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    console.print(f"Leaderboard: {out}")
