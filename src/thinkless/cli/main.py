"""The ``thinkless`` command line."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
import time
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from .._version import __version__
from ..logs import configure_logging
from ..settings import Settings

app = typer.Typer(
    name="thinkless",
    help="ThinkLess: fast typed decisions for AI agents, with the LLM only where thinking is needed.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
bench_app = typer.Typer(help="Run the benchmarks.", no_args_is_help=True)
trace_app = typer.Typer(help="Inspect saved traces.", no_args_is_help=True)
app.add_typer(bench_app, name="bench")
app.add_typer(trace_app, name="trace")

console = Console()
err = Console(stderr=True)

LlmOption = Annotated[
    str,
    typer.Option(
        "--llm",
        help="Reasoning model as backend[:model]: local, local:Qwen/Qwen3-4B, anthropic, "
        "anthropic:claude-haiku-4-5, openai:<model>, ollama:qwen3:8b, vllm:<model>.",
    ),
]
DeviceOption = Annotated[str, typer.Option(help="Device for local models: auto, cpu, cuda or mps.")]
VerboseOption = Annotated[bool, typer.Option("--verbose", "-v", help="Show info logs.")]


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"thinkless {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version_callback, is_eager=True, help="Show the version."
        ),
    ] = False,
) -> None:
    """ThinkLess command line."""


def _setup(verbose: bool) -> None:
    configure_logging("INFO" if verbose else Settings().log_level)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if not verbose:
        import warnings

        warnings.filterwarnings("ignore")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def _installed(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


# ----------------------------------------------------------------- doctor


@app.command()
def doctor() -> None:
    """Check the environment: accelerators, optional backends, keys and settings."""
    settings = Settings()
    table = Table(show_header=False, box=None)
    table.add_column(style="dim")
    table.add_column()
    table.add_row("thinkless", __version__)
    table.add_row("python", f"{platform.python_version()} ({sys.executable})")
    table.add_row("platform", platform.platform())

    torch_version = _installed("torch")
    if torch_version:
        try:
            import torch

            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                accel = f"CUDA: {props.name}, {props.total_memory / 1e9:.1f} GB"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                accel = "Apple MPS"
            else:
                accel = "CPU only"
        except Exception as exc:  # pragma: no cover - environment specific
            accel = f"torch failed to initialize: {exc}"
        table.add_row("torch", f"{torch_version} ({accel})")
    else:
        table.add_row("torch", "not installed (local models unavailable)")

    extras = {
        "laya": "laya",
        "gliner": "gliner2",
        "transformers": "transformers",
        "openai": "openai",
        "anthropic": "anthropic",
        "otel": "opentelemetry-sdk",
        "bench": "datasets",
    }
    for label, package in extras.items():
        version = _installed(package)
        table.add_row(label, f"[green]{version}[/]" if version else "[yellow]not installed[/]")

    for key in ("TYPESAFE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        table.add_row(key, "[green]set[/]" if os.environ.get(key) else "[dim]not set[/]")
    table.add_row("trace dir", str(settings.trace_dir))
    table.add_row("device", settings.device)
    table.add_row("capture content", str(settings.capture_content))
    table.add_row(
        "HF cache", os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    )
    console.print(table)
    if platform.system() == "Windows":
        console.print(
            "[dim]Windows note: if a first model download fails with WinError 1314 (symlink privilege), "
            "run the command again or enable Developer Mode. See docs/guides/troubleshooting.md.[/]"
        )


# ------------------------------------------------------------------- demo


@app.command()
def demo(
    ticket: Annotated[
        str, typer.Option(help="Scenario id from the bundled set, for example T-016.")
    ] = "T-001",
    message: Annotated[
        str | None, typer.Option(help="Your own ticket text instead of a scenario.")
    ] = None,
    customer: Annotated[str, typer.Option(help="Customer id used with --message.")] = "C-1001",
    mode: Annotated[
        list[str] | None, typer.Option(help="hybrid, llm or models. Repeat to compare.")
    ] = None,
    llm: LlmOption = "local",
    device: DeviceOption = "auto",
    threshold: Annotated[float, typer.Option(help="Engine confidence threshold.")] = 0.8,
    view: Annotated[bool, typer.Option(help="Open the HTML trace viewer afterwards.")] = False,
    verbose: VerboseOption = False,
) -> None:
    """Run the support agent on one ticket and print its trace."""
    _setup(verbose)
    from ..demo.support import MODES, SupportAgent, SupportStack, World, load_scenarios
    from ..llm import from_spec
    from ..tracing import JSONLSink, MemorySink, Tracer
    from ..tracing.console import print_trace

    modes = mode or ["hybrid"]
    for m in modes:
        if m not in MODES:
            raise typer.BadParameter(f"unknown mode {m!r}; choose from {', '.join(MODES)}")
    if message:
        scenario: dict[str, Any] = {
            "id": "custom",
            "customer_id": customer,
            "subject": "",
            "message": message,
        }
    else:
        scenarios = {s["id"]: s for s in load_scenarios()}
        if ticket not in scenarios:
            raise typer.BadParameter(
                f"unknown ticket {ticket!r}; ids run from T-001 to T-{len(scenarios):03d}"
            )
        scenario = scenarios[ticket]

    settings = Settings()
    err.print(f"[dim]Loading models for {', '.join(modes)} ({llm})...[/]")
    stack = SupportStack(from_spec(llm, device=device), device=device)
    stack.warmup(tuple(modes))
    trace_dir = settings.trace_dir / "demo"
    written: list[str] = []
    for m in modes:
        memory = MemorySink()
        sink = JSONLSink(trace_dir)
        agent = SupportAgent(
            stack.engine(
                m,
                tracer=Tracer([memory, sink], capture_content=settings.capture_content),
                threshold=threshold,
            )
        )
        outcome, _ = agent.handle(scenario, World(), mode=m)
        console.rule(f"[bold]{m}[/]  {scenario['id']}")
        console.print(f"[dim]customer:[/] {scenario['message']}")
        print_trace(memory.trace(outcome.trace_id or ""), console=console)
        expected = (scenario.get("expected") or {}).get("action")
        verdict = (
            ""
            if not expected
            else (
                " [green]matches expected[/]"
                if expected == outcome.action
                else f" [red]expected {expected}[/]"
            )
        )
        console.print(f"\n[bold]action[/] {outcome.action}{verdict}")
        console.print(f"[bold]reply[/] ({outcome.reply_source}) {outcome.reply}\n")
        written.append(outcome.trace_id or "")
    console.print(f"[dim]Traces written to {trace_dir}[/]")
    if view:
        from ..tracing.viewer import write_viewer

        output = write_viewer(
            [trace_dir], settings.trace_dir / "viewer.html", title="ThinkLess demo", limit=20
        )
        console.print(f"Viewer: {output}")
        webbrowser.open(output.resolve().as_uri())


# ------------------------------------------------------------------ bench


@bench_app.command("support")
def bench_support(
    mode: Annotated[
        list[str] | None,
        typer.Option(help="Modes to run. Repeat the option. Default: llm, hybrid, models."),
    ] = None,
    llm: LlmOption = "local",
    device: DeviceOption = "auto",
    limit: Annotated[int | None, typer.Option(help="Only the first N tickets.")] = None,
    out: Annotated[
        Path | None,
        typer.Option(help="Output directory. Default: .thinkless/bench/support-<timestamp>."),
    ] = None,
    reference: Annotated[
        str, typer.Option(help="provider:model whose prices turn tokens into an estimated cost.")
    ] = "anthropic:claude-sonnet-5",
    threshold: Annotated[float, typer.Option(help="Engine confidence threshold.")] = 0.8,
    jev: Annotated[
        bool, typer.Option(help="Add TypeSafe Jev to the cascade (needs TYPESAFE_API_KEY).")
    ] = False,
    verbose: VerboseOption = False,
) -> None:
    """Run the support agent on every scenario in each mode and compare."""
    _setup(verbose)
    from ..bench.report import support_markdown, support_table
    from ..bench.support import run_support_benchmark
    from ..demo.support import MODES, SupportStack, load_scenarios
    from ..engine import Engine
    from ..llm import from_spec
    from ..providers import SystemOne
    from ..tracing import Tracer
    from ..tracing.viewer import write_viewer

    modes = mode or ["llm", "hybrid", "models"]
    for m in modes:
        if m not in MODES:
            raise typer.BadParameter(f"unknown mode {m!r}; choose from {', '.join(MODES)}")
    scenarios = load_scenarios()[:limit] if limit else load_scenarios()
    output = out or Settings().trace_dir.parent / "bench" / time.strftime("support-%Y%m%dT%H%M%S")
    stack = SupportStack(
        from_spec(llm, device=device), device=device, jev=SystemOne.jev() if jev else None
    )
    err.print(f"[dim]Loading models for {', '.join(modes)} ({llm})...[/]")
    stack.warmup(tuple(modes))

    total = len(scenarios) * len(modes)
    done = 0
    started = time.perf_counter()

    def progress(result: Any) -> None:
        nonlocal done
        done += 1
        mark = "[green]ok[/]" if result.correct else "[red]miss[/]"
        err.print(
            f"[dim]{done:>3}/{total}  {time.perf_counter() - started:6.1f}s[/]  {result.mode:<7} {result.ticket_id} "
            f"{mark} {result.action}  [dim]llm calls {result.summary.llm_calls}[/]"
        )

    def factory(mode_name: str) -> Callable[[Tracer], Engine]:
        return lambda tracer: stack.engine(mode_name, tracer=tracer, threshold=threshold)

    bench = run_support_benchmark(
        {m: factory(m) for m in modes},
        reasoning_model=stack.llm.name,
        scenarios=scenarios,
        output_dir=output,
        reference_price=reference,
        threshold=threshold,
        on_ticket=progress,
    )
    (output / "report.md").write_text(support_markdown(bench), encoding="utf-8")
    write_viewer([output / "traces"], output / "viewer.html", title="Support benchmark")
    console.print(support_table(bench))
    console.print(
        f"\nResults: {output / 'results.json'}\nReport:  {output / 'report.md'}\nViewer:  {output / 'viewer.html'}"
    )


@bench_app.command("intents")
def bench_intents(
    dataset: Annotated[str, typer.Option(help="banking77, clinc150 or emotion.")] = "banking77",
    provider: Annotated[
        list[str] | None, typer.Option(help="gliner, laya or llm. Repeat the option.")
    ] = None,
    llm: LlmOption = "local",
    device: DeviceOption = "auto",
    limit: Annotated[int, typer.Option(help="Examples to sample from the test split.")] = 500,
    seed: Annotated[int, typer.Option(help="Sampling seed.")] = 13,
    target: Annotated[
        float, typer.Option(help="Target accuracy for the threshold recommendation.")
    ] = 0.95,
    out: Annotated[Path | None, typer.Option(help="Output directory.")] = None,
    verbose: VerboseOption = False,
) -> None:
    """Accuracy, calibration and cascade trade-offs on a public intent dataset."""
    _setup(verbose)
    from ..bench.intents import run_intents_benchmark
    from ..bench.intents_report import intents_markdown, intents_table

    providers = provider or ["gliner", "laya", "llm"]
    output = out or Settings().trace_dir.parent / "bench" / time.strftime(
        f"intents-{dataset}-%Y%m%dT%H%M%S"
    )
    result = run_intents_benchmark(
        dataset,
        providers=providers,
        llm_spec=llm,
        device=device,
        limit=limit,
        seed=seed,
        target_accuracy=target,
        output_dir=output,
        progress=lambda msg: err.print(f"[dim]{msg}[/]"),
    )
    (output / "report.md").write_text(intents_markdown(result), encoding="utf-8")
    console.print(intents_table(result))
    console.print(f"\nResults: {output / 'results.json'}\nReport:  {output / 'report.md'}")


# -------------------------------------------------------------- calibrate


@app.command()
def calibrate(
    data: Annotated[Path, typer.Argument(help="JSONL file with one labeled example per line.")],
    provider: Annotated[str, typer.Option(help="gliner, laya, llm or jev.")] = "gliner",
    kind: Annotated[str, typer.Option(help="choice or yes_no.")] = "choice",
    question: Annotated[
        str, typer.Option(help="The question to ask.")
    ] = "Which category best fits the text?",
    labels: Annotated[
        str | None,
        typer.Option(help="Choice options, comma-separated. Default: every label in the data."),
    ] = None,
    demo_question: Annotated[
        str | None,
        typer.Option(help="Use a question from the support demo by name, for example injection."),
    ] = None,
    text_field: Annotated[str, typer.Option(help="Field holding the input text.")] = "text",
    label_field: Annotated[
        str | None,
        typer.Option(help="Field holding the label. Default: label, or the demo question name."),
    ] = None,
    target: Annotated[float, typer.Option(help="Accuracy the accepted answers must reach.")] = 0.95,
    llm: LlmOption = "local",
    device: DeviceOption = "auto",
    verbose: VerboseOption = False,
) -> None:
    """Find the threshold that meets a target accuracy on your own labeled data.

    Rows look like {"text": "...", "label": "refund"} for a choice question, or
    {"text": "...", "label": true} for a yes/no question.
    """
    _setup(verbose)
    from ..bench.intents import build_provider, evaluate_question
    from ..bench.metrics import expected_calibration_error, recommend_threshold, threshold_sweep
    from ..questions import Choice, Question, YesNo

    rows = [
        json.loads(line) for line in data.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        raise typer.BadParameter(f"{data} has no rows")
    asked: Question
    if demo_question:
        from ..demo.support import questions as demo

        candidates = {q.key: q for q in (*demo.TRIAGE, demo.KB_MATCH)}
        if demo_question not in candidates:
            raise typer.BadParameter(
                f"unknown demo question {demo_question!r}; choose from {', '.join(candidates)}"
            )
        asked = candidates[demo_question]
        field = label_field or demo_question
    else:
        field = label_field or "label"
        if kind == "yes_no":
            asked = YesNo(question, name=field)
        elif kind == "choice":
            options = (
                [x.strip() for x in labels.split(",")]
                if labels
                else sorted({str(r[field]) for r in rows})
            )
            asked = Choice(question, options=options, name=field)
        else:
            raise typer.BadParameter("kind must be choice or yes_no")
    if not isinstance(asked, (Choice, YesNo)):
        raise typer.BadParameter("calibration supports choice and yes/no questions")

    evaluation = evaluate_question(
        build_provider(provider, llm_spec=llm, device=device),
        asked,
        rows,
        text_field=text_field,
        label_field=field,
    )
    if any(c is None for c in evaluation.confidences):
        console.print(
            f"[yellow]{provider} reports no calibrated confidence, so there is no threshold to tune.[/] "
            f"Accuracy {evaluation.accuracy * 100:.1f}%."
        )
        raise typer.Exit()
    confidences = [float(c) for c in evaluation.confidences if c is not None]
    points = threshold_sweep(confidences, evaluation.correct)
    best = recommend_threshold(points, target)
    table = Table(title=f"{provider} on {data.name}: {asked.key} ({len(rows)} rows)")
    for column in ("threshold", "answers alone", "accuracy of those answers"):
        table.add_column(column, justify="right")
    for point in points:
        style = "bold green" if best and point.threshold == best.threshold else None
        table.add_row(
            f"{point.threshold:.2f}",
            f"{point.coverage * 100:.1f}%",
            "n/a" if point.accuracy is None else f"{point.accuracy * 100:.1f}%",
            style=style,
        )
    console.print(table)
    console.print(
        f"overall accuracy {evaluation.accuracy * 100:.1f}%, "
        f"ECE {expected_calibration_error(confidences, evaluation.correct):.3f}, "
        f"p50 latency {evaluation.latency_p50_ms:.1f} ms"
    )
    if best:
        console.print(
            f"[bold]Recommended threshold {best.threshold:.2f}[/]: {provider} answers "
            f"{best.coverage * 100:.1f}% of inputs on its own at {(best.accuracy or 0.0) * 100:.1f}% accuracy; "
            "the rest go to the next provider."
        )
    else:
        console.print(
            f"[yellow]No threshold reaches {target * 100:.0f}% accuracy.[/] Route this question to a "
            f'stronger provider, for example Question(..., providers=("rules", "llm")).'
        )


# ------------------------------------------------------------------ trace


@trace_app.command("ls")
def trace_ls(
    directory: Annotated[
        Path | None, typer.Argument(help="Trace directory. Default: THINKLESS_TRACE_DIR.")
    ] = None,
    limit: Annotated[int, typer.Option(help="How many to show.")] = 20,
) -> None:
    """List recent traces with their headline numbers."""
    from ..tracing import iter_trace_files, read_trace, summarize

    root = directory or Settings().trace_dir
    table = Table(title=str(root))
    for column in ("file", "name", "mode", "time", "LLM calls", "decisions", "cost"):
        table.add_column(column)
    for index, file in enumerate(iter_trace_files(root)):
        if index >= limit:
            break
        spans = read_trace(file)
        if not spans:
            continue
        s = summarize(spans)
        table.add_row(
            file.name,
            s.name,
            str(s.attributes.get("mode", "")),
            f"{s.duration_ms:.0f} ms",
            str(s.llm_calls),
            str(s.decisions),
            f"${s.cost_usd:.6f}",
        )
    console.print(table)


@trace_app.command("show")
def trace_show(path: Annotated[Path, typer.Argument(help="A .jsonl trace file.")]) -> None:
    """Print one trace as a tree with its summary."""
    from ..tracing import read_trace
    from ..tracing.console import print_trace

    print_trace(read_trace(path), console=console)


@trace_app.command("view")
def trace_view(
    paths: Annotated[
        list[Path] | None,
        typer.Argument(help="Trace files or directories. Default: THINKLESS_TRACE_DIR."),
    ] = None,
    out: Annotated[Path | None, typer.Option(help="Where to write the HTML file.")] = None,
    limit: Annotated[
        int | None, typer.Option(help="At most this many traces, newest first.")
    ] = 500,
    open_browser: Annotated[
        bool, typer.Option("--open/--no-open", help="Open the viewer in a browser.")
    ] = True,
) -> None:
    """Write a self-contained HTML viewer for traces and open it."""
    from ..tracing.viewer import write_viewer

    sources = paths or [Settings().trace_dir]
    output = out or Settings().trace_dir / "viewer.html"
    write_viewer(sources, output, title="ThinkLess traces", limit=limit)
    console.print(f"Viewer: {output}")
    if open_browser:
        webbrowser.open(output.resolve().as_uri())


if __name__ == "__main__":  # pragma: no cover
    app()
