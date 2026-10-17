"""``thinkless habits``: find the decisions in an agent's LLM calls and take them over."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

habits_app = typer.Typer(
    help="Find the decisions in your LLM calls, learn them, and answer them locally once proven.",
    no_args_is_help=True,
)

console = Console()
err = Console(stderr=True)

DEFAULT_DIR = Path(".thinkless/habits")
DirOption = Annotated[
    Path, typer.Option("--dir", help="Habits directory: calls, plan and students.")
]
ConfirmOption = Annotated[
    int | None,
    typer.Option(
        "--confirm-calls",
        help="New calls a student on probation is checked on before it answers any. "
        "Defaults to 394 at the default target and confidence; 0 skips probation.",
    ),
]
PriceOption = Annotated[
    list[str] | None,
    typer.Option(
        "--price",
        help="Your price for a model as MODEL=INPUT,OUTPUT in USD per million tokens, "
        "for calls with no reported cost. Repeat for several models.",
    ),
]


def _rates(specs: list[str] | None) -> dict[str, tuple[float, float]]:
    from ..habits.report import parse_rates

    try:
        return parse_rates(specs or [])
    except ValueError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc


def _plural(count: int, word: str) -> str:
    return f"{count:,} {word}{'' if count == 1 else 's'}"


def _tokens(count: int) -> str:
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.0f}k"
    return str(count)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _usd(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.2f}" if value >= 1 else f"${value:.4f}"


def _load(paths: list[Path] | None, directory: Path) -> list:
    from ..habits.calls import read_calls

    sources = paths or [directory / "calls"]
    missing = [p for p in sources if not p.exists()]
    if missing:
        err.print(f"[red]Not found:[/red] {', '.join(str(p) for p in missing)}")
        raise typer.Exit(1)
    observations = list(read_calls(sources))
    if not observations:
        err.print("[yellow]No chat completion calls found in those files.[/yellow]")
        raise typer.Exit(1)
    return observations


def _answer(site) -> str:  # type: ignore[no-untyped-def]
    answer = "\n".join(f.describe() for f in site.fields[:3]) or "no answers"
    if len(site.fields) > 3:
        answer += f"\n+{len(site.fields) - 3} more fields"
    return answer


def _print_report(report, *, with_plan: bool) -> None:  # type: ignore[no-untyped-def]
    title = f"{_plural(report.calls, 'call')}, {_plural(len(report.sites), 'site')}"
    table = Table(title=title, title_justify="left")
    table.add_column("Site")
    table.add_column("Calls", justify="right")
    if with_plan:
        for column in ("Status", "Local", "Audits agreed", "Agreement before activation"):
            table.add_column(column, justify="left" if column == "Status" else "right")
    else:
        for column in ("LLM cost", "Answer", "Verdict"):
            table.add_column(column, justify="right" if column == "LLM cost" else "left")
    for site in report.sites:
        name = f"{site.name}\n[dim]{site.id} {site.model}[/dim]"
        calls = f"{site.calls:,}\n[dim]{_pct(site.share)}[/dim]"
        if with_plan:
            audits = f"{site.audit_agreed}/{site.audits}" if site.audits else "-"
            evidence = site.evidence or {}
            held = (
                f"held-out {evidence['agreed']}/{evidence['accepted']}\n[dim]lower bound "
                f"{_pct(evidence.get('lower_bound'))}[/dim]"
                if evidence.get("accepted")
                else "-"
            )
            confirmation = evidence.get("confirmation")
            if confirmation:
                held += (
                    f"\nnew calls {confirmation['agreed']}/{confirmation['calls']}"
                    f"\n[dim]lower bound {_pct(confirmation.get('lower_bound'))}[/dim]"
                )
            table.add_row(name, calls, site.status or site.verdict, f"{site.local:,}", audits, held)
        else:
            llm_calls = site.calls - site.local
            tokens = (
                f"[dim]{_tokens(site.input_tokens)} in, {_tokens(site.output_tokens)} out[/dim]"
            )
            cost = _usd(site.cost_usd) if site.unknown_cost < llm_calls else "unknown"
            table.add_row(name, calls, f"{cost}\n{tokens}", _answer(site), site.verdict)
    console.print(table)


@habits_app.command("scan")
def scan(
    paths: Annotated[
        list[Path] | None,
        typer.Argument(
            help="Call logs (JSONL files or directories). Defaults to the Habits directory."
        ),
    ] = None,
    directory: DirOption = DEFAULT_DIR,
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
    examples: Annotated[
        bool, typer.Option(help="Show example inputs and answers per site.")
    ] = False,
    price: PriceOption = None,
) -> None:
    """Which of your LLM calls are decisions, and what they cost.

    Reads Habits call logs, chat fine-tuning files ({"messages": [...]}) or
    request and response pairs. Nothing is changed or sent anywhere.
    """
    from ..habits.calls import local_answers
    from ..habits.report import build_report

    sources = paths or [directory / "calls"]
    report = build_report(
        _load(paths, directory), local=local_answers(sources), rates=_rates(price)
    )
    if as_json:
        console.print_json(json.dumps(report.to_dict(), default=str))
        return
    _print_report(report, with_plan=False)
    if examples:
        for site in report.sites:
            if site.verdict == "generation":
                continue
            console.print(f"\n[bold]{site.name}[/bold] [dim]{site.id}[/dim]")
            for example in site.examples:
                console.print(f"  {example['input'][:120]!r}  ->  {example['answer']}")
    calls = report.calls
    decision_share = report.decision_calls / calls if calls else 0.0
    known = report.cost_usd
    cost_share = report.decision_cost / known if known else None
    console.print()
    line = f"[bold]{_pct(decision_share)}[/bold] of calls"
    if cost_share is not None:
        line += f" ({_pct(cost_share)} of known LLM cost)"
    console.print(line + " come from sites whose answers are decisions.")
    if report.mixed_calls:
        console.print(
            f"{_pct(report.mixed_calls / calls)} of calls pack a decision together with free "
            "text; splitting those calls would let the decision be answered locally."
        )
    if report.unknown_cost:
        console.print(
            f"[dim]{_plural(report.unknown_cost, 'LLM call')} with no known cost: pass "
            "--price MODEL=INPUT,OUTPUT (USD per million tokens) to estimate it.[/dim]"
        )
    console.print(
        "\nNext: [bold]thinkless habits learn[/bold] measures how many of them a local student can answer."
    )


@habits_app.command("learn")
def learn_command(
    directory: DirOption = DEFAULT_DIR,
    calls: Annotated[
        list[Path] | None,
        typer.Option("--calls", help="Learn from these logs instead of the Habits directory."),
    ] = None,
    target: Annotated[
        float,
        typer.Option(help="Agreement with the LLM a site needs, as a lower confidence bound."),
    ] = 0.97,
    confidence: Annotated[float, typer.Option(help="Level of the lower bound.")] = 0.95,
    min_calls: Annotated[
        int, typer.Option(help="Answered calls a site needs before training.")
    ] = 100,
    confirm_calls: ConfirmOption = None,
) -> None:
    """Train a student per decision site and decide which sites may be answered locally."""
    from ..habits.learn import LearnOptions, learn

    if calls:
        _load(calls, directory)
    elif not (directory / "calls").exists():
        err.print(f"[red]No calls recorded in {directory / 'calls'} yet.[/red]")
        raise typer.Exit(1)
    options = LearnOptions(
        target=target, confidence=confidence, min_calls=min_calls, confirm_calls=confirm_calls
    )
    with console.status("Learning"):
        plan = learn(directory, calls=calls, options=options)
    table = Table(title=f"Plan: {directory / 'plan.json'}", title_justify="left")
    for column in (
        "Site",
        "Status",
        "Threshold",
        "Held-out",
        "Coverage",
        "Agreement",
        "Lower bound",
    ):
        table.add_column(column, justify="right" if column not in ("Site", "Status") else "left")
    active = 0
    unpredictable = []
    for entry in sorted(plan["sites"].values(), key=lambda e: e.get("calls", 0), reverse=True):
        evidence = entry.get("evidence") or {}
        active += entry["status"] == "active"
        held = f"{evidence.get('calls', 0):,}"
        predictability = evidence.get("predictability")
        if predictability is not None:
            baseline = evidence.get("baseline") or 0.0
            held += (
                f"\n[dim]predictable {_pct(predictability)}"
                f"\nmost common answer {_pct(baseline)}[/dim]"
            )
            # Closing less than half the gap between always giving the most
            # common answer and matching every time.
            if predictability < baseline + (1.0 - baseline) / 2:
                unpredictable.append(entry["name"])
        table.add_row(
            f"{entry['name']}\n[dim]{entry['id']}[/dim]",
            f"{entry['status']}\n[dim]{entry.get('reason') or ''}[/dim]",
            "-" if entry.get("threshold") is None else f"{entry['threshold']:.3f}",
            held,
            _pct(evidence.get("coverage")),
            _pct(evidence.get("agreement")),
            _pct(evidence.get("lower_bound")),
        )
    console.print(table)
    console.print(
        f"{_plural(active, 'site')} active at {target:.0%} agreement with the LLM "
        f"({confidence:.0%} lower bound). Running agents pick the plan up within seconds."
    )
    if unpredictable:
        console.print(
            f"[yellow]Hard to predict:[/yellow] {', '.join(unpredictable)}. A student trained on "
            "the LLM's answers does little better on its next answers than always giving the "
            "most common one; similar inputs may be getting different answers. A clearer prompt "
            "or labeled examples usually help, whether or not ThinkLess answers the site."
        )


@habits_app.command("status")
def status(
    directory: DirOption = DEFAULT_DIR,
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
    price: PriceOption = None,
) -> None:
    """What ThinkLess has answered, per site, and what it saved."""
    from ..habits.calls import local_answers
    from ..habits.learn import load_plan
    from ..habits.report import build_report

    plan = load_plan(directory)
    report = build_report(
        _load(None, directory),
        plan=plan,
        local=local_answers([directory / "calls"]),
        rates=_rates(price),
    )
    if as_json:
        console.print_json(json.dumps(report.to_dict(), default=str))
        return
    _print_report(report, with_plan=True)
    if plan is None:
        console.print("\nNo plan yet: run [bold]thinkless habits learn[/bold].")
        return
    saved = report.saved_usd
    cost = (
        f"about {_usd(saved)} of LLM cost."
        if saved is not None
        else "LLM cost unknown; pass --price MODEL=INPUT,OUTPUT to estimate it."
    )
    console.print(
        f"\nAnswered locally: [bold]{report.local:,}[/bold] of {report.calls:,} calls "
        f"({_pct(report.local / report.calls if report.calls else 0)}), {cost}"
    )


@habits_app.command("proxy")
def proxy(
    upstream: Annotated[
        str, typer.Option(help="The provider's API base URL, such as https://api.openai.com/v1.")
    ] = "https://api.openai.com/v1",
    directory: DirOption = DEFAULT_DIR,
    host: Annotated[str, typer.Option(help="Interface to listen on.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8787,
    mode: Annotated[str, typer.Option(help="auto, shadow, record or off.")] = "auto",
    audit: Annotated[
        float, typer.Option(help="Share of local answers still checked by the LLM.")
    ] = 0.05,
    learn_every: Annotated[
        int, typer.Option(help="Learn again after this many new LLM answers.")
    ] = 200,
    confirm_calls: ConfirmOption = None,
) -> None:
    """Serve an OpenAI-compatible endpoint with Habits in front of the upstream."""
    try:
        import uvicorn

        from ..habits.proxy import create_proxy
    except ImportError as exc:
        err.print('[red]The proxy needs the server extra:[/red] pip install "thinkless[server]"')
        raise typer.Exit(1) from exc
    from ..habits.runtime import Habits

    habits = Habits(
        directory, mode=mode, audit=audit, learn_every=learn_every, confirm_calls=confirm_calls
    )
    console.print(
        f"Habits proxy on http://{host}:{port}/v1 in front of {upstream} ({mode} mode).\n"
        f"Point clients at it: OPENAI_BASE_URL=http://{host}:{port}/v1"
    )
    uvicorn.run(create_proxy(upstream, habits), host=host, port=port, log_level="warning")


@habits_app.command("reset")
def reset(
    directory: DirOption = DEFAULT_DIR,
    calls: Annotated[bool, typer.Option("--calls", help="Also delete the recorded calls.")] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Do not ask for confirmation.")] = False,
) -> None:
    """Forget the plan and the students, so every call goes to the LLM again."""
    targets = [directory / "plan.json", directory / "models", directory / ".learning.lock"]
    if calls:
        targets.append(directory / "calls")
    present = [t for t in targets if t.exists()]
    if not present:
        console.print("Nothing to reset.")
        return
    if not yes:
        typer.confirm(f"Delete {', '.join(str(t) for t in present)}?", abort=True)
    for target in present:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    console.print("Reset. Every call goes to the LLM until the next plan activates a site.")
