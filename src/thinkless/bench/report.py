"""Human-readable reports for benchmark results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rich.table import Table

from .support import ModeReport, SupportBenchmark

__all__ = ["support_markdown", "support_table"]


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _ms(value: float) -> str:
    return f"{value / 1000:.2f} s" if value >= 1000 else f"{value:.0f} ms"


ROWS: Sequence[tuple[str, Any]] = (
    ("Task success (correct action)", lambda m: _pct(m.task_success)),
    ("Intent accuracy", lambda m: _pct(m.intent_accuracy)),
    ("Order id accuracy", lambda m: _pct(m.order_id_accuracy)),
    ("LLM calls per ticket", lambda m: f"{m.llm_calls:.2f}"),
    ("  for decisions", lambda m: f"{m.decision_llm_calls:.2f}"),
    ("  for replies", lambda m: f"{m.generation_calls:.2f}"),
    ("Decision time per ticket", lambda m: _ms(m.decision_ms)),
    ("Reply generation time per ticket", lambda m: _ms(m.generation_ms)),
    ("End-to-end latency p50", lambda m: _ms(m.latency_p50_ms)),
    ("End-to-end latency p95", lambda m: _ms(m.latency_p95_ms)),
    (
        "LLM tokens per ticket (in / out)",
        lambda m: f"{m.llm_input_tokens:.0f} / {m.llm_output_tokens:.0f}",
    ),
    ("LLM tokens spent on decisions", lambda m: f"{m.decision_llm_tokens:.0f}"),
    (
        "Billed cost per 1k tickets",
        lambda m: f"${m.cost_usd * 1000:.3f}" if m.cost_usd else "n/a (local)",
    ),
    ("Reference cost per 1k tickets", lambda m: f"${m.reference_cost_per_1k_tickets:.2f}"),
    ("Replies passing the grounding check", lambda m: _pct(m.grounded_rate)),
)


def _planes(mode: ModeReport) -> str:
    total = sum(mode.decisions_by_plane.values()) or 1
    order = ("rule", "model", "llm", "unresolved")
    return ", ".join(
        f"{plane} {mode.decisions_by_plane.get(plane, 0) / total * 100:.0f}%"
        for plane in order
        if mode.decisions_by_plane.get(plane)
    )


def support_table(bench: SupportBenchmark) -> Table:
    """A Rich table comparing modes side by side."""
    table = Table(
        title=f"Support benchmark ({bench.modes[next(iter(bench.modes))].tickets} tickets)"
    )
    table.add_column("metric", style="dim")
    for name in bench.modes:
        table.add_column(name, justify="right")
    for label, fn in ROWS:
        table.add_row(label, *(fn(mode) for mode in bench.modes.values()))
    table.add_row("Decisions by plane", *(_planes(m) for m in bench.modes.values()))
    return table


def support_markdown(bench: SupportBenchmark) -> str:
    """The benchmark as a Markdown report."""
    modes = list(bench.modes.values())
    lines = [
        "# Support benchmark",
        "",
        f"- ThinkLess {bench.thinkless_version}, run {bench.created_at}",
        f"- Reasoning model: `{bench.reasoning_model}`",
        f"- Engine threshold: {bench.threshold}",
        f"- Reference price for cost estimates: `{bench.reference_price}`",
        f"- Environment: {bench.environment.get('device', 'cpu')}, Python {bench.environment.get('python')}, "
        f"torch {bench.environment.get('torch', 'n/a')}",
        f"- Tickets: {modes[0].tickets}",
        "- Billed cost is what the provider reported for each call (OpenRouter does). Reference cost "
        "prices the same LLM tokens at the reference model's published rates, so runs on different "
        "models and local runs can be compared; tokenizers differ, so compare it by ratio.",
        "",
        "| Metric | " + " | ".join(f"`{m.mode}`" for m in modes) + " |",
        "|---|" + "---:|" * len(modes),
    ]
    for label, fn in ROWS:
        lines.append(f"| {label.strip()} | " + " | ".join(fn(m) for m in modes) + " |")
    lines.append("| Decisions by plane | " + " | ".join(_planes(m) for m in modes) + " |")
    lines.append("")

    for mode in modes:
        lines += [
            f"## Questions in `{mode.mode}` mode",
            "",
            "| Question | Answered by | Escalation rate | Accuracy (labeled) |",
            "|---|---|---:|---:|",
        ]
        for name, stats in mode.questions.items():
            answered = ", ".join(
                f"{k} {v}" for k, v in sorted(stats["answered_by"].items(), key=lambda kv: -kv[1])
            )
            accuracy = _pct(stats["accuracy"]) + (
                f" of {stats['labeled']}" if stats["labeled"] else ""
            )
            lines.append(
                f"| `{name}` | {answered} | {_pct(stats['escalation_rate'])} | {accuracy} |"
            )
        lines.append("")
        if mode.failures:
            lines += [f"Failures in `{mode.mode}`:", ""]
            for failure in mode.failures:
                lines.append(
                    f"- {failure['ticket_id']}: expected `{failure['expected']}`, got `{failure['got']}` "
                    f"(intent `{failure['intent']}` from {failure['intent_provider']})"
                )
            lines.append("")
    return "\n".join(lines)
