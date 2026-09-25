"""Reports for the intents benchmark."""

from __future__ import annotations

from rich.table import Table

from .intents import CascadePoint, IntentsBenchmark

__all__ = ["cascade_highlights", "intents_markdown", "intents_table"]


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def cascade_highlights(result: IntentsBenchmark) -> list[CascadePoint]:
    """A few informative cascade operating points, lowest threshold first."""
    wanted = (0.5, 0.7, 0.8, 0.9, 0.95)
    return [p for p in result.cascade if p.threshold in wanted]


def intents_table(result: IntentsBenchmark) -> Table:
    table = Table(title=f"{result.dataset_title}: {result.examples} examples")
    for column in (
        "provider",
        "accuracy",
        "ECE",
        "p50 latency",
        "p95 latency",
        f"threshold for {result.target_accuracy:.0%}",
    ):
        table.add_column(column, justify="right")
    for name, report in result.providers.items():
        rec = report.recommended
        table.add_row(
            name,
            _pct(report.accuracy),
            "n/a" if report.ece is None else f"{report.ece:.3f}",
            f"{report.latency_p50_ms:.1f} ms",
            f"{report.latency_p95_ms:.1f} ms",
            f"{rec.threshold:.2f} (covers {_pct(rec.coverage)})" if rec else "not reached",
        )
    return table


def intents_markdown(result: IntentsBenchmark) -> str:
    lines = [
        f"# Intent benchmark: {result.dataset}",
        "",
        f"- Dataset: {result.dataset_title} ([source]({result.source}))",
        f"- Sample: {result.examples} test examples, seed {result.seed}, {result.labels} labels",
        f"- LLM: `{result.llm}`",
        f"- Environment: {result.environment.get('device', 'cpu')}, torch {result.environment.get('torch', 'n/a')}",
        f"- ThinkLess {result.thinkless_version}, run {result.created_at}",
        "",
        "## Providers",
        "",
        f"| Provider | Accuracy | ECE | p50 latency | p95 latency | Threshold for {result.target_accuracy:.0%} accuracy |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, report in result.providers.items():
        rec = report.recommended
        lines.append(
            f"| `{name}` | {_pct(report.accuracy)} | {'n/a' if report.ece is None else f'{report.ece:.3f}'} | "
            f"{report.latency_p50_ms:.1f} ms | {report.latency_p95_ms:.1f} ms | "
            + (f"{rec.threshold:.2f}, answers {_pct(rec.coverage)} alone" if rec else "not reached")
            + " |"
        )
    if result.cascade:
        lines += [
            "",
            "## Cascade",
            "",
            "Small models are tried in order; an answer below the threshold goes to the next one, and "
            "finally to the LLM. Accuracy is for the whole cascade.",
            "",
            "| Threshold | Accuracy | Calls reaching the LLM | Mean latency | Answered by |",
            "|---:|---:|---:|---:|---|",
        ]
        for point in result.cascade:
            answered = ", ".join(f"{k} {_pct(v)}" for k, v in point.answered_by.items() if v)
            label = "never accept" if point.threshold > 1 else f"{point.threshold:.2f}"
            lines.append(
                f"| {label} | {_pct(point.accuracy)} | {_pct(point.llm_share)} | {point.mean_latency_ms:.0f} ms | {answered} |"
            )
    return "\n".join(lines) + "\n"
