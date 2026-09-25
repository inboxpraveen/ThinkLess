"""The support benchmark: the same agent and tickets, one run per decision-plane mode."""

from __future__ import annotations

import json
import platform
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .._version import __version__
from ..decision import Usage
from ..engine import Engine
from ..pricing import PriceTable, default_prices
from ..tracing import JSONLSink, MemorySink, Tracer
from ..tracing.summary import TraceSummary
from .metrics import percentile

__all__ = ["ModeReport", "SupportBenchmark", "TicketResult", "run_support_benchmark"]

# For these yes/no questions the expected action implies the true answer.
LABELED_YES_NO: dict[str, Callable[[Mapping[str, Any]], bool]] = {
    "wants_human": lambda expected: expected["action"] == "escalate_human",
    "injection": lambda expected: expected["action"] == "escalate_security",
}


class TicketResult(BaseModel):
    """One ticket processed in one mode."""

    ticket_id: str
    mode: str
    expected_action: str
    action: str
    correct: bool
    expected_intent: str | None = None
    intent: Any = None
    intent_provider: str | None = None
    expected_order_id: str | None = None
    order_id: str | None = None
    reply_source: str
    grounded: bool
    decisions: dict[str, Any] = Field(default_factory=dict)
    escalated: list[str] = Field(default_factory=list)
    decision_llm_calls: int = 0
    decision_llm_tokens: int = 0
    decision_llm_ms: float = 0.0
    summary: TraceSummary


class ModeReport(BaseModel):
    """Aggregates for one mode. Per-ticket values are means."""

    mode: str
    tickets: int
    task_success: float
    intent_accuracy: float | None
    order_id_accuracy: float | None
    llm_calls: float
    decision_llm_calls: float
    generation_calls: float
    decisions_by_plane: dict[str, int]
    escalations: float
    latency_p50_ms: float
    latency_p95_ms: float
    decision_ms: float
    generation_ms: float
    llm_input_tokens: float
    llm_output_tokens: float
    decision_llm_tokens: float
    cost_usd: float
    reference_cost_per_1k_tickets: float
    grounded_rate: float
    template_fallbacks: int
    questions: dict[str, dict[str, Any]]
    failures: list[dict[str, Any]]


class SupportBenchmark(BaseModel):
    """A complete benchmark run, as written to ``results.json``."""

    thinkless_version: str = __version__
    created_at: str
    reasoning_model: str
    reference_price: str
    threshold: float
    environment: dict[str, Any]
    modes: dict[str, ModeReport]
    tickets: list[TicketResult]


def environment_info(extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Python, platform and accelerator, recorded with every result file."""
    env: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        env["torch"] = torch.__version__
        env["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        env["device"] = "cpu"
    env.update(extra or {})
    return env


def _truth(name: str, expected: Mapping[str, Any]) -> tuple[bool, Any]:
    if name == "intent":
        return expected.get("intent") is not None, expected.get("intent")
    if name in LABELED_YES_NO:
        return True, LABELED_YES_NO[name](expected)
    if name == "order":
        return "order_id" in expected, expected.get("order_id")
    return False, None


def _question_stats(
    results: Sequence[TicketResult], scenarios: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    providers: dict[str, Counter[str]] = defaultdict(Counter)
    asked: Counter[str] = Counter()
    escalated: Counter[str] = Counter()
    hits: Counter[str] = Counter()
    labeled: Counter[str] = Counter()
    for result in results:
        expected = scenarios[result.ticket_id]["expected"]
        for name, view in result.decisions.items():
            asked[name] += 1
            providers[name][view.get("provider") or "none"] += 1
            escalated[name] += int(name in result.escalated)
            has_label, truth = _truth(name, expected)
            if not has_label:
                continue
            value = view.get("value")
            if name == "order":
                value = value.get("order_id") if isinstance(value, Mapping) else None
            labeled[name] += 1
            hits[name] += int(value == truth)
    return {
        name: {
            "asked": asked[name],
            "answered_by": dict(providers[name]),
            "escalation_rate": round(escalated[name] / asked[name], 4),
            "accuracy": round(hits[name] / labeled[name], 4) if labeled[name] else None,
            "labeled": labeled[name],
        }
        for name in asked
    }


def _mode_report(
    mode: str,
    results: Sequence[TicketResult],
    scenarios: Mapping[str, Mapping[str, Any]],
    reference: tuple[str, str],
    prices: PriceTable,
) -> ModeReport:
    n = len(results)
    summaries = [r.summary for r in results]
    planes: Counter[str] = Counter()
    for summary in summaries:
        planes.update(summary.decisions_by_plane)
    intent_rows = [r for r in results if r.expected_intent is not None and "intent" in r.decisions]
    order_rows = [r for r in results if "order" in r.decisions]
    llm_in = sum(s.llm_input_tokens for s in summaries)
    llm_out = sum(s.llm_output_tokens for s in summaries)
    reference_cost, _ = prices.cost(
        reference[0], reference[1], Usage(input_tokens=llm_in, output_tokens=llm_out)
    )
    small_model_ms = sum(
        s.time_by_plane_ms.get("model", 0.0) + s.time_by_plane_ms.get("rule", 0.0)
        for s in summaries
    )
    decision_llm_ms = sum(r.decision_llm_ms for r in results)
    all_llm_ms = sum(s.time_by_plane_ms.get("llm", 0.0) for s in summaries)

    def mean(total: float, digits: int = 3) -> float:
        return round(total / n, digits) if n else 0.0

    return ModeReport(
        mode=mode,
        tickets=n,
        task_success=mean(sum(r.correct for r in results), 4),
        intent_accuracy=round(
            sum(r.intent == r.expected_intent for r in intent_rows) / len(intent_rows), 4
        )
        if intent_rows
        else None,
        order_id_accuracy=round(
            sum(r.order_id == r.expected_order_id for r in order_rows) / len(order_rows), 4
        )
        if order_rows
        else None,
        llm_calls=mean(sum(s.llm_calls for s in summaries)),
        decision_llm_calls=mean(sum(r.decision_llm_calls for r in results)),
        generation_calls=mean(sum(s.generation_calls for s in summaries)),
        decisions_by_plane=dict(planes),
        escalations=mean(sum(s.escalations for s in summaries)),
        latency_p50_ms=round(percentile([s.duration_ms for s in summaries], 50), 1),
        latency_p95_ms=round(percentile([s.duration_ms for s in summaries], 95), 1),
        decision_ms=mean(small_model_ms + decision_llm_ms, 1),
        generation_ms=mean(all_llm_ms - decision_llm_ms, 1),
        llm_input_tokens=mean(llm_in, 1),
        llm_output_tokens=mean(llm_out, 1),
        decision_llm_tokens=mean(sum(r.decision_llm_tokens for r in results), 1),
        cost_usd=mean(sum(s.cost_usd for s in summaries), 8),
        reference_cost_per_1k_tickets=round(reference_cost / n * 1000, 4) if n else 0.0,
        grounded_rate=mean(sum(r.grounded for r in results), 4),
        template_fallbacks=sum(r.reply_source == "template_fallback" for r in results),
        questions=_question_stats(results, scenarios),
        failures=[
            {
                "ticket_id": r.ticket_id,
                "expected": r.expected_action,
                "got": r.action,
                "intent": r.decisions.get("intent", {}).get("value"),
                "intent_provider": r.intent_provider,
            }
            for r in results
            if not r.correct
        ],
    )


def _trace_facts(spans: Sequence[Mapping[str, Any]]) -> tuple[list[str], int, int, float]:
    """Escalated question names and LLM decision usage, read from a trace."""
    escalated: set[str] = set()
    calls = tokens = 0
    ms = 0.0
    for span in spans:
        attrs = span.get("attributes") or {}
        if span.get("kind") == "decide":
            escalated.update(d["name"] for d in attrs.get("decisions") or [] if d.get("escalated"))
        if span.get("kind") != "attempt":
            continue
        if span.get("plane") == "llm":
            usage = attrs.get("usage") or {}
            calls += 1
            tokens += int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
            ms += float(span.get("duration_ms") or 0.0)
    return sorted(escalated), calls, tokens, ms


def run_support_benchmark(
    engines: Mapping[str, Callable[[Tracer], Engine]],
    *,
    reasoning_model: str,
    scenarios: Sequence[Mapping[str, Any]] | None = None,
    output_dir: str | Path | None = None,
    reference_price: str = "anthropic:claude-sonnet-5",
    threshold: float = 0.8,
    prices: PriceTable | None = None,
    on_ticket: Callable[[TicketResult], None] | None = None,
    environment: Mapping[str, Any] | None = None,
) -> SupportBenchmark:
    """Run every ticket through every mode.

    Args:
        engines: Mode name to a factory that builds that mode's engine around
            the tracer it is given.
        reasoning_model: Label of the LLM, recorded in the results.
        scenarios: Tickets with expected outcomes. Defaults to the bundled set.
        output_dir: Where to write ``results.json`` and one trace file per
            ticket and mode. Nothing is written when omitted.
        reference_price: ``provider:model`` whose price list turns LLM tokens
            into an estimated dollar cost, so local runs can be compared with
            a hosted deployment.
        threshold: The engine threshold, recorded in the results.
        on_ticket: Progress callback.
    """
    from ..demo.support import SupportAgent, World, load_scenarios

    tickets = list(scenarios or load_scenarios())
    by_id = {str(t["id"]): t for t in tickets}
    prices = prices or default_prices()
    provider, _, model = reference_price.partition(":")
    out = Path(output_dir) if output_dir else None

    results: list[TicketResult] = []
    for mode, factory in engines.items():
        memory = MemorySink()
        sinks: list[Any] = [memory]
        if out is not None:
            sinks.append(JSONLSink(out / "traces" / mode))
        agent = SupportAgent(factory(Tracer(sinks)))
        for ticket in tickets:
            outcome, summary = agent.handle(
                ticket, World(), mode=mode, expected_action=ticket["expected"]["action"]
            )
            escalated, calls, tokens, ms = _trace_facts(memory.trace(outcome.trace_id or ""))
            memory.clear()
            expected = ticket["expected"]
            intent = outcome.decisions.get("intent") or {}
            result = TicketResult(
                ticket_id=str(ticket["id"]),
                mode=mode,
                expected_action=expected["action"],
                action=outcome.action,
                correct=outcome.action == expected["action"],
                expected_intent=expected.get("intent"),
                intent=intent.get("value") if intent.get("status") == "accepted" else None,
                intent_provider=intent.get("provider"),
                expected_order_id=expected.get("order_id"),
                order_id=outcome.order_id,
                reply_source=outcome.reply_source,
                grounded=outcome.grounded,
                decisions=outcome.decisions,
                escalated=escalated,
                decision_llm_calls=calls,
                decision_llm_tokens=tokens,
                decision_llm_ms=ms,
                summary=summary,
            )
            results.append(result)
            if on_ticket is not None:
                on_ticket(result)

    modes = {
        mode: _mode_report(
            mode, [r for r in results if r.mode == mode], by_id, (provider, model), prices
        )
        for mode in engines
    }
    benchmark = SupportBenchmark(
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        reasoning_model=reasoning_model,
        reference_price=reference_price,
        threshold=threshold,
        environment=environment_info(environment),
        modes=modes,
        tickets=results,
    )
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(
            json.dumps(benchmark.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return benchmark
