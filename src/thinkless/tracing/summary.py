"""Roll-ups computed from the spans of one trace."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, Field

__all__ = ["TraceSummary", "summarize"]

PLANES = ("rule", "model", "llm", "tool", "code")


class TraceSummary(BaseModel):
    """What a run cost and where its time went.

    Attributes:
        llm_calls: Every call to a generative model, whether it generated text
            or answered decision questions.
        generation_calls: LLM calls made through ``Engine.generate``.
        decisions: Questions resolved in the run, plus deterministic checks
            recorded with ``Engine.rule``.
        decisions_by_plane: How many decisions each plane resolved.
            ``unresolved`` counts abstentions.
        uncertain: Decisions that ended below threshold on every provider.
        escalations: Decisions where a provider answered below threshold (or
            failed) and the next provider was asked.
        llm_input_tokens, llm_output_tokens: Tokens spent on generative models.
        model_input_tokens: Tokens processed by small decision models.
        cost_usd: Estimated spend, from the configured price table.
        time_by_plane_ms: Time inside leaf work (provider attempts, generations,
            tools and rules) grouped by plane.
    """

    trace_id: str
    name: str
    status: str
    started_at: str
    duration_ms: float
    attributes: dict[str, Any] = Field(default_factory=dict)
    llm_calls: int = 0
    generation_calls: int = 0
    decisions: int = 0
    decisions_by_plane: dict[str, int] = Field(default_factory=dict)
    uncertain: int = 0
    escalations: int = 0
    tool_calls: int = 0
    errors: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    model_input_tokens: int = 0
    cost_usd: float = 0.0
    time_by_plane_ms: dict[str, float] = Field(default_factory=dict)

    @property
    def llm_tokens(self) -> int:
        return self.llm_input_tokens + self.llm_output_tokens


def _usage(attrs: Mapping[str, Any]) -> tuple[int, int]:
    usage = attrs.get("usage") or {}
    return int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)


def summarize(spans: Iterable[Mapping[str, Any]]) -> TraceSummary:
    """Summarize one trace.

    Args:
        spans: Span dictionaries of a single trace, in any order.

    Raises:
        ValueError: If there are no spans.
    """
    spans = list(spans)
    if not spans:
        raise ValueError("cannot summarize an empty trace")
    root = next((s for s in spans if s.get("parent_id") is None), None)
    if root is None:
        root = min(spans, key=lambda s: s.get("start_ms") or 0.0)

    by_plane: Counter[str] = Counter()
    time_by_plane: defaultdict[str, float] = defaultdict(float)
    summary = TraceSummary(
        trace_id=root["trace_id"],
        name=root["name"],
        status=root.get("status", "ok"),
        started_at=time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime((root.get("start_ms") or 0.0) / 1000.0)
        ),
        duration_ms=float(root.get("duration_ms") or 0.0),
        attributes={k: v for k, v in (root.get("attributes") or {}).items() if k != "input"},
    )

    for span in spans:
        kind = span.get("kind")
        plane = span.get("plane")
        attrs = span.get("attributes") or {}
        duration = float(span.get("duration_ms") or 0.0)
        if span.get("status") == "error":
            summary.errors += 1
        summary.cost_usd += float(attrs.get("cost_usd") or 0.0)

        if kind in ("attempt", "llm", "tool", "rule") and plane:
            time_by_plane[plane] += duration

        if kind == "llm":
            summary.llm_calls += 1
            summary.generation_calls += 1
            inp, out = _usage(attrs)
            summary.llm_input_tokens += inp
            summary.llm_output_tokens += out
        elif kind == "attempt" and not attrs.get("skipped"):
            inp, out = _usage(attrs)
            if plane == "llm":
                summary.llm_calls += 1
                summary.llm_input_tokens += inp
                summary.llm_output_tokens += out
            elif plane == "model":
                summary.model_input_tokens += inp
        elif kind == "tool":
            summary.tool_calls += 1
        elif kind == "rule":
            summary.decisions += 1
            by_plane["rule"] += 1
        elif kind == "decide":
            for decision in attrs.get("decisions") or []:
                summary.decisions += 1
                by_plane[decision.get("plane") or "unresolved"] += 1
                if decision.get("status") == "uncertain":
                    summary.uncertain += 1
                escalated = decision.get("escalated")
                if escalated is None:
                    escalated = len(decision.get("attempts") or []) > 1
                if escalated:
                    summary.escalations += 1

    summary.cost_usd = round(summary.cost_usd, 8)
    summary.decisions_by_plane = dict(by_plane)
    summary.time_by_plane_ms = {k: round(v, 3) for k, v in time_by_plane.items()}
    return summary
