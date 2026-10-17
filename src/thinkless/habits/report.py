"""Reports over a call log: what the calls are, what they cost, what ThinkLess did.

:func:`build_report` works on any call log, with or without a plan. Without a
plan it answers the first question a team has: how many of our LLM calls are
decisions, and what do they cost? With a plan it adds each site's status,
its threshold and evidence, and what ThinkLess answered locally.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..decision import Usage
from ..pricing import PriceTable, default_prices
from .calls import Observation
from .sites import SiteProfile, discover

__all__ = ["FieldSummary", "Report", "SiteSummary", "build_report", "estimate_cost"]

HOSTS = {
    "api.openai.com": "openai",
    "openrouter.ai": "openrouter",
    "api.anthropic.com": "anthropic",
    "generativelanguage.googleapis.com": "google",
    "api.groq.com": "groq",
    "api.together.xyz": "together",
    "api.mistral.ai": "mistral",
    "api.deepseek.com": "deepseek",
    "api.fireworks.ai": "fireworks",
}
LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1", "host.docker.internal")


Rates = Mapping[str, tuple[float, float]]
"""Model id to (input, output) USD per million tokens, overriding the price table."""


def parse_rates(specs: Iterable[str]) -> dict[str, tuple[float, float]]:
    """Read ``MODEL=INPUT,OUTPUT`` specs (USD per million tokens)."""
    rates: dict[str, tuple[float, float]] = {}
    for spec in specs:
        model, _, numbers = spec.partition("=")
        parts = numbers.split(",")
        if not model or len(parts) != 2:
            raise ValueError(f"expected MODEL=INPUT,OUTPUT, got {spec!r}")
        rates[model.strip()] = (float(parts[0]), float(parts[1]))
    return rates


def estimate_cost(
    observation: Observation, prices: PriceTable, rates: Rates | None = None
) -> float | None:
    """The call's reported cost, or an estimate from ``rates`` or the price table, or ``None``."""
    if observation.cost_usd is not None:
        return observation.cost_usd
    output = observation.output
    model = (output.model if output is not None else None) or observation.call.model
    if rates and output is not None:
        for name in (model, observation.call.model):
            if name in rates:
                rate_in, rate_out = rates[name]
                return (output.input_tokens * rate_in + output.output_tokens * rate_out) / 1e6
    host = (observation.host or "").lower()
    if host in LOCAL_HOSTS:
        return 0.0
    provider = HOSTS.get(host)
    if provider is None or output is None:
        return None
    usage = Usage(input_tokens=output.input_tokens, output_tokens=output.output_tokens)
    dollars, known = prices.cost(provider, output.model or observation.call.model, usage)
    return dollars if known else None


@dataclass
class FieldSummary:
    name: str
    kind: str
    options: int
    top: list[tuple[str, int]]

    def describe(self) -> str:
        if self.kind == "yes_no":
            return f"{self.name}: yes/no"
        if self.kind in ("choice", "score"):
            return f"{self.name}: {self.options} options"
        if self.kind == "constant":
            return f"{self.name}: always {self.top[0][0] if self.top else '?'}"
        if self.kind == "extract":
            return f"{self.name}: copied from the input"
        return f"{self.name}: free text"


@dataclass
class SiteSummary:
    """One site in a report.

    Attributes:
        cost_usd: What the LLM calls of the site cost, where known.
        unknown_cost: LLM calls whose cost is not known.
        local: Calls ThinkLess answered.
        audits: Calls ThinkLess answered but sent to the LLM to check.
        audit_agreed: Audits where the LLM gave the same answer.
        shadow: Calls the LLM answered while a student existed.
        shadow_agreed: Of those, how many the student answered the same way.
        saved_usd: The site's mean LLM cost per call times ``local``.
    """

    id: str
    name: str
    model: str
    contract: str
    calls: int
    share: float
    verdict: str
    fields: list[FieldSummary]
    questions: list[dict[str, Any]]
    cost_usd: float = 0.0
    unknown_cost: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_latency_ms: float | None = None
    local: int = 0
    audits: int = 0
    audit_agreed: int = 0
    shadow: int = 0
    shadow_agreed: int = 0
    saved_usd: float | None = None
    status: str | None = None
    threshold: float | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    examples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.__dict__)
        data["fields"] = [f.__dict__ for f in self.fields]
        return data


@dataclass
class Report:
    calls: int
    sites: list[SiteSummary]
    cost_usd: float
    unknown_cost: int

    @property
    def decision_calls(self) -> int:
        return sum(s.calls for s in self.sites if s.verdict == "decision")

    @property
    def decision_cost(self) -> float:
        return sum(s.cost_usd for s in self.sites if s.verdict == "decision")

    @property
    def mixed_calls(self) -> int:
        return sum(s.calls for s in self.sites if s.verdict == "mixed")

    @property
    def local(self) -> int:
        return sum(s.local for s in self.sites)

    @property
    def saved_usd(self) -> float | None:
        """LLM cost of the calls ThinkLess answered, where the site's cost per call is known."""
        known = [s.saved_usd for s in self.sites if s.saved_usd is not None]
        return sum(known) if known else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "cost_usd": self.cost_usd,
            "unknown_cost": self.unknown_cost,
            "decision_calls": self.decision_calls,
            "decision_cost_usd": self.decision_cost,
            "mixed_calls": self.mixed_calls,
            "local": self.local,
            "saved_usd": self.saved_usd,
            "sites": [s.to_dict() for s in self.sites],
        }


def _summary(
    site: SiteProfile,
    total: int,
    prices: PriceTable,
    plan: Mapping[str, Any] | None,
    answered_locally: int = 0,
    rates: Rates | None = None,
) -> SiteSummary:
    fields = [
        FieldSummary(
            name=f.name,
            kind=f.kind,
            options=len(f.options),
            top=[(f.display(label), count) for label, count in f.counts.most_common(3)],
        )
        for f in site.fields.values()
    ]
    summary = SiteSummary(
        id=site.id,
        name=site.name,
        model=site.call.model,
        contract=site.call.contract,
        calls=site.calls,
        share=site.calls / total if total else 0.0,
        verdict=site.verdict,
        fields=fields,
        questions=[q.spec() for q in site.questions()],
        examples=site.examples(),
    )
    summary.local = answered_locally
    summary.calls += answered_locally
    summary.share = summary.calls / total if total else 0.0
    latencies = []
    llm_costs = []
    for observation in site.observations:
        if observation.answered_by == "habit":
            summary.local += 1
            continue
        if observation.output is not None:
            summary.input_tokens += observation.output.input_tokens
            summary.output_tokens += observation.output.output_tokens
        cost = estimate_cost(observation, prices, rates)
        if cost is None:
            summary.unknown_cost += 1
        else:
            summary.cost_usd += cost
            llm_costs.append(cost)
        if observation.latency_ms is not None:
            latencies.append(observation.latency_ms)
        local = observation.local
        if observation.answered_by == "audit":
            summary.audits += 1
        if local and observation.output is not None:
            fields_local = local.get("fields") or {}
            agreed = all(
                observation.output.fields.get(name) == label for name, label in fields_local.items()
            )
            if observation.answered_by == "audit":
                summary.audit_agreed += agreed
            else:
                summary.shadow += 1
                summary.shadow_agreed += agreed
    if latencies:
        summary.llm_latency_ms = sum(latencies) / len(latencies)
    if llm_costs and summary.local:
        summary.saved_usd = sum(llm_costs) / len(llm_costs) * summary.local
    entry = (plan or {}).get("sites", {}).get(site.id)
    if entry:
        summary.status = entry.get("status")
        summary.threshold = entry.get("threshold")
        summary.evidence = dict(entry.get("evidence") or {})
        summary.reason = entry.get("reason")
    return summary


def build_report(
    observations: Iterable[Observation],
    *,
    plan: Mapping[str, Any] | None = None,
    prices: PriceTable | None = None,
    local: Mapping[str, int] | None = None,
    rates: Rates | None = None,
) -> Report:
    """Group the calls into sites and summarize each.

    Args:
        observations: The recorded calls.
        plan: The Habits plan, for each site's status and evidence.
        prices: Price table for calls with no reported cost.
        local: Calls ThinkLess answered per site id, from
            :func:`~thinkless.habits.calls.local_answers`.
        rates: Prices per model that override the price table.
    """
    items = list(observations)
    prices = prices or default_prices()
    sites = discover(items)
    local = dict(local or {})
    total = len(items) + sum(local.values())
    summaries = [
        _summary(site, total, prices, plan, local.pop(site.id, 0), rates) for site in sites
    ]
    return Report(
        calls=total,
        sites=summaries,
        cost_usd=sum(s.cost_usd for s in summaries),
        unknown_cost=sum(s.unknown_cost for s in summaries),
    )
