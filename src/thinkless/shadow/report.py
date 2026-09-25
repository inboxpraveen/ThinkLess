"""Turn a shadow log into agreement, projected savings and a verdict per question."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..bench.metrics import percentile, threshold_sweep, wilson_interval
from ..questions import Kind, Question, question_from_spec
from .compare import agree

__all__ = [
    "PlaneAgreement",
    "QuestionReport",
    "ShadowReport",
    "SideStats",
    "ThresholdAdvice",
    "build_report",
    "export_labels",
    "read_log",
]

FAST_PLANES = ("rule", "model")
SKIPPED = ("deadline", "spend_limit")


class SideStats(BaseModel):
    """Cost, latency and planes of one side of the comparison.

    Attributes:
        calls: Records with this side present.
        cost_known: Share of calls with a known cost.
        cost_per_1k_usd: Mean cost per 1,000 calls, over calls with a known cost.
        latency_p50_ms, latency_p95_ms: Latency of the calls that report one.
        planes: Share of calls settled by each plane (ThinkLess sides only).
        fast_share: Share settled by rules or small models, without an LLM.
        llm_share: Share that reached an LLM at any point in the cascade.
    """

    calls: int = 0
    cost_known: float = 0.0
    cost_per_1k_usd: float | None = None
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    planes: dict[str, float] | None = None
    fast_share: float | None = None
    llm_share: float | None = None


class PlaneAgreement(BaseModel):
    """Agreement on the calls one plane of the shadow engine settled."""

    calls: int
    agreement: float | None
    low: float | None


class ThresholdAdvice(BaseModel):
    """A threshold for one provider, fitted on production traffic.

    The recommendation is the lowest threshold whose accepted answers agree
    with the primary at the target rate or better, as a 95% lower bound. The
    primary's answers stand in for labels, so the advice is only as good as
    the primary. It uses the calls where this provider was asked, which
    excludes calls an earlier provider settled.
    """

    provider: str
    calls: int
    recommended: float | None
    coverage: float | None
    agreement: float | None


class QuestionReport(BaseModel):
    """Everything the report says about one question.

    ``by_plane`` groups agreement by the plane that answered in the shadow
    engine. ``primary_by_plane`` does the same for the primary, when the
    primary is a ThinkLess engine: that is the view for auditing a live
    engine against an LLM.
    """

    question: str
    kind: str
    records: int
    compared: int
    agreement: float | None
    agreement_low: float | None
    agreement_high: float | None
    fast_agreement: float | None = None
    fast_agreement_low: float | None = None
    errors: int = 0
    primary: SideStats
    shadow: SideStats
    by_plane: dict[str, PlaneAgreement] = Field(default_factory=dict)
    primary_by_plane: dict[str, PlaneAgreement] = Field(default_factory=dict)
    field_agreement: dict[str, float] | None = None
    savings_per_1k_usd: float | None = None
    savings_share: float | None = None
    thresholds: list[ThresholdAdvice] = Field(default_factory=list)
    verdict: str
    ready: bool


class ShadowReport(BaseModel):
    """The whole log, per question and in total."""

    source: str
    records: int
    target: float
    min_calls: int
    questions: list[QuestionReport]
    primary_cost_per_1k_usd: float | None
    shadow_cost_per_1k_usd: float | None
    savings_per_1k_usd: float | None
    savings_share: float | None
    monthly_volume: int | None = None
    monthly_savings_usd: float | None = None


def read_log(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield the records of a shadow log, skipping lines that do not parse."""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and "question" in record:
                yield record


def _question(record: dict[str, Any]) -> Question | None:
    spec = record.get("spec")
    if not isinstance(spec, dict):
        return None
    try:
        return question_from_spec(spec, name=record.get("question"))
    except ValueError:
        return None


def _comparable(question: Question | None, side: dict[str, Any]) -> Any:
    if question is not None and question.kind is Kind.SCORE and side.get("level") is not None:
        return side["level"]
    return side.get("value")


def _side_stats(sides: Sequence[dict[str, Any]]) -> SideStats:
    present = [s for s in sides if s and "error" not in s]
    if not present:
        return SideStats()
    costs = [float(s["cost_usd"]) for s in present if s.get("cost_usd") is not None]
    latencies = [float(s["latency_ms"]) for s in present if s.get("latency_ms") is not None]
    planes = [s.get("plane") for s in present if "status" in s]
    stats = SideStats(
        calls=len(present),
        cost_known=len(costs) / len(present),
        cost_per_1k_usd=sum(costs) / len(costs) * 1000 if costs else None,
        latency_p50_ms=percentile(latencies, 50) if latencies else None,
        latency_p95_ms=percentile(latencies, 95) if latencies else None,
    )
    if planes:
        counts: dict[str, int] = defaultdict(int)
        for plane in planes:
            counts[plane or "unresolved"] += 1
        stats.planes = {plane: n / len(planes) for plane, n in sorted(counts.items())}
        stats.fast_share = sum(counts.get(p, 0) for p in FAST_PLANES) / len(planes)
        reached_llm = sum(
            1
            for s in present
            if "status" in s
            and (
                s.get("plane") == "llm"
                or any(
                    a.get("plane") == "llm" and a.get("reason") not in SKIPPED
                    for a in s.get("attempts") or ()
                )
            )
        )
        stats.llm_share = reached_llm / len(planes)
    return stats


def _thresholds(
    question: Question | None, records: Sequence[dict[str, Any]], target: float
) -> list[ThresholdAdvice]:
    if question is None or question.kind not in (Kind.CHOICE, Kind.YES_NO):
        return []
    samples: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for record in records:
        shadow, primary = record.get("shadow") or {}, record.get("primary") or {}
        reference = primary.get("value")
        if reference is None:
            continue
        for attempt in shadow.get("attempts") or ():
            if attempt.get("plane") != "model" or attempt.get("confidence") is None:
                continue
            ok, _ = agree(question, reference, attempt.get("value"))
            if ok is None:
                continue
            samples[attempt["provider"]].append((float(attempt["confidence"]), ok))
    advice = []
    for provider, pairs in samples.items():
        points = threshold_sweep([c for c, _ in pairs], [ok for _, ok in pairs])
        # The lowest threshold whose accepted answers clear the target at 95%
        # confidence, not just on the point estimate: sampled traffic is noisy.
        best = None
        for point in sorted(points, key=lambda p: p.threshold):
            accepted = round(point.coverage * len(pairs))
            if point.accuracy is None or accepted == 0:
                continue
            if wilson_interval(round(point.accuracy * accepted), accepted)[0] >= target:
                best = point
                break
        advice.append(
            ThresholdAdvice(
                provider=provider,
                calls=len(pairs),
                recommended=None if best is None else best.threshold,
                coverage=None if best is None else best.coverage,
                agreement=None if best is None else best.accuracy,
            )
        )
    return advice


def _plane_agreement(groups: dict[str, list[bool]]) -> dict[str, PlaneAgreement]:
    return {
        plane: PlaneAgreement(
            calls=len(oks),
            agreement=sum(oks) / len(oks),
            low=wilson_interval(sum(oks), len(oks))[0],
        )
        for plane, oks in sorted(groups.items())
    }


def _question_report(
    key: str, records: Sequence[dict[str, Any]], target: float, min_calls: int
) -> QuestionReport:
    question = _question(records[-1])
    kind = str(records[-1].get("kind", ""))
    errors = sum(1 for r in records if "error" in (r.get("shadow") or {}))
    outcomes: list[bool] = []
    by_plane: dict[str, list[bool]] = defaultdict(list)
    primary_by_plane: dict[str, list[bool]] = defaultdict(list)
    fields: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        shadow, primary = record.get("shadow") or {}, record.get("primary") or {}
        if "error" in shadow:
            continue
        agreed = record.get("agree")
        if agreed is None and question is not None:
            agreed, _ = agree(
                question, _comparable(question, primary), _comparable(question, shadow)
            )
        if agreed is None:
            continue
        outcomes.append(bool(agreed))
        by_plane[shadow.get("plane") or "unresolved"].append(bool(agreed))
        if "status" in primary:
            primary_by_plane[primary.get("plane") or "unresolved"].append(bool(agreed))
        for name, ok in (record.get("fields") or {}).items():
            fields[name].append(bool(ok))

    compared = len(outcomes)
    hits = sum(outcomes)
    low, high = wilson_interval(hits, compared) if compared else (None, None)
    fast = [ok for plane in FAST_PLANES for ok in by_plane.get(plane, [])]
    fast_low = wilson_interval(sum(fast), len(fast))[0] if fast else None

    primary_stats = _side_stats([r.get("primary") or {} for r in records])
    shadow_stats = _side_stats([r.get("shadow") or {} for r in records])
    savings = share = None
    if (
        primary_stats.cost_per_1k_usd is not None
        and shadow_stats.cost_per_1k_usd is not None
        and primary_stats.cost_known == 1.0
    ):
        savings = primary_stats.cost_per_1k_usd - shadow_stats.cost_per_1k_usd
        if primary_stats.cost_per_1k_usd > 0:
            share = savings / primary_stats.cost_per_1k_usd

    if compared < min_calls:
        verdict, ready = f"collect more: {compared} of {min_calls} compared calls", False
    elif low is not None and low >= target:
        verdict, ready = f"ready: agreement is at least {low * 100:.1f}% at 95% confidence", True
    else:
        verdict = (
            f"not yet: agreement {hits / compared * 100:.1f}%, "
            f"lower bound {(low or 0.0) * 100:.1f}% is under the {target * 100:.0f}% target"
        )
        ready = False

    return QuestionReport(
        question=key,
        kind=kind,
        records=len(records),
        compared=compared,
        agreement=hits / compared if compared else None,
        agreement_low=low,
        agreement_high=high,
        fast_agreement=sum(fast) / len(fast) if fast else None,
        fast_agreement_low=fast_low,
        errors=errors,
        primary=primary_stats,
        shadow=shadow_stats,
        by_plane=_plane_agreement(by_plane),
        primary_by_plane=_plane_agreement(primary_by_plane),
        field_agreement={name: sum(oks) / len(oks) for name, oks in fields.items()} or None,
        savings_per_1k_usd=savings,
        savings_share=share,
        thresholds=_thresholds(question, records, target),
        verdict=verdict,
        ready=ready,
    )


def build_report(
    records: Iterable[dict[str, Any]],
    *,
    source: str = "",
    target: float = 0.95,
    min_calls: int = 100,
    monthly_volume: int | None = None,
) -> ShadowReport:
    """Summarize shadow records.

    Args:
        records: Records from :func:`read_log`.
        source: Where they came from, for the report header.
        target: Agreement the shadow must reach, as a 95% lower bound, for a
            question to be called ready.
        min_calls: Compared calls needed before any verdict.
        monthly_volume: Decisions a month, to project the monthly saving.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total = 0
    for record in records:
        grouped[str(record["question"])].append(record)
        total += 1
    questions = [
        _question_report(key, rows, target, min_calls) for key, rows in sorted(grouped.items())
    ]

    def weighted(attr: str) -> float | None:
        parts = [(getattr(q, attr).cost_per_1k_usd, q.records) for q in questions]
        if not parts or any(cost is None for cost, _ in parts):
            return None
        weight = sum(n for _, n in parts)
        return sum(float(cost or 0.0) * n for cost, n in parts) / weight if weight else None

    primary_cost, shadow_cost = weighted("primary"), weighted("shadow")
    savings = share = monthly = None
    if primary_cost is not None and shadow_cost is not None:
        savings = primary_cost - shadow_cost
        share = savings / primary_cost if primary_cost > 0 else None
        if monthly_volume:
            monthly = savings * monthly_volume / 1000
    return ShadowReport(
        source=source,
        records=total,
        target=target,
        min_calls=min_calls,
        questions=questions,
        primary_cost_per_1k_usd=primary_cost,
        shadow_cost_per_1k_usd=shadow_cost,
        savings_per_1k_usd=savings,
        savings_share=share,
        monthly_volume=monthly_volume,
        monthly_savings_usd=monthly,
    )


def export_labels(
    records: Iterable[dict[str, Any]],
    out: str | Path,
    *,
    question: str | None = None,
    disagreements_only: bool = True,
    label_from: str = "primary",
) -> int:
    """Write records as labeling rows that ``thinkless calibrate`` reads.

    Each row has ``text`` (the logged input), ``label`` (pre-filled from
    ``label_from``, or ``null``), both answers, the shadow's confidence and
    ``agree``. Correct the labels, then calibrate on the file.

    Args:
        records: Records from :func:`read_log`.
        out: Output JSONL path.
        question: Only this question.
        disagreements_only: Only calls where the two systems disagreed, which
            is where a person's label teaches the most.
        label_from: ``primary``, ``shadow`` or ``none``.

    Returns:
        Rows written. Records logged without content are skipped, since
        there is no input to label.
    """
    if label_from not in ("primary", "shadow", "none"):
        raise ValueError("label_from must be primary, shadow or none")
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            if question and record.get("question") != question:
                continue
            if disagreements_only and record.get("agree") is not False:
                continue
            if record.get("state") in (None, "[redacted]"):
                continue
            primary, shadow = record.get("primary") or {}, record.get("shadow") or {}
            label = (
                None if label_from == "none" else (primary if label_from == "primary" else shadow)
            )
            row = {
                "text": record["state"],
                "label": None if label is None else _comparable(_question(record), label),
                "question": record.get("question"),
                "primary": _comparable(_question(record), primary),
                "shadow": _comparable(_question(record), shadow),
                "shadow_confidence": shadow.get("confidence"),
                "shadow_plane": shadow.get("plane"),
                "agree": record.get("agree"),
                "state_sha": record.get("state_sha"),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    return written
