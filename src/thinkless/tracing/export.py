"""Production traces as data: labeling rows, and drift between two periods."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .sinks import iter_trace_files, read_trace
from .tracer import REDACTED

__all__ = [
    "DriftReport",
    "QuestionDrift",
    "drift_report",
    "export_trace_labels",
    "iter_decisions",
]


def _files(paths: Iterable[str | Path]) -> Iterator[Path]:
    for path in paths:
        target = Path(path)
        if target.is_dir():
            yield from iter_trace_files(target)
        elif target.suffix == ".jsonl":
            yield target


def iter_decisions(paths: Iterable[str | Path]) -> Iterator[dict[str, Any]]:
    """Every decision recorded in trace files or directories, one dict each.

    Each dict has the question name and kind, the input (``state``, or
    ``None`` when content capture was off), the value, status, plane,
    provider, confidence and whether it escalated, plus the trace and span
    ids to find it again.
    """
    for file in _files(paths):
        try:
            spans = read_trace(file)
        except (OSError, json.JSONDecodeError):
            continue
        shadow = any(
            s.get("parent_id") is None and (s.get("attributes") or {}).get("shadow") for s in spans
        )
        if shadow:
            continue
        for span in spans:
            if span.get("kind") != "decide":
                continue
            attrs = span.get("attributes") or {}
            state = attrs.get("state")
            kinds = {k: (v or {}).get("kind") for k, v in (attrs.get("questions") or {}).items()}
            for decision in attrs.get("decisions") or ():
                yield {
                    "trace_id": span.get("trace_id"),
                    "span_id": span.get("span_id"),
                    "start_ms": span.get("start_ms"),
                    "question": decision.get("name"),
                    "kind": decision.get("kind") or kinds.get(decision.get("name")),
                    "state": None if state in (None, REDACTED) else state,
                    "value": decision.get("value"),
                    "level": decision.get("level"),
                    "status": decision.get("status"),
                    "plane": decision.get("plane"),
                    "provider": decision.get("provider"),
                    "confidence": decision.get("confidence"),
                    "escalated": bool(decision.get("escalated")),
                    "attempts": decision.get("attempts") or [],
                }


def export_trace_labels(
    paths: Iterable[str | Path],
    out: str | Path,
    *,
    question: str | None = None,
    statuses: Sequence[str] | None = None,
    planes: Sequence[str] | None = None,
    limit: int | None = None,
    seed: int = 0,
) -> int:
    """Write traced decisions as rows to label, in the format ``thinkless calibrate`` reads.

    Rows hold ``text`` (the input), ``label`` (the decision's value when it
    was accepted, else ``null``), and the decision's status, plane, provider
    and confidence. Decisions traced without content are skipped.

    Args:
        paths: Trace files or directories.
        out: Output JSONL path.
        question: Only this question.
        statuses: Only these statuses, for example ``["uncertain"]``.
        planes: Only decisions settled by these planes.
        limit: At most this many rows, sampled at random with ``seed``.

    Returns:
        Rows written.
    """
    rows = []
    for record in iter_decisions(paths):
        if record["state"] is None:
            continue
        if question and record["question"] != question:
            continue
        if statuses and record["status"] not in statuses:
            continue
        if planes and (record["plane"] or "unresolved") not in planes:
            continue
        value = (
            record["level"] if record["kind"] == "score" and record["level"] else record["value"]
        )
        rows.append(
            {
                "text": record["state"],
                "label": value if record["status"] == "accepted" else None,
                "question": record["question"],
                "value": value,
                "status": record["status"],
                "plane": record["plane"],
                "provider": record["provider"],
                "confidence": record["confidence"],
                "trace_id": record["trace_id"],
            }
        )
    if limit is not None and len(rows) > limit:
        rows = random.Random(seed).sample(rows, limit)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


class QuestionDrift(BaseModel):
    """How one question's decisions moved between two periods.

    Attributes:
        llm_share: Share of decisions an LLM settled, before and after.
        uncertain_share: Share that came back uncertain or abstained.
        escalation_share: Share where an earlier provider was passed over.
        mean_confidence: Mean confidence of calibrated answers.
        answer_shift: Total variation distance between the two answer
            distributions: 0 when identical, 1 when disjoint.
        flags: What crossed an alert threshold.
    """

    question: str
    baseline: int
    current: int
    llm_share: tuple[float, float]
    uncertain_share: tuple[float, float]
    escalation_share: tuple[float, float]
    mean_confidence: tuple[float | None, float | None]
    answer_shift: float
    flags: list[str]


class DriftReport(BaseModel):
    questions: list[QuestionDrift]

    @property
    def drifted(self) -> list[QuestionDrift]:
        return [q for q in self.questions if q.flags]


def _period(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["question"])].append(record)
    return grouped


def _share(rows: Sequence[dict[str, Any]], test: Any) -> float:
    return sum(1 for r in rows if test(r)) / len(rows) if rows else 0.0


def _answer(record: dict[str, Any]) -> str:
    value = record["level"] if record.get("level") else record["value"]
    return json.dumps(value, sort_keys=True, default=str)


def _mean_confidence(rows: Sequence[dict[str, Any]]) -> float | None:
    values = [float(r["confidence"]) for r in rows if r.get("confidence") is not None]
    return sum(values) / len(values) if values else None


def drift_report(
    baseline: Iterable[str | Path],
    current: Iterable[str | Path],
    *,
    llm_share_points: float = 0.10,
    unsure_points: float = 0.10,
    answer_shift_limit: float = 0.15,
    confidence_drop: float = 0.05,
    min_decisions: int = 30,
) -> DriftReport:
    """Compare decisions in two sets of traces, question by question.

    A question is flagged when the share reaching an LLM rises by more than
    ``llm_share_points``, when the share not accepted (uncertain or
    abstained) rises by more than ``unsure_points``, when the answer
    distribution moves by more than
    ``answer_shift_limit`` (total variation distance), or when the mean
    confidence of calibrated answers drops by more than ``confidence_drop``.
    Questions with fewer than ``min_decisions`` in either period are
    reported without flags.
    """
    before = _period(iter_decisions(baseline))
    after = _period(iter_decisions(current))
    reports = []
    for name in sorted(set(before) | set(after)):
        a, b = before.get(name, []), after.get(name, [])
        llm = (_share(a, lambda r: r["plane"] == "llm"), _share(b, lambda r: r["plane"] == "llm"))
        unsure = (
            _share(a, lambda r: r["status"] != "accepted"),
            _share(b, lambda r: r["status"] != "accepted"),
        )
        escalated = (_share(a, lambda r: r["escalated"]), _share(b, lambda r: r["escalated"]))
        confidence = (_mean_confidence(a), _mean_confidence(b))
        dist_a, dist_b = Counter(map(_answer, a)), Counter(map(_answer, b))
        shift = 0.0
        if a and b:
            labels = set(dist_a) | set(dist_b)
            shift = 0.5 * sum(abs(dist_a[x] / len(a) - dist_b[x] / len(b)) for x in labels)
        flags: list[str] = []
        if len(a) >= min_decisions and len(b) >= min_decisions:
            if llm[1] - llm[0] > llm_share_points:
                flags.append(f"LLM share rose from {llm[0] * 100:.0f}% to {llm[1] * 100:.0f}%")
            if unsure[1] - unsure[0] > unsure_points:
                flags.append(
                    f"share not accepted rose from {unsure[0] * 100:.0f}% to {unsure[1] * 100:.0f}%"
                )
            if shift > answer_shift_limit:
                flags.append(f"answers shifted by {shift:.2f} (total variation)")
            if (
                confidence[0] is not None
                and confidence[1] is not None
                and confidence[0] - confidence[1] > confidence_drop
            ):
                flags.append(
                    f"mean confidence fell from {confidence[0]:.2f} to {confidence[1]:.2f}"
                )
        reports.append(
            QuestionDrift(
                question=name,
                baseline=len(a),
                current=len(b),
                llm_share=llm,
                uncertain_share=unsure,
                escalation_share=escalated,
                mean_confidence=confidence,
                answer_shift=round(shift, 4),
                flags=flags,
            )
        )
    return DriftReport(questions=reports)
