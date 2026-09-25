"""Metrics for the decision benchmark, computed only from raw per-row predictions.

Nothing here trusts a number a submitter reported: every metric is derived
from the predictions in a result file and the labels in the published task
files, which is what makes results checkable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from ...questions import Question, Score
from ...shadow.compare import comparable
from ..metrics import expected_calibration_error, percentile, threshold_sweep, wilson_interval

__all__ = ["TaskMetrics", "correct_rows", "score_task"]


class TaskMetrics(BaseModel):
    """What one provider scored on one task.

    Attributes:
        rows: Test rows.
        answered: Rows with an answer (not abstained, no error).
        accuracy: Share of all test rows answered correctly; an abstention
            counts as wrong. ``accuracy_low``/``accuracy_high`` bound it at 95%.
        macro_f1: Choice tasks, averaged over the classes present in the test rows.
        balanced_accuracy: Yes/no tasks, the mean of the recall on each answer.
        mae_levels, within_one: Score tasks: mean absolute error in levels, and
            the share within one level of the label.
        field_precision, field_recall, field_f1: Extract tasks, over non-empty fields.
        ece, brier: Calibration of the reported confidence against correctness.
        threshold: The lowest threshold whose accepted calibration answers reach
            ``target`` accuracy; ``None`` when none does or there is no confidence.
        coverage, selective_accuracy: At that threshold on the test rows: the
            share answered alone, and the accuracy of those answers.
        aurc: Area under the risk-coverage curve on the test rows (lower is
            better): the mean error rate over all coverage levels when answers
            are taken in order of confidence.
        latency_p50_ms, latency_p95_ms: Per-row wall time.
        cost_per_1k_usd: Billed or estimated cost per 1,000 test rows.
        errors: Rows where the provider failed.
    """

    rows: int
    answered: int
    accuracy: float
    accuracy_low: float
    accuracy_high: float
    macro_f1: float | None = None
    balanced_accuracy: float | None = None
    mae_levels: float | None = None
    within_one: float | None = None
    field_precision: float | None = None
    field_recall: float | None = None
    field_f1: float | None = None
    calibrated: bool
    ece: float | None = None
    brier: float | None = None
    target: float
    threshold: float | None = None
    coverage: float | None = None
    selective_accuracy: float | None = None
    aurc: float | None = None
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    cost_per_1k_usd: float
    input_tokens_per_row: float
    errors: int


def _prediction(question: Question, row: Mapping[str, Any]) -> Any:
    return comparable(question, row.get("value"))


def correct_rows(
    question: Question, predictions: Sequence[Mapping[str, Any]], truth: Mapping[str, Any]
) -> list[bool]:
    """Whether each prediction matches its label; missing answers are wrong."""
    out = []
    for row in predictions:
        predicted = _prediction(question, row)
        expected = comparable(question, truth[row["id"]])
        out.append(predicted is not None and predicted == expected)
    return out


def _macro_f1(pairs: Sequence[tuple[Any, Any]]) -> float:
    classes = {t for t, _ in pairs}
    scores = []
    for label in classes:
        tp = sum(1 for t, p in pairs if t == label and p == label)
        fp = sum(1 for t, p in pairs if t != label and p == label)
        fn = sum(1 for t, p in pairs if t == label and p != label)
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def _aurc(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    order = sorted(range(len(confidences)), key=lambda i: -confidences[i])
    errors = 0
    total = 0.0
    for rank, index in enumerate(order, start=1):
        errors += 0 if correct[index] else 1
        total += errors / rank
    return total / len(order) if order else 0.0


def _fit_threshold(
    confidences: Sequence[float], correct: Sequence[bool], target: float
) -> float | None:
    points = threshold_sweep(
        confidences, correct, thresholds=[round(x * 0.01, 2) for x in range(0, 100)]
    )
    for point in sorted(points, key=lambda p: p.threshold):
        if point.accuracy is not None and point.accuracy >= target and point.coverage > 0:
            return point.threshold
    return None


def score_task(
    question: Question,
    test: Sequence[Mapping[str, Any]],
    truth: Mapping[str, Any],
    *,
    calibration: Sequence[Mapping[str, Any]] | None = None,
    calibration_truth: Mapping[str, Any] | None = None,
    target: float = 0.95,
) -> TaskMetrics:
    """Score test predictions, fitting the selective threshold on calibration predictions."""
    correct = correct_rows(question, test, truth)
    n = len(test)
    hits = sum(correct)
    low, high = wilson_interval(hits, n)
    answered = sum(1 for r in test if r.get("value") is not None and not r.get("error"))
    latencies = [float(r["latency_ms"]) for r in test if r.get("latency_ms") is not None]
    metrics: dict[str, Any] = {
        "rows": n,
        "answered": answered,
        "accuracy": hits / n if n else 0.0,
        "accuracy_low": low,
        "accuracy_high": high,
        "latency_p50_ms": percentile(latencies, 50) if latencies else None,
        "latency_p95_ms": percentile(latencies, 95) if latencies else None,
        "cost_per_1k_usd": sum(float(r.get("cost_usd") or 0.0) for r in test) / n * 1000
        if n
        else 0.0,
        "input_tokens_per_row": sum(int(r.get("input_tokens") or 0) for r in test) / n
        if n
        else 0.0,
        "errors": sum(1 for r in test if r.get("error")),
        "target": target,
    }

    kind = question.kind.value
    pairs = [(comparable(question, truth[r["id"]]), _prediction(question, r)) for r in test]
    if kind == "choice":
        metrics["macro_f1"] = _macro_f1(pairs)
    elif kind == "yes_no":
        recalls = []
        for answer in (True, False):
            relevant = [p for t, p in pairs if t is answer]
            if relevant:
                recalls.append(sum(1 for p in relevant if p is answer) / len(relevant))
        metrics["balanced_accuracy"] = sum(recalls) / len(recalls) if recalls else None
    elif kind == "score" and isinstance(question, Score):
        levels = [comparable(question, level) for level in question.levels]
        distances = [
            abs(levels.index(t) - levels.index(p)) if p in levels else len(levels) - 1
            for t, p in pairs
        ]
        metrics["mae_levels"] = sum(distances) / len(distances) if distances else None
        metrics["within_one"] = (
            sum(1 for d in distances if d <= 1) / len(distances) if distances else None
        )
    elif kind == "extract":
        tp = fp = fn = 0
        for t, p in pairs:
            t = t or {}
            p = p or {}
            for field, expected in t.items():
                got = p.get(field)
                if expected is not None and got == expected:
                    tp += 1
                else:
                    if expected is not None:
                        fn += 1
                    if got is not None:
                        fp += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        metrics["field_precision"] = precision
        metrics["field_recall"] = recall
        metrics["field_f1"] = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )

    # A provider is calibrated here when every answer it gave carries a confidence.
    confidences = [r.get("confidence") for r in test]
    given = [c for c, r in zip(confidences, test, strict=True) if r.get("value") is not None]
    calibrated = bool(given) and all(c is not None for c in given)
    metrics["calibrated"] = calibrated
    if calibrated:
        values = [float(c) if c is not None else 0.0 for c in confidences]
        metrics["ece"] = expected_calibration_error(values, correct)
        metrics["brier"] = (
            sum((v - (1.0 if ok else 0.0)) ** 2 for v, ok in zip(values, correct, strict=True)) / n
        )
        metrics["aurc"] = _aurc(values, correct)
        if calibration and calibration_truth:
            cal_correct = correct_rows(question, calibration, calibration_truth)
            cal_values = [float(r.get("confidence") or 0.0) for r in calibration]
            threshold = _fit_threshold(cal_values, cal_correct, target)
            metrics["threshold"] = threshold
            if threshold is not None:
                accepted = [ok for v, ok in zip(values, correct, strict=True) if v >= threshold]
                metrics["coverage"] = len(accepted) / n
                metrics["selective_accuracy"] = sum(accepted) / len(accepted) if accepted else None
            else:
                metrics["coverage"] = 0.0
    return TaskMetrics.model_validate(metrics)
