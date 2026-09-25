"""Metrics shared by the benchmarks and the calibration tool."""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel

__all__ = [
    "ThresholdPoint",
    "expected_calibration_error",
    "percentile",
    "recommend_threshold",
    "threshold_sweep",
    "wilson_interval",
]


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, ``q`` in [0, 100]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * q / 100.0
    low, high = math.floor(rank), math.ceil(rank)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (rank - low))


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], bins: int = 10
) -> float:
    """ECE: the gap between confidence and accuracy, averaged over equal-width bins."""
    if not confidences:
        return 0.0
    total = len(confidences)
    error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [
            i
            for i, c in enumerate(confidences)
            if (low <= c < high) or (index == bins - 1 and c == 1.0)
        ]
        if not members:
            continue
        accuracy = sum(correct[i] for i in members) / len(members)
        confidence = sum(confidences[i] for i in members) / len(members)
        error += abs(accuracy - confidence) * len(members) / total
    return error


class ThresholdPoint(BaseModel):
    """What happens at one threshold.

    Attributes:
        threshold: Minimum confidence to accept the model's answer.
        coverage: Share of inputs the model answers on its own.
        accuracy: Accuracy on the inputs it answers.
        cascade_accuracy: Accuracy of the whole cascade when the rest go to
            the fallback (only when fallback predictions are supplied).
    """

    threshold: float
    coverage: float
    accuracy: float | None
    cascade_accuracy: float | None = None


def threshold_sweep(
    confidences: Sequence[float],
    correct: Sequence[bool],
    fallback_correct: Sequence[bool] | None = None,
    thresholds: Sequence[float] | None = None,
) -> list[ThresholdPoint]:
    """Coverage and accuracy of a model across thresholds.

    With ``fallback_correct``, also reports the accuracy of a cascade that
    sends every input below threshold to the fallback.
    """
    grid = thresholds or [round(x * 0.05, 2) for x in range(0, 20)] + [0.97, 0.99]
    points = []
    n = len(confidences)
    for threshold in grid:
        accepted = [i for i in range(n) if confidences[i] >= threshold]
        coverage = len(accepted) / n if n else 0.0
        accuracy = sum(correct[i] for i in accepted) / len(accepted) if accepted else None
        cascade = None
        if fallback_correct is not None and n:
            hits = sum(
                correct[i] if confidences[i] >= threshold else fallback_correct[i] for i in range(n)
            )
            cascade = hits / n
        points.append(
            ThresholdPoint(
                threshold=threshold, coverage=coverage, accuracy=accuracy, cascade_accuracy=cascade
            )
        )
    return points


def recommend_threshold(
    points: Sequence[ThresholdPoint], target_accuracy: float
) -> ThresholdPoint | None:
    """The lowest threshold whose accepted answers reach ``target_accuracy``.

    Lowest threshold means the highest coverage, so the fewest escalations.
    """
    for point in sorted(points, key=lambda p: p.threshold):
        if point.accuracy is not None and point.accuracy >= target_accuracy and point.coverage > 0:
            return point
    return None


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion, 95% by default.

    It stays inside [0, 1] and behaves at small ``n`` and at rates near 0 or
    1, which is where agreement numbers usually sit.
    """
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)
