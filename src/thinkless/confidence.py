"""Confidence normalization.

Providers disagree about what "confidence" means. Measured on the same input,
Laya reports ``1 - normalized entropy`` for choices and ``max(p, 1 - p)`` for
yes/no questions, while TypeSafe documents ``(k * p_max - 1) / (k - 1)``. A
threshold of 0.8 would therefore mean something different depending on which
backend answered.

ThinkLess computes one confidence from each provider's probability
distribution and applies thresholds to that number only. The provider's own
fields are kept untouched in ``Decision.raw``.

The formula is the normalized maximum probability, the same one TypeSafe
documents for Jev::

    confidence = (k * p_max - 1) / (k - 1)

It is 0 for a uniform distribution over ``k`` outcomes and 1 when all mass sits
on one outcome. For a yes/no question (``k = 2``) it reduces to
``2 * max(p, 1 - p) - 1``, so a threshold of 0.8 requires ``p >= 0.9``.
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = ["from_distribution", "from_yes_probability", "normalize_distribution"]


def _clip(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def normalize_distribution(scores: Mapping[str, float]) -> dict[str, float]:
    """Rescale non-negative scores so they sum to 1.

    Useful for providers that emit independent per-label scores (for example
    sigmoid outputs) rather than a softmax distribution.
    """
    cleaned = {label: max(0.0, float(score)) for label, score in scores.items()}
    total = sum(cleaned.values())
    if total <= 0.0:
        uniform = 1.0 / len(cleaned) if cleaned else 0.0
        return dict.fromkeys(cleaned, uniform)
    return {label: score / total for label, score in cleaned.items()}


def from_distribution(probabilities: Mapping[str, float]) -> float:
    """Normalized confidence of a categorical distribution."""
    k = len(probabilities)
    if k == 0:
        return 0.0
    if k == 1:
        return 1.0
    p_max = max(probabilities.values())
    return _clip((k * p_max - 1.0) / (k - 1.0))


def from_yes_probability(p_yes: float) -> float:
    """Normalized confidence of a binary answer given ``P(yes)``."""
    p = _clip(float(p_yes))
    return _clip(2.0 * max(p, 1.0 - p) - 1.0)
