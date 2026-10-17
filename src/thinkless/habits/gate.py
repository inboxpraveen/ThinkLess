"""The gate: when a local answer is trusted, with a stated error bound.

ThinkLess answers a site locally only at confidence levels where, on calls
the student had not been trained on, its answers agreed with the LLM at
least ``target`` of the time, as a one-sided lower confidence bound
(Clopper-Pearson) at level ``confidence``.

Candidate thresholds are tested in a fixed order, from the most selective
(only the student's most confident answers) to the least, and testing stops
at the first threshold that fails. Testing in a fixed sequence controls the
chance of wrongly passing any threshold at ``1 - confidence`` without a
multiple-testing correction (the fixed-sequence procedure used by Learn Then
Test, Angelopoulos et al. 2021).

The bound is about agreement with the LLM on traffic like the traffic the
calls were drawn from. Live audits check that it still holds after the
student starts answering.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

__all__ = [
    "GateResult",
    "binomial_lower_bound",
    "binomial_upper_bound",
    "first_candidate",
    "min_evidence",
    "select_threshold",
]


def _log_binom_tail(k: int, n: int, p: float) -> float:
    """log P(X >= k) for X ~ Binomial(n, p)."""
    if k <= 0:
        return 0.0
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    terms = [
        math.lgamma(n + 1)
        - math.lgamma(i + 1)
        - math.lgamma(n - i + 1)
        + i * log_p
        + (n - i) * log_q
        for i in range(k, n + 1)
    ]
    top = max(terms)
    return top + math.log(sum(math.exp(t - top) for t in terms))


def binomial_lower_bound(successes: int, n: int, confidence: float = 0.95) -> float:
    """One-sided Clopper-Pearson lower bound on a success rate.

    The largest ``p`` such that seeing ``successes`` or more out of ``n``
    has probability at most ``1 - confidence``; the true rate is at least
    this with probability ``confidence``.
    """
    if n <= 0 or successes <= 0:
        return 0.0
    alpha = 1.0 - confidence
    if successes == n:
        return float(alpha ** (1.0 / n))
    log_alpha = math.log(alpha)
    low, high = 0.0, successes / n
    for _ in range(60):
        mid = (low + high) / 2
        if _log_binom_tail(successes, n, mid) > log_alpha:
            high = mid
        else:
            low = mid
    return low


def binomial_upper_bound(successes: int, n: int, confidence: float = 0.95) -> float:
    """One-sided Clopper-Pearson upper bound on a success rate."""
    if n <= 0:
        return 1.0
    return 1.0 - binomial_lower_bound(n - successes, n, confidence)


def min_evidence(target: float, confidence: float) -> int:
    """Fewest agreeing answers that can clear ``target`` even with no disagreement."""
    return math.ceil(math.log(1.0 - confidence) / math.log(target))


@lru_cache(maxsize=64)
def first_candidate(target: float, confidence: float) -> int:
    """Size of the first threshold tested.

    The smallest sample on which an observed agreement halfway between the
    target and 1 clears the bound. Starting smaller would make the first test
    demand perfect agreement, so a single disagreement among the most
    confident answers would switch a good site off.
    """
    halfway = (1.0 + target) / 2
    n = min_evidence(target, confidence)
    while binomial_lower_bound(math.floor(n * halfway), n, confidence) < target:
        n = math.ceil(n * 1.05)
    return n


@dataclass(frozen=True)
class GateResult:
    """The chosen threshold and the evidence behind it.

    Attributes:
        threshold: Answers with at least this joint confidence are given locally.
        accepted: Held-out calls at or above the threshold.
        agreed: How many of those matched the LLM.
        calls: Held-out calls in total.
        lower_bound: The Clopper-Pearson lower bound on agreement at the threshold.
    """

    threshold: float
    accepted: int
    agreed: int
    calls: int
    lower_bound: float

    @property
    def coverage(self) -> float:
        return self.accepted / self.calls if self.calls else 0.0

    @property
    def agreement(self) -> float:
        return self.agreed / self.accepted if self.accepted else 0.0


def select_threshold(
    scores: Sequence[float],
    agreed: Sequence[bool],
    *,
    target: float = 0.97,
    confidence: float = 0.95,
    step: float = 1.1,
) -> GateResult | None:
    """The least selective threshold that passes, testing in fixed sequence.

    Candidates accept the ``k`` most confident held-out answers, for ``k``
    starting at :func:`first_candidate` (or half the answers, when there are
    too few) and growing by ``step``. The order depends only on the scores and
    the number of answers, never on the agreement, which is what makes the
    fixed sequence valid.

    Args:
        scores: Joint confidence of each held-out answer.
        agreed: Whether each answer matched the LLM.
        target: Required agreement.
        confidence: Level of the lower bound, and one minus the chance of
            passing a threshold whose true agreement is below ``target``.

    Returns:
        ``None`` when even the most selective candidate fails.
    """
    pairs = sorted(zip(scores, agreed, strict=True), key=lambda pair: pair[0], reverse=True)
    n = len(pairs)
    if n < min_evidence(target, confidence):
        return None
    first = max(min_evidence(target, confidence), min(n // 2, first_candidate(target, confidence)))
    prefix = [0]
    for _, ok in pairs:
        prefix.append(prefix[-1] + int(ok))
    best: GateResult | None = None
    k = first
    while k <= n:
        # Ties share a threshold, so the candidate takes every answer tied with the k-th.
        threshold = pairs[k - 1][0]
        end = k
        while end < n and pairs[end][0] >= threshold:
            end += 1
        bound = binomial_lower_bound(prefix[end], end, confidence)
        if bound < target:
            break
        best = GateResult(threshold, end, prefix[end], n, bound)
        if end == n:
            break
        k = max(end + 1, math.ceil(end * step))
    return best
