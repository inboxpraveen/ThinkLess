"""Results returned by the decision plane."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .questions import Kind

__all__ = ["Answer", "Attempt", "Decision", "Plane", "Status", "Usage"]


class Plane(str, Enum):
    """Where a piece of work ran.

    ``rule`` is deterministic code, ``model`` is a small decision model,
    ``llm`` is a generative model, ``tool`` is an action with side effects and
    ``code`` is application logic recorded for the trace.
    """

    RULE = "rule"
    MODEL = "model"
    LLM = "llm"
    TOOL = "tool"
    CODE = "code"


class Status(str, Enum):
    """Outcome of a decision after the cascade finished.

    ``accepted`` means an answer met its threshold (or came from a provider the
    policy trusts without calibration). ``uncertain`` means every provider
    answered below threshold; the best answer is returned and the application
    decides what to do. ``abstained`` means no provider produced an answer.
    """

    ACCEPTED = "accepted"
    UNCERTAIN = "uncertain"
    ABSTAINED = "abstained"


class Usage(BaseModel):
    """Token accounting for one call."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class Answer(BaseModel):
    """What a provider returns for one question, before policy is applied.

    Providers fill ``probabilities`` whenever they have a distribution; the
    engine derives the normalized confidence from it. ``confidence`` is only
    used when no distribution is available (extraction, for example).
    """

    model_config = ConfigDict(extra="forbid")

    value: Any
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    fields: dict[str, float | None] | None = None
    raw: dict[str, Any] | None = None


class Attempt(BaseModel):
    """One provider's try at one question."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    plane: Plane
    value: Any = None
    confidence: float | None = None
    latency_ms: float = 0.0
    accepted: bool = False
    reason: str = ""


class Decision(BaseModel):
    """The resolved answer to a question.

    Attributes:
        name: The question key.
        kind: The question kind.
        value: The answer. A label for ``Choice``, a float for ``Score``, a bool
            for ``YesNo`` and a dict of field values for ``Extract``. ``None``
            when the decision abstained.
        status: ``accepted``, ``uncertain`` or ``abstained``.
        confidence: Normalized confidence in [0, 1]. ``None`` when the
            answering provider is not calibrated (a prompted LLM, for example).
        probability: Probability of the chosen answer, when known.
        probabilities: Full distribution over answers, when known.
        level: For ``Score`` questions, the most likely level label.
        threshold: The threshold that was applied.
        provider: Name of the provider whose answer was used.
        plane: Plane of that provider.
        model: Model identifier reported by that provider.
        latency_ms: Wall time spent on this question across all attempts. When
            several questions share a provider call, the call time is counted
            once per question.
        attempts: Every provider that was tried, in order.
        usage: Tokens consumed by the answering provider's call.
        cost_usd: Estimated cost of the answering provider's call.
        raw: The provider's native payload for this question.
        span_id: Trace span that recorded this decision.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Kind
    value: Any = None
    status: Status
    confidence: float | None = None
    probability: float | None = None
    probabilities: dict[str, float] | None = None
    level: str | None = None
    fields: dict[str, float | None] | None = None
    threshold: float
    provider: str | None = None
    plane: Plane | None = None
    model: str | None = None
    latency_ms: float = 0.0
    attempts: list[Attempt] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float = 0.0
    raw: dict[str, Any] | None = None
    span_id: str | None = None

    @property
    def accepted(self) -> bool:
        return self.status is Status.ACCEPTED

    @property
    def escalated(self) -> bool:
        """True when an earlier provider answered (or failed) and was passed over.

        A provider that abstains, such as a rule that does not fire, hands the
        question on without an escalation: that is the cascade working as
        designed, at no cost.
        """
        return any(
            attempt.reason in ("below_threshold", "no_confidence")
            or attempt.reason.startswith("error")
            for attempt in self.attempts[:-1]
        )

    def is_(self, value: Any) -> bool:
        """True when the decision was accepted and equals ``value``.

        Convenient for routing code: ``if intent.is_("refund"): ...`` never
        fires on an uncertain answer.
        """
        return self.accepted and self.value == value

    def summary(self) -> dict[str, Any]:
        """Compact form used in trace attributes."""
        return {
            "name": self.name,
            "kind": self.kind.value,
            "value": self.value,
            "status": self.status.value,
            "confidence": None if self.confidence is None else round(self.confidence, 4),
            "threshold": self.threshold,
            "provider": self.provider,
            "plane": None if self.plane is None else self.plane.value,
            "attempts": [a.provider for a in self.attempts],
            "escalated": self.escalated,
            "level": self.level,
        }
