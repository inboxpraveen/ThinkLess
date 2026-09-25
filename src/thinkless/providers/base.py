"""The decision provider interface."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ..decision import Answer, Plane, Usage
from ..questions import Kind, Question

__all__ = ["DecisionProvider", "ProviderResult", "State", "render_state"]

State = str | Mapping[str, Any] | Sequence[Any]
"""What a question is asked about: text, a JSON-like object, or a list of either."""


def render_state(state: State) -> str:
    """Flatten a state into text for providers that only read strings.

    Mappings become ``key: value`` lines (nested values as compact JSON) so
    field names stay visible to the model.
    """
    if isinstance(state, str):
        return state
    if isinstance(state, Mapping):
        lines = []
        for key, value in state.items():
            if value is None or value == "":
                continue
            text = (
                value
                if isinstance(value, str)
                else json.dumps(value, ensure_ascii=False, default=str)
            )
            lines.append(f"{key}: {text}")
        return "\n".join(lines)
    if isinstance(state, Sequence):
        return "\n".join(render_state(item) for item in state)
    return str(state)


class ProviderResult(BaseModel):
    """What a provider returns for one batched call.

    Attributes:
        answers: One entry per question asked. ``None`` means the provider
            abstained on that question.
        model: The model that served the call, as reported by the backend.
        usage: Tokens consumed by the call.
        meta: Structural details recorded on the trace span.
        content: Prompts and raw completions; recorded only when content
            capture is on.
    """

    model_config = ConfigDict(extra="forbid")

    answers: dict[str, Answer | None]
    model: str | None = None
    usage: Usage = Field(default_factory=Usage)
    meta: dict[str, Any] = Field(default_factory=dict)
    content: dict[str, Any] = Field(default_factory=dict)


class DecisionProvider(ABC):
    """Answers typed questions.

    Subclasses set:
        name: Unique name within an engine, used in traces and allowlists.
        plane: ``rule``, ``model`` or ``llm``.
        kinds: Question kinds the provider can answer.
        calibrated: Whether its probabilities are meaningful enough to
            threshold. Prompted LLMs are not.
        price_key: Provider id used for price lookup.
    """

    name: str = "provider"
    plane: ClassVar[Plane] = Plane.MODEL
    kinds: ClassVar[frozenset[Kind]] = frozenset()
    calibrated: ClassVar[bool] = True
    price_key: str = "local"

    def supports(self, question: Question) -> bool:
        return question.kind in self.kinds and question.allows(self.name)

    @abstractmethod
    def answer(self, state: State, questions: Mapping[str, Question]) -> ProviderResult:
        """Answer every question in one call where the backend allows it."""
        raise NotImplementedError

    def warmup(self) -> None:  # noqa: B027 - optional hook
        """Load weights ahead of the first call."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
