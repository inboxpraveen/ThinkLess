from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, ClassVar

import pytest

from thinkless import Answer, Engine, Kind, MemorySink, Plane, Question, Tracer
from thinkless.providers import DecisionProvider, ProviderResult, State


class FakeProvider(DecisionProvider):
    """A provider whose answers are scripted per question name.

    ``answers`` maps a question name to an ``Answer``, to ``None`` (abstain) or
    to a callable ``(state, question) -> Answer | None``.
    """

    plane: ClassVar[Plane] = Plane.MODEL
    kinds: ClassVar[frozenset[Kind]] = frozenset(Kind)

    def __init__(
        self,
        name: str,
        answers: Mapping[str, Answer | Callable[[State, Question], Answer | None] | None]
        | None = None,
        *,
        plane: Plane | None = None,
        calibrated: bool = True,
        kinds: frozenset[Kind] | None = None,
        error: Exception | None = None,
        model: str = "fake-model",
        input_tokens: int = 10,
    ) -> None:
        self.name = name
        self._answers = dict(answers or {})
        if plane is not None:
            self.plane = plane  # type: ignore[misc]
        self.calibrated = calibrated  # type: ignore[misc]
        if kinds is not None:
            self.kinds = kinds  # type: ignore[misc]
        self._error = error
        self._model = model
        self._input_tokens = input_tokens
        self.calls: list[list[str]] = []

    def answer(self, state: State, questions: Mapping[str, Question]) -> ProviderResult:
        self.calls.append(list(questions))
        if self._error is not None:
            raise self._error
        answers: dict[str, Answer | None] = {}
        for key, question in questions.items():
            scripted = self._answers.get(key)
            answers[key] = scripted(state, question) if callable(scripted) else scripted
        from thinkless import Usage

        return ProviderResult(
            answers=answers, model=self._model, usage=Usage(input_tokens=self._input_tokens)
        )


@pytest.fixture
def memory() -> MemorySink:
    return MemorySink()


@pytest.fixture
def make_engine(memory: MemorySink) -> Callable[..., Engine]:
    def build(providers: list[DecisionProvider], **kwargs: Any) -> Engine:
        kwargs.setdefault("tracer", Tracer([memory]))
        return Engine(providers, **kwargs)

    return build


def choice_answer(value: str, probabilities: dict[str, float]) -> Answer:
    return Answer(value=value, probabilities=probabilities)


def yes_answer(p_yes: float) -> Answer:
    return Answer(value=p_yes >= 0.5, probabilities={"yes": p_yes, "no": 1 - p_yes})
