"""Deterministic rules: the cheapest and most predictable provider."""

from __future__ import annotations

import inspect
import re
from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import Any, ClassVar

from ..decision import Answer, Plane
from ..questions import Choice, Extract, Kind, Question, Score, YesNo
from .base import DecisionProvider, ProviderResult, State, render_state

__all__ = ["RuleFunction", "Rules"]

RuleFunction = Callable[..., Any]


def _one_hot(labels: list[str], chosen: str) -> dict[str, float]:
    return {label: 1.0 if label == chosen else 0.0 for label in labels}


class Rules(DecisionProvider):
    """Python functions that answer questions when they can.

    A rule is registered for a question name and called with the state (and,
    if it accepts a second argument, the question). It returns an answer to
    settle the question with full confidence, or ``None`` to pass it to the
    next provider in the cascade.

    Example:
        >>> rules = Rules()
        >>> @rules.rule("wants_human")
        ... def asks_for_person(state):
        ...     return True if "real person" in state["message"].lower() else None
        >>> rules.match("intent", r"\\bunsubscribe\\b", "cancel_subscription")
        >>> rules.extract("order", order_id=r"#\\s?(\\d{4,6})")
    """

    plane: ClassVar[Plane] = Plane.RULE
    kinds: ClassVar[frozenset[Kind]] = frozenset(Kind)

    def __init__(self, name: str = "rules") -> None:
        self.name = name
        self._rules: dict[str, list[RuleFunction]] = defaultdict(list)

    def add(self, question: str, fn: RuleFunction) -> RuleFunction:
        self._rules[question].append(fn)
        return fn

    def rule(self, question: str) -> Callable[[RuleFunction], RuleFunction]:
        """Decorator form of :meth:`add`."""

        def register(fn: RuleFunction) -> RuleFunction:
            return self.add(question, fn)

        return register

    def match(
        self,
        question: str,
        pattern: str,
        value: Any,
        *,
        field: str | None = None,
        flags: int = re.IGNORECASE,
    ) -> None:
        """Answer ``value`` when ``pattern`` matches the state text.

        Args:
            question: Question name.
            pattern: Regular expression searched in the rendered state, or in
                ``state[field]`` when ``field`` is given.
            value: The answer to return on a match.
        """
        compiled = re.compile(pattern, flags)

        def matcher(state: State) -> Any:
            text = _field_text(state, field)
            return value if compiled.search(text) else None

        matcher.__name__ = f"match_{question}"
        self.add(question, matcher)

    def extract(
        self,
        question: str,
        *,
        field: str | None = None,
        flags: int = re.IGNORECASE,
        **patterns: str,
    ) -> None:
        """Extract fields with regular expressions.

        Each keyword maps a field name to a pattern; the first capture group
        (or the whole match) becomes the value. The rule answers when at least
        one pattern matches.
        """
        compiled = {name: re.compile(p, flags) for name, p in patterns.items()}

        def extractor(state: State) -> dict[str, str] | None:
            text = _field_text(state, field)
            found = {}
            for name, regex in compiled.items():
                match = regex.search(text)
                if match:
                    found[name] = match.group(1) if match.groups() else match.group(0)
            return found or None

        extractor.__name__ = f"extract_{question}"
        self.add(question, extractor)

    def supports(self, question: Question) -> bool:
        return question.key in self._rules and question.allows(self.name)

    def answer(self, state: State, questions: Mapping[str, Question]) -> ProviderResult:
        answers: dict[str, Answer | None] = {}
        fired: dict[str, str] = {}
        for key, question in questions.items():
            answers[key] = None
            for fn in self._rules.get(question.key, ()):
                value = _call(fn, state, question)
                if value is None:
                    continue
                answers[key] = _to_answer(question, value)
                fired[key] = getattr(fn, "__name__", "rule")
                break
        return ProviderResult(answers=answers, model="rules", meta={"fired": fired})


def _field_text(state: State, field: str | None) -> str:
    if field is not None and isinstance(state, Mapping):
        return str(state.get(field) or "")
    return render_state(state)


def _call(fn: RuleFunction, state: State, question: Question) -> Any:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(state)
    positional = [
        p for p in params.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) >= 2 or any(p.kind is p.VAR_POSITIONAL for p in params.values()):
        return fn(state, question)
    return fn(state)


def _to_answer(question: Question, value: Any) -> Answer:
    """Validate a rule's return value against the question and wrap it."""
    if isinstance(question, Choice):
        label = str(value)
        if label not in question.options:
            raise ValueError(f"rule answered {label!r}, which is not an option of {question.key!r}")
        return Answer(value=label, probabilities=_one_hot(question.labels, label))
    if isinstance(question, Score):
        levels = list(question.levels)
        if isinstance(value, str):
            if value not in levels:
                raise ValueError(
                    f"rule answered {value!r}, which is not a level of {question.key!r}"
                )
            index = levels.index(value)
        else:
            index = round(float(value))
            if not 0 <= index < len(levels):
                raise ValueError(f"rule answered level {value!r} outside 0..{len(levels) - 1}")
        return Answer(value=float(index), probabilities=_one_hot(levels, levels[index]))
    if isinstance(question, YesNo):
        if not isinstance(value, bool):
            raise ValueError(f"rule for {question.key!r} must return a bool or None")
        return Answer(value=value, probabilities={"yes": float(value), "no": float(not value)})
    if isinstance(question, Extract):
        if not isinstance(value, Mapping):
            raise ValueError(f"rule for {question.key!r} must return a dict of fields or None")
        values = {name: value.get(name) for name in question.fields}
        confidences = {name: (1.0 if values[name] is not None else None) for name in values}
        return Answer(value=values, fields=confidences)
    raise TypeError(f"unsupported question type {type(question).__name__}")
