"""When two answers to the same question count as the same answer."""

from __future__ import annotations

from typing import Any

from ..questions import Choice, Extract, Kind, Question, Score

__all__ = ["agree", "comparable"]

_TRUE = {"true", "yes", "y", "1"}
_FALSE = {"false", "no", "n", "0"}


def _text(value: Any) -> str:
    return " ".join(str(value).split()).casefold()


def comparable(question: Question, value: Any) -> Any:
    """Normalize an answer so answers from different systems can be compared.

    Labels and levels compare case-insensitively, booleans accept ``yes`` and
    ``no`` spellings, a score given as a number maps to the nearest level, and
    extracted fields compare as trimmed, case-folded text. Returns ``None``
    when there is no answer to compare.
    """
    if value is None:
        return None
    kind = question.kind
    if kind is Kind.YES_NO:
        if isinstance(value, bool):
            return value
        text = _text(value)
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        return text
    if kind is Kind.SCORE and isinstance(question, Score):
        levels = [_text(level) for level in question.levels]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            index = min(max(round(float(value)), 0), len(levels) - 1)
            return levels[index]
        return _text(value)
    if kind is Kind.CHOICE and isinstance(question, Choice):
        text = _text(value)
        for label in question.labels:
            if _text(label) == text:
                return _text(label)
        return text
    if kind is Kind.EXTRACT and isinstance(question, Extract):
        if not isinstance(value, dict):
            return None
        return {field: _field(value.get(field)) for field in question.fields}
    return value


def _field(value: Any) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _text(value)
    return _text(value) if isinstance(value, str) else value


def agree(question: Question, a: Any, b: Any) -> tuple[bool | None, dict[str, bool] | None]:
    """Whether two answers agree, and for ``Extract``, which fields agree.

    Returns ``(None, None)`` when either side has no answer.
    """
    left, right = comparable(question, a), comparable(question, b)
    if left is None or right is None:
        return None, None
    if isinstance(left, dict) and isinstance(right, dict):
        fields = {name: left.get(name) == right.get(name) for name in left}
        return all(fields.values()), fields
    return left == right, None
