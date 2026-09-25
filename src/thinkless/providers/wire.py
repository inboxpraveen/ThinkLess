"""Conversion to and from the System One wire format.

The format was introduced by TypeSafe for Jev (``POST /v1/systemone``) and is
also spoken by Kev, OpenJev and Laya. Question types are ``choice``, ``score``
and ``noul`` (yes/no); answers carry per-option probabilities.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..decision import Answer
from ..questions import Choice, Question, Score, YesNo

__all__ = ["from_wire", "to_wire"]


def to_wire(questions: Mapping[str, Question]) -> dict[str, dict[str, Any]]:
    """Encode questions as a System One ``questions`` map."""
    wire: dict[str, dict[str, Any]] = {}
    for key, question in questions.items():
        if isinstance(question, Choice):
            wire[key] = {
                "type": "choice",
                "instructions": question.instructions,
                "criteria": dict(question.options.items()),
            }
        elif isinstance(question, Score):
            wire[key] = {
                "type": "score",
                "instructions": question.instructions,
                "criteria": list(question.levels),
            }
        elif isinstance(question, YesNo):
            item: dict[str, Any] = {"type": "noul", "instructions": question.instructions}
            if question.yes_means or question.no_means:
                item["criteria"] = {"true": question.yes_means, "false": question.no_means}
            wire[key] = item
        else:
            raise TypeError(f"System One cannot encode {type(question).__name__}")
    return wire


def _score_probabilities(raw: Mapping[str, Any], levels: list[str]) -> dict[str, float] | None:
    probs = raw.get("probabilities")
    if not isinstance(probs, Mapping):
        return None
    out: dict[str, float] = {}
    for key, value in probs.items():
        label = key
        if str(key).isdigit() and int(key) < len(levels):
            label = levels[int(key)]
        out[str(label)] = float(value)
    return out


def from_wire(
    answers: Mapping[str, Any], questions: Mapping[str, Question]
) -> dict[str, Answer | None]:
    """Decode a System One ``answers`` map. Missing or malformed entries become ``None``."""
    decoded: dict[str, Answer | None] = {}
    for key, question in questions.items():
        raw = answers.get(key)
        if not isinstance(raw, Mapping):
            decoded[key] = None
            continue
        raw_dict = dict(raw)
        if isinstance(question, Choice):
            choice = raw.get("choice")
            probs = raw.get("probabilities")
            if choice not in question.options:
                decoded[key] = None
                continue
            decoded[key] = Answer(
                value=choice,
                probabilities={str(k): float(v) for k, v in probs.items()}
                if isinstance(probs, Mapping)
                else None,
                confidence=raw.get("confidence"),
                raw=raw_dict,
            )
        elif isinstance(question, Score):
            score = raw.get("score")
            if score is None:
                decoded[key] = None
                continue
            decoded[key] = Answer(
                value=float(score),
                probabilities=_score_probabilities(raw, list(question.levels)),
                confidence=raw.get("confidence"),
                raw=raw_dict,
            )
        elif isinstance(question, YesNo):
            p_yes = raw.get("noul")
            if p_yes is None:
                decoded[key] = None
                continue
            p = float(p_yes)
            decoded[key] = Answer(
                value=p >= 0.5,
                probabilities={"yes": p, "no": 1.0 - p},
                confidence=raw.get("confidence"),
                raw=raw_dict,
            )
        else:
            decoded[key] = None
    return decoded
