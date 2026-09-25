"""Conversion to and from the System One wire format.

The format was introduced by TypeSafe for Jev (``POST /v1/systemone``) and is
also spoken by Kev, OpenJev and Laya. Question types are ``choice``, ``score``
and ``noul`` (yes/no); answers carry per-option probabilities.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..decision import Answer, Decision, Status
from ..questions import Choice, Question, Score, YesNo

__all__ = ["decisions_to_wire", "from_wire", "questions_from_wire", "to_wire"]


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


# ------------------------------------------------------------- server side


def questions_from_wire(wire: Mapping[str, Any]) -> dict[str, Question]:
    """Decode a System One ``questions`` map, for serving the format.

    Raises:
        ValueError: On an unknown type or a malformed question.
    """
    questions: dict[str, Question] = {}
    for key, item in wire.items():
        if not isinstance(item, Mapping):
            raise ValueError(f"question {key!r} must be an object")
        kind = item.get("type")
        instructions = item.get("instructions") or key
        criteria = item.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, (Mapping, list)):
                raise ValueError(f"choice {key!r} needs criteria")
            questions[key] = Choice(instructions, options=criteria, name=key)
        elif kind == "score":
            if not isinstance(criteria, list):
                raise ValueError(f"score {key!r} needs a list of levels as criteria")
            questions[key] = Score(instructions, levels=[str(c) for c in criteria], name=key)
        elif kind == "noul":
            meanings = criteria if isinstance(criteria, Mapping) else {}
            questions[key] = YesNo(
                instructions,
                yes_means=meanings.get("true"),
                no_means=meanings.get("false"),
                name=key,
            )
        else:
            raise ValueError(f"question {key!r} has unknown type {kind!r}")
    return questions


def _wire_confidence(decision: Decision) -> float:
    if decision.confidence is not None:
        return float(decision.confidence)
    if decision.probability is not None:
        return float(decision.probability)
    return 1.0 if decision.status is Status.ACCEPTED else 0.0


def decisions_to_wire(
    decisions: Mapping[str, Decision], questions: Mapping[str, Question]
) -> dict[str, dict[str, Any]]:
    """Encode decisions as a System One ``answers`` map.

    Abstained decisions are left out, as a System One server leaves out
    questions it cannot answer. Answers from providers without probabilities
    get the chosen answer at probability 1.
    """
    answers: dict[str, dict[str, Any]] = {}
    for key, decision in decisions.items():
        question = questions[key]
        if decision.value is None:
            continue
        probs = decision.probabilities or {}
        if isinstance(question, Choice):
            answers[key] = {
                "type": "choice",
                "choice": str(decision.value),
                "confidence": _wire_confidence(decision),
                "probabilities": {k: float(v) for k, v in probs.items()}
                or {str(decision.value): 1.0},
            }
        elif isinstance(question, Score):
            levels = list(question.levels)
            by_index = {
                str(levels.index(level)): float(p) for level, p in probs.items() if level in levels
            }
            if not by_index:
                by_index = {str(min(max(round(float(decision.value)), 0), len(levels) - 1)): 1.0}
            answers[key] = {
                "type": "score",
                "score": float(decision.value),
                "confidence": _wire_confidence(decision),
                "legend": {str(i): level for i, level in enumerate(levels)},
                "probabilities": by_index,
            }
        elif isinstance(question, YesNo):
            p_yes = probs.get("yes")
            if p_yes is None:
                p_yes = 1.0 if decision.value else 0.0
            answers[key] = {"type": "noul", "noul": float(p_yes)}
    return answers
