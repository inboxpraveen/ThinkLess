"""Eject a proven Habits site into an Engine with typed questions.

Habits learns a decision inside an LLM call. Once a site is proven, its
decision can live in code instead: :func:`eject` writes a Python module that
declares the site's typed questions and an :class:`~thinkless.Engine` whose
first provider is the site's trained student, so the application can call
``engine.decide_many(text, QUESTIONS)`` directly and keep the LLM as the
fallback for inputs the student is unsure of.

The student answers every question with the site's joint confidence, the same
score the gate measured, so an engine with the site's threshold accepts
exactly the answers Habits would have served.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from ..decision import Answer, Plane
from ..providers.base import DecisionProvider, ProviderResult, State
from ..questions import Choice, Extract, Kind, Question, Score, YesNo
from .learn import load_plan
from .student import SiteModel
from .wire import MISSING, ROOT_FIELD, TEXT_FIELD, TOOLS_FIELD, value_of

__all__ = ["SiteStudent", "eject", "site_questions"]

ANSWER_KEY = "answer"


def _key(field: str) -> str:
    return ANSWER_KEY if field in (TEXT_FIELD, ROOT_FIELD) else field


def _input_text(state: State) -> str:
    """The text a student reads: what Habits takes from a call's user message."""
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    lines = (line.strip() for line in text.splitlines())
    return "\n".join(line for line in lines if len(line) >= 2)


class SiteStudent(DecisionProvider):
    """A Habits student as a decision provider.

    Args:
        path: The student's files without their suffix (``<path>.json`` and
            ``<path>.npz``), as :func:`eject` copies them.
        name: Provider name in traces.

    Each question is answered from the field of the same name (``answer``
    for a site whose whole reply is the decision, ``extract`` for its
    extracted fields). The confidence of every answer is the student's joint
    score over all of the site's fields.
    """

    plane: ClassVar[Plane] = Plane.MODEL
    kinds: ClassVar[frozenset[Kind]] = frozenset(
        {Kind.CHOICE, Kind.YES_NO, Kind.SCORE, Kind.EXTRACT}
    )

    def __init__(self, path: str | Path, *, name: str = "habits") -> None:
        self.name = name
        self.path = Path(path)
        self.model = SiteModel.load(self.path)

    def supports(self, question: Question) -> bool:
        if not super().supports(question):
            return False
        if isinstance(question, Extract):
            return all(field in self.model.fields for field in question.fields)
        return self._field(question) is not None

    def _field(self, question: Question) -> str | None:
        key = question.key
        if key == ANSWER_KEY:
            for field in (TEXT_FIELD, ROOT_FIELD):
                if field in self.model.fields:
                    return field
        return key if key in self.model.fields else None

    def answer(self, state: State, questions: Mapping[str, Question]) -> ProviderResult:
        labels, score = self.model.predict(_input_text(state))
        # None abstains: the question moves on to the next provider.
        answers: dict[str, Answer | None] = dict.fromkeys(questions)
        for key, question in questions.items():
            if isinstance(question, Extract):
                values = {f: _value(labels.get(f)) for f in question.fields}
                answers[key] = Answer(
                    value=values,
                    fields={f: score for f, v in values.items() if v is not None},
                    confidence=score,
                )
                continue
            field = self._field(question)
            label = labels.get(field) if field is not None else None
            if label is None or label == MISSING:
                continue
            value = _value(label)
            if isinstance(question, Score):
                level = str(value)
                if level not in question.levels:
                    continue
                answers[key] = Answer(value=float(question.levels.index(level)), confidence=score)
            elif isinstance(question, YesNo):
                answers[key] = Answer(value=bool(value), confidence=score)
            elif isinstance(question, Choice):
                shown = value if isinstance(value, str) else label
                if shown not in question.options:
                    continue
                answers[key] = Answer(value=shown, confidence=score)
        return ProviderResult(answers=answers, model=self.path.name)


def _value(label: str | None) -> Any:
    if label is None or label == MISSING:
        return None
    try:
        return value_of(label)
    except ValueError:
        return label


def site_questions(entry: Mapping[str, Any]) -> dict[str, Question]:
    """The typed questions of a plan entry, keyed by name."""
    from ..questions import question_from_spec

    found = {}
    for spec in entry.get("questions") or []:
        question = question_from_spec(spec)
        found[question.key] = question
    return found


def _find(plan: Mapping[str, Any], site: str) -> dict[str, Any]:
    sites = plan.get("sites", {})
    if site in sites:
        return dict(sites[site])
    named = [entry for entry in sites.values() if entry.get("name") == site]
    if len(named) == 1:
        return dict(named[0])
    known = ", ".join(f"{e['id']} ({e.get('name') or 'unnamed'})" for e in sites.values())
    raise KeyError(f"no site {site!r} in the plan; sites: {known or 'none'}")


def eject(
    directory: str | Path,
    site: str,
    out: str | Path,
    *,
    allow_unproven: bool = False,
) -> Path:
    """Write a module with ``site``'s questions and an engine its student answers.

    Args:
        directory: The Habits directory.
        site: The site's id or name.
        out: Path of the module to write. The student's files are copied
            next to it, into ``<module>_student.json`` and ``.npz``, so the
            module keeps working after Habits retires the student.
        allow_unproven: Eject a site that has not passed its gate. Its
            threshold is then whatever the last learning run chose, if any.

    Raises:
        KeyError: If the site is not in the plan.
        ValueError: If it has not passed its gate, has no student, or has
            no closed questions.
    """
    root = Path(directory)
    plan = load_plan(root)
    if plan is None:
        raise KeyError(f"no plan in {root}; run thinkless habits learn first")
    entry = _find(plan, site)
    if entry.get("status") != "active" and not allow_unproven:
        raise ValueError(
            f"site {entry['id']} is {entry.get('status')!r}, not active: it has not passed "
            "its gate (pass allow_unproven to eject it anyway)"
        )
    stem = entry.get("student")
    threshold = entry.get("threshold")
    if not stem or threshold is None:
        raise ValueError(f"site {entry['id']} has no gated student to eject")
    questions = site_questions(entry)
    if TOOLS_FIELD in (entry.get("fields") or {}):
        raise ValueError(f"site {entry['id']} decides which tools to call; it cannot be ejected")
    if not questions:
        raise ValueError(f"site {entry['id']} has no closed decisions to eject")

    target = Path(out)
    if target.suffix != ".py":
        target = target.with_suffix(".py")
    student = target.with_name(f"{target.stem}_student")
    target.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".json", ".npz"):
        shutil.copyfile(
            (root / "models" / str(stem)).with_suffix(suffix), student.with_suffix(suffix)
        )
    target.write_text(_source(entry, questions, student.name, float(threshold)), encoding="utf-8")
    return target


def _source(
    entry: Mapping[str, Any], questions: Mapping[str, Question], student: str, threshold: float
) -> str:
    evidence = entry.get("evidence") or {}
    title = entry.get("name") or entry["id"]
    lines = [
        f'"""Decisions of the Habits site {title}, ejected {time.strftime("%Y-%m-%d")}.',
        "",
        f"Site {entry['id']}, student {entry.get('student')}, model {entry.get('model')}.",
    ]
    if evidence.get("calls"):
        lines += [
            f"On {evidence['calls']} held-out calls the student answered "
            f"{evidence.get('coverage', 0):.1%} of them at or above the threshold,",
            f"agreeing with the LLM on {evidence.get('agreement', 0):.1%} "
            f"(lower bound {evidence.get('lower_bound', 0):.1%}).",
        ]
    lines += [
        "",
        "Pass the text the LLM was given as the user message:",
        "",
        "    decisions = build_engine(fallback).decide_many(text, QUESTIONS)",
        '"""',
        "",
        "from pathlib import Path",
        "",
        f"from thinkless import {', '.join(sorted({'Engine', *_kinds(questions)}))}",
        "from thinkless.habits.eject import SiteStudent",
        "from thinkless.providers.base import DecisionProvider",
        "",
        f'STUDENT = Path(__file__).with_name("{student}")',
        "# The gate's threshold: the student's answers at or above it are the ones",
        "# Habits served.",
        f"THRESHOLD = {threshold!r}",
        "",
        "QUESTIONS = {",
    ]
    for key, question in questions.items():
        lines.append(f"    {key!r}: {_constructor(question)},")
    lines += [
        "}",
        "",
        "",
        "def build_engine(*fallback: DecisionProvider) -> Engine:",
        '    """The site\'s student first; below the threshold, the ``fallback`` providers."""',
        "    return Engine([SiteStudent(STUDENT), *fallback], threshold=THRESHOLD)",
        "",
    ]
    return "\n".join(lines)


def _kinds(questions: Mapping[str, Question]) -> set[str]:
    return {type(question).__name__ for question in questions.values()}


def _constructor(question: Question) -> str:
    name = question.key
    if isinstance(question, Choice):
        return f"Choice({question.instructions!r}, {list(question.options)!r}, name={name!r})"
    if isinstance(question, YesNo):
        return f"YesNo({question.instructions!r}, name={name!r})"
    if isinstance(question, Score):
        return f"Score({question.instructions!r}, {list(question.levels)!r}, name={name!r})"
    if isinstance(question, Extract):
        return f"Extract(fields={dict(question.fields)!r}, name={name!r})"
    raise TypeError(f"cannot eject a {type(question).__name__} question")
