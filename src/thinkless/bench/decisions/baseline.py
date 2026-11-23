"""The majority-class baseline: the floor every result on the leaderboard should clear."""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from ...decision import Answer, Plane
from ...providers.base import DecisionProvider, ProviderResult, State
from ...questions import Choice, Extract, Kind, Question, Score, YesNo
from .tasks import load_rows, load_tasks

__all__ = ["Majority"]


class Majority(DecisionProvider):
    """Answers every question with the most common label of its calibration rows.

    It never reads the input. Its accuracy is what a model scores with no
    skill at all, which is the number to beat: on ``jailbreak``, always
    answering "no" is already right on 65 percent of the test rows.

    The reported probabilities are the label frequencies, so the provider is
    honestly calibrated and every row gets the same confidence.

    Args:
        labels: Labels per question name. Questions without an entry are
            looked up as decision benchmark tasks of the same name, and their
            calibration rows are counted.
        data_dir: Local task files, as for :func:`load_rows`.
    """

    name = "majority"
    plane: ClassVar[Plane] = Plane.MODEL
    kinds: ClassVar[frozenset[Kind]] = frozenset(Kind)

    def __init__(
        self,
        labels: Mapping[str, Sequence[Any]] | None = None,
        *,
        data_dir: str | Path | None = None,
    ) -> None:
        self._labels = {key: list(values) for key, values in (labels or {}).items()}
        self._data_dir = data_dir
        self._lock = threading.Lock()

    def _labels_for(self, question: Question) -> list[Any]:
        with self._lock:
            if question.key not in self._labels:
                task = load_tasks([question.key])[0]
                rows = load_rows(task, "calibration", self._data_dir)
                self._labels[question.key] = [row["label"] for row in rows]
            return self._labels[question.key]

    def supports(self, question: Question) -> bool:
        if not super().supports(question):
            return False
        if question.key in self._labels:
            return True
        return question.key in {task.name for task in load_tasks()}

    def answer(self, state: State, questions: Mapping[str, Question]) -> ProviderResult:
        return ProviderResult(
            answers={key: self._answer(q, self._labels_for(q)) for key, q in questions.items()},
            model="majority-class",
        )

    @staticmethod
    def _answer(question: Question, labels: Sequence[Any]) -> Answer | None:
        if not labels:
            return None
        if isinstance(question, Extract):
            # The most common value of every field is "not mentioned"; the
            # confidence is how often a calibration row mentions nothing.
            empty = sum(1 for label in labels if not any(dict(label or {}).values()))
            return Answer(value=dict.fromkeys(question.fields), confidence=empty / len(labels))
        if isinstance(question, YesNo):
            p_yes = sum(1 for label in labels if label is True) / len(labels)
            return Answer(value=p_yes > 0.5, probabilities={"yes": p_yes, "no": 1.0 - p_yes})
        order: list[str]
        if isinstance(question, Choice):
            order = question.labels
        elif isinstance(question, Score):
            order = list(question.levels)
        else:  # pragma: no cover - every kind is handled above
            return None
        counts = Counter(str(label) for label in labels)
        probabilities = {option: counts.get(option, 0) / len(labels) for option in order}
        # Ties go to the option listed first, so the answer never depends on row order.
        best = max(order, key=lambda option: (counts.get(option, 0), -order.index(option)))
        if isinstance(question, Score):
            return Answer(value=float(order.index(best)), probabilities=probabilities)
        return Answer(value=best, probabilities=probabilities)
