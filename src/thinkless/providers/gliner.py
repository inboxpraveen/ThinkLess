"""GLiNER 2.5: schema-driven classification and extraction on small encoders."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from typing import Any, ClassVar

from .._hub import ensure_downloaded
from ..confidence import normalize_distribution
from ..decision import Answer, Plane
from ..logs import get_logger
from ..questions import Choice, Extract, Kind, Question
from ..settings import resolve_device
from .base import DecisionProvider, ProviderResult, State, render_state

__all__ = ["GLiNER"]

logger = get_logger("providers.gliner")

# Representative length for warmup: GPU kernels are chosen per input shape.
_WARMUP_TEXT = (
    "Hello, I placed order 4471 last Tuesday and the tracking page has not changed since. "
    "Could you tell me when it will arrive, or refund it if it is lost? Thanks for your help."
)


def _field_spec(name: str, description: str) -> str:
    return f"{name}::str::{description.replace('::', ':')}"


class GLiNER(DecisionProvider):
    """Answers choice and extract questions with a GLiNER2 model.

    Choice questions are scored per label (the model emits an independent
    score for every option) and the scores are normalized into a distribution,
    so confidence is comparable with other providers. All choice questions of
    a call share one forward pass, and so do all extract questions.

    Measured on an RTX 5060 laptop GPU with ``gliner2.5-base-v1``: about 16 ms
    per classification and 38 ms per extraction; about 85 ms each on CPU.

    Args:
        model: ``fastino/gliner2.5-base-v1`` (English, 0.2B),
            ``fastino/gliner2.5-small-v1`` (74M, fastest on CPU) or
            ``fastino/gliner2.5-multi-v1`` (multilingual).
        device: ``auto``, ``cpu``, ``cuda`` or ``mps``.
        name: Provider name in traces.
        extract_threshold: Minimum span confidence for an extracted field.

    Requires the ``gliner`` extra. Weights download on first use.
    """

    plane: ClassVar[Plane] = Plane.MODEL
    kinds: ClassVar[frozenset[Kind]] = frozenset({Kind.CHOICE, Kind.EXTRACT})

    def __init__(
        self,
        model: str = "fastino/gliner2.5-base-v1",
        *,
        device: str = "auto",
        name: str = "gliner",
        extract_threshold: float = 0.5,
    ) -> None:
        self.name = name
        self.model = model
        self._device_request = device
        self._extract_threshold = extract_threshold
        self._model: Any = None
        self._lock = threading.Lock()
        self.device: str | None = None

    def warmup(self) -> None:
        """Load the weights and run representative inferences of each kind.

        The first forward passes on a GPU are several times slower than the
        rest while kernels initialize, and GLiNER's TorchScript parts only
        specialize after a couple of runs; paying for that here keeps it off
        the first requests.
        """
        model = self._load()
        labels = {f"option {i}": f"a description of option number {i}" for i in range(8)}
        for _ in range(2):
            self._warm(model, labels)

    def _warm(self, model: Any, labels: dict[str, str]) -> None:
        with self._lock:
            model.classify_text(
                _WARMUP_TEXT, {"w": {"labels": labels, "multi_label": True, "cls_threshold": 0.0}}
            )
            model.extract_json(
                _WARMUP_TEXT, {"w": ["id::str::the reference number", "day::str::the day"]}
            )

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                try:
                    from gliner2 import AutoExtractor
                except ImportError as exc:  # pragma: no cover - depends on the extra
                    raise ImportError(
                        'The GLiNER provider needs the gliner extra: pip install "thinkless[gliner]"'
                    ) from exc
                device = resolve_device(self._device_request)
                started = time.perf_counter()
                ensure_downloaded(self.model)
                model = AutoExtractor.from_pretrained(self.model)
                if device != "cpu":
                    model = model.to(device)
                self._model = model
                self.device = device
                logger.info(
                    "loaded %s on %s in %.1f s", self.model, device, time.perf_counter() - started
                )
        return self._model

    def answer(self, state: State, questions: Mapping[str, Question]) -> ProviderResult:
        model = self._load()
        text = render_state(state)
        answers: dict[str, Answer | None] = {}

        choices = {k: q for k, q in questions.items() if isinstance(q, Choice)}
        extracts = {k: q for k, q in questions.items() if isinstance(q, Extract)}

        if choices:
            tasks: dict[str, Any] = {}
            for key, question in choices.items():
                has_descriptions = all(question.options.values())
                labels: Any = dict(question.options) if has_descriptions else question.labels
                tasks[key] = {"labels": labels, "multi_label": True, "cls_threshold": 0.0}
            with self._lock:
                result = model.classify_text(text, tasks, include_confidence=True)
            for key, question in choices.items():
                answers[key] = self._choice_answer(result.get(key), question)

        if extracts:
            schema = {
                key: [_field_spec(name, desc) for name, desc in q.fields.items()]
                for key, q in extracts.items()
            }
            with self._lock:
                result = model.extract_json(
                    text, schema, threshold=self._extract_threshold, include_confidence=True
                )
            for key, extract in extracts.items():
                answers[key] = self._extract_answer(result.get(key), extract)

        return ProviderResult(answers=answers, model=self.model, meta={"device": self.device})

    @staticmethod
    def _choice_answer(raw: Any, question: Choice) -> Answer | None:
        if not raw:
            return None
        items = raw if isinstance(raw, list) else [raw]
        scores = dict.fromkeys(question.labels, 0.0)
        for item in items:
            label = item.get("label") if isinstance(item, Mapping) else None
            if label in scores:
                scores[label] = float(item.get("confidence") or 0.0)
        if not any(scores.values()):
            return None
        probabilities = normalize_distribution(scores)
        best = max(probabilities, key=probabilities.__getitem__)
        return Answer(value=best, probabilities=probabilities, raw={"scores": scores})

    @staticmethod
    def _extract_answer(raw: Any, question: Extract) -> Answer:
        record: Mapping[str, Any] = {}
        if isinstance(raw, list) and raw:
            record = raw[0] if isinstance(raw[0], Mapping) else {}
        elif isinstance(raw, Mapping):
            record = raw
        values: dict[str, Any] = {}
        confidences: dict[str, float | None] = {}
        for name in question.fields:
            item = record.get(name)
            if isinstance(item, Mapping):
                values[name] = item.get("text")
                confidences[name] = (
                    float(item["confidence"]) if item.get("confidence") is not None else None
                )
            elif isinstance(item, list) and item and isinstance(item[0], Mapping):
                values[name] = item[0].get("text")
                confidences[name] = item[0].get("confidence")
            else:
                values[name] = item if isinstance(item, str) else None
                confidences[name] = None
        return Answer(value=values, fields=confidences, raw={"record": dict(record)})

    def close(self) -> None:
        self._model = None
