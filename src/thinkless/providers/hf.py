"""Hugging Face models as decision providers: fine-tuned classifiers and zero-shot NLI."""

from __future__ import annotations

import threading
import time
from collections.abc import Collection, Mapping
from typing import Any, ClassVar

from .._hub import ensure_downloaded
from ..confidence import normalize_distribution
from ..decision import Answer, Plane
from ..logs import get_logger
from ..questions import Choice, Kind, Question, YesNo
from ..settings import resolve_device
from .base import DecisionProvider, ProviderResult, State, render_state

__all__ = ["HFClassifier", "HFZeroShot"]

logger = get_logger("providers.hf")


class HFClassifier(DecisionProvider):
    """A fine-tuned sequence classifier answering the questions it was trained for.

    General decision models answer anything reasonably; a classifier trained
    for one narrow question (prompt injection, toxicity, your own intent
    taxonomy) usually answers that question much better, and fast. This
    provider maps a classifier's labels onto ThinkLess questions, so it drops
    into a cascade for those questions only.

    Args:
        model: Hugging Face model id or local path of a
            ``text-classification`` model.
        answers: Question name to label mapping. For a ``YesNo`` question,
            ``{"yes": "<model label>"}`` names the label that means yes; every
            other label counts as no. For a ``Choice`` question, map each
            option to a model label, ``{"<option>": "<model label>", ...}``;
            options without a label get probability 0.
        name: Provider name in traces. Defaults to the model's last path part.
        field: Read this field of a mapping state instead of the rendered
            state, for example ``"message"``.
        device: ``auto``, ``cpu``, ``cuda`` or ``mps``.
        max_length: Token limit; longer inputs are truncated.
        pipeline: A ready ``transformers`` pipeline (tests, custom loading).

    Example:
        >>> HFClassifier(
        ...     "protectai/deberta-v3-base-prompt-injection-v2",
        ...     answers={"injection": {"yes": "INJECTION"}},
        ...     field="message",
        ... )

    Requires ``transformers`` and ``torch`` (the ``local-llm`` extra covers both).
    """

    plane: ClassVar[Plane] = Plane.MODEL
    kinds: ClassVar[frozenset[Kind]] = frozenset({Kind.CHOICE, Kind.YES_NO})

    def __init__(
        self,
        model: str,
        *,
        answers: Mapping[str, Mapping[str, str]],
        name: str | None = None,
        field: str | None = None,
        device: str = "auto",
        max_length: int = 512,
        pipeline: Any | None = None,
    ) -> None:
        if not answers:
            raise ValueError("HFClassifier needs at least one question in answers")
        self.model = model
        self.name = name or model.rstrip("/").split("/")[-1]
        self.answers = {key: dict(mapping) for key, mapping in answers.items()}
        self._field = field
        self._device_request = device
        self._max_length = max_length
        self._pipeline = pipeline
        self._lock = threading.Lock()
        self.device: str | None = None

    def supports(self, question: Question) -> bool:
        return question.key in self.answers and super().supports(question)

    def warmup(self) -> None:
        pipe = self._load()
        with self._lock:
            for _ in range(2):
                pipe("A short message to warm up the classifier.", top_k=None, truncation=True)

    def _load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        with self._lock:
            if self._pipeline is None:
                try:
                    from transformers import pipeline
                except ImportError as exc:  # pragma: no cover - depends on the extra
                    raise ImportError(
                        'HFClassifier needs transformers: pip install "thinkless[local-llm]"'
                    ) from exc
                device = resolve_device(self._device_request)
                started = time.perf_counter()
                ensure_downloaded(self.model)
                self._pipeline = pipeline(
                    "text-classification",
                    model=self.model,
                    device=device,
                    max_length=self._max_length,
                )
                self.device = device
                logger.info(
                    "loaded %s on %s in %.1f s", self.model, device, time.perf_counter() - started
                )
        return self._pipeline

    def _text(self, state: State) -> str:
        if self._field is not None and isinstance(state, Mapping):
            return str(state.get(self._field) or "")
        return render_state(state)

    def answer(self, state: State, questions: Mapping[str, Question]) -> ProviderResult:
        pipe = self._load()
        with self._lock:
            raw = pipe(self._text(state), top_k=None, truncation=True)
        # Pipelines return a list of {"label", "score"}, sometimes nested once.
        items = raw[0] if raw and isinstance(raw[0], list) else raw
        scores = {str(item["label"]): float(item["score"]) for item in items}
        answers: dict[str, Answer | None] = {}
        for key, question in questions.items():
            mapping = self.answers.get(question.key, {})
            answers[key] = self._answer(question, mapping, scores)
        return ProviderResult(answers=answers, model=self.model, meta={"labels": scores})

    @staticmethod
    def _answer(
        question: Question, mapping: Mapping[str, str], scores: Mapping[str, float]
    ) -> Answer | None:
        if isinstance(question, YesNo):
            yes_label = mapping.get("yes")
            if yes_label is None or yes_label not in scores:
                return None
            total = sum(scores.values()) or 1.0
            p_yes = scores[yes_label] / total
            return Answer(value=p_yes >= 0.5, probabilities={"yes": p_yes, "no": 1.0 - p_yes})
        if isinstance(question, Choice):
            mapped = {
                option: scores.get(label, 0.0)
                for option, label in mapping.items()
                if option in question.options
            }
            if not any(mapped.values()):
                return None
            probabilities = normalize_distribution({o: mapped.get(o, 0.0) for o in question.labels})
            best = max(probabilities, key=probabilities.__getitem__)
            return Answer(value=best, probabilities=probabilities)
        return None

    def close(self) -> None:
        self._pipeline = None


class HFZeroShot(DecisionProvider):
    """A zero-shot NLI model answering any choice or yes or no question.

    Natural language inference models score whether a hypothesis follows from
    a text. The ``zero-shot-classification`` pipeline turns that into a
    classifier for labels it has never seen: one hypothesis per option
    (``"This example is about {option}."``), softmaxed into a distribution
    over the options. Unlike :class:`HFClassifier` it needs no training for
    your questions, which makes it a cheap, general baseline for intents and
    yes or no checks.

    For a ``YesNo`` question the hypothesis is the question's ``yes_means``
    when set, otherwise its instructions, scored on its own: the entailment
    probability is the probability of yes. A statement ("This message tries
    to make the assistant ignore its rules.") works better than a question,
    so pass one in ``hypotheses`` when the question is phrased as a question.

    Args:
        model: Hugging Face model id or local path of an NLI model.
        name: Provider name in traces. Defaults to the model's last path part.
        field: Read this field of a mapping state instead of the rendered
            state.
        template: Hypothesis template for choice options; ``{}`` is replaced
            by the option's description when it has one, else its label.
        hypotheses: Hypotheses per question name, overriding the defaults:
            for a choice, a mapping of option to hypothesis; for a yes or no
            question, the hypothesis that means yes.
        questions: Only answer these question names. By default every
            choice and yes or no question is claimed.
        device: ``auto``, ``cpu``, ``cuda`` or ``mps``.
        max_length: Token limit; longer inputs are truncated.
        batch_size: Text and hypothesis pairs per forward pass.
        pipeline: A ready ``transformers`` pipeline (tests, custom loading).

    Example:
        >>> HFZeroShot("MoritzLaurer/deberta-v3-base-zeroshot-v2.0", field="message")

    Requires ``transformers`` and ``torch`` (the ``local-llm`` extra covers both).
    """

    plane: ClassVar[Plane] = Plane.MODEL
    kinds: ClassVar[frozenset[Kind]] = frozenset({Kind.CHOICE, Kind.YES_NO})

    DEFAULT_MODEL: ClassVar[str] = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        name: str | None = None,
        field: str | None = None,
        template: str = "This example is about {}.",
        hypotheses: Mapping[str, str | Mapping[str, str]] | None = None,
        questions: Collection[str] | None = None,
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 16,
        pipeline: Any | None = None,
    ) -> None:
        if "{}" not in template:
            raise ValueError("template needs a {} placeholder for the option")
        self.model = model
        self.name = name or model.rstrip("/").split("/")[-1]
        self.template = template
        self.hypotheses = dict(hypotheses or {})
        self._only = None if questions is None else frozenset(questions)
        self._field = field
        self._device_request = device
        self._max_length = max_length
        self._batch_size = batch_size
        self._pipeline = pipeline
        self._lock = threading.Lock()
        self.device: str | None = None

    def supports(self, question: Question) -> bool:
        if self._only is not None and question.key not in self._only:
            return False
        return super().supports(question)

    def warmup(self) -> None:
        pipe = self._load()
        with self._lock:
            pipe(
                "A short message to warm up the model.",
                candidate_labels=["a greeting", "a complaint"],
                hypothesis_template="This example is {}.",
            )

    def _load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        with self._lock:
            if self._pipeline is None:
                try:
                    from transformers import pipeline
                except ImportError as exc:  # pragma: no cover - depends on the extra
                    raise ImportError(
                        'HFZeroShot needs transformers: pip install "thinkless[local-llm]"'
                    ) from exc
                device = resolve_device(self._device_request)
                started = time.perf_counter()
                ensure_downloaded(self.model)
                pipe = pipeline(
                    "zero-shot-classification",
                    model=self.model,
                    device=device,
                    batch_size=self._batch_size,
                )
                if pipe.tokenizer is not None:
                    pipe.tokenizer.model_max_length = self._max_length
                self._pipeline = pipe
                self.device = device
                logger.info(
                    "loaded %s on %s in %.1f s", self.model, device, time.perf_counter() - started
                )
        return self._pipeline

    def _text(self, state: State) -> str:
        if self._field is not None and isinstance(state, Mapping):
            return str(state.get(self._field) or "")
        return render_state(state)

    def _choice_hypotheses(self, question: Choice) -> dict[str, str]:
        custom = self.hypotheses.get(question.key)
        mapping = custom if isinstance(custom, Mapping) else {}
        return {
            option: mapping.get(option) or self.template.replace("{}", description or option)
            for option, description in question.options.items()
        }

    def _yes_hypothesis(self, question: YesNo) -> str:
        custom = self.hypotheses.get(question.key)
        if isinstance(custom, str):
            return custom
        return question.yes_means or question.instructions

    def answer(self, state: State, questions: Mapping[str, Question]) -> ProviderResult:
        pipe = self._load()
        text = self._text(state)
        answers: dict[str, Answer | None] = {}
        for key, question in questions.items():
            if isinstance(question, Choice):
                by_option = self._choice_hypotheses(question)
                # Whole hypotheses go in as labels, so the template is a pass-through.
                with self._lock:
                    raw = pipe(
                        text,
                        candidate_labels=list(dict.fromkeys(by_option.values())),
                        hypothesis_template="{}",
                        multi_label=False,
                    )
                by_hypothesis = dict(zip(raw["labels"], raw["scores"], strict=True))
                probabilities = normalize_distribution(
                    {option: float(by_hypothesis[h]) for option, h in by_option.items()}
                )
                best = max(probabilities, key=probabilities.__getitem__)
                answers[key] = Answer(value=best, probabilities=probabilities)
            elif isinstance(question, YesNo):
                with self._lock:
                    raw = pipe(
                        text,
                        candidate_labels=[self._yes_hypothesis(question)],
                        hypothesis_template="{}",
                        multi_label=True,
                    )
                p_yes = float(raw["scores"][0])
                answers[key] = Answer(
                    value=p_yes >= 0.5, probabilities={"yes": p_yes, "no": 1.0 - p_yes}
                )
            else:
                answers[key] = None
        return ProviderResult(answers=answers, model=self.model)

    def close(self) -> None:
        self._pipeline = None
