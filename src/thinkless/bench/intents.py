"""Intent classification benchmark on public datasets.

Measures each provider's accuracy, calibration and latency on one ``Choice``
question, then simulates the cascade: at every threshold, how often the small
models answer on their own, how accurate the combined system is, and how many
calls reach the LLM.
"""

from __future__ import annotations

import csv
import io
import json
import random
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from .._version import __version__
from ..engine import Engine
from ..providers.base import DecisionProvider
from ..questions import Choice, YesNo
from ..tracing import Tracer
from .metrics import (
    ThresholdPoint,
    expected_calibration_error,
    percentile,
    recommend_threshold,
    threshold_sweep,
)
from .support import environment_info

__all__ = [
    "DATASETS",
    "CascadePoint",
    "ChoiceEvaluation",
    "IntentsBenchmark",
    "ProviderEval",
    "build_provider",
    "evaluate_choice",
    "evaluate_question",
    "load_dataset_rows",
    "run_intents_benchmark",
]

BANKING77_TEST = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"

DATASETS: dict[str, dict[str, Any]] = {
    "banking77": {
        "title": "Banking77 (PolyAI), 77 banking intents",
        "instructions": "What is the customer's banking request about?",
        "source": "https://github.com/PolyAI-LDN/task-specific-datasets",
    },
    "clinc150": {
        "title": "CLINC150 plus (clinc/clinc_oos), 150 intents and out of scope",
        "instructions": "What does the user want?",
        "source": "https://huggingface.co/datasets/clinc/clinc_oos",
        "repo": "clinc/clinc_oos",
        "config": "plus",
        "label": "intent",
    },
    "emotion": {
        "title": "Emotion (dair-ai/emotion), 6 emotions",
        "instructions": "Which emotion does the text express?",
        "source": "https://huggingface.co/datasets/dair-ai/emotion",
        "repo": "dair-ai/emotion",
        "config": "split",
        "label": "label",
    },
}


def _humanize(label: str) -> str:
    return "out of scope" if label == "oos" else label.replace("_", " ")


def _cache_dir() -> Path:
    path = Path.home() / ".cache" / "thinkless" / "datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_dataset_rows(name: str) -> list[dict[str, str]]:
    """Test split of a supported dataset as ``{"text", "label"}`` rows with readable labels."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose one of {', '.join(DATASETS)}")
    if name == "banking77":
        cached = _cache_dir() / "banking77-test.csv"
        if not cached.exists():
            response = httpx.get(BANKING77_TEST, timeout=60.0, follow_redirects=True)
            response.raise_for_status()
            cached.write_text(response.text, encoding="utf-8")
        reader = csv.DictReader(io.StringIO(cached.read_text(encoding="utf-8")))
        return [{"text": row["text"], "label": _humanize(row["category"])} for row in reader]
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise ImportError(
            f'The {name} dataset needs the bench extra: pip install "thinkless[bench]"'
        ) from exc
    spec = DATASETS[name]
    dataset = load_dataset(spec["repo"], spec["config"], split="test")
    names = dataset.features[spec["label"]].names
    return [{"text": row["text"], "label": _humanize(names[row[spec["label"]]])} for row in dataset]


def build_provider(name: str, *, llm_spec: str = "local", device: str = "auto") -> DecisionProvider:
    """A provider by short name: ``gliner``, ``laya``, ``llm`` or ``jev``."""
    if name == "gliner":
        from ..providers.gliner import GLiNER

        return GLiNER(device=device)
    if name == "laya":
        from ..providers.laya import Laya

        return Laya(device=device)
    if name == "llm":
        from ..llm import from_spec
        from ..providers import LLMDecider

        return LLMDecider(from_spec(llm_spec, device=device))
    if name == "jev":
        from ..providers import SystemOne

        return SystemOne.jev()
    raise ValueError(f"unknown provider {name!r}; choose gliner, laya, llm or jev")


class ChoiceEvaluation(BaseModel):
    """One provider's answers to one question over a labeled set."""

    provider: str
    predictions: list[Any]
    confidences: list[float | None]
    correct: list[bool]
    latencies_ms: list[float]
    input_tokens: list[int]

    @property
    def accuracy(self) -> float:
        return sum(self.correct) / len(self.correct) if self.correct else 0.0

    @property
    def latency_p50_ms(self) -> float:
        return percentile(self.latencies_ms, 50)


def evaluate_question(
    provider: DecisionProvider,
    question: Choice | YesNo,
    rows: Sequence[dict[str, Any]],
    *,
    text_field: str = "text",
    label_field: str = "label",
    progress: Callable[[int, int], None] | None = None,
) -> ChoiceEvaluation:
    """Ask ``question`` about every row and score the answer against the row's label.

    For a ``Choice`` the label is an option; for a ``YesNo`` it is a boolean.
    """
    provider.warmup()
    engine = Engine([provider], threshold=0.0, tracer=Tracer())
    predictions: list[Any] = []
    confidences: list[float | None] = []
    correct: list[bool] = []
    latencies: list[float] = []
    tokens: list[int] = []
    for index, row in enumerate(rows):
        truth = row[label_field]
        if isinstance(question, YesNo):
            truth = (
                truth
                if isinstance(truth, bool)
                else str(truth).strip().lower() in ("true", "yes", "1")
            )
        started = time.perf_counter()
        decision = engine.decide(row[text_field], question)
        latencies.append((time.perf_counter() - started) * 1000.0)
        predictions.append(decision.value)
        confidences.append(decision.confidence)
        correct.append(decision.value == truth)
        tokens.append(decision.usage.input_tokens)
        if progress is not None:
            progress(index + 1, len(rows))
    return ChoiceEvaluation(
        provider=provider.name,
        predictions=predictions,
        confidences=confidences,
        correct=correct,
        latencies_ms=latencies,
        input_tokens=tokens,
    )


def evaluate_choice(
    provider: DecisionProvider,
    question: Choice,
    rows: Sequence[dict[str, Any]],
    *,
    progress: Callable[[int, int], None] | None = None,
) -> ChoiceEvaluation:
    """Shorthand for :func:`evaluate_question` on ``{"text", "label"}`` rows."""
    return evaluate_question(provider, question, rows, progress=progress)


class ProviderEval(BaseModel):
    provider: str
    accuracy: float
    ece: float | None
    latency_p50_ms: float
    latency_p95_ms: float
    mean_input_tokens: float
    abstained: int
    sweep: list[ThresholdPoint] = Field(default_factory=list)
    recommended: ThresholdPoint | None = None


class CascadePoint(BaseModel):
    """The simulated cascade (small models in order, then the LLM) at one threshold."""

    threshold: float
    accuracy: float
    llm_share: float
    mean_latency_ms: float
    answered_by: dict[str, float]


class IntentsBenchmark(BaseModel):
    thinkless_version: str = __version__
    created_at: str
    dataset: str
    dataset_title: str
    source: str
    examples: int
    labels: int
    seed: int
    llm: str
    target_accuracy: float
    environment: dict[str, Any]
    providers: dict[str, ProviderEval]
    cascade: list[CascadePoint]


def _cascade(
    evaluations: dict[str, ChoiceEvaluation], order: Sequence[str], fallback: str | None
) -> list[CascadePoint]:
    small = [p for p in order if p != fallback]
    n = len(next(iter(evaluations.values())).correct)
    points = []
    for threshold in [round(x * 0.05, 2) for x in range(0, 20)] + [0.97, 0.99, 1.01]:
        hits = 0
        llm_calls = 0
        latency = 0.0
        answered: dict[str, int] = dict.fromkeys([*small, fallback or "unresolved"], 0)
        for i in range(n):
            chosen = None
            for name in small:
                latency += evaluations[name].latencies_ms[i]
                confidence = evaluations[name].confidences[i]
                if confidence is not None and confidence >= threshold:
                    chosen = name
                    break
            if chosen is None and fallback is not None:
                chosen = fallback
                llm_calls += 1
                latency += evaluations[fallback].latencies_ms[i]
            if chosen is None:
                answered["unresolved"] += 1
                continue
            answered[chosen] += 1
            hits += int(evaluations[chosen].correct[i])
        points.append(
            CascadePoint(
                threshold=threshold,
                accuracy=round(hits / n, 4),
                llm_share=round(llm_calls / n, 4),
                mean_latency_ms=round(latency / n, 2),
                answered_by={k: round(v / n, 4) for k, v in answered.items()},
            )
        )
    return points


def _resolved_llm(spec: str) -> str:
    from ..llm.factory import DEFAULT_LOCAL_MODEL

    backend, _, model = spec.partition(":")
    if backend == "local" and not model:
        return f"local:{DEFAULT_LOCAL_MODEL}"
    return spec


def run_intents_benchmark(
    dataset: str,
    *,
    providers: Sequence[str] = ("gliner", "laya", "llm"),
    llm_spec: str = "local",
    device: str = "auto",
    limit: int = 500,
    seed: int = 13,
    target_accuracy: float = 0.95,
    output_dir: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> IntentsBenchmark:
    """Evaluate providers on a dataset sample and simulate the cascade.

    The sample is drawn with a fixed seed so runs are comparable. When ``llm``
    is among the providers it acts as the cascade's fallback.
    """
    say = progress or (lambda message: None)
    rows = load_dataset_rows(dataset)
    rng = random.Random(seed)
    sample = rng.sample(rows, min(limit, len(rows)))
    labels = sorted({row["label"] for row in rows})
    question = Choice(DATASETS[dataset]["instructions"], options=labels, name="label")
    say(f"{dataset}: {len(sample)} of {len(rows)} test examples, {len(labels)} labels")

    evaluations: dict[str, ChoiceEvaluation] = {}
    reports: dict[str, ProviderEval] = {}
    for name in providers:
        provider = build_provider(name, llm_spec=llm_spec, device=device)
        started = time.perf_counter()

        def tick(done: int, total: int, name: str = name, started: float = started) -> None:
            if done % 50 == 0 or done == total:
                say(f"{name}: {done}/{total}  {time.perf_counter() - started:.0f}s")

        evaluation = evaluate_choice(provider, question, sample, progress=tick)
        evaluations[name] = evaluation
        calibrated = all(c is not None for c in evaluation.confidences)
        confidences = [c if c is not None else 1.0 for c in evaluation.confidences]
        sweep = threshold_sweep(confidences, evaluation.correct) if calibrated else []
        reports[name] = ProviderEval(
            provider=name,
            accuracy=round(evaluation.accuracy, 4),
            ece=round(expected_calibration_error(confidences, evaluation.correct), 4)
            if calibrated
            else None,
            latency_p50_ms=round(evaluation.latency_p50_ms, 2),
            latency_p95_ms=round(percentile(evaluation.latencies_ms, 95), 2),
            mean_input_tokens=round(sum(evaluation.input_tokens) / len(sample), 1),
            abstained=sum(p is None for p in evaluation.predictions),
            sweep=sweep,
            recommended=recommend_threshold(sweep, target_accuracy) if sweep else None,
        )
        provider.close()

    fallback = "llm" if "llm" in providers else None
    result = IntentsBenchmark(
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        dataset=dataset,
        dataset_title=DATASETS[dataset]["title"],
        source=DATASETS[dataset]["source"],
        examples=len(sample),
        labels=len(labels),
        seed=seed,
        llm=_resolved_llm(llm_spec) if "llm" in providers else "not used",
        target_accuracy=target_accuracy,
        environment=environment_info(),
        providers=reports,
        cascade=_cascade(evaluations, list(providers), fallback),
    )
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(
            json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8"
        )
        with (out / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for i, row in enumerate(sample):
                record: dict[str, Any] = {"text": row["text"], "label": row["label"]}
                for name, evaluation in evaluations.items():
                    record[name] = {
                        "prediction": evaluation.predictions[i],
                        "confidence": evaluation.confidences[i],
                        "latency_ms": round(evaluation.latencies_ms[i], 2),
                    }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return result
