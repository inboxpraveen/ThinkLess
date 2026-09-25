"""Run a provider on the decision benchmark and write a result file."""

from __future__ import annotations

import importlib
import os
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..._json import jsonable
from ..._version import __version__
from ...engine import Engine
from ...limits import SpendLimit
from ...providers.base import DecisionProvider
from ...questions import Question
from ...tracing import Tracer
from ..support import environment_info
from .score import TaskMetrics, score_task
from .tasks import TaskSpec, benchmark_version, load_rows, load_tasks, question_for

__all__ = [
    "SCHEMA",
    "BenchmarkResult",
    "Submission",
    "TaskResult",
    "build_submission_provider",
    "run_benchmark",
]

SCHEMA = "thinkless-decision-benchmark/1"
Progress = Callable[[str, str, int, int], None]


class Submission(BaseModel):
    """Who and what was benchmarked."""

    name: str
    spec: str
    provider: str
    model: str | None = None
    submitted_by: str | None = None
    url: str | None = None
    notes: str | None = None
    trained_on_task_data: bool = False


class TaskResult(BaseModel):
    fingerprint: str
    supported: bool
    predictions: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    metrics: TaskMetrics | None = None


class BenchmarkResult(BaseModel):
    """A result file. ``metrics`` are informative; verification recomputes them."""

    schema_: str = Field(default=SCHEMA, alias="schema")
    benchmark_version: str
    thinkless_version: str
    created_at: str
    partial: bool = False
    target: float
    submission: Submission
    environment: dict[str, Any]
    tasks: dict[str, TaskResult]

    model_config = {"populate_by_name": True}

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            self.model_dump_json(by_alias=True, exclude_none=False) + "\n",
            encoding="utf-8",
        )
        return target


def _load(path: str) -> Any:
    module_name, _, attribute = path.partition(":")
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    obj = getattr(importlib.import_module(module_name), attribute)
    return obj() if callable(obj) and not isinstance(obj, DecisionProvider) else obj


def build_submission_provider(
    spec: str,
    *,
    llm: str = "local",
    device: str = "auto",
    reasoning: str | None = None,
) -> DecisionProvider:
    """A provider from a submission spec.

    Specs: ``gliner``, ``laya``, ``llm`` (with ``llm=`` as the model spec),
    ``jev``, ``systemone:<base url>`` for any System One server, or
    ``module:attribute`` for your own provider (an instance or a factory).
    """
    if spec.startswith("systemone:"):
        from ...providers import SystemOne

        url = spec.removeprefix("systemone:")
        return SystemOne.self_hosted(url, api_key_env="SYSTEMONE_API_KEY")
    if ":" in spec and not spec.startswith(("http:", "https:")):
        provider = _load(spec)
        if not isinstance(provider, DecisionProvider):
            raise TypeError(f"{spec} is not a DecisionProvider")
        return provider
    from ..intents import build_provider

    return build_provider(spec, llm_spec=llm, device=device, reasoning=reasoning)


def _row_result(question: Question, row: dict[str, Any], engine: Engine) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        decision = engine.decide(row["state"], question)
    except Exception as exc:
        return {
            "id": row["id"],
            "value": None,
            "confidence": None,
            "status": "error",
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "cost_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "model": None,
            "error": f"{type(exc).__name__}: {exc}"[:300],
        }
    latency = (time.perf_counter() - started) * 1000
    failed = next((a.reason for a in decision.attempts if a.reason.startswith("error")), None)
    value = decision.level if question.kind.value == "score" and decision.level else decision.value
    return {
        "id": row["id"],
        "value": jsonable(value),
        "confidence": None if decision.confidence is None else round(decision.confidence, 6),
        "status": decision.status.value,
        "latency_ms": round(latency, 3),
        "cost_usd": decision.cost_usd,
        "input_tokens": decision.usage.input_tokens,
        "output_tokens": decision.usage.output_tokens,
        "model": decision.model,
        "error": failed if decision.value is None else None,
    }


def _run_split(
    task: TaskSpec,
    split: str,
    question: Question,
    engine: Engine,
    rows: list[dict[str, Any]],
    concurrency: int,
    progress: Progress | None,
) -> list[dict[str, Any]]:
    done = 0
    lock = threading.Lock()

    def one(row: dict[str, Any]) -> dict[str, Any]:
        nonlocal done
        result = _row_result(question, row, engine)
        with lock:
            done += 1
            count = done
        if progress is not None:
            progress(task.name, split, count, len(rows))
        return result

    if concurrency <= 1:
        results = [one(row) for row in rows]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(one, rows))
    # A rate limit or a dropped connection says nothing about the model: ask
    # those rows again, one at a time, after the rest are done.
    for index, (row, result) in enumerate(zip(rows, results, strict=True)):
        if _transient(result.get("error")):
            time.sleep(RETRY_DELAY_S)
            results[index] = _row_result(question, row, engine)
    return results


RETRY_DELAY_S = 1.0
TRANSIENT = ("RateLimit", "Timeout", "APIConnection", "ConnectError", "ReadError", "InternalServer")


def _transient(error: str | None) -> bool:
    return bool(error) and any(marker in str(error) for marker in TRANSIENT)


def run_benchmark(
    provider: DecisionProvider,
    *,
    name: str,
    spec: str,
    tasks: list[str] | None = None,
    data_dir: str | Path | None = None,
    target: float = 0.95,
    max_cost_usd: float = 2.0,
    concurrency: int = 1,
    limit: int | None = None,
    submitted_by: str | None = None,
    notes: str | None = None,
    trained_on_task_data: bool = False,
    progress: Progress | None = None,
) -> BenchmarkResult:
    """Ask ``provider`` every question of every task and score the answers.

    Providers without calibrated confidence (prompted LLMs) skip the
    calibration split, which only serves to fit a threshold. Tasks whose
    question kind the provider does not support are recorded as unsupported.

    Args:
        provider: What to benchmark.
        name: Display name on the leaderboard.
        spec: How to rebuild the provider, recorded for reproduction.
        tasks: Task names; all by default.
        data_dir: Local task files; found or downloaded otherwise.
        target: Accuracy target for the selective threshold.
        max_cost_usd: Spend cap for the whole run.
        concurrency: Parallel requests, for hosted providers.
        limit: Only the first rows of each split, for smoke tests. The result
            is marked partial and never ranked.
        submitted_by: Name or handle of whoever ran it.
        notes: Anything a reader should know, such as hardware or settings.
        trained_on_task_data: The model was trained or fine-tuned on the
            source datasets of these tasks (their train splits). Such results
            are marked on the leaderboard: they measure a supervised model,
            not a general one.
    """
    limit_usd = SpendLimit(max_cost_usd)
    engine = Engine([provider], threshold=0.0, tracer=Tracer(), spend_limit=limit_usd)
    provider.warmup()
    results: dict[str, TaskResult] = {}
    model: str | None = None
    for task in load_tasks(tasks):
        question = question_for(task)
        if not provider.supports(question):
            results[task.name] = TaskResult(fingerprint=task.fingerprint(), supported=False)
            continue
        splits = ["test", "calibration"] if provider.calibrated else ["test"]
        predictions: dict[str, list[dict[str, Any]]] = {}
        truth: dict[str, dict[str, Any]] = {}
        for split in splits:
            rows = load_rows(task, split, data_dir)
            if limit is not None:
                rows = rows[:limit]
            truth[split] = {row["id"]: row["label"] for row in rows}
            predictions[split] = _run_split(
                task, split, question, engine, rows, concurrency, progress
            )
            model = model or next((p["model"] for p in predictions[split] if p["model"]), None)
        metrics = score_task(
            question,
            predictions["test"],
            truth["test"],
            calibration=predictions.get("calibration"),
            calibration_truth=truth.get("calibration"),
            target=target,
        )
        results[task.name] = TaskResult(
            fingerprint=task.fingerprint(), supported=True, predictions=predictions, metrics=metrics
        )
    partial = limit is not None or limit_usd.exhausted
    return BenchmarkResult(
        benchmark_version=benchmark_version(),
        thinkless_version=__version__,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        partial=partial,
        target=target,
        submission=Submission(
            name=name,
            spec=spec,
            provider=provider.name,
            model=model,
            submitted_by=submitted_by,
            notes=notes,
            trained_on_task_data=trained_on_task_data,
        ),
        environment=environment_info(),
        tasks=results,
    )
