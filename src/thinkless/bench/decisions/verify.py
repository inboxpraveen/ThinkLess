"""Check a result file against the published tasks and recompute every metric."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .run import SCHEMA, BenchmarkResult
from .score import TaskMetrics, score_task
from .tasks import benchmark_version, load_rows, load_tasks, question_for

__all__ = ["Verification", "verify_result"]

TOLERANCE = 1e-6


class Verification(BaseModel):
    """The outcome of checking one result file.

    ``metrics`` are always recomputed from the predictions; the leaderboard
    uses these, never the numbers written in the file.
    """

    path: str
    name: str
    ok: bool
    rankable: bool
    problems: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    metrics: dict[str, TaskMetrics] = Field(default_factory=dict)
    supported: dict[str, bool] = Field(default_factory=dict)
    result: BenchmarkResult | None = None


def _differs(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) > TOLERANCE
    return bool(a != b)


def verify_result(path: str | Path, *, data_dir: str | Path | None = None) -> Verification:
    """Verify one result file.

    Checks the schema and benchmark version, that every task's question and
    row files match the published fingerprint, that the predictions cover
    exactly the published rows, and that the metrics in the file match the
    ones recomputed from the predictions.
    """
    problems: list[str] = []
    notes: list[str] = []
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        result = BenchmarkResult.model_validate(raw)
    except (OSError, ValueError) as exc:
        return Verification(path=str(path), name="?", ok=False, rankable=False, problems=[str(exc)])

    if result.schema_ != SCHEMA:
        problems.append(f"schema is {result.schema_!r}, expected {SCHEMA!r}")
    if result.benchmark_version != benchmark_version():
        problems.append(
            f"benchmark version {result.benchmark_version} does not match {benchmark_version()}"
        )

    metrics: dict[str, TaskMetrics] = {}
    supported: dict[str, bool] = {}
    for task in load_tasks():
        entry = result.tasks.get(task.name)
        if entry is None:
            problems.append(f"{task.name}: missing")
            continue
        supported[task.name] = entry.supported
        if entry.fingerprint != task.fingerprint():
            problems.append(f"{task.name}: fingerprint {entry.fingerprint} does not match the task")
            continue
        if not entry.supported:
            continue
        question = question_for(task)
        truth: dict[str, dict[str, Any]] = {}
        for split, predictions in entry.predictions.items():
            rows = load_rows(task, split, data_dir)
            expected_ids = [row["id"] for row in rows]
            got_ids = [p.get("id") for p in predictions]
            if not result.partial and got_ids != expected_ids:
                problems.append(f"{task.name}/{split}: predictions do not cover the published rows")
            if result.partial and not set(got_ids) <= set(expected_ids):
                problems.append(f"{task.name}/{split}: predictions for unknown rows")
            truth[split] = {row["id"]: row["label"] for row in rows}
        if "test" not in entry.predictions:
            problems.append(f"{task.name}: no test predictions")
            continue
        recomputed = score_task(
            question,
            entry.predictions["test"],
            truth["test"],
            calibration=entry.predictions.get("calibration"),
            calibration_truth=truth.get("calibration"),
            target=result.target,
        )
        metrics[task.name] = recomputed
        if entry.metrics is not None:
            stored = entry.metrics.model_dump()
            fresh = recomputed.model_dump()
            changed = [k for k in fresh if _differs(stored.get(k), fresh[k])]
            if changed:
                notes.append(f"{task.name}: stored metrics differ from recomputed ones: {changed}")
    if result.partial:
        notes.append("partial run: checked, but never ranked")
    return Verification(
        path=str(path),
        name=result.submission.name,
        ok=not problems,
        rankable=not problems and not result.partial and result.target == 0.95,
        problems=problems,
        notes=notes,
        metrics=metrics,
        supported=supported,
        result=result,
    )
