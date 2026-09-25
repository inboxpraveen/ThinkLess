"""The tasks of the decision benchmark: specs, questions and verified rows."""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from ...questions import Choice, Extract, Question, Score, YesNo

__all__ = [
    "DATA_URL",
    "TaskFile",
    "TaskSpec",
    "benchmark_version",
    "find_data_dir",
    "load_rows",
    "load_tasks",
    "question_for",
]

DATA_URL = (
    "https://raw.githubusercontent.com/inboxpraveen/ThinkLess/main/benchmarks/decisions/tasks"
)
SPLITS = ("calibration", "test")


class TaskFile(BaseModel):
    path: str
    rows: int
    sha256: str


class TaskSpec(BaseModel):
    """One task: where the rows came from and the question every provider is asked."""

    name: str
    title: str
    domain: str
    question: dict[str, Any]
    source: str
    revision: str
    license: str
    attribution: str
    notes: str
    files: dict[str, TaskFile]

    @property
    def kind(self) -> str:
        return str(self.question["kind"])

    def fingerprint(self) -> str:
        """Hash of the question and the row files, recorded in every result."""
        payload = json.dumps(
            {"question": self.question, "files": {k: v.sha256 for k, v in self.files.items()}},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@lru_cache(maxsize=1)
def _manifest() -> dict[str, Any]:
    text = resources.files("thinkless.bench.decisions").joinpath("tasks.json").read_text("utf-8")
    data: dict[str, Any] = json.loads(text)
    return data


def benchmark_version() -> str:
    return str(_manifest()["version"])


def load_tasks(names: list[str] | None = None) -> list[TaskSpec]:
    """Task specs in benchmark order, optionally only ``names``."""
    tasks = [TaskSpec.model_validate(t) for t in _manifest()["tasks"]]
    if not names or names == ["all"]:
        return tasks
    known = {t.name: t for t in tasks}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"unknown tasks {unknown}; choose from {', '.join(known)}")
    return [known[n] for n in names]


def question_for(task: TaskSpec) -> Question:
    """The question asked on every row of ``task``: the same for every provider."""
    q = task.question
    kind = q["kind"]
    name = task.name
    if kind == "choice":
        return Choice(q["instructions"], options=q["options"], name=name)
    if kind == "yes_no":
        return YesNo(q["instructions"], name=name)
    if kind == "score":
        return Score(q["instructions"], levels=q["levels"], name=name)
    if kind == "extract":
        return Extract(q["instructions"], fields=q["fields"], name=name)
    raise ValueError(f"task {name} has unknown kind {kind!r}")


def find_data_dir(explicit: str | Path | None = None) -> Path | None:
    """Where task files live locally: an explicit path, the environment, or a checkout."""
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    if os.environ.get("THINKLESS_BENCH_DATA"):
        candidates.append(Path(os.environ["THINKLESS_BENCH_DATA"]))
    here = Path.cwd().resolve()
    candidates += [p / "benchmarks" / "decisions" / "tasks" for p in (here, *here.parents)]
    for candidate in candidates:
        if (candidate / "banking77").is_dir():
            return candidate
    return None


def _cache_dir() -> Path:
    path = Path.home() / ".cache" / "thinkless" / "bench" / f"decisions-{benchmark_version()}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task_file(task: TaskSpec, split: str, data_dir: str | Path | None) -> Path:
    entry = task.files[split]
    local = find_data_dir(data_dir)
    if local is not None:
        path = local / entry.path
    else:
        path = _cache_dir() / entry.path
        if not path.exists() or _sha256(path) != entry.sha256:
            response = httpx.get(f"{DATA_URL}/{entry.path}", timeout=120, follow_redirects=True)
            response.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
    if _sha256(path) != entry.sha256:
        raise ValueError(
            f"{path} does not match benchmark {benchmark_version()} (sha256 differs). "
            "Task files never change within a version: re-download them, or upgrade thinkless."
        )
    return path


def load_rows(
    task: TaskSpec, split: str, data_dir: str | Path | None = None
) -> list[dict[str, Any]]:
    """The rows of one split, checked against the published hash.

    Each row has ``id``, ``state`` (text, a mapping, or a list of messages),
    ``label`` and ``source``.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    path = _task_file(task, split, data_dir)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
