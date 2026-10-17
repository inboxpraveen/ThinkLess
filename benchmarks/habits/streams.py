"""Long streams of real inputs for the Habits replays.

The decision benchmark freezes 500 test rows per task, which is a few hours of
traffic for a busy agent. These streams use the full public datasets
instead, so the replay shows what happens over thousands of calls.

Each stream returns a task spec (the question an agent would ask) and rows
with ``id``, ``state`` and ``label``, in the decision benchmark's format.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import httpx

from thinkless.bench.decisions.tasks import load_tasks

CACHE = Path.home() / ".cache" / "thinkless" / "datasets"
# The same revisions the decision benchmark pins.
BANKING77_REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
BANKING77 = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    f"{BANKING77_REVISION}/banking_data/{{split}}.csv"
)
CLINC_REVISION = "155b9c710419136e17307b80d0a13e68cd46b4ec"


def _banking77(split: str) -> list[dict[str, Any]]:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"banking77-{split}-{BANKING77_REVISION[:8]}.csv"
    if not path.exists():
        response = httpx.get(BANKING77.format(split=split), timeout=60, follow_redirects=True)
        response.raise_for_status()
        path.write_text(response.text, encoding="utf-8")
    reader = csv.DictReader(io.StringIO(path.read_text(encoding="utf-8")))
    return [
        {
            "id": f"banking77/{split}/{i:05d}",
            "state": row["text"],
            "label": row["category"].replace("_", " "),
        }
        for i, row in enumerate(reader)
    ]


def _clinc150(split: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    data = load_dataset("clinc/clinc_oos", "plus", split=split, revision=CLINC_REVISION)
    names = data.features["intent"].names
    return [
        {
            "id": f"clinc150/{split}/{i:05d}",
            "state": row["text"],
            "label": "out of scope"
            if names[row["intent"]] == "oos"
            else names[row["intent"]].replace("_", " "),
        }
        for i, row in enumerate(data)
    ]


CIVIL_REVISION = "f2970eb3a55777454c94069077cc8d9b5866312d"
CIVIL_ROWS = 10000


def _civil_comments() -> list[dict[str, Any]]:
    """10,000 validation comments in their natural proportion of toxic ones."""
    import random

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "google/civil_comments",
        "data/validation-00000-of-00001.parquet",
        repo_type="dataset",
        revision=CIVIL_REVISION,
    )
    rows = [
        {
            "id": f"civil_comments/validation/{i:05d}",
            "state": r["text"],
            "label": r["toxicity"] >= 0.5,
        }
        for i, r in enumerate(pq.read_table(path).to_pylist())
        if r["text"] and r["text"].strip()
    ]
    return random.Random(0).sample(rows, CIVIL_ROWS)


STREAMS = {
    "banking77": (lambda: _banking77("train") + _banking77("test"), "banking77"),
    "civil_comments": (_civil_comments, "civil_comments"),
    "clinc150": (
        lambda: _clinc150("train") + _clinc150("validation") + _clinc150("test"),
        "clinc150",
    ),
}


def load_stream(name: str) -> tuple[Any, list[dict[str, Any]]]:
    """The task spec and every row of a long stream."""
    if name not in STREAMS:
        raise SystemExit(f"unknown stream {name!r}; choose from {', '.join(STREAMS)}")
    loader, task_name = STREAMS[name]
    task = load_tasks([task_name])[0]
    rows = loader()
    options = set(task.question.get("options") or [True, False])
    unknown = {row["label"] for row in rows} - options
    if unknown:
        raise SystemExit(f"{name}: labels not in the task's options: {sorted(unknown)[:5]}")
    return task, rows
