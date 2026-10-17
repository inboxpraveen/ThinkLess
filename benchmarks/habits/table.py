"""Print the Habits replay results as the Markdown tables used in the docs.

    python benchmarks/habits/table.py results/*.json

Every number comes from the result files, which hold the settings, the
timeline, the final plan and the events of each replay. Long streams go in
the first table; the 500-row decision benchmark tasks, where the question is
where each site stands after a few hundred calls, go in the second.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

TITLES = {
    "banking77": "Banking77, 77 intents",
    "civil_comments": "Civil Comments, is it toxic",
}


def _teacher(spec: str) -> str:
    return "dataset labels" if spec == "labels" else "Qwen 3.7 Flash"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _evidence(result: dict[str, Any]) -> dict[str, Any]:
    sites = result["sites"]
    return (sites[0].get("evidence") or {}) if sites else {}


def _last_window(result: dict[str, Any]) -> str:
    timeline = result["timeline"]
    if len(timeline) < 2:
        return "n/a"
    before, last = timeline[-2], timeline[-1]
    calls = last["calls"] - before["calls"]
    return _pct((last["local"] - before["local"]) / calls) if calls else "n/a"


def long_row(result: dict[str, Any]) -> str:
    first = f"{result['activated_at']:,}" if result["activated_at"] else "never"
    suspensions = sum(1 for e in result.get("events", []) if e.get("event") == "suspended")
    return (
        f"| {TITLES[result['source']]} | {_teacher(result['teacher'])} | {result['calls']:,} "
        f"| {_pct(_evidence(result).get('predictability'))} | {first} "
        f"| **{_pct(result['local_share'])}** | {_last_window(result)} "
        f"| {_pct(result['local_agreement'])} "
        f"| {_pct(result['accuracy_teacher'])} to {_pct(result['accuracy_pipeline'])} "
        f"| {suspensions} |"
    )


def short_row(result: dict[str, Any]) -> str:
    site = result["sites"][0] if result["sites"] else {}
    return (
        f"| {result['task']} | {result['kind'].replace('_', '/')} | {result['calls']:,} "
        f"| {_pct(_evidence(result).get('predictability'))} | {site.get('status', 'n/a')} "
        f"| {site.get('reason') or ''} |"
    )


def main(paths: list[str]) -> None:
    results = []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        results.extend(data if isinstance(data, list) else [data])
    print(
        "| Stream | Teacher | Calls | Predictability | First local answer at call "
        "| Answered locally, whole stream | Answered locally, last 500 calls "
        "| Local answers matching the teacher | Accuracy vs labels, teacher alone "
        "to with Habits | Suspensions |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for result in results:
        if result["source"] in TITLES:
            print(long_row(result))
    print()
    print("| Task | Kind | Calls | Predictability | Status after the stream | Why |")
    print("|---|---|---:|---:|---|---|")
    for result in results:
        if result["source"].startswith("task:"):
            print(short_row(result))
    print()
    for result in results:
        rounds = len(result["timeline"])
        print(
            f"<!-- {result['source']} ({_teacher(result['teacher'])}): {rounds} learning rounds, "
            f"{result['learn_seconds']} s in total -->"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
