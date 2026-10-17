"""Replay traffic through Habits and measure what it answers locally.

An agent calls an LLM for a decision on every input. The calls go through the
real OpenAI SDK with Habits under it; behind Habits, a stand-in LLM
returns the answer a real model gave for that input. Because the teacher's
answer is known for every input, the replay measures what production cannot:
whether each local answer matched what the LLM would have said.

Teachers:

* ``results:<file>``: answers recorded in a decision benchmark result file,
  for example the Qwen 3.7 Flash run in
  ``benchmarks/decisions/results/verified/qwen3.7-flash.json``. Free.
* ``cache:<file>``: answers a live LLM gave, recorded by ``teach.py``.
* ``labels``: the dataset's own labels, a teacher that is never wrong.

Usage:

    python benchmarks/habits/replay.py --source task:jailbreak \\
        --teacher results:benchmarks/decisions/results/verified/qwen3.7-flash.json

The stand-in LLM reports the tokens and cost the real model reported for
that input, so the cost columns are what those calls cost.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import httpx2
import openai
from pydantic import BaseModel, create_model

from thinkless.bench.decisions.tasks import load_rows, load_tasks
from thinkless.habits import Habits, wrap

ROOT = Path(__file__).resolve().parent
SYSTEM = "You are the decision step of a production assistant.\n{instructions}"


def teacher_answers(
    spec: str, task: str, rows: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """``row id -> {"value", "cost_usd", "input_tokens", "output_tokens"}``."""
    if spec == "labels":
        return {row["id"]: {"value": row["label"]} for row in rows}
    kind, _, path = spec.partition(":")
    if kind == "results":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        predictions = data["tasks"][task]["predictions"]["test"]
        # A strict schema never lets a real call come back empty; skip rows the model failed on.
        return {p["id"]: p for p in predictions if p.get("value") is not None}
    if kind == "cache":
        found = {}
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                found[record["id"]] = record
        return found
    raise SystemExit(f"unknown teacher {spec!r}")


def contract(task: Any) -> tuple[str, type[BaseModel] | None]:
    """How the agent asks: a JSON schema for choices and scores, plain yes or no text otherwise."""
    q = task.question
    if q["kind"] == "choice":
        labels = tuple(q["options"])
        return "json", create_model("Decision", label=(Literal[labels], ...))  # type: ignore[valid-type]
    if q["kind"] == "score":
        levels = tuple(q["levels"])
        return "json", create_model("Rating", rating=(Literal[levels], ...))  # type: ignore[valid-type]
    if q["kind"] == "yes_no":
        return "text", None
    return "json", create_model("Entities", entities=(dict[str, Any], ...))


def messages_for(task: Any, row: Mapping[str, Any]) -> list[dict[str, str]]:
    q = task.question
    instructions = q["instructions"]
    if q["kind"] == "yes_no":
        instructions += "\nAnswer with yes or no only."
    elif q["kind"] in ("choice", "score"):
        # A router prompt names its options; structured output alone only
        # constrains the tokens, it does not show the model the list.
        labels = q.get("options") or q.get("levels")
        instructions += "\nPick exactly one of these labels:\n" + "; ".join(labels)
    system = {"role": "system", "content": SYSTEM.format(instructions=instructions)}
    state = row["state"]
    if isinstance(state, list):
        return [system, *({"role": m["role"], "content": m["content"]} for m in state)]
    if isinstance(state, Mapping):
        text = "\n".join(f"{k}: {v}" for k, v in state.items())
        return [system, {"role": "user", "content": text}]
    return [system, {"role": "user", "content": str(state)}]


def reply_for(task: Any, value: Any) -> str:
    kind = task.question["kind"]
    if kind == "choice":
        return json.dumps({"label": value})
    if kind == "score":
        return json.dumps({"rating": value})
    if kind == "yes_no":
        return "yes" if value else "no"
    return json.dumps({"entities": value or {}})


def answer_value(task: Any, text: str | None, parsed: Any) -> Any:
    kind = task.question["kind"]
    if kind == "yes_no":
        return (text or "").strip().lower().startswith("yes")
    if parsed is None:
        return None
    if kind == "choice":
        return parsed.label
    if kind == "score":
        return parsed.rating
    return parsed.entities


def same(task: Any, a: Any, b: Any) -> bool:
    return bool(a == b)


def source_rows(source: str) -> tuple[Any, list[dict[str, Any]]]:
    """``task:<name>`` for a decision benchmark task's frozen rows, or a long stream by name."""
    if source.startswith("task:"):
        task = load_tasks([source[5:]])[0]
        return task, load_rows(task, "test") + load_rows(task, "calibration")
    from streams import load_stream

    return load_stream(source)


def run(
    source: str,
    teacher: str,
    *,
    learn_every: int,
    audit: float,
    seed: int,
    limit: int | None = None,
    target: float = 0.97,
) -> dict[str, Any]:
    task, rows = source_rows(source)
    task_name = task.name
    answers = teacher_answers(teacher, task_name, rows)
    rows = [r for r in rows if r["id"] in answers]
    random.Random(seed).shuffle(rows)
    if limit:
        rows = rows[:limit]
    style, model = contract(task)

    current: dict[str, Any] = {}

    def llm(request: httpx2.Request) -> httpx2.Response:
        record = answers[current["id"]]
        usage = {
            "prompt_tokens": int(record.get("input_tokens") or 0),
            "completion_tokens": int(record.get("output_tokens") or 0),
        }
        if record.get("cost_usd") is not None:
            usage["cost"] = record["cost_usd"]
        return httpx2.Response(
            200,
            json={
                "id": "chatcmpl-replay",
                "object": "chat.completion",
                "created": 0,
                "model": "teacher",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": reply_for(task, record["value"]),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage,
            },
        )

    workdir = Path(tempfile.mkdtemp(prefix=f"habits-{task_name}-"))
    habits = Habits(workdir, learn_every=None, audit=audit, seed=seed, target=target)
    client = wrap(
        openai.OpenAI(
            api_key="replay", http_client=httpx2.Client(transport=httpx2.MockTransport(llm))
        ),
        habits,
    )
    timeline = []
    local = agreed_local = audits = 0
    correct_pipeline = correct_teacher = 0
    cost_teacher = cost_pipeline = 0.0
    local_ms: list[float] = []
    activated_at = None
    learn_s = 0.0
    for i, row in enumerate(rows, 1):
        current["id"] = row["id"]
        record = answers[row["id"]]
        messages = messages_for(task, row)
        started = time.perf_counter()
        if style == "json":
            raw = client.chat.completions.with_raw_response.parse(
                model="teacher", messages=messages, response_format=model
            )
            completion = raw.parse()
            message = completion.choices[0].message
            value = answer_value(task, message.content, message.parsed)
        else:
            raw = client.chat.completions.with_raw_response.create(
                model="teacher", messages=messages
            )
            completion = raw.parse()
            value = answer_value(task, completion.choices[0].message.content, None)
        elapsed = (time.perf_counter() - started) * 1000
        answered_locally = raw.headers.get("x-thinkless-habit") == "local"
        cost = float(record.get("cost_usd") or 0.0)
        cost_teacher += cost
        if answered_locally:
            local += 1
            local_ms.append(elapsed)
            agreed_local += same(task, value, record["value"])
            if activated_at is None:
                activated_at = i
        else:
            cost_pipeline += cost
        correct_pipeline += same(task, value, row["label"])
        correct_teacher += same(task, record["value"], row["label"])
        if i % learn_every == 0:
            t = time.perf_counter()
            plan_now = habits.learn()
            learn_s += time.perf_counter() - t
            statuses = [s.get("status") for s in plan_now["sites"].values()]
            timeline.append(
                {"calls": i, "local": local, "agreed_local": agreed_local, "status": statuses}
            )
    audits = habits.counts["audit"]
    plan = habits.plan or {}
    sites = [
        {k: s.get(k) for k in ("name", "verdict", "status", "threshold", "evidence", "reason")}
        for s in plan.get("sites", {}).values()
    ]
    habits.close()
    events_file = workdir / "events.jsonl"
    events = (
        [json.loads(line) for line in events_file.read_text(encoding="utf-8").splitlines()]
        if events_file.exists()
        else []
    )
    shutil.rmtree(workdir, ignore_errors=True)
    n = len(rows)
    return {
        "source": source,
        "task": task_name,
        "kind": task.question["kind"],
        "teacher": teacher,
        "calls": n,
        "local": local,
        "local_share": local / n,
        "local_agreement": agreed_local / local if local else None,
        "audits": audits,
        "activated_at": activated_at,
        "accuracy_teacher": correct_teacher / n,
        "accuracy_pipeline": correct_pipeline / n,
        "cost_teacher_usd": cost_teacher,
        "cost_pipeline_usd": cost_pipeline,
        "local_latency_ms_median": sorted(local_ms)[len(local_ms) // 2] if local_ms else None,
        "learn_seconds": round(learn_s, 2),
        "timeline": timeline,
        "sites": sites,
        "events": [
            {k: e.get(k) for k in ("event", "threshold", "evidence", "audits", "agreed")}
            for e in events
        ],
        "settings": {"learn_every": learn_every, "audit": audit, "seed": seed, "target": target},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="task:<name> or a stream: banking77, clinc150",
    )
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--learn-every", type=int, default=100)
    parser.add_argument("--audit", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit", type=int, default=None, help="Use only the first N rows after shuffling."
    )
    parser.add_argument("--target", type=float, default=0.97)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    results = []
    for name in args.source:
        result = run(
            name,
            args.teacher,
            learn_every=args.learn_every,
            audit=args.audit,
            seed=args.seed,
            limit=args.limit,
            target=args.target,
        )
        results.append(result)
        agreement = result["local_agreement"]
        print(
            f"{name:15} calls {result['calls']:5}  local {result['local_share']:6.1%}  "
            f"local agreement {'n/a' if agreement is None else f'{agreement:.1%}'}  "
            f"accuracy {result['accuracy_teacher']:.1%} -> {result['accuracy_pipeline']:.1%}  "
            f"cost ${result['cost_teacher_usd']:.4f} -> ${result['cost_pipeline_usd']:.4f}  "
            f"activated at {result['activated_at']}",
            flush=True,
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
