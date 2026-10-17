"""Record a hosted LLM's answers on a long stream, for replays with a real teacher.

Sends each input to the model the way an agent would (a system prompt, the
input, and a JSON schema whose only field is the label), and appends the
answer, tokens and billed cost to a cache file. Rows already in the cache are
skipped, so an interrupted run resumes and a finished one is never paid for
twice.

    python benchmarks/habits/teach.py --stream banking77 \\
        --model qwen/qwen3.7-flash --max-cost-usd 0.40

Needs ``OPENROUTER_API_KEY`` (read from ``.env`` too).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import openai
from replay import answer_value, contract, messages_for
from streams import load_stream

from thinkless import load_env

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stream", required=True)
    parser.add_argument("--model", default="qwen/qwen3.7-flash")
    parser.add_argument("--max-cost-usd", type=float, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    load_env(ROOT.parent.parent / ".env")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    task, rows = load_stream(args.stream)
    random.Random(args.seed).shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{args.stream}-{args.model.replace('/', '_')}.jsonl"
    done: set[str] = set()
    spent = 0.0
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            done.add(record["id"])
            spent += float(record.get("cost_usd") or 0.0)
    todo = [row for row in rows if row["id"] not in done]
    print(f"{len(done)} cached, {len(todo)} to ask, ${spent:.4f} spent so far", flush=True)

    style, model = contract(task)
    response_format: dict[str, Any] | None = None
    if model is not None:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": model.__name__,
                "schema": model.model_json_schema(),
                "strict": True,
            },
        }
    client = openai.OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", max_retries=6)
    lock = threading.Lock()
    stop = threading.Event()
    state = {"spent": spent, "asked": 0, "errors": 0}
    handle = path.open("a", encoding="utf-8")

    def ask(row: dict[str, Any]) -> None:
        if stop.is_set():
            return
        try:
            extra: dict[str, Any] = {}
            if response_format is not None:
                extra["response_format"] = response_format
            completion = client.chat.completions.create(
                model=args.model,
                messages=messages_for(task, row),  # type: ignore[arg-type]
                temperature=0,
                extra_body={"reasoning": {"enabled": False}, "usage": {"include": True}},
                **extra,
            )
            content = completion.choices[0].message.content or ""
            if style == "text":
                value = answer_value(task, content, None)
            else:
                value = json.loads(content).get("label")
        except (openai.APIError, json.JSONDecodeError, AttributeError) as exc:
            with lock:
                state["errors"] += 1
            print(
                f"{row['id']}: {type(exc).__name__}: {str(exc)[:120]}", file=sys.stderr, flush=True
            )
            return
        usage = completion.usage
        extra = getattr(usage, "model_extra", None) or {}
        cost = float(extra.get("cost") or 0.0)
        record = {
            "id": row["id"],
            "value": value,
            "input_tokens": getattr(usage, "prompt_tokens", 0),
            "output_tokens": getattr(usage, "completion_tokens", 0),
            "cost_usd": cost,
            "model": completion.model,
        }
        with lock:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            state["spent"] += cost
            state["asked"] += 1
            if state["spent"] >= args.max_cost_usd:
                stop.set()
            if state["asked"] % 250 == 0:
                print(f"asked {state['asked']}, ${state['spent']:.4f}", flush=True)

    started = time.time()
    with ThreadPoolExecutor(args.concurrency) as pool:
        list(pool.map(ask, todo))
    handle.close()
    print(
        f"done: asked {state['asked']}, errors {state['errors']}, total ${state['spent']:.4f}, "
        f"{time.time() - started:.0f} s{' (stopped at the spending cap)' if stop.is_set() else ''}"
    )


if __name__ == "__main__":
    main()
