"""Measure batched decisions against one-at-a-time decisions on real local models.

For each provider, the same benchmark rows are answered twice: with
``Engine.decide_many`` one row at a time, and with ``Engine.decide_batch``.
The script reports throughput for both and how often the answers differ
(batching pads inputs together, which can move a model's scores slightly).
It then sends the rows as concurrent requests through a ``Batched`` wrapper,
the path the decision server uses.

    python benchmarks/batching.py --provider gliner --task banking77 --rows 300
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

from thinkless import Engine
from thinkless.bench.decisions import build_submission_provider, load_rows, load_tasks, question_for
from thinkless.providers import Batched
from thinkless.tracing import Tracer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--provider", required=True)
    parser.add_argument("--task", default="banking77")
    parser.add_argument("--rows", type=int, default=300)
    parser.add_argument("--chunk", type=int, default=32, help="states per decide_batch call")
    parser.add_argument("--threads", type=int, default=16, help="concurrent callers")
    args = parser.parse_args()

    task = load_tasks([args.task])[0]
    question = question_for(task)
    rows = load_rows(task, "test")[: args.rows]
    states = [row["state"] for row in rows]
    provider = build_submission_provider(args.provider)
    engine = Engine([provider], threshold=0.8, tracer=Tracer())
    engine.warmup([question])

    started = time.perf_counter()
    one = [engine.decide(state, question) for state in states]
    sequential_s = time.perf_counter() - started

    started = time.perf_counter()
    many = []
    for start in range(0, len(states), args.chunk):
        chunk = engine.decide_batch(states[start : start + args.chunk], [question])
        many += [d[question.key] for d in chunk]
    batched_s = time.perf_counter() - started

    wrapped = Batched(provider, max_batch=args.chunk)
    served = Engine([wrapped], threshold=0.8, tracer=Tracer())
    started = time.perf_counter()
    with ThreadPoolExecutor(args.threads) as pool:
        concurrent = list(pool.map(lambda s: served.decide(s, question), states))
    concurrent_s = time.perf_counter() - started
    wrapped.stop()

    def value(d):
        return d.level if d.level is not None else d.value

    def differ(a, b):
        return sum(value(x) != value(y) for x, y in zip(a, b, strict=True))

    def flips(a, b):
        return sum(x.status != y.status for x, y in zip(a, b, strict=True))

    def gap(a, b):
        return max(
            abs((x.confidence or 0.0) - (y.confidence or 0.0)) for x, y in zip(a, b, strict=True)
        )

    report = {
        "provider": args.provider,
        "task": args.task,
        "rows": len(states),
        "device": getattr(provider, "device", None),
        "one_at_a_time_per_s": round(len(states) / sequential_s, 1),
        "decide_batch_per_s": round(len(states) / batched_s, 1),
        "batched_server_per_s": round(len(states) / concurrent_s, 1),
        "server_batches": wrapped.batches_run,
        "server_calls_batched": wrapped.calls_batched,
        "answers_differ": {
            "decide_batch": differ(one, many),
            "server": differ(one, concurrent),
        },
        "accept_flips": {"decide_batch": flips(one, many), "server": flips(one, concurrent)},
        "max_confidence_gap": {
            "decide_batch": round(gap(one, many), 5),
            "server": round(gap(one, concurrent), 5),
        },
    }
    print(json.dumps(report, indent=1))
    engine.close()


if __name__ == "__main__":
    main()
