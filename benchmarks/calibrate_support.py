"""Reproduce the support demo's calibrated thresholds.

Runs Laya and GLiNER on the labeled calibration set that ships with the demo
(written separately from the benchmark tickets) and prints, for each
question and provider, the lowest threshold whose accepted answers meet the
accuracy target. The targets follow the policy stated in
``thinkless/demo/support/questions.py``: 100 percent for the security
question, 95 percent for routing questions.

    python benchmarks/calibrate_support.py
"""

from __future__ import annotations

import json
from importlib import resources

from rich.console import Console
from rich.table import Table

from thinkless import Engine, Tracer
from thinkless.bench.metrics import recommend_threshold, threshold_sweep
from thinkless.demo.support import questions as q
from thinkless.providers.gliner import GLiNER
from thinkless.providers.laya import Laya


def main() -> None:
    text = (
        resources.files("thinkless.demo.support")
        .joinpath("data/calibration.jsonl")
        .read_text(encoding="utf-8")
    )
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    routing = [r for r in rows if not r["injection"]]
    intent_rows = [r for r in routing if not r["wants_human"]]

    providers = {"gliner": GLiNER(), "laya": Laya()}
    jobs = [
        ("intent", "gliner", q.INTENT, intent_rows, 0.95),
        ("intent", "laya", q.INTENT, intent_rows, 0.95),
        ("wants_human", "laya", q.WANTS_HUMAN, routing, 0.95),
        ("injection", "laya", q.INJECTION, rows, 1.0),
    ]
    table = Table(title="Support demo calibration")
    for column in (
        "key",
        "rows",
        "target",
        "threshold",
        "answers alone",
        "accuracy",
        "in the demo",
    ):
        table.add_column(column, justify="right")
    for name, provider_name, question, subset, target in jobs:
        engine = Engine([providers[provider_name]], threshold=0.0, tracer=Tracer())
        confidences, correct = [], []
        for row in subset:
            decision = engine.decide({"message": row["text"]}, question)
            confidences.append(decision.confidence or 0.0)
            correct.append(decision.value == row[name])
        best = recommend_threshold(threshold_sweep(confidences, correct), target)
        key = f"{name}@{provider_name}"
        table.add_row(
            key,
            str(len(subset)),
            f"{target:.0%}",
            "none" if best is None else f"{best.threshold:.2f}",
            "n/a" if best is None else f"{best.coverage:.0%}",
            "n/a" if best is None or best.accuracy is None else f"{best.accuracy:.0%}",
            str(q.CALIBRATED_THRESHOLDS.get(key, "default")),
        )
    Console().print(table)


if __name__ == "__main__":
    main()
