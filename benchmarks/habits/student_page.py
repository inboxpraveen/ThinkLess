"""Build the data behind the docs page "Test a student in your browser".

The page runs a Habits student in the browser: the same hashed features and
the same linear model as ``thinkless.habits``, trained on Qwen 3.7 Flash's
recorded answers to Banking77 queries. This script trains it, fits the gate
threshold on calls it was not trained on, measures it on a third set of
calls, and writes a compact copy of the weights for the page.

Calls, in the replay's shuffled order (seed 0):

* the first 8,000 train the student,
* the next 2,500 choose the threshold with the Habits gate (97% agreement as
  a 95% lower bound, the published default),
* the rest (2,583) are the test set the page reports and replays.

The page's copy stores weights as 8-bit integers with a scale per feature
(``--features`` keeps only the strongest features, for a smaller file). The script reports how
often that copy gives the same label and the same local-or-LLM call as the
full student on the test set, and the page prints those numbers.

Run from ``benchmarks/habits`` (free, from the recorded answers):

    python student_page.py --out ../../docs/assets/student
"""

from __future__ import annotations

import argparse
import json
import random
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
from streams import load_stream

from thinkless.habits.features import featurize
from thinkless.habits.gate import select_threshold
from thinkless.habits.student import LinearModel

CACHE = Path(__file__).parent / "cache" / "banking77-qwen_qwen3.7-flash.jsonl"
TRAIN, GATE = 8000, 2500
TARGET, CONFIDENCE = 0.97, 0.95
MAGIC = b"TLS1"


def _rows() -> list[dict[str, Any]]:
    _, rows = load_stream("banking77")
    teacher = {}
    with CACHE.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            teacher[record["id"]] = record
    usable = [dict(row, teacher=teacher[row["id"]]) for row in rows if row["id"] in teacher]
    usable = [row for row in usable if row["teacher"].get("value")]
    random.Random(0).shuffle(usable)
    return usable


def _compact(model: LinearModel, keep: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keys, int8 weights and per-row scales of the ``keep`` strongest features."""
    keys = np.array(list(model.vocab), dtype=np.int64)
    rows = np.array([model.vocab[k] for k in keys], dtype=np.int64)
    weights = model.weights[rows]
    strength = np.abs(weights).max(axis=1)
    order = np.argsort(-strength, kind="stable")[:keep]
    keys, weights = keys[order], weights[order]
    scales = np.abs(weights).max(axis=1) / 127.0
    scales[scales == 0] = 1.0
    quantized = np.clip(np.rint(weights / scales[:, None]), -127, 127).astype(np.int8)
    return keys.astype(np.uint32), quantized, scales.astype(np.float32)


def _compact_predict(
    keys: np.ndarray, weights: np.ndarray, scales: np.ndarray, labels: list[str], text: str
) -> tuple[str, float]:
    index = {int(k): i for i, k in enumerate(keys)}
    logits = np.zeros(len(labels))
    for feature, value in featurize(text).items():
        row = index.get(feature)
        if row is not None:
            logits += weights[row].astype(np.float64) * float(scales[row]) * value
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    best = int(probs.argmax())
    return labels[best], float(probs[best])


def build(out: Path, keep: int, samples: int) -> dict[str, Any]:
    rows = _rows()
    train, gate, test = rows[:TRAIN], rows[TRAIN : TRAIN + GATE], rows[TRAIN + GATE :]
    started = time.perf_counter()
    model = LinearModel.fit(
        [featurize(r["state"]) for r in train], [r["teacher"]["value"] for r in train], seed=0
    )
    seconds = time.perf_counter() - started

    def judge(subset: list[dict[str, Any]]) -> list[tuple[str, float]]:
        return [model.predict(featurize(r["state"]), r["state"]) for r in subset]

    gated = judge(gate)
    result = select_threshold(
        [p for _, p in gated],
        [label == r["teacher"]["value"] for (label, _), r in zip(gated, gate, strict=True)],
        target=TARGET,
        confidence=CONFIDENCE,
    )
    if result is None:
        raise SystemExit("the gate did not pass; nothing to publish")
    threshold = result.threshold

    keys, weights, scales = _compact(model, keep)
    full = judge(test)
    small = [_compact_predict(keys, weights, scales, model.labels, r["state"]) for r in test]
    local = [p >= threshold for _, p in full]
    agreed_local = sum(
        1
        for (label, p), r in zip(full, test, strict=True)
        if p >= threshold and label == r["teacher"]["value"]
    )
    same_label = sum(a[0] == b[0] for a, b in zip(full, small, strict=True))
    same_route = sum(
        (a[1] >= threshold) == (b[1] >= threshold) for a, b in zip(full, small, strict=True)
    )
    teacher_cost = sum(float(r["teacher"].get("cost_usd") or 0.0) for r in test)
    saved_cost = sum(
        float(r["teacher"].get("cost_usd") or 0.0)
        for r, is_local in zip(test, local, strict=True)
        if is_local
    )
    summary = {
        "task": "banking77",
        "teacher": "qwen/qwen3.7-flash (OpenRouter, reasoning off, temperature 0)",
        "labels": model.labels,
        "threshold": threshold,
        "target": TARGET,
        "confidence": CONFIDENCE,
        "calls": {"train": len(train), "gate": len(gate), "test": len(test)},
        "train_seconds": round(seconds, 1),
        "gate": {
            "accepted": result.accepted,
            "agreed": result.agreed,
            "calls": result.calls,
            "lower_bound": result.lower_bound,
        },
        "test": {
            "local": sum(local),
            "agreed_local": agreed_local,
            "teacher_cost_usd": teacher_cost,
            "saved_cost_usd": saved_cost,
            "teacher_accuracy": sum(r["teacher"]["value"] == r["label"] for r in test) / len(test),
        },
        "compact": {
            "features": len(keys),
            "full_features": len(model.vocab),
            "same_label": same_label / len(test),
            "same_route": same_route / len(test),
        },
        "weights": "banking77.bin",
    }
    # The replay shows test calls the visitor can step through.
    picks = random.Random(1).sample(range(len(test)), samples)
    summary["samples"] = [
        {
            "text": test[i]["state"],
            "teacher": test[i]["teacher"]["value"],
            "label": test[i]["label"],
            "cost_usd": test[i]["teacher"].get("cost_usd"),
        }
        for i in sorted(picks)
    ]
    out.mkdir(parents=True, exist_ok=True)
    # Binary layout, little-endian: magic, feature count, class count, then
    # uint32 keys, float32 scales and int8 weights (row-major, features x classes).
    with (out / "banking77.bin").open("wb") as handle:
        handle.write(MAGIC + struct.pack("<II", len(keys), len(model.labels)))
        handle.write(keys.astype("<u4").tobytes())
        handle.write(scales.astype("<f4").tobytes())
        handle.write(weights.astype(np.int8).tobytes())
    (out / "banking77.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--features", type=int, default=0, help="0 keeps every feature")
    parser.add_argument("--samples", type=int, default=60)
    args = parser.parse_args()
    summary = build(args.out, args.features or 1 << 30, args.samples)
    shown = {k: v for k, v in summary.items() if k not in ("labels", "samples")}
    print(json.dumps(shown, indent=1))


if __name__ == "__main__":
    main()
