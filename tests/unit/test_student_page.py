"""The browser student on the docs page must answer exactly like the Python one."""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from thinkless.habits.features import featurize

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs" / "assets" / "student"
NODE = shutil.which("node")

TEXTS = [
    "I still have not received my new card, I ordered over a week ago.",
    "Why was I charged a fee for withdrawing cash?",
    "",
    "   ",
    "snake_case_words and 12345 numbers 3.14",
    "ΣΟΦΙΑ ΚΑΙ ΛΟΓΟΣ",  # a final sigma lowers to ς in both
    "İstanbul ŞİMDİ",  # dotted capital I lowers to two code points
    "मेरा कार्ड काम नहीं कर रहा है",  # combining vowel signs split \w runs
    "我的卡丢了，请帮我冻结",
    "card 💳 lost 😱 help!!!",
    "repeat repeat repeat repeat " * 30,
    "x" * 5000 + " middle " + "y" * 5000,  # truncated to the first and last 2,000 chars
    "Ünïcödé café naïve façade",
    "tab\tseparated\nnew line\r\nwindows",
]


def _run_node(script: str, tmp_path: Path) -> object:
    module = tmp_path / "student.mjs"
    module.write_text((ASSETS / "student.js").read_text(encoding="utf-8"), encoding="utf-8")
    runner = tmp_path / "run.mjs"
    runner.write_text(script, encoding="utf-8")
    done = subprocess.run(
        [NODE, str(runner)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@pytest.mark.skipif(NODE is None, reason="needs node")
def test_browser_features_match_python(tmp_path: Path) -> None:
    (tmp_path / "texts.json").write_text(json.dumps(TEXTS), encoding="utf-8")
    script = """
import { featurize } from "./student.mjs";
import { readFileSync } from "node:fs";
const texts = JSON.parse(readFileSync(new URL("./texts.json", import.meta.url), "utf-8"));
console.log(JSON.stringify(texts.map((t) => Object.fromEntries(featurize(t)))));
"""
    browser = _run_node(script, tmp_path)
    assert isinstance(browser, list)
    for text, js in zip(TEXTS, browser, strict=True):
        py = featurize(text)
        assert {int(k) for k in js} == set(py), text[:40]
        for key, value in py.items():
            assert js[str(key)] == pytest.approx(value, rel=1e-9, abs=1e-12), text[:40]


def _read_weights() -> tuple[list[str], dict[int, int], np.ndarray, np.ndarray]:
    meta = json.loads((ASSETS / "banking77.json").read_text(encoding="utf-8"))
    data = (ASSETS / "banking77.bin").read_bytes()
    assert data[:4] == b"TLS1"
    features, classes = struct.unpack("<II", data[4:12])
    offset = 12
    keys = np.frombuffer(data, "<u4", features, offset)
    offset += 4 * features
    scales = np.frombuffer(data, "<f4", features, offset)
    offset += 4 * features
    weights = np.frombuffer(data, np.int8, features * classes, offset).reshape(features, classes)
    assert offset + features * classes == len(data)
    return meta["labels"], {int(k): i for i, k in enumerate(keys)}, scales, weights


def _python_predict(text: str) -> tuple[str, float]:
    labels, index, scales, weights = _read_weights()
    logits = np.zeros(len(labels))
    for feature, value in featurize(text).items():
        row = index.get(feature)
        if row is not None:
            logits += weights[row].astype(np.float64) * float(scales[row]) * value
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    best = int(probs.argmax())
    return labels[best], float(probs[best])


def test_published_student_is_consistent() -> None:
    meta = json.loads((ASSETS / "banking77.json").read_text(encoding="utf-8"))
    labels, index, _, _ = _read_weights()
    assert 70 <= len(labels) <= 77  # labels the teacher actually gave in training
    assert len(index) == meta["compact"]["features"]
    assert meta["compact"]["same_route"] > 0.99
    assert 0.5 < meta["threshold"] < 1.0
    test = meta["test"]
    assert test["agreed_local"] / test["local"] >= meta["target"] - 0.01
    assert all(s["teacher"] in labels for s in meta["samples"])


@pytest.mark.skipif(NODE is None, reason="needs node")
def test_browser_predictions_match_python(tmp_path: Path) -> None:
    meta = json.loads((ASSETS / "banking77.json").read_text(encoding="utf-8"))
    texts = [s["text"] for s in meta["samples"]] + TEXTS
    (tmp_path / "texts.json").write_text(json.dumps(texts), encoding="utf-8")
    shutil.copy(ASSETS / "banking77.bin", tmp_path / "banking77.bin")
    shutil.copy(ASSETS / "banking77.json", tmp_path / "banking77.json")
    script = """
import { loadStudent, predict } from "./student.mjs";
import { readFileSync } from "node:fs";
const here = (name) => new URL("./" + name, import.meta.url);
const meta = JSON.parse(readFileSync(here("banking77.json"), "utf-8"));
const bytes = readFileSync(here("banking77.bin"));
const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
const student = loadStudent(buffer, meta.labels);
const texts = JSON.parse(readFileSync(here("texts.json"), "utf-8"));
console.log(JSON.stringify(texts.map((t) => predict(student, t).ranked[0])));
"""
    browser = _run_node(script, tmp_path)
    assert isinstance(browser, list)
    for text, js in zip(texts, browser, strict=True):
        label, probability = _python_predict(text)
        assert js["label"] == label, text[:40]
        assert js["probability"] == pytest.approx(probability, abs=1e-6), text[:40]
