from __future__ import annotations

import json

import pytest

from tests.unit.test_support_demo import SCENARIOS, Oracle, _reply
from thinkless import Engine
from thinkless.bench import (
    expected_calibration_error,
    percentile,
    recommend_threshold,
    run_support_benchmark,
    threshold_sweep,
)
from thinkless.bench.intents import ChoiceEvaluation, _cascade
from thinkless.bench.report import support_markdown, support_table
from thinkless.demo.support import support_rules
from thinkless.llm import ScriptedLLM


def test_percentile() -> None:
    assert percentile([], 50) == 0.0
    assert percentile([5.0], 95) == 5.0
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([1, 2, 3, 4, 5], 100) == 5


def test_ece_is_zero_when_confidence_matches_accuracy() -> None:
    assert expected_calibration_error([1.0, 1.0], [True, True]) == pytest.approx(0.0)
    assert expected_calibration_error([0.9, 0.9], [False, False]) == pytest.approx(0.9)


def test_sweep_and_recommendation() -> None:
    confidences = [0.95, 0.9, 0.6, 0.3]
    correct = [True, True, False, True]
    points = threshold_sweep(
        confidences, correct, fallback_correct=[True, True, True, False], thresholds=[0.0, 0.5, 0.8]
    )
    assert [p.coverage for p in points] == [1.0, 0.75, 0.5]
    assert points[2].accuracy == 1.0
    assert points[2].cascade_accuracy == pytest.approx(0.75)
    best = recommend_threshold(points, 0.99)
    assert best is not None
    assert best.threshold == 0.8
    assert recommend_threshold(points[:1], 0.99) is None


def test_cascade_simulation() -> None:
    small = ChoiceEvaluation(
        provider="small",
        predictions=["a", "b"],
        confidences=[0.9, 0.2],
        correct=[True, False],
        latencies_ms=[10.0, 10.0],
        input_tokens=[5, 5],
    )
    llm = ChoiceEvaluation(
        provider="llm",
        predictions=["a", "c"],
        confidences=[None, None],
        correct=[True, True],
        latencies_ms=[500.0, 500.0],
        input_tokens=[50, 50],
    )
    points = {
        p.threshold: p for p in _cascade({"small": small, "llm": llm}, ["small", "llm"], "llm")
    }
    assert points[0.5].accuracy == 1.0
    assert points[0.5].llm_share == 0.5
    assert points[0.5].mean_latency_ms == pytest.approx(260.0)
    assert points[0.0].llm_share == 0.0
    assert points[1.01].llm_share == 1.0


def test_support_benchmark_end_to_end(tmp_path) -> None:
    def factory(tracer):
        return Engine([support_rules(), Oracle()], llm=ScriptedLLM(_reply), tracer=tracer)

    bench = run_support_benchmark(
        {"oracle": factory},
        reasoning_model="scripted",
        scenarios=SCENARIOS[:8],
        output_dir=tmp_path,
    )
    report = bench.modes["oracle"]
    assert report.tickets == 8
    assert report.task_success == 1.0
    assert report.failures == []
    assert report.questions["intent"]["accuracy"] == 1.0
    saved = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert saved["modes"]["oracle"]["tickets"] == 8
    assert len(list((tmp_path / "traces" / "oracle").glob("*.jsonl"))) == 8
    markdown = support_markdown(bench)
    assert "| Task success (correct action) | 100.0% |" in markdown
    assert support_table(bench).row_count > 10
