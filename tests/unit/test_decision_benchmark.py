from __future__ import annotations

import json

import pytest

from tests.conftest import FakeProvider
from thinkless import Answer, Choice, Extract, Kind, Score, YesNo
from thinkless.bench.decisions import (
    leaderboard_markdown,
    load_rows,
    load_tasks,
    question_for,
    run_benchmark,
    score_task,
    verify_result,
)
from thinkless.providers.base import render_state


def _key(state) -> str:
    return json.dumps(state, sort_keys=True, ensure_ascii=False)


def test_tasks_are_published_consistently() -> None:
    tasks = load_tasks()
    assert [t.name for t in tasks] == [
        "banking77",
        "clinc150",
        "massive",
        "multiwoz",
        "jailbreak",
        "civil_comments",
        "helpsteer2",
        "wnut17",
    ]
    assert {t.kind for t in tasks} == {"choice", "yes_no", "score", "extract"}
    for task in tasks:
        question = question_for(task)
        for split in ("calibration", "test"):
            rows = load_rows(task, split)  # also checks the sha256
            assert len(rows) == task.files[split].rows
            assert len({r["id"] for r in rows}) == len(rows)
            if isinstance(question, Choice):
                assert {r["label"] for r in rows} <= set(question.options)
            if isinstance(question, Score):
                assert {r["label"] for r in rows} <= set(question.levels)
            if isinstance(question, YesNo):
                assert {r["label"] for r in rows} <= {True, False}
            if isinstance(question, Extract):
                assert all(set(r["label"]) == set(question.fields) for r in rows)
        assert len(task.fingerprint()) == 16


def test_calibration_and_test_rows_never_overlap() -> None:
    for task in load_tasks():
        test = {_key(r["state"]) for r in load_rows(task, "test")}
        calibration = {_key(r["state"]) for r in load_rows(task, "calibration")}
        assert not test & calibration, task.name


def test_conversations_render_as_role_lines() -> None:
    state = [
        {"role": "user", "content": "I need a train"},
        {"role": "assistant", "content": "Where to?"},
        {"role": "user", "content": [{"type": "text", "text": "Cambridge"}]},
    ]
    assert render_state(state) == "user: I need a train\nassistant: Where to?\nuser: Cambridge"
    assert render_state([{"a": 1}, {"b": 2}]) == "a: 1\nb: 2"


INTENT = Choice("Which?", options=["a", "b", "c"], name="t")


def _pred(i: int, value, confidence=None) -> dict:
    return {
        "id": f"r{i}",
        "value": value,
        "confidence": confidence,
        "latency_ms": 10.0,
        "cost_usd": 0.001,
    }


def test_score_choice_with_selective_threshold() -> None:
    truth = {f"r{i}": "a" if i % 2 else "b" for i in range(10)}
    # Confident answers are right, unsure ones wrong.
    test = [_pred(i, truth[f"r{i}"] if i < 8 else "c", 0.9 if i < 8 else 0.3) for i in range(10)]
    calibration = [
        _pred(i, truth[f"r{i}"] if i < 8 else "c", 0.9 if i < 8 else 0.3) for i in range(10)
    ]
    m = score_task(INTENT, test, truth, calibration=calibration, calibration_truth=truth)
    assert m.accuracy == pytest.approx(0.8)
    assert m.calibrated
    assert m.threshold is not None
    assert 0.3 < m.threshold <= 0.9
    assert m.coverage == pytest.approx(0.8)
    assert m.selective_accuracy == 1.0
    assert m.cost_per_1k_usd == pytest.approx(1.0)
    assert m.aurc < 0.1
    assert 0 < m.macro_f1 < 1


def test_score_uncalibrated_and_abstentions() -> None:
    truth = {"r0": "a", "r1": "b"}
    m = score_task(INTENT, [_pred(0, "A"), _pred(1, None)], truth)
    assert m.accuracy == 0.5  # case-insensitive match; the abstention counts as wrong
    assert not m.calibrated
    assert m.coverage is None
    assert m.ece is None


def test_score_kinds() -> None:
    yes = YesNo("Toxic?", name="t")
    truth = {"r0": True, "r1": True, "r2": False, "r3": False}
    m = score_task(yes, [_pred(0, True), _pred(1, True), _pred(2, True), _pred(3, False)], truth)
    assert m.balanced_accuracy == pytest.approx(0.75)

    score = Score("How helpful?", levels=["low", "mid", "high"], name="t")
    truth = {"r0": "low", "r1": "high"}
    m = score_task(score, [_pred(0, "mid"), _pred(1, "high")], truth)
    assert m.accuracy == 0.5
    assert m.mae_levels == 0.5
    assert m.within_one == 1.0

    extract = Extract(fields={"person": "", "location": ""}, name="t")
    truth = {"r0": {"person": "Ada", "location": None}, "r1": {"person": None, "location": "Paris"}}
    preds = [
        _pred(0, {"person": "ada", "location": "London"}),
        _pred(1, {"person": None, "location": "Paris"}),
    ]
    m = score_task(extract, preds, truth)
    assert m.accuracy == 0.5
    assert m.field_precision == pytest.approx(2 / 3)
    assert m.field_recall == 1.0


def _perfect_provider(task_name: str) -> FakeProvider:
    task = load_tasks([task_name])[0]
    labels = {
        _key(r["state"]): r["label"]
        for split in ("test", "calibration")
        for r in load_rows(task, split)
    }

    def answer(state, question):
        label = labels[_key(state)]
        return Answer(
            value=label,
            probabilities={"yes": 0.99 if label else 0.01, "no": 0.01 if label else 0.99},
        )

    return FakeProvider("perfect", {task_name: answer}, kinds=frozenset({Kind.YES_NO}))


def test_run_verify_and_detect_tampering(tmp_path) -> None:
    result = run_benchmark(
        _perfect_provider("jailbreak"), name="Perfect", spec="test", tasks=["jailbreak"]
    )
    assert result.tasks["jailbreak"].metrics.accuracy == 1.0
    assert list(result.tasks) == ["jailbreak"]

    # A run on one task is incomplete for the benchmark, so fill the rest as unsupported.
    full = run_benchmark(_perfect_provider("jailbreak"), name="Perfect", spec="test")
    assert not full.tasks["banking77"].supported
    good = full.write(tmp_path / "good.json")
    check = verify_result(good)
    assert check.ok
    assert check.rankable
    assert check.metrics["jailbreak"].accuracy == 1.0

    data = json.loads(good.read_text(encoding="utf-8"))
    data["tasks"]["jailbreak"]["predictions"]["test"][0]["value"] = not data["tasks"]["jailbreak"][
        "predictions"
    ]["test"][0]["value"]
    (tmp_path / "edited.json").write_text(json.dumps(data), encoding="utf-8")
    edited = verify_result(tmp_path / "edited.json")
    assert edited.metrics["jailbreak"].accuracy < 1.0  # the recomputed number, not the stored one
    assert any("stored metrics differ" in n for n in edited.notes)

    data["tasks"]["jailbreak"]["predictions"]["test"].pop()
    (tmp_path / "short.json").write_text(json.dumps(data), encoding="utf-8")
    assert not verify_result(tmp_path / "short.json").ok

    data = json.loads(good.read_text(encoding="utf-8"))
    data["tasks"]["jailbreak"]["fingerprint"] = "0" * 16
    (tmp_path / "old.json").write_text(json.dumps(data), encoding="utf-8")
    assert any("fingerprint" in p for p in verify_result(tmp_path / "old.json").problems)

    text = leaderboard_markdown([check, verify_result(tmp_path / "short.json")])
    assert "| Perfect |" in text
    assert "## Not ranked" in text


def test_limit_marks_the_run_partial(tmp_path) -> None:
    result = run_benchmark(_perfect_provider("jailbreak"), name="Smoke", spec="test", limit=3)
    assert result.partial
    check = verify_result(result.write(tmp_path / "smoke.json"))
    assert check.ok
    assert not check.rankable


def test_cli_verify_and_leaderboard(tmp_path) -> None:
    from typer.testing import CliRunner

    from thinkless.cli.decisions import decisions_app

    result = run_benchmark(_perfect_provider("jailbreak"), name="Perfect", spec="test")
    good = result.write(tmp_path / "verified" / "perfect.json")
    runner = CliRunner()
    checked = runner.invoke(decisions_app, ["verify", str(good)])
    assert checked.exit_code == 0, checked.output
    assert "ranked" in checked.output
    out = tmp_path / "leaderboard.md"
    built = runner.invoke(
        decisions_app, ["leaderboard", str(tmp_path / "verified"), "--out", str(out)]
    )
    assert built.exit_code == 0, built.output
    text = out.read_text(encoding="utf-8")
    assert "| Perfect | verified |" in text
    assert "Accuracy by language" not in text  # massive is unsupported by this provider

    broken = tmp_path / "broken.json"
    broken.write_text("{}", encoding="utf-8")
    assert runner.invoke(decisions_app, ["verify", str(broken)]).exit_code == 1


def test_transient_errors_are_retried(monkeypatch) -> None:
    from thinkless.bench.decisions import run as run_module
    from thinkless.bench.decisions.run import _transient

    monkeypatch.setattr(run_module, "RETRY_DELAY_S", 0.0)

    assert _transient("error: RateLimitError")
    assert _transient("error: APITimeoutError")
    assert not _transient("error: ValueError")
    assert not _transient(None)

    task = load_tasks(["jailbreak"])[0]
    labels = {_key(r["state"]): r["label"] for r in load_rows(task, "test")}
    calls: dict[str, int] = {}

    class RateLimitError(Exception):
        """Named like the OpenAI SDK's error, which is what the runner looks for."""

    def flaky(state, question):
        key = _key(state)
        calls[key] = calls.get(key, 0) + 1
        if calls[key] == 1 and len(calls) % 2:
            raise RateLimitError("slow down")
        return Answer(value=labels[key], probabilities={"yes": 0.9, "no": 0.1})

    provider = FakeProvider(
        "flaky", {"jailbreak": flaky}, kinds=frozenset({Kind.YES_NO}), calibrated=False
    )
    result = run_benchmark(provider, name="Flaky", spec="test", tasks=["jailbreak"], limit=10)
    rows = result.tasks["jailbreak"].predictions["test"]
    assert all(r["error"] is None for r in rows)
    assert result.tasks["jailbreak"].metrics.accuracy == 1.0
