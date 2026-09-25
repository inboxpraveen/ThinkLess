from __future__ import annotations

import json

from tests.conftest import FakeProvider, choice_answer
from thinkless import Choice, Engine, JSONLSink, Plane, Tracer
from thinkless.tracing.export import drift_report, export_trace_labels, iter_decisions

INTENT = Choice(
    "What does the customer want?", options=["refund", "status", "other"], name="intent"
)


def _traffic(directory, *, sure: float, count: int, value: str = "refund", capture: bool = True):
    rest = (1 - sure) / 2
    fast = FakeProvider(
        "fast",
        {
            "intent": choice_answer(
                value, {"refund": rest, "status": rest, "other": rest} | {value: sure}
            )
        },
    )
    llm = FakeProvider(
        "llm",
        {"intent": choice_answer("status", {"refund": 0.0, "status": 1.0, "other": 0.0})},
        plane=Plane.LLM,
    )
    engine = Engine(
        [fast, llm], threshold=0.8, tracer=Tracer([JSONLSink(directory)], capture_content=capture)
    )
    for i in range(count):
        with engine.run("ticket"):
            engine.decide(f"ticket {i}", INTENT)


def test_iter_and_export(tmp_path) -> None:
    _traffic(tmp_path / "t", sure=0.98, count=5)
    _traffic(tmp_path / "t", sure=0.5, count=3)
    records = list(iter_decisions([tmp_path / "t"]))
    assert len(records) == 8
    assert {r["plane"] for r in records} == {"model", "llm"}

    out = tmp_path / "rows.jsonl"
    assert export_trace_labels([tmp_path / "t"], out, planes=["llm"]) == 3
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert all(row["text"].startswith("ticket") for row in rows)
    assert {row["label"] for row in rows} == {"status"}
    assert export_trace_labels([tmp_path / "t"], out, limit=2) == 2


def test_export_skips_traces_without_content(tmp_path) -> None:
    _traffic(tmp_path / "t", sure=0.98, count=3, capture=False)
    assert export_trace_labels([tmp_path / "t"], tmp_path / "rows.jsonl") == 0


def test_drift_flags_a_rising_llm_share(tmp_path) -> None:
    _traffic(tmp_path / "before", sure=0.98, count=40)
    _traffic(tmp_path / "after", sure=0.98, count=20)
    _traffic(tmp_path / "after", sure=0.5, count=20)
    report = drift_report([tmp_path / "before"], [tmp_path / "after"], min_decisions=30)
    (q,) = report.questions
    assert q.llm_share == (0.0, 0.5)
    assert any("LLM share rose" in flag for flag in q.flags)
    assert not any("not accepted" in flag for flag in q.flags)  # the LLM accepted them
    assert any("answers shifted" in flag for flag in q.flags)

    steady = drift_report([tmp_path / "before"], [tmp_path / "before"])
    assert steady.drifted == []


def test_drift_flags_a_rising_share_not_accepted(tmp_path) -> None:
    from thinkless.providers import Rules

    rules = Rules()
    rules.match("intent", r"refund", "refund")
    for directory, messages in (("a", ["refund"] * 40), ("b", ["refund"] * 10 + ["hi"] * 30)):
        engine = Engine([rules], tracer=Tracer([JSONLSink(tmp_path / directory)]))
        for message in messages:
            with engine.run("t"):
                engine.decide(message, INTENT)
    (q,) = drift_report([tmp_path / "a"], [tmp_path / "b"]).questions
    assert q.uncertain_share == (0.0, 0.75)
    assert any("share not accepted rose from 0% to 75%" in flag for flag in q.flags)
