from __future__ import annotations

import asyncio
import json
import threading

import pytest
from typer.testing import CliRunner

from tests.conftest import FakeProvider, choice_answer
from thinkless import Answer, Choice, Engine, Extract, MemorySink, Plane, Score, Tracer, YesNo
from thinkless.bench.metrics import wilson_interval
from thinkless.cli.shadow import shadow_app
from thinkless.shadow import Shadow, agree, build_report, export_labels, read_log

INTENT = Choice(
    "What does the customer want?", options=["refund", "status", "other"], name="intent"
)
URGENT = YesNo("Is it urgent?", name="urgent")
ORDER = Extract(name="order", fields={"order_id": "the order number"})


def _candidate(value: str = "refund", p: float = 0.97, tracer: Tracer | None = None) -> Engine:
    rest = (1 - p) / 2
    fake = FakeProvider(
        "fake",
        {
            "intent": choice_answer(
                value, {"refund": rest, "status": rest, "other": rest} | {value: p}
            )
        },
    )
    return Engine([fake], threshold=0.8, tracer=tracer or Tracer())


def _records(path) -> list[dict]:
    return list(read_log(path))


def test_agreement_normalizes_answers() -> None:
    assert agree(INTENT, "Refund", "refund") == (True, None)
    assert agree(INTENT, "refund", "status") == (False, None)
    assert agree(URGENT, "yes", True) == (True, None)
    assert agree(URGENT, None, True) == (None, None)
    score = Score("How urgent?", levels=["low", "medium", "high"], name="urgency")
    assert agree(score, 2.2, "high") == (True, None)
    ok, fields = agree(
        Extract(name="o", fields={"id": "", "city": ""}),
        {"id": " 4471 ", "city": None},
        {"id": 4471, "city": ""},
    )
    assert ok is True
    assert fields == {"id": True, "city": True}


def test_compare_returns_primary_result_and_logs_both_sides(tmp_path) -> None:
    log = tmp_path / "shadow.jsonl"
    with Shadow(_candidate("refund"), log=log, background=False) as shadow:
        result = shadow.compare("I was charged twice", INTENT, lambda: "status", cost_usd=0.002)
    assert result == "status"
    (record,) = _records(log)
    assert record["question"] == "intent"
    assert record["primary"] == {
        "source": "callable",
        "value": "status",
        "latency_ms": record["primary"]["latency_ms"],
        "cost_usd": 0.002,
    }
    assert record["shadow"]["value"] == "refund"
    assert record["shadow"]["plane"] == "model"
    assert record["shadow"]["status"] == "accepted"
    assert record["agree"] is False
    assert record["state"] == "I was charged twice"
    assert len(record["state_sha"]) == 16


def test_watch_passes_results_and_exceptions_through(tmp_path) -> None:
    log = tmp_path / "shadow.jsonl"
    shadow = Shadow(_candidate("refund"), log=log, background=False)

    @shadow.watch(INTENT, cost_usd=0.001)
    def classify(ticket: str) -> str:
        if ticket == "boom":
            raise RuntimeError("primary failed")
        return "refund"

    assert classify("charged twice") == "refund"
    with pytest.raises(RuntimeError, match="primary failed"):
        classify("boom")
    shadow.close()
    (record,) = _records(log)
    assert record["agree"] is True
    assert shadow.stats.recorded == 1


def test_watch_async_function(tmp_path) -> None:
    log = tmp_path / "shadow.jsonl"
    shadow = Shadow(_candidate("refund"), log=log)

    @shadow.watch(INTENT, state=lambda ticket, **_: ticket["text"], to_value=lambda r: r["intent"])
    async def classify(ticket: dict, *, user: str) -> dict:
        return {"intent": "refund", "user": user}

    result = asyncio.run(classify({"text": "money back please"}, user="u1"))
    shadow.close()
    assert result == {"intent": "refund", "user": "u1"}
    (record,) = _records(log)
    assert record["state"] == "money back please"
    assert record["primary"]["value"] == "refund"
    assert record["agree"] is True


def test_background_runs_stay_out_of_the_callers_trace(tmp_path) -> None:
    primary_sink, shadow_sink = MemorySink(), MemorySink()
    production = Engine([], tracer=Tracer([primary_sink]))
    shadow = Shadow(
        _candidate(tracer=Tracer([shadow_sink])), log=tmp_path / "s.jsonl", background=True
    )
    with production.run("ticket") as run:
        shadow.compare("charged twice", INTENT, lambda: "refund")
    shadow.close()
    assert all(span.trace_id == run.trace_id for span in primary_sink.spans)
    roots = [s for s in shadow_sink.spans if s.parent_id is None]
    assert len(roots) == 1
    assert roots[0].attributes["shadow"] is True
    assert roots[0].trace_id != run.trace_id


def test_inline_runs_also_start_a_separate_trace(tmp_path) -> None:
    sink = MemorySink()
    tracer = Tracer([sink])
    production = Engine([], tracer=tracer)
    shadow = Shadow(_candidate(tracer=tracer), log=tmp_path / "s.jsonl", background=False)
    with production.run("ticket") as run:
        shadow.compare("charged twice", INTENT, lambda: "refund")
    assert run.summary().decisions == 0
    shadow.close()


def test_sampling_spend_cap_and_backpressure(tmp_path) -> None:
    shadow = Shadow(_candidate(), log=tmp_path / "a.jsonl", sample=0.0, background=False)
    for _ in range(5):
        shadow.observe("x", INTENT, "refund")
    assert shadow.stats.skipped == 5
    assert shadow.stats.recorded == 0
    shadow.close()

    capped = Shadow(_candidate(), log=tmp_path / "b.jsonl", background=False, max_cost_usd=0.0)
    capped.observe("x", INTENT, "refund")
    assert capped.stats.capped == 1
    capped.close()

    gate = threading.Event()

    def slow(state, question):
        gate.wait(5)
        return choice_answer("refund", {"refund": 0.98, "status": 0.01, "other": 0.01})

    engine = Engine([FakeProvider("slow", {"intent": slow})], tracer=Tracer())
    busy = Shadow(engine, log=tmp_path / "c.jsonl", workers=1, max_pending=2)
    for _ in range(5):
        busy.observe("x", INTENT, "refund")
    assert busy.stats.dropped == 3
    gate.set()
    busy.close()
    assert busy.stats.recorded == 2


def test_candidate_errors_are_logged_not_raised(tmp_path) -> None:
    engine = Engine(
        [FakeProvider("bad", error=RuntimeError("down"))], tracer=Tracer(), on_error="raise"
    )
    log = tmp_path / "s.jsonl"
    with Shadow(engine, log=log, background=False) as shadow:
        assert shadow.compare("x", INTENT, lambda: "refund") == "refund"
    (record,) = _records(log)
    assert "down" in record["shadow"]["error"]
    assert shadow.stats.errors == 1


def test_capture_off_keeps_only_a_fingerprint(tmp_path) -> None:
    fake = FakeProvider(
        "fake",
        {"order": lambda s, q: Answer(value={"order_id": "4471"}, confidence=0.99)},
    )
    engine = Engine([fake], tracer=Tracer(capture_content=False))
    log = tmp_path / "s.jsonl"
    with Shadow(engine, log=log, background=False) as shadow:
        shadow.observe("order 4471 for jane@example.com", ORDER, {"order_id": "4471"})
    (record,) = _records(log)
    text = json.dumps(record)
    assert "jane@example.com" not in text
    assert "4471" not in text
    assert record["agree"] is True
    assert record["fields"] == {"order_id": True}


def test_engine_as_primary_audits_thinkless_against_an_llm(tmp_path) -> None:
    thinkless = _candidate("refund")
    reference = Engine(
        [
            FakeProvider(
                "llm",
                {"intent": choice_answer("refund", {"refund": 1.0, "status": 0, "other": 0})},
                plane=Plane.LLM,
            )
        ],
        tracer=Tracer(),
    )
    log = tmp_path / "audit.jsonl"
    with Shadow(reference, log=log, background=False) as audit:
        decision = audit.compare("charged twice", INTENT, thinkless)
    assert decision.value == "refund"
    (record,) = _records(log)
    assert record["primary"]["source"] == "engine"
    assert record["primary"]["plane"] == "model"
    assert record["shadow"]["plane"] == "llm"
    assert record["agree"] is True
    (q,) = build_report(read_log(log), min_calls=1).questions
    assert q.primary_by_plane["model"].calls == 1
    assert q.primary_by_plane["model"].agreement == 1.0


def _synthetic_log(path, n: int, agree_every: int, confidence_split: bool = False) -> None:
    rows = []
    for i in range(n):
        agreed = i % agree_every != 0
        conf = 0.95 if agreed or not confidence_split else 0.55
        rows.append(
            {
                "v": 1,
                "question": "intent",
                "kind": "choice",
                "spec": INTENT.spec(),
                "state": f"ticket {i}",
                "state_sha": f"{i:016x}",
                "primary": {
                    "source": "callable",
                    "value": "refund",
                    "cost_usd": 0.001,
                    "latency_ms": 800,
                },
                "shadow": {
                    "source": "engine",
                    "value": "refund" if agreed else "status",
                    "status": "accepted",
                    "plane": "model" if i % 5 else "llm",
                    "provider": "fake",
                    "confidence": conf,
                    "cost_usd": 0.0 if i % 5 else 0.0005,
                    "latency_ms": 20,
                    "attempts": [
                        {
                            "provider": "fake",
                            "plane": "model",
                            "value": "refund" if agreed else "status",
                            "confidence": conf,
                            "accepted": True,
                            "reason": "met_threshold",
                        }
                    ],
                },
                "agree": agreed,
            }
        )
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_report_agreement_savings_and_verdict(tmp_path) -> None:
    log = tmp_path / "s.jsonl"
    _synthetic_log(log, n=400, agree_every=50, confidence_split=True)
    report = build_report(read_log(log), target=0.95, min_calls=100, monthly_volume=1_000_000)
    (q,) = report.questions
    assert q.compared == 400
    assert q.agreement == pytest.approx(392 / 400)
    low, _ = wilson_interval(392, 400)
    assert q.agreement_low == pytest.approx(low)
    assert q.ready
    assert q.shadow.fast_share == pytest.approx(0.8)
    assert q.primary.cost_per_1k_usd == pytest.approx(1.0)
    assert q.shadow.cost_per_1k_usd == pytest.approx(0.1)
    assert report.savings_share == pytest.approx(0.9)
    assert report.monthly_savings_usd == pytest.approx(900.0)
    (loose,) = q.thresholds
    assert loose.recommended == 0.0

    (strict,) = build_report(read_log(log), target=0.99).questions[0].thresholds
    assert strict.recommended == pytest.approx(0.6)
    assert strict.agreement == 1.0
    assert strict.coverage == pytest.approx(392 / 400)


def test_report_needs_enough_calls(tmp_path) -> None:
    log = tmp_path / "s.jsonl"
    _synthetic_log(log, n=30, agree_every=1000)
    (q,) = build_report(read_log(log), min_calls=100).questions
    assert not q.ready
    assert q.verdict.startswith("collect more")


def test_report_without_primary_cost_projects_no_saving(tmp_path) -> None:
    log = tmp_path / "s.jsonl"
    with Shadow(_candidate(), log=log, background=False) as shadow:
        shadow.observe("x", INTENT, "refund")
    report = build_report(read_log(log))
    assert report.savings_per_1k_usd is None


def test_export_labels_feeds_calibration(tmp_path) -> None:
    log = tmp_path / "s.jsonl"
    _synthetic_log(log, n=100, agree_every=10)
    out = tmp_path / "label.jsonl"
    assert export_labels(read_log(log), out) == 10
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert {row["label"] for row in rows} == {"refund"}
    assert {row["shadow"] for row in rows} == {"status"}
    assert all(row["text"].startswith("ticket") for row in rows)


def test_cli_report_and_export(tmp_path) -> None:
    log = tmp_path / "s.jsonl"
    _synthetic_log(log, n=200, agree_every=40)
    runner = CliRunner()
    result = runner.invoke(shadow_app, ["report", str(log), "--volume", "100000"])
    assert result.exit_code == 0, result.output
    assert "intent" in result.output
    assert "per 1,000 decisions" in result.output
    as_json = runner.invoke(shadow_app, ["report", str(log), "--json"])
    assert json.loads(as_json.output)["records"] == 200
    out = tmp_path / "rows.jsonl"
    exported = runner.invoke(shadow_app, ["export", str(log), "--out", str(out)])
    assert exported.exit_code == 0, exported.output
    assert "Wrote 5 rows" in exported.output
