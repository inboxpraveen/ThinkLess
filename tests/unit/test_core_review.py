"""Regressions from the 0.6.0 review of the engine, providers, server and shadow mode."""

from __future__ import annotations

import math

import pytest

from tests.conftest import FakeProvider, choice_answer
from thinkless import (
    Choice,
    Decision,
    Engine,
    MemorySink,
    Plane,
    Score,
    Status,
    Tracer,
    YesNo,
)
from thinkless.pricing import Price, PriceTable
from thinkless.providers import Rules
from thinkless.providers.systemone import SystemOne
from thinkless.providers.wire import decisions_to_wire, from_wire

PRICES = PriceTable([Price(provider="paid", model="m", input=1000.0, output=0.0)])
SURE = choice_answer("status", {"refund": 0.01, "status": 0.98, "other": 0.01})
UNSURE = choice_answer("refund", {"refund": 0.5, "status": 0.3, "other": 0.2})


class PaidProvider(FakeProvider):
    price_key = "paid"


def _paid(answers: dict) -> PaidProvider:
    # 10 input tokens at $1000 per million: $0.01 per call.
    return PaidProvider("paid", answers, plane=Plane.LLM, model="m", input_tokens=10)


def test_a_rule_fires_for_a_question_named_by_its_mapping_key() -> None:
    rules = Rules()
    rules.match("wants_human", r"\bperson\b", True, field=None)
    paid = _paid({})
    engine = Engine([rules, paid], tracer=Tracer())
    decisions = engine.decide_many(
        "let me talk to a person", {"wants_human": YesNo("Do they want a person?")}
    )
    assert decisions["wants_human"].provider == "rules"
    assert decisions["wants_human"].value is True
    assert paid.calls == []


def test_a_nested_run_keeps_the_outer_spend_cap() -> None:
    intent = Choice("Intent?", options=["refund", "status", "other"], name="intent")
    paid = _paid({"intent": SURE})
    engine = Engine([paid], prices=PRICES, tracer=Tracer())
    with engine.run("ticket", max_cost_usd=0.01):
        assert engine.decide("x", intent).provider == "paid"
        with engine.run("subtask"):
            blocked = engine.decide("x", intent)
    assert blocked.status is Status.ABSTAINED
    assert len(paid.calls) == 1


def test_a_bad_run_cap_leaves_no_sink_behind() -> None:
    engine = Engine([], tracer=Tracer())
    before = len(engine.tracer.sinks)
    with pytest.raises(ValueError), engine.run("x", max_cost_usd=-1):
        pass
    assert len(engine.tracer.sinks) == before


def test_uncertain_answers_without_probabilities_stay_off_the_wire() -> None:
    question = YesNo("Urgent?", name="urgent")
    common = {"name": "urgent", "kind": question.kind, "value": True, "threshold": 0.8}
    unsure = Decision(status=Status.UNCERTAIN, **common)
    sure = Decision(status=Status.ACCEPTED, **common)
    assert decisions_to_wire({"urgent": unsure}, {"urgent": question}) == {}
    assert decisions_to_wire({"urgent": sure}, {"urgent": question}) == {
        "urgent": {"type": "noul", "noul": 1.0}
    }


def test_one_malformed_wire_answer_spoils_only_itself() -> None:
    questions = {
        "urgent": YesNo("Urgent?", name="urgent"),
        "level": Score("How bad?", levels=["low", "high"], name="level"),
        "intent": Choice("Intent?", options=["refund", "other"], name="intent"),
        "odd": Choice("Odd?", options=["a", "b"], name="odd"),
    }
    answers = {
        "urgent": {"noul": "high"},
        "level": {"score": "x"},
        "odd": {"choice": ["a"]},
        "intent": {"choice": "refund", "probabilities": {"refund": 0.9, "other": 0.1}},
    }
    decoded = from_wire(answers, questions)
    assert decoded["urgent"] is None
    assert decoded["level"] is None
    assert decoded["odd"] is None
    assert decoded["intent"] is not None
    assert decoded["intent"].value == "refund"


def test_retry_after_that_is_not_a_wait_falls_back_to_backoff() -> None:
    for header in ("-1", "nan", "inf", "soon"):
        delay = SystemOne._backoff(0, header)
        assert math.isfinite(delay)
        assert delay >= 0.0
    assert SystemOne._backoff(0, "3") == 3.0


def test_a_score_that_is_not_a_number_is_no_answer() -> None:
    from thinkless.providers.llm import parse_reply

    score = Score("How bad?", levels=["low", "mid", "high"], name="level")
    intent = Choice("Intent?", options=["refund", "other"], name="intent")
    parsed = parse_reply('{"level": NaN, "intent": "refund"}', {"level": score, "intent": intent})
    assert parsed["level"] is None
    assert parsed["intent"] is not None


def test_the_server_reports_each_paid_call_once() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from thinkless.server.app import create_app

    # One paid call answers both questions: $0.01, not $0.02.
    paid = _paid({"intent": SURE, "level": SURE})
    engine = Engine([paid], prices=PRICES, tracer=Tracer([MemorySink()]))
    intent = Choice("Intent?", options=["refund", "status", "other"], name="intent")
    level = Choice("Level?", options=["refund", "status", "other"], name="level")
    client = TestClient(create_app(engine, [intent, level], warmup=False))
    reply = client.post("/v1/decide", json={"state": "x", "questions": ["intent", "level"]})
    assert reply.status_code == 200, reply.text
    assert reply.json()["cost_usd"] == pytest.approx(0.01)


def test_shadow_counts_paid_attempts_that_were_not_accepted(tmp_path) -> None:
    from thinkless.shadow import Shadow

    intent = Choice("Intent?", options=["refund", "status", "other"], name="intent")
    first = _paid({"intent": UNSURE})
    first.name = "first"
    second = _paid({"intent": SURE})
    engine = Engine([first, second], prices=PRICES, tracer=Tracer())
    shadow = Shadow(engine, log=tmp_path / "shadow.jsonl", sample=1.0, background=False)
    shadow.observe("x", intent, "status")
    assert shadow.stats.spent_usd == pytest.approx(0.02)
    shadow.close()


def test_shadow_report_with_nothing_compared() -> None:
    from thinkless.shadow.report import build_report

    record = {
        "v": 1,
        "question": "intent",
        "kind": "choice",
        "agree": None,
        "primary": {"value": "a"},
        "shadow": {"source": "engine", "error": "boom"},
    }
    report = build_report([record], min_calls=0)
    assert report is not None


def test_side_by_side_runs_under_one_span_count_only_their_own_spans() -> None:
    import threading

    from thinkless import Engine, Tracer, YesNo

    gate = threading.Barrier(2)

    def slow(state, question):
        gate.wait(5)  # both runs are open at once
        return None  # noqa: RET501 - the provider abstains

    engine = Engine([FakeProvider("fast", {"urgent": slow})], tracer=Tracer())
    question = YesNo("Urgent?", name="urgent")
    summaries: dict[str, object] = {}

    def work(name: str) -> None:
        with engine.run(name) as run:
            engine.decide(name, question)
        summaries[name] = run.summary()

    with engine.tracer.span("request", "outer"):
        import contextvars

        threads = [
            threading.Thread(target=contextvars.copy_context().run, args=(work, name))
            for name in ("a", "b")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
    for name in ("a", "b"):
        summary = summaries[name]
        assert summary.name == name
        assert summary.decisions == 1
