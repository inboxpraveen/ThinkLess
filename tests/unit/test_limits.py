from __future__ import annotations

import time

import pytest

from tests.conftest import FakeProvider, choice_answer
from thinkless import (
    Choice,
    ConfigurationError,
    Engine,
    MemorySink,
    Plane,
    SpendLimit,
    SpendLimitError,
    Status,
    Tracer,
    summarize,
)
from thinkless.llm import ScriptedLLM
from thinkless.pricing import Price, PriceTable

INTENT = Choice(
    "What does the customer want?", options=["refund", "status", "other"], name="intent"
)
UNSURE = choice_answer("refund", {"refund": 0.5, "status": 0.3, "other": 0.2})
SURE = choice_answer("status", {"refund": 0.01, "status": 0.98, "other": 0.01})
PRICES = PriceTable([Price(provider="paid", model="m", input=1000.0, output=0.0)])


class PaidProvider(FakeProvider):
    price_key = "paid"


def _paid(answer=SURE) -> PaidProvider:
    return PaidProvider("paid", {"intent": answer}, plane=Plane.LLM, model="m", input_tokens=10)


def test_spend_limit_window_and_lifetime() -> None:
    lifetime = SpendLimit(1.0)
    lifetime.add(0.4)
    lifetime.add(0.7)
    assert lifetime.exhausted
    assert lifetime.remaining_usd == 0.0

    windowed = SpendLimit(1.0, window_s=0.05)
    windowed.add(1.0)
    assert windowed.exhausted
    time.sleep(0.08)
    assert not windowed.exhausted
    assert windowed.spent_usd == pytest.approx(0.0)
    with pytest.raises(ValueError):
        SpendLimit(-1)


def test_engine_spend_limit_skips_paid_providers() -> None:
    # 10 input tokens at $1000 per million: $0.01 per call.
    limit = SpendLimit(0.015)
    fast = FakeProvider("fast", {"intent": UNSURE})
    paid = _paid()
    engine = Engine([fast, paid], prices=PRICES, spend_limit=limit, tracer=Tracer())

    first = engine.decide("x", INTENT)
    second = engine.decide("x", INTENT)
    assert first.provider == "paid"
    assert second.provider == "paid"
    assert limit.exhausted

    third = engine.decide("x", INTENT)
    assert third.status is Status.UNCERTAIN
    assert third.provider == "fast"
    assert [a.reason for a in third.attempts] == ["below_threshold", "spend_limit"]
    assert len(paid.calls) == 2


def test_run_cap_applies_inside_the_run_only() -> None:
    sink = MemorySink()
    paid = _paid()
    engine = Engine([paid], prices=PRICES, tracer=Tracer([sink]))
    with engine.run("ticket", max_cost_usd=0.01) as run:
        assert engine.decide("x", INTENT).provider == "paid"
        blocked = engine.decide("x", INTENT)
    assert blocked.status is Status.ABSTAINED
    assert summarize(run.spans).llm_calls == 1
    assert engine.decide("x", INTENT).provider == "paid"


def test_generate_refuses_past_the_limit() -> None:
    class PaidLLM(ScriptedLLM):
        provider = "paid"

    llm = PaidLLM(["hello"], model="m")
    engine = Engine([], llm=llm, prices=PRICES, spend_limit=SpendLimit(0.0))
    with pytest.raises(SpendLimitError):
        engine.generate("hi")

    free = Engine([], llm=ScriptedLLM(["hello"]), spend_limit=SpendLimit(0.0))
    assert free.generate("hi").text == "hello"


def test_deadline_skips_the_rest_of_the_cascade() -> None:
    def slow(state, question):
        time.sleep(0.05)
        return UNSURE

    first = FakeProvider("slow", {"intent": slow})
    second = FakeProvider("second", {"intent": SURE})
    engine = Engine([first, second], deadline_ms=20, tracer=Tracer())
    decision = engine.decide("x", INTENT)
    assert decision.status is Status.UNCERTAIN
    assert [a.reason for a in decision.attempts] == ["below_threshold", "deadline"]
    assert second.calls == []

    relaxed = engine.decide("x", INTENT, deadline_ms=5000)
    assert relaxed.provider == "second"
    with pytest.raises(ConfigurationError):
        Engine([], deadline_ms=0)
