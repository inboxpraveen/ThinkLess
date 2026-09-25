"""Real inference with local weights. Opt in with ``pytest -m local``.

These download models on first run (about 2.5 GB for all three) and run on
the GPU when one is available.
"""

from __future__ import annotations

import pytest

from thinkless import Choice, Engine, Extract, Kind, MemorySink, Score, Status, Tracer, YesNo

pytestmark = pytest.mark.local

TICKET = {
    "subject": "Charged twice",
    "message": "I was charged twice for my March order #4471. Refund the second payment or I will cancel.",
}
INTENT = Choice(
    "What does the customer want?",
    options={
        "refund_duplicate_charge": "billed twice for one purchase and wants the extra charge back",
        "order_status": "asks where an order is or when it arrives",
        "cancel_subscription": "wants to end a subscription or membership",
        "other": None,
    },
    name="intent",
)
URGENCY = Score("How urgent is this?", levels=["low", "medium", "high"], name="urgency")
CHURN = YesNo("Is the customer threatening to leave?", name="churn")


@pytest.fixture(scope="module")
def laya():
    from thinkless.providers import Laya

    provider = Laya()
    provider.warmup()
    return provider


@pytest.fixture(scope="module")
def gliner():
    from thinkless.providers import GLiNER

    provider = GLiNER()
    provider.warmup()
    return provider


def test_laya_answers_all_three_kinds(laya) -> None:
    engine = Engine([laya], threshold=0.0, tracer=Tracer([MemorySink()]))
    decisions = engine.decide_many(TICKET, [INTENT, URGENCY, CHURN])
    assert decisions["intent"].value == "refund_duplicate_charge"
    assert decisions["intent"].confidence > 0.9
    assert decisions["urgency"].level in URGENCY.levels
    assert isinstance(decisions["churn"].value, bool)
    assert all(d.status is Status.ACCEPTED for d in decisions.values())
    assert decisions["intent"].usage.input_tokens > 0


def test_gliner_choice_distribution_and_extraction(gliner) -> None:
    engine = Engine([gliner], threshold=0.0, tracer=Tracer([MemorySink()]))
    intent = engine.decide(TICKET, INTENT)
    assert intent.value == "refund_duplicate_charge"
    assert set(intent.probabilities) == set(INTENT.options)
    assert sum(intent.probabilities.values()) == pytest.approx(1.0)

    order = engine.decide(TICKET, Extract(fields={"order_id": "the order number"}, name="order"))
    assert order.value["order_id"] == "4471"
    assert order.fields["order_id"] > 0.5


def test_gliner_skips_unsupported_kinds(gliner) -> None:
    assert Kind.SCORE not in gliner.kinds
    assert not gliner.supports(URGENCY)


def test_transformers_llm_generates() -> None:
    from thinkless.llm import TransformersLLM

    llm = TransformersLLM("Qwen/Qwen3-1.7B")
    completion = llm.complete(
        [{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=8
    )
    assert "ready" in completion.text.lower()
    assert completion.usage.output_tokens > 0
