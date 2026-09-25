from __future__ import annotations

import pytest

from thinkless import Choice, Extract, Score, Status, YesNo
from thinkless.providers import Rules


def test_match_and_decorator(make_engine) -> None:
    rules = Rules()
    rules.match("wants_human", r"\b(real person|human)\b", True, field="message")

    @rules.rule("intent")
    def empty_is_other(state):
        return "other" if not state["message"].strip() else None

    engine = make_engine([rules])
    human = engine.decide(
        {"message": "Let me talk to a real person"}, YesNo("Human?", name="wants_human")
    )
    assert human.value is True
    assert human.confidence == 1.0
    assert human.provider == "rules"

    intent = Choice("Intent?", options=["refund", "other"], name="intent")
    assert engine.decide({"message": "   "}, intent).value == "other"
    assert engine.decide({"message": "refund please"}, intent).status is Status.ABSTAINED


def test_rule_can_take_the_question(make_engine) -> None:
    rules = Rules()

    @rules.rule("urgency")
    def by_keyword(state, question):
        return question.levels[-1] if "asap" in state else None

    decision = make_engine([rules]).decide(
        "need this asap", Score("Urgency?", levels=["low", "high"], name="urgency")
    )
    assert decision.level == "high"
    assert decision.value == 1.0


def test_extract_regex(make_engine) -> None:
    rules = Rules()
    rules.extract("order", order_id=r"#\s?(\d{4,6})", email=r"[\w.+-]+@[\w-]+\.[\w.]+")
    question = Extract(fields={"order_id": "order number", "email": "email"}, name="order")
    engine = make_engine([rules])

    decision = engine.decide("Order #4471, reach me at pk@example.com", question)
    assert decision.value == {"order_id": "4471", "email": "pk@example.com"}
    assert decision.fields == {"order_id": 1.0, "email": 1.0}
    assert engine.decide("no ids here", question).status is Status.ABSTAINED


def test_invalid_rule_values_raise_and_fall_through(make_engine) -> None:
    rules = Rules()
    rules.add("intent", lambda state: "not-an-option")
    decision = make_engine([rules]).decide(
        "x", Choice("Intent?", options=["a", "b"], name="intent")
    )
    assert decision.status is Status.ABSTAINED
    assert decision.attempts[0].reason == "error: ValueError"


def test_rules_only_claim_questions_they_know() -> None:
    rules = Rules()
    rules.match("known", "x", True)
    assert rules.supports(YesNo("?", name="known"))
    assert not rules.supports(YesNo("?", name="unknown"))


def test_yes_no_rule_must_return_bool(make_engine) -> None:
    rules = Rules()
    rules.add("flag", lambda state: "yes")
    with pytest.raises(ValueError):
        make_engine([rules], on_error="raise").decide("x", YesNo("?", name="flag"))
