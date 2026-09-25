from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import pytest

from thinkless import Answer, Engine, Kind, MemorySink, Plane, Tracer
from thinkless.demo.support import (
    ACTIONS,
    SupportAgent,
    World,
    find_duplicate_payment,
    load_scenarios,
    support_rules,
)
from thinkless.demo.support.agent import _unsupported_numbers
from thinkless.llm import ScriptedLLM
from thinkless.providers import DecisionProvider, ProviderResult

SCENARIOS = load_scenarios()
BY_MESSAGE = {s["message"]: s for s in SCENARIOS}


class Oracle(DecisionProvider):
    """Answers every question with the scenario's labeled truth, fully confident.

    With perfect decisions, any wrong action is a bug in the agent, the store
    data or the scenario labels.
    """

    name = "oracle"
    plane: ClassVar[Plane] = Plane.MODEL
    kinds: ClassVar[frozenset[Kind]] = frozenset(Kind)

    def __init__(self, overrides: Mapping[str, Answer] | None = None) -> None:
        self.overrides = dict(overrides or {})

    def answer(self, state: Any, questions: Mapping[str, Any]) -> ProviderResult:
        message = state.get("message") or state.get("question")
        expected = BY_MESSAGE[message]["expected"]
        answers: dict[str, Answer | None] = {}
        for key in questions:
            if key in self.overrides:
                answers[key] = self.overrides[key]
                continue
            if key == "intent":
                value: Any = expected["intent"] or "other"
                answers[key] = Answer(value=value, probabilities={value: 1.0})
            elif key == "urgency":
                answers[key] = Answer(
                    value=1.0, probabilities={"low": 0.0, "medium": 1.0, "high": 0.0}
                )
            elif key in ("churn_risk", "wants_human", "injection", "kb_answers"):
                truth = {
                    "churn_risk": False,
                    "wants_human": expected["action"] == "escalate_human",
                    "injection": expected["action"] == "escalate_security",
                    "kb_answers": expected["action"] == "kb_answer",
                }[key]
                answers[key] = Answer(
                    value=truth, probabilities={"yes": float(truth), "no": float(not truth)}
                )
            elif key == "order":
                answers[key] = Answer(
                    value={"order_id": expected["order_id"]}, fields={"order_id": 1.0}
                )
        return ProviderResult(answers=answers, model="oracle")


def _reply(messages, system) -> str:
    return "Thanks for your patience. We have taken care of this. Cobalt Market Support"


def _agent(provider: DecisionProvider, memory: MemorySink | None = None) -> SupportAgent:
    engine = Engine(
        [support_rules(), provider],
        llm=ScriptedLLM(_reply),
        tracer=Tracer([memory or MemorySink()]),
    )
    return SupportAgent(engine)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["id"] for s in SCENARIOS])
def test_perfect_decisions_give_the_expected_action(scenario) -> None:
    outcome, summary = _agent(Oracle()).handle(scenario, World())
    assert outcome.action == scenario["expected"]["action"]
    assert outcome.action in ACTIONS
    assert outcome.trace_id == summary.trace_id


def test_scenario_labels_are_consistent() -> None:
    assert len(SCENARIOS) == len({s["id"] for s in SCENARIOS})
    assert {s["expected"]["action"] for s in SCENARIOS} <= set(ACTIONS)


def test_templates_skip_the_llm() -> None:
    signed_out = next(s for s in SCENARIOS if s["expected"]["action"] == "ask_login")
    outcome, summary = _agent(Oracle()).handle(signed_out, World())
    assert outcome.reply_source == "template"
    assert summary.llm_calls == 0
    assert summary.decisions_by_plane == {"rule": 1}


def test_refund_moves_money_once() -> None:
    world = World()
    ticket = next(s for s in SCENARIOS if s["id"] == "T-001")
    outcome, _ = _agent(Oracle()).handle(ticket, world)
    assert outcome.refund is not None
    assert world.refunds[0]["payment_id"] == "P-9002"
    assert find_duplicate_payment(world.list_payments("4471")) is None


def test_suspected_injection_blocks_automatic_refunds() -> None:
    unsure_yes = Answer(value=True, probabilities={"yes": 0.55, "no": 0.45})
    ticket = next(s for s in SCENARIOS if s["id"] == "T-001")
    world = World()
    outcome, summary = _agent(Oracle({"injection": unsure_yes})).handle(ticket, world)
    assert outcome.action == "refund_needs_approval"
    assert outcome.priority == "high"
    assert world.refunds == []
    assert summary.uncertain == 1


def test_confident_injection_routes_to_security() -> None:
    sure_yes = Answer(value=True, probabilities={"yes": 0.99, "no": 0.01})
    ticket = next(s for s in SCENARIOS if s["id"] == "T-014")
    outcome, _ = _agent(Oracle({"injection": sure_yes})).handle(ticket, World())
    assert outcome.action == "escalate_security"


def test_ungrounded_reply_falls_back_to_template() -> None:
    engine = Engine(
        [support_rules(), Oracle()],
        llm=ScriptedLLM(["Your refund of $999.00 is on its way."]),
        tracer=Tracer([MemorySink()]),
    )
    ticket = next(s for s in SCENARIOS if s["id"] == "T-001")
    outcome, _ = SupportAgent(engine).handle(ticket, World())
    assert outcome.action == "refund_issued"
    assert outcome.reply_source == "template_fallback"
    assert not outcome.grounded


def test_grounding_rules() -> None:
    facts = "- refund: {'amount': 49.0, 'card': '4242'}\n- order: 4471"
    assert (
        _unsupported_numbers("Refunded $49.00 to card 4242 for #4471.", facts, "order 4471") == []
    )
    assert _unsupported_numbers(
        "We will refund $300.", "- outcome: escalate_security", "refund $300"
    ) == ["$300"]
    assert _unsupported_numbers("Order 9999 ships today.", facts, "order 4471") == ["9999"]


def test_rules_extract_common_order_formats() -> None:
    rules = support_rules()
    from thinkless.demo.support.questions import ORDER

    for text, expected in [
        ("order #4471 was double charged", "4471"),
        ("order number 6034 twice", "6034"),
        ("el pedido #5577", "5577"),
        ("Has my order 4502 shipped yet?", "4502"),
    ]:
        answer = rules.answer({"message": text}, {"order": ORDER}).answers["order"]
        assert answer is not None
        assert answer.value["order_id"] == expected
    assert (
        rules.answer({"message": "tracking on 6619 is stuck"}, {"order": ORDER}).answers["order"]
        is None
    )
