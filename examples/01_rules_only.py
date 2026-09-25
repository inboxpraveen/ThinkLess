"""The ThinkLess API with no models at all.

Rules answer what they can; a scripted stand-in plays the LLM for the rest.
Nothing downloads, so this runs anywhere in a second:

    pip install thinkless
    python examples/01_rules_only.py
"""

from __future__ import annotations

from thinkless import Choice, ConsoleSink, Engine, Extract, Tracer, YesNo
from thinkless.llm import ScriptedLLM
from thinkless.providers import LLMDecider, Rules

INTENT = Choice(
    "What does the customer want?",
    name="intent",
    options={
        "refund": "wants money back",
        "order_status": "asks where an order is",
        "other": "anything else",
    },
)
WANTS_HUMAN = YesNo("Does the customer ask for a person?", name="wants_human")
ORDER = Extract(name="order", fields={"order_id": "the order number"})

rules = Rules()
rules.match("intent", r"\b(refund|money back|charged twice)\b", "refund", field="message")
rules.match("wants_human", r"\b(real person|a human|agent)\b", True, field="message")
rules.extract("order", field="message", order_id=r"#\s?(\d{4,6})")

# Stands in for an LLM: answers whatever the rules could not.
fallback = ScriptedLLM(
    ['{"intent": "order_status", "wants_human": false, "order": {"order_id": null}}'],
    model="stand-in",
)

engine = Engine(
    [rules, LLMDecider(fallback)],
    llm=fallback,
    tracer=Tracer([ConsoleSink()]),
)

for message in (
    "I was charged twice for #4471, I want my money back.",
    "Where is my parcel?",
):
    with engine.run("ticket", mode="rules-first"):
        decisions = engine.decide_many({"message": message}, [INTENT, WANTS_HUMAN, ORDER])
    for name, decision in decisions.items():
        print(f"{name:12} {decision.value!s:28} {decision.status.value:9} via {decision.provider}")
    print()
