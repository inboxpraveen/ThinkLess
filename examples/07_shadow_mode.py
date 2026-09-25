"""Shadow mode: measure what ThinkLess would save before it decides anything.

An existing classifier keeps answering. A ThinkLess engine (rules first, the
same LLM for the rest) runs next to it on every call and logs both answers.
The report then shows how often they agree and what each costs. Nothing
downloads and no API is called:

    pip install thinkless
    python examples/07_shadow_mode.py
    thinkless shadow report .thinkless/shadow/intent.jsonl --min-calls 20
    thinkless shadow export .thinkless/shadow/intent.jsonl --out to_label.jsonl

The LLM is a scripted stand-in and the prices are illustrative, so the
numbers show how the report reads, not what your traffic will do. Both sides
use the same LLM, the same price table and prompts of similar length, so the
saving comes only from the tickets the rules settle.
"""

from __future__ import annotations

import json
from pathlib import Path

from thinkless import Choice, Engine
from thinkless.llm import Completion, ScriptedLLM
from thinkless.pricing import Price, PriceTable
from thinkless.providers import LLMDecider, Rules
from thinkless.shadow import Shadow, build_report, read_log

INTENT = Choice(
    "What does the customer want?",
    name="intent",
    options={
        "refund": "wants money back",
        "order_status": "asks where an order is",
        "cancel": "wants to cancel a subscription",
        "other": "anything else",
    },
)

TICKETS = [
    "I was charged twice for order 4471, please refund one.",
    "Where is my parcel? It was due on Monday.",
    "Please cancel my subscription from next month.",
    "Can I get my money back for the broken kettle?",
    "Tracking has not updated in five days.",
    "Do you ship to Norway?",
    "I don't want a refund, I just want to know where my order is.",
    "Cancel the plan, I no longer use it.",
    "The app logs me out every few minutes.",
    "When will order 5520 arrive?",
]
LOG = Path(".thinkless/shadow/intent.jsonl")


def llm_reads(message: str) -> str:
    """What the LLM answers. A stand-in for a real model call."""
    text = message.lower()
    if "where" in text or "tracking" in text or "arrive" in text:
        return "order_status"
    if "refund" in text or "money back" in text or "charged twice" in text:
        return "refund"
    if "cancel" in text:
        return "cancel"
    return "other"


class StandIn(ScriptedLLM):
    provider = "example"  # priced below, so every call shows a realistic cost


def respond(messages, system) -> str:
    ticket = messages[-1]["content"].split("<<<\n", 1)[1].split("\n>>>", 1)[0]
    return json.dumps({"intent": llm_reads(ticket)})


llm = StandIn(respond, model="stand-in", latency_ms=40)
PRICES = PriceTable([Price(provider="example", model="stand-in", input=0.40, output=1.60)])


# ---------------------------------------------------------------- today's code


def classify_today(message: str) -> Completion:
    """The existing code: one LLM call per ticket."""
    prompt = (
        "What does the customer want? Answer with one of these labels:\n"
        "- refund: wants money back\n"
        "- order_status: asks where an order is\n"
        "- cancel: wants to cancel a subscription\n"
        "- other: anything else\n\n"
        f"Customer message:\n<<<\n{message}\n>>>\n\n"
        'Reply with JSON only: {"intent": "<label>"}'
    )
    return llm.complete([{"role": "user", "content": prompt}], max_tokens=20)


# ----------------------------------------------------- the candidate: rules first

rules = Rules()
rules.match("intent", r"\b(refund|money back|charged twice)\b", "refund")
rules.match("intent", r"\bcancel\b", "cancel")
rules.match("intent", r"\b(where is|tracking)\b", "order_status")

candidate = Engine([rules, LLMDecider(llm)], prices=PRICES)


def main() -> None:
    LOG.unlink(missing_ok=True)
    shadow = Shadow(candidate, log=LOG)
    watched = shadow.watch(
        INTENT,
        to_value=lambda completion: json.loads(completion.text)["intent"],
        cost_usd=lambda completion: PRICES.cost("example", completion.model, completion.usage)[0],
    )(classify_today)

    for _ in range(5):
        for ticket in TICKETS:
            watched(ticket)  # returns today's answer, exactly as before
    shadow.close()

    report = build_report(read_log(LOG), min_calls=20, monthly_volume=500_000)
    (intent,) = report.questions
    print(f"shadowed calls:        {intent.compared}")
    print(f"settled without LLM:   {intent.shadow.fast_share:.0%}")
    print(
        f"agreement:             {intent.agreement:.1%} "
        f"(95% CI {intent.agreement_low:.1%} to {intent.agreement_high:.1%})"
    )
    print(f"cost per 1k today:     ${intent.primary.cost_per_1k_usd:.4f}")
    print(f"cost per 1k shadow:    ${intent.shadow.cost_per_1k_usd:.4f}")
    print(f"monthly saving (500k): ${report.monthly_savings_usd:.2f}")
    print(f"verdict:               {intent.verdict}")
    print(f"\nlog: {LOG}")


if __name__ == "__main__":
    main()
