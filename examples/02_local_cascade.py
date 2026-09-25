"""A full local cascade: rules, GLiNER, Laya, then a local LLM.

Every model runs on this machine; no API key is needed. The first run
downloads about 5 GB of weights.

    pip install "thinkless[local]"
    python examples/02_local_cascade.py
"""

from __future__ import annotations

from thinkless import Choice, Engine, Extract, MemorySink, Score, Tracer, YesNo, summarize
from thinkless.llm import TransformersLLM
from thinkless.providers import GLiNER, Laya, LLMDecider, Rules

QUESTIONS = [
    Choice(
        "What does the customer want?",
        name="intent",
        options={
            "refund": "wants money back for a charge or a purchase",
            "order_status": "asks where an order is or when it arrives",
            "cancel": "wants to cancel a subscription",
            "other": "anything else",
        },
    ),
    Score(
        "How urgent is the message?",
        name="urgency",
        levels=("low", "medium", "high"),
        threshold=0.2,
    ),
    YesNo("Is the customer threatening to leave?", name="churn_risk", threshold=0.5),
    Extract(name="order", fields={"order_id": "the order number, digits only"}),
]

MESSAGES = [
    "I was charged twice for order #4471. Refund the second payment or I'm cancelling.",
    "The tracking on my grinder, 9105, hasn't moved in a week.",
    "Please cancel my Plus membership.",
]

rules = Rules()
rules.extract("order", field="message", order_id=r"#\s?(\d{4,5})\b")

llm = TransformersLLM("Qwen/Qwen3-1.7B")
memory = MemorySink()
engine = Engine(
    [rules, GLiNER(), Laya(), LLMDecider(llm)], llm=llm, threshold=0.8, tracer=Tracer([memory])
)
engine.warmup()

for message in MESSAGES:
    with engine.run("ticket") as run:
        decisions = engine.decide_many({"message": message}, QUESTIONS)
    print(message)
    for name, d in decisions.items():
        value = d.level or d.value
        conf = "n/a" if d.confidence is None else f"{d.confidence:.2f}"
        print(
            f"  {name:11} {value!s:26} conf {conf:5} via {d.provider:6} tried {' > '.join(a.provider for a in d.attempts)}"
        )
    s = summarize(run.spans)
    print(
        f"  {s.duration_ms:.0f} ms, LLM calls {s.llm_calls}, decisions by plane {s.decisions_by_plane}\n"
    )
