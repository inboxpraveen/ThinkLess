"""Local models first, a hosted model when they are unsure.

Uses TypeSafe Jev in the cascade when TYPESAFE_API_KEY is set, and Claude as
the final decider and the reply writer.

    pip install "thinkless[gliner,anthropic]"
    export ANTHROPIC_API_KEY=...        # or run `ant auth login`
    export TYPESAFE_API_KEY=...         # optional
    python examples/03_hosted_fallback.py
"""

from __future__ import annotations

import os

from thinkless import Choice, Engine, JSONLSink, Tracer
from thinkless.llm import AnthropicLLM
from thinkless.providers import GLiNER, LLMDecider, SystemOne

INTENT = Choice(
    "What is the user asking the travel assistant to do?",
    name="intent",
    options={
        "book_flight": "find or book a flight",
        "change_booking": "change or cancel an existing booking",
        "baggage": "questions about luggage allowance or lost bags",
        "visa": "entry requirements, visas or passports",
        "other": "anything else",
    },
)

# Claude Opus 5 at low effort. For high volume, AnthropicLLM("claude-haiku-4-5")
# is the cheaper option; Haiku does not accept effort, so leave it unset there.
claude = AnthropicLLM(effort="low")
cascade = [GLiNER()]
if os.environ.get("TYPESAFE_API_KEY"):
    cascade.append(SystemOne.jev())
cascade.append(LLMDecider(claude))

engine = Engine(
    cascade, llm=claude, threshold=0.85, tracer=Tracer([JSONLSink(".thinkless/traces/examples")])
)

message = "My suitcase didn't come off the belt in Lisbon, flight TP1351. Who do I talk to?"
with engine.run("travel_request", input=message) as run:
    intent = engine.decide(message, INTENT)
    reply = engine.generate(
        f"The traveler wrote: {message}\nTheir request is about: {intent.value}.\n"
        "Reply in two sentences with the next step they should take.",
        max_tokens=200,
    )

print(
    f"intent {intent.value} via {intent.provider} (tried {[a.provider for a in intent.attempts]})"
)
print(reply.text)
summary = run.summary()
print(f"LLM calls {summary.llm_calls}, cost ${summary.cost_usd:.5f}, {summary.duration_ms:.0f} ms")
