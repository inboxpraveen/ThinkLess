"""Habits under an agent written the usual way, with no API key.

The agent routes support messages with the OpenAI SDK and a Pydantic schema,
exactly as it would without ThinkLess. One line puts Habits under the
client. A stand-in plays the LLM, so this runs offline in a few seconds:

    pip install thinkless openai
    python examples/10_habits.py

What happens:

1. The first 300 messages go to the "LLM", and ThinkLess records them.
2. ``learn()`` trains a local student on the LLM's answers and tests it on
   calls it was not trained on: it must agree with the LLM at least 97% of
   the time, as a 95% lower bound. Passing puts it on probation.
3. The next 450 messages still go to the LLM. The student answers each one
   in shadow, and the next ``learn()`` tests it once more on the first 394 of
   those new calls. Passing that switches the site on.
4. The last 300 messages: most are answered locally in about half a
   millisecond; the rest, and a 5% audit sample, still go to the LLM.

With a real model, drop the stand-in and keep the one line:
``client = habits.wrap(OpenAI())``.
"""

from __future__ import annotations

import importlib
import json
import random
import tempfile
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel

from thinkless import habits


class Route(BaseModel):
    intent: Literal["refund", "order_status", "cancel", "other"]
    wants_human: bool


PHRASES = {
    "refund": ["refund", "my money back", "charged twice", "reimburse me"],
    "order_status": ["where is my order", "tracking number", "has not arrived", "delivery date"],
    "cancel": ["cancel", "stop my order", "do not want it anymore"],
    "other": ["hello there", "change my password", "update my address", "opening hours"],
}


def message(rng: random.Random) -> str:
    intent = rng.choice(list(PHRASES))
    opener = rng.choice(["Hi,", "Hello team.", "", "Quick one:"])
    order = rng.choice(["", f" for order #{rng.randint(1000, 99999)}"])
    human = " and let me talk to a person" if rng.random() < 0.2 else ""
    return f"{opener} {rng.choice(PHRASES[intent])}{order}{human} please".strip()


# The stand-in LLM: answers like a model would, over HTTP.
lib: Any = importlib.import_module(habits.sdk_http_library())
llm_calls = 0


def fake_llm(request: Any) -> Any:
    global llm_calls
    llm_calls += 1
    text = json.loads(request.content)["messages"][-1]["content"]
    intent = next((k for k, ps in PHRASES.items() if any(p in text for p in ps)), "other")
    content = json.dumps({"intent": intent, "wants_human": "person" in text})
    return lib.Response(
        200,
        json={
            "id": "chatcmpl-demo",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-5-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 180, "completion_tokens": 14, "total_tokens": 194},
        },
    )


runtime = habits.Habits(tempfile.mkdtemp(prefix="habits-demo-"), learn_every=None, seed=0)
client = OpenAI(api_key="demo", http_client=lib.Client(transport=lib.MockTransport(fake_llm)))
client = habits.wrap(client, runtime)  # the one line


def route(text: str) -> tuple[Route, bool]:
    raw = client.chat.completions.with_raw_response.parse(
        model="gpt-5-mini",
        messages=[
            {
                "role": "system",
                "content": "Route the customer's message. Say whether they want a person.",
            },
            {"role": "user", "content": text},
        ],
        response_format=Route,
    )
    parsed = raw.parse().choices[0].message.parsed
    assert parsed is not None
    return parsed, raw.headers.get("x-thinkless-habit") == "local"


rng = random.Random(7)
for _ in range(300):
    route(message(rng))
print(f"First 300 messages: {llm_calls} LLM calls, all recorded.")


def report(plan: dict[str, Any]) -> None:
    for site in plan["sites"].values():
        print(f"  Site {site['name']!r}: {site['status']}. {site['reason']}.")


report(runtime.learn())

before = llm_calls
for _ in range(450):
    route(message(rng))
print(f"Next 450 messages, on probation: {llm_calls - before} LLM calls, none answered locally.")
report(runtime.learn())

before = llm_calls
local = 0
for i in range(300):
    text = message(rng)
    answer, was_local = route(text)
    local += was_local
    if i < 3:
        print(
            f"  {text!r} -> {answer.intent}, wants_human={answer.wants_human} "
            f"({'local' if was_local else 'LLM'})"
        )
print(
    f"Last 300 messages: {local} answered locally, {llm_calls - before} sent to the LLM "
    f"(including {runtime.counts['audit']} audits)."
)
runtime.close()
