"""``thinkless habits demo``: watch Habits take over a decision, offline.

A support agent routes customer messages with a JSON schema through an
OpenAI-compatible endpoint. A stand-in plays the LLM, so the demo needs no
API key, no model download and no extra packages, and runs in seconds. It
goes through the three stages a real site goes through: recording, probation
in shadow, and answering locally with audits.
"""

from __future__ import annotations

import json
import random
import shutil
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console

from .runtime import Habits
from .transport import transport_classes

__all__ = ["run_demo"]

PHRASES = {
    "refund": ["refund", "my money back", "charged twice", "reimburse me"],
    "order_status": ["where is my order", "tracking number", "has not arrived", "delivery date"],
    "cancel": ["cancel", "stop my order", "do not want it anymore"],
    "other": ["hello there", "change my password", "update my address", "opening hours"],
}
SYSTEM = "Route the customer's message. Say whether they want a person."
ROUTE = {
    "type": "json_schema",
    "json_schema": {
        "name": "Route",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": list(PHRASES)},
                "wants_human": {"type": "boolean"},
            },
            "required": ["intent", "wants_human"],
            "additionalProperties": False,
        },
    },
}
DOCS = "https://inboxpraveen.github.io/ThinkLess/habits/"


def _message(rng: random.Random) -> str:
    intent = rng.choice(list(PHRASES))
    opener = rng.choice(["Hi,", "Hello team.", "", "Quick one:"])
    order = rng.choice(["", f" for order #{rng.randint(1000, 99999)}"])
    human = " and let me talk to a person" if rng.random() < 0.2 else ""
    return f"{opener} {rng.choice(PHRASES[intent])}{order}{human} please".strip()


def _teacher(text: str) -> dict[str, Any]:
    """What the stand-in LLM answers."""
    intent = next((k for k, ps in PHRASES.items() if any(p in text for p in ps)), "other")
    return {"intent": intent, "wants_human": "person" in text}


class _StandIn:
    """Plays the LLM over HTTP, in the Chat Completions format."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        text = json.loads(request.content)["messages"][-1]["content"]
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-demo",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-5-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": json.dumps(_teacher(text))},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 180, "completion_tokens": 14, "total_tokens": 194},
            },
        )


def run_demo(
    console: Console | None = None,
    *,
    directory: str | Path | None = None,
    stages: tuple[int, int, int] = (300, 450, 300),
    seed: int = 7,
) -> dict[str, Any]:
    """Run the demo and print what happens. Returns the numbers it printed.

    Args:
        console: Where to print. Defaults to standard output.
        directory: Keep the Habits directory here, to look at it afterwards
            with ``thinkless habits status --dir``. Defaults to a temporary one.
        stages: Messages sent while recording, on probation, and once live.
        seed: Seed for the generated messages.
    """
    out = console or Console()
    root = Path(directory) if directory is not None else Path(tempfile.mkdtemp(prefix="habits-"))
    llm = _StandIn()
    habits = Habits(root, learn_every=None, seed=0)
    sync, _ = transport_classes("httpx")
    client = httpx.Client(
        base_url="https://llm.example/v1",
        transport=sync(habits, inner=httpx.MockTransport(llm)),
    )
    rng = random.Random(seed)

    def ask(text: str) -> tuple[dict[str, Any], bool, float]:
        started = time.perf_counter()
        response = client.post(
            "/chat/completions",
            json={
                "model": "gpt-5-mini",
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": text},
                ],
                "response_format": ROUTE,
            },
        )
        elapsed = (time.perf_counter() - started) * 1000
        answer = json.loads(response.json()["choices"][0]["message"]["content"])
        return answer, response.headers.get("x-thinkless-habit") == "local", elapsed

    def site_line(plan: dict[str, Any]) -> str:
        site = next(iter(plan["sites"].values()), None)
        if site is None:
            return "no decision site found yet"
        return (
            f"site [bold]{site['name']}[/bold] is [bold]{site['status']}[/bold]: {site['reason']}"
        )

    out.print(
        "\n[bold]ThinkLess Habits demo[/bold]\n"
        "A support agent routes each customer message with a JSON schema "
        "(intent, and whether they want a person). A stand-in plays the LLM, so "
        "this runs offline.\n"
    )

    recording, probation, live = stages
    for _ in range(recording):
        ask(_message(rng))
    out.print(f"1. [bold]Recording.[/bold] {recording} messages went to the LLM and were recorded.")
    plan = habits.learn()
    out.print(f"   ThinkLess trained a student on them; {site_line(plan)}.\n")

    before = llm.calls
    for _ in range(probation):
        ask(_message(rng))
    out.print(
        f"2. [bold]Probation.[/bold] {probation} more messages, "
        f"{llm.calls - before} sent to the LLM: the student answered each in shadow only."
    )
    plan = habits.learn()
    out.print(f"   After that check on new calls, {site_line(plan)}.\n")

    before = llm.calls
    audits_before = habits.counts["audit"]
    local_ms: list[float] = []
    matched = 0
    examples: list[str] = []
    for _ in range(live):
        text = _message(rng)
        answer, was_local, elapsed = ask(text)
        if was_local:
            local_ms.append(elapsed)
            matched += answer == _teacher(text)
        if len(examples) < 3:
            where = "local" if was_local else "LLM"
            examples.append(f"   {text!r} -> {answer['intent']} ({where})")
    sent = llm.calls - before
    audits = habits.counts["audit"] - audits_before
    local = len(local_ms)
    median = statistics.median(local_ms) if local_ms else None
    timing = f" (median {median:.2f} ms for the whole HTTP call)" if median is not None else ""
    rest = (
        f"the other {sent} went to the LLM as random audits, so agreement keeps being measured"
        if sent == audits
        else f"{sent} went to the LLM, {audits} of them random audits"
    )
    out.print(
        f"3. [bold]Live.[/bold] {live} messages: "
        f"[bold green]{local} answered locally[/bold green]{timing}; {rest}."
    )
    out.print("\n".join(examples))
    if local:
        out.print(
            f"   Local answers that match what the LLM would have said: {matched} of {local}."
        )
    habits.close()
    client.close()

    out.print(
        "\nWith a real model the change to your code is one line:\n\n"
        "    from openai import OpenAI\n"
        "    from thinkless import habits\n\n"
        "    client = habits.wrap(OpenAI())\n\n"
        "Already have LLM logs? [bold]thinkless habits scan logs/[/bold] shows which calls "
        f"are decisions, without sending anything anywhere.\nDocs: {DOCS}"
    )
    if directory is not None:
        out.print(f"The demo's calls and plan are in {root}: thinkless habits status --dir {root}")
    else:
        shutil.rmtree(root, ignore_errors=True)
    site: dict[str, Any] = next(iter(plan["sites"].values()), {})
    return {
        "directory": str(root),
        "status": site.get("status"),
        "llm_calls": llm.calls,
        "live": live,
        "local": local,
        "sent": sent,
        "audits": audits,
        "matched": matched,
        "median_local_ms": median,
    }
