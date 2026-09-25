"""A LangGraph support graph where decisions pick the path and gate the tools.

The graph you would build anyway, with two changes: a ThinkLess node answers
the triage questions, and a ThinkLess router picks the next node. Order
lookups and plain refunds never reach the LLM; everything uncertain goes to
the LLM agent node exactly as before. A policy gate checks each refund before
it runs. Nothing downloads and no API is called:

    pip install "thinkless[langgraph]"
    python examples/08_langgraph_router.py
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from thinkless import Choice, Engine, Extract, YesNo, summarize
from thinkless.integrations import gate
from thinkless.integrations.langgraph import decision_node, router
from thinkless.llm import ScriptedLLM
from thinkless.providers import LLMDecider, Rules
from thinkless.tracing import MemorySink, Tracer

INTENT = Choice(
    "What does the customer want?",
    name="intent",
    options={
        "refund": "wants money back",
        "order_status": "asks where an order is",
        "other": "anything else",
    },
)
ORDER = Extract(name="order", fields={"order_id": "the order number"})
REFUND_OK = YesNo("Is this refund within the automatic refund policy?", name="refund_ok")

rules = Rules()
rules.match("intent", r"\b(refund|money back|charged twice)\b", "refund")
rules.match("intent", r"\b(where is|tracking|arrive)\b", "order_status")
rules.extract("order", order_id=r"#?\b(\d{4,6})\b")


@rules.rule("refund_ok")
def within_policy(state: dict[str, Any]) -> bool:
    return float(state["arguments"]["amount"]) <= 100


# The LLM: a scripted stand-in for the model your agent node already calls.
llm = ScriptedLLM(
    lambda messages, system: (
        '{"intent": "other", "order": {"order_id": null}}'
        if "Reply with JSON" in messages[-1]["content"]
        else "Happy to help. Could you tell me a little more?"
    ),
    model="stand-in",
)
sink = MemorySink()
engine = Engine([rules, LLMDecider(llm)], llm=llm, tracer=Tracer([sink]))

ORDERS = {"4471": {"status": "shipped", "eta": "Thursday", "paid": 42.0}}


@gate(engine, REFUND_OK, on_block=lambda d: "Refunds over $100 go to a specialist.")
def refund(order_id: str, amount: float) -> str:
    return f"Refunded ${amount:.2f} on order {order_id}."


class State(MessagesState):
    decisions: dict


def tracking(state: State) -> dict:
    order_id = state["decisions"]["order"]["value"]["order_id"]
    order = ORDERS.get(order_id or "")
    if not order:
        return {"messages": [AIMessage("I could not find that order number.")]}
    return {"messages": [AIMessage(f"Order {order_id} is {order['status']}, due {order['eta']}.")]}


def refunds(state: State) -> dict:
    order_id = state["decisions"]["order"]["value"]["order_id"]
    order = ORDERS.get(order_id or "")
    if not order:
        return {"messages": [AIMessage("Which order should I refund?")]}
    asked = re.search(r"\$(\d+(?:\.\d+)?)", state["messages"][-1].content)
    amount = float(asked.group(1)) if asked else order["paid"]
    return {"messages": [AIMessage(refund(order_id, amount))]}


def agent(state: State) -> dict:
    """The LLM agent node you have today, unchanged."""
    reply = engine.generate(state["messages"][-1].content, name="agent")
    return {"messages": [AIMessage(reply.text)]}


intent = router(engine, INTENT, {"order_status": "tracking", "refund": "refunds"}, default="agent")

builder = StateGraph(State)
builder.add_node("triage", decision_node(engine, [INTENT, ORDER]))
builder.add_node("tracking", tracking)
builder.add_node("refunds", refunds)
builder.add_node("agent", agent)
builder.add_edge(START, "triage")
builder.add_conditional_edges("triage", intent, intent.destinations)
for node in ("tracking", "refunds", "agent"):
    builder.add_edge(node, END)
graph = builder.compile()


def main() -> None:
    for message in (
        "Where is order #4471?",
        "I was charged twice for #4471, I want my money back.",
        "Please refund $250 on order 4471.",
        "Can you recommend a gift for my dad?",
    ):
        with engine.run("ticket") as run:
            result = graph.invoke({"messages": [HumanMessage(message)]})
        summary = summarize(run.spans)
        path = result["decisions"]["intent"]
        print(f"> {message}")
        print(f"  intent {path['value']} ({path['status']}, {path['plane']})")
        print(f"  reply  {result['messages'][-1].content}")
        print(f"  LLM calls {summary.llm_calls}\n")


if __name__ == "__main__":
    main()
