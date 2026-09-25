# LangGraph

```bash
pip install "thinkless[langgraph]"
```

The adapter lives in `thinkless.integrations.langgraph`. It adds no node
types of its own: a router is a plain function for `add_conditional_edges`,
and a decision node is a plain node function. Graphs, checkpointers,
streaming and LangSmith tracing keep working as before.

The complete example is
[`examples/08_langgraph_router.py`](https://github.com/inboxpraveen/ThinkLess/blob/main/examples/08_langgraph_router.py).
It runs with no model downloads and no API key.

## Route on a decision

A common graph starts with an LLM node that reads the message and decides
where to go. Replace the decision, keep the node:

```python
from langgraph.graph import START, MessagesState, StateGraph

from thinkless import Choice, Engine
from thinkless.integrations.langgraph import router
from thinkless.providers import GLiNER, LLMDecider, Rules

INTENT = Choice(
    "What does the customer want?",
    name="intent",
    options={
        "refund": "wants money back",
        "order_status": "asks where an order is",
        "other": "anything else",
    },
)

engine = Engine([Rules(), GLiNER(), LLMDecider(llm)], llm=llm)

intent = router(
    engine,
    INTENT,
    {"order_status": "tracking", "refund": "refunds"},
    default="agent",  # your existing LLM agent node
)

builder = StateGraph(MessagesState)
builder.add_node("tracking", tracking)
builder.add_node("refunds", refunds)
builder.add_node("agent", agent)
builder.add_conditional_edges(START, intent, intent.destinations)
```

The router reads the latest user message from `state["messages"]`, decides
the intent, and returns the node for an accepted answer. Uncertain answers,
abstentions and answers without a route go to `default`. Passing
`intent.destinations` as the path map lets `graph.get_graph().draw_mermaid()`
show every edge.

For inputs other than the last message, pass `state=`:

```python
router(engine, INTENT, routes, default="agent",
       state=lambda s: {"message": s["messages"][-1].content, "plan": s["plan"]})
```

## Answer several questions in one node

Most turns need more than one decision: the intent, an order number, whether
the customer asks for a person. `decision_node` answers them as one batch
(one call per provider) and writes plain dictionaries into the state:

```python
from thinkless import Extract, YesNo
from thinkless.integrations.langgraph import decision_node

ORDER = Extract(name="order", fields={"order_id": "the order number"})
WANTS_HUMAN = YesNo("Does the customer ask for a person?", name="wants_human")


class State(MessagesState):
    decisions: dict


builder = StateGraph(State)
builder.add_node("triage", decision_node(engine, [INTENT, ORDER, WANTS_HUMAN]))
builder.add_edge(START, "triage")
builder.add_conditional_edges("triage", intent, intent.destinations)
```

Each entry looks like this, so it survives any checkpointer:

```python
state["decisions"]["intent"]
# {"value": "refund", "status": "accepted", "accepted": True,
#  "confidence": 0.97, "plane": "model", "provider": "gliner", "level": None}
```

A router placed after the decision node reads the stored answer instead of
asking again, so the batch is the only decision call in the turn. Pass
`key=None` to the router to always ask.

Nodes downstream read the values directly, and should check `accepted`
before acting on one:

```python
def tracking(state: State) -> dict:
    order = state["decisions"]["order"]
    if not order["accepted"]:
        return {"messages": [AIMessage("Which order do you mean?")]}
    ...
```

## Gate a tool

Put `gate` under LangChain's `@tool`. The tool keeps its name, docstring and
argument schema, so `ToolNode` and `bind_tools` see no difference:

```python
from langchain_core.tools import tool

from thinkless import YesNo
from thinkless.integrations import gate

REFUND_OK = YesNo("Is this refund within policy?", name="refund_ok")

rules = Rules()

@rules.rule("refund_ok")
def within_policy(state):
    return state["arguments"]["amount"] <= 100


@tool
@gate(Engine([rules]), REFUND_OK, on_block=lambda d: "Refunds over $100 need a person.")
def refund(order_id: str, amount: float) -> str:
    """Refund an amount on an order."""
    return payments.refund(order_id, amount)
```

The question is asked about `{"tool": "refund", "arguments": {...}}`. A
blocked call returns the `on_block` text as the tool result, so the model can
tell the user what happened. Without `on_block`, `ToolBlockedError` is raised.
For policy checks, prefer rules: a limit that must hold should be code.

## Async graphs

`router` and `decision_node` call the engine synchronously. In a graph run
with `ainvoke` or `astream`, use the async forms, which run the engine on a
worker thread so model inference never blocks the event loop:

```python
from thinkless.integrations.langgraph import adecision_node

builder.add_node("triage", adecision_node(engine, [INTENT, ORDER, WANTS_HUMAN]))


async def route_intent(state):
    return await intent.aroute(state)


builder.add_conditional_edges("triage", route_intent, intent.destinations)
```

## Tracing

ThinkLess spans go to the engine's tracer, LangGraph's to LangSmith. To see a
whole turn in one ThinkLess trace, invoke the graph inside a run:

```python
with engine.run("ticket", user=user_id) as run:
    result = graph.invoke({"messages": [HumanMessage(text)]})
print(run.summary().llm_calls, run.summary().cost_usd)
```

Every decision, `engine.generate` call and function decorated with
`thinkless.tool` inside the block becomes part of that trace.
