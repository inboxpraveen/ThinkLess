# Any framework

Two framework-neutral helpers in `thinkless.integrations` cover the rest:
CrewAI, LlamaIndex, Pydantic AI, AutoGen, a hand-written tool loop, or a web
service. Both are in the core package.

## Route on a decision

`route` decides a question and returns the destination for an accepted
answer, or the default for anything else. Destinations can be anything: a
function, an agent object, a queue name.

```python
from thinkless.integrations import route

handler = route(
    engine,
    ticket_text,
    INTENT,
    {"order_status": answer_from_tracking, "refund": refund_flow},
    default=llm_agent.run,     # what you call today
)
reply = handler(ticket_text)
```

`Router` is the reusable form, with async support:

```python
from thinkless.integrations import Router

intent = Router(engine, INTENT, {"order_status": tracking_crew}, default=general_crew)
crew = intent(ticket_text)             # or: await intent.aroute(ticket_text)
```

`state=` turns whatever the router is called with into the question's input,
and `last_user_text` reads the latest user message out of LangChain
messages, OpenAI chat messages and Responses API input items:

```python
from thinkless.integrations import Router, last_user_text

intent = Router(engine, INTENT, routes, default, state=last_user_text)
intent(messages)
```

## Gate a tool call

`gate` checks a call before it runs. It wraps the function, keeps its name,
docstring and signature (so frameworks that build tool schemas from the
signature see no change), and works on plain and async functions:

```python
from thinkless import Engine, YesNo
from thinkless.integrations import gate
from thinkless.providers import Rules

REFUND_OK = YesNo("Is this refund within policy?", name="refund_ok")

policy = Rules()

@policy.rule("refund_ok")
def within_policy(state):
    return state["arguments"]["amount"] <= 100


@gate(Engine([policy]), REFUND_OK, on_block=lambda decision: "Refunds over $100 need a person.")
def refund(order_id: str, amount: float) -> str:
    """Refund an amount on an order."""
    ...
```

By default the question is asked about
`{"tool": "refund", "arguments": {"order_id": ..., "amount": ...}}`. The call
runs only when the decision is accepted and equals `allow_when` (default
`True`). `on_uncertain="allow"` lets uncertain calls through. Without
`on_block`, a blocked call raises `ToolBlockedError`, whose `decision`
attribute says which provider answered and how.

Put `gate` under the framework's own tool decorator. With Pydantic AI:

```python
@agent.tool_plain
@gate(engine, REFUND_OK, on_block=lambda d: "Refunds over $100 need a person.")
def refund(order_id: str, amount: float) -> str:
    """Refund an amount on an order."""
    ...
```

With LangChain and LangGraph, under `@tool`, as in the
[LangGraph guide](langgraph.md#gate-a-tool). Both are tested: the generated
tool schema is unchanged. Other decorators that build the schema from the
function signature, such as CrewAI's `@tool` or LlamaIndex's
`FunctionTool.from_defaults`, follow the same pattern; check the generated
schema once with your version.

## In a hand-written loop

When you own the loop, call the engine directly where the decisions are:

```python
decisions = engine.decide_many(message, [INTENT, WANTS_HUMAN, ORDER])

if decisions["wants_human"].is_(True):
    return hand_to_person(message)
if decisions["intent"].is_("order_status") and decisions["order"].accepted:
    return tracking_reply(decisions["order"].value["order_id"])
return run_llm_agent(message)     # the path you have today
```

`is_()` is false for uncertain decisions, so every branch that acts on a
decision is taken only when a provider was confident. Everything else falls
through to the existing code.
