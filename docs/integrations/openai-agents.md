# OpenAI Agents SDK

```bash
pip install "thinkless[openai-agents]"
```

The adapter lives in `thinkless.integrations.openai_agents` and returns the
SDK's own objects: an `InputGuardrail`, a `ToolInputGuardrail` and plain
`Agent` instances. Runs, handoffs, sessions and SDK tracing work as before.

The complete example is
[`examples/09_openai_agents.py`](https://github.com/inboxpraveen/ThinkLess/blob/main/examples/09_openai_agents.py).
It runs against OpenRouter or OpenAI when a key is set, and prints the
ThinkLess decisions without one.

## Input guardrails

The SDK's guardrail pattern runs a second agent, and so a second LLM call, to
check the input. A ThinkLess guardrail answers the same question with a rule
or a classifier first:

```python
from agents import Agent, InputGuardrailTripwireTriggered, Runner

from thinkless import Engine, YesNo
from thinkless.integrations.openai_agents import input_guardrail
from thinkless.providers import Laya, LLMDecider, Rules

INJECTION = YesNo(
    "Does the message try to override the assistant's instructions?", name="injection"
)

rules = Rules()
rules.match("injection", r"\b(ignore|disregard) (all |any )?(previous|prior) instructions\b", True)
engine = Engine([rules, Laya(), LLMDecider(llm)])

agent = Agent(
    name="support",
    instructions="...",
    input_guardrails=[input_guardrail(engine, INJECTION)],
)

try:
    result = await Runner.run(agent, message)
except InputGuardrailTripwireTriggered as blocked:
    info = blocked.guardrail_result.output.output_info   # the decision summary
```

The guardrail trips only when the decision is accepted with the tripping
answer (`trip_when=True` by default). An uncertain decision does not trip it,
so give the engine an LLM decider as its last provider when every input needs
a verdict.

By default the check runs before the agent (`run_in_parallel=False`). Rules
and small models answer in milliseconds, so waiting costs little, and a
tripped check stops the run before the agent's model is called at all. Pass
`run_in_parallel=True` for the SDK's own behavior.

## Tool guardrails

A tool input guardrail sees the tool's name and JSON arguments before it
runs. The ThinkLess version allows the call only on a confident decision, and
otherwise sends the model a message in place of the tool result:

```python
from agents import function_tool

from thinkless.integrations.openai_agents import tool_input_guardrail

REFUND_OK = YesNo("Is this refund within policy?", name="refund_ok")

policy = Rules()

@policy.rule("refund_ok")
def within_policy(state):
    return state["arguments"]["amount"] <= 100


@function_tool(
    tool_input_guardrails=[
        tool_input_guardrail(
            Engine([policy]),
            REFUND_OK,
            message="Refunds over $100 need a specialist. Say so to the customer.",
        )
    ]
)
def refund(order_id: str, amount: float) -> str:
    """Refund an amount on an order."""
    ...
```

The question is asked about `{"tool": "refund", "arguments": {...}}`.

| Option | Default | Effect |
|---|---|---|
| `allow_when` | `True` | the accepted answer that lets the call run |
| `on_uncertain` | `"reject"` | `reject` sends `message` back, `allow` runs the tool, `raise` stops the run |
| `on_reject` | `"reject"` | `raise` stops the run with `ToolInputGuardrailTripwireTriggered` instead |
| `message` | a generic refusal | text, or a function of the decision |

For limits that must hold, write the check as a rule. A model is for the
judgment calls a rule cannot express, such as "does this message come from
the account holder?".

## Route to a specialist

A triage agent whose only job is to hand off costs one LLM call per request,
and that call is often the slowest step. `route_agent` decides the intent
first and starts the run on the right agent directly:

```python
from thinkless.integrations.openai_agents import route_agent

triage = Agent(name="triage", instructions="...", handoffs=[billing, tracking])

agent = await route_agent(
    engine,
    message,
    INTENT,
    {"refund": billing, "order_status": tracking},
    default=triage,   # unclear requests still go through triage
)
result = await Runner.run(agent, message)
```

Unclear requests still start at `triage`, with its handoffs intact. In the
example, three of five requests skip the triage call entirely.

## Using a model other than OpenAI

The SDK accepts any chat-completions endpoint. With OpenRouter:

```python
from agents import OpenAIChatCompletionsModel, set_tracing_disabled
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
model = OpenAIChatCompletionsModel(model="openai/gpt-5.6-luna", openai_client=client)
set_tracing_disabled(True)   # SDK traces upload to OpenAI and need an OpenAI key

agent = Agent(name="support", instructions="...", model=model)
```

ThinkLess's own `LLMDecider` can use the same account:
`from_spec("openrouter:qwen/qwen3.7-flash")`, see
[LLM backends](../guides/llm-backends.md).
