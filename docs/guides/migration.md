# Migrating an existing agent

This guide moves the decisions of a working agent onto ThinkLess one at a
time, measuring each move before it happens and keeping the current code as
the fallback. Nothing about the agent's replies, tools or framework changes.

## What moves and what stays

| Moves to ThinkLess | Stays with your agent and its LLM |
|---|---|
| Picking a label: intent, category, route, next step | Writing replies |
| Yes or no checks: injection, off topic, asks for a person, churn risk | Planning multi-step work |
| Scores on a scale: urgency, sentiment, risk | Choosing which tool to call next in an open-ended loop |
| Pulling fields out of text: order numbers, dates, amounts | Summaries, drafts and anything generative |
| Policy checks on tool arguments | The tools themselves |

A good first candidate is a decision that runs on every request, has a fixed
set of answers, and is made by an LLM today.

## Step 1: find the decisions

Search the code for LLM calls whose output is a label, a boolean, a number or
a handful of fields. They tend to look like this:

- prompts containing "classify", "one of", "yes or no", "respond with only",
  "return JSON with";
- `response_format` or JSON mode with a small schema;
- a triage or router agent whose instructions are a list of handoffs;
- a guardrail agent that returns `is_safe` or `tripwire`;
- several small LLM calls in a row on the same message.

For each one, write down how often it runs, what it costs per call and which
code acts on the answer. That list is the migration plan.

## Step 2: write each decision as a question

A classification prompt:

```python
INTENT_PROMPT = """Classify the customer message into one of:
- refund: the customer wants money back
- order_status: the customer asks where an order is
- cancel: the customer wants to cancel a subscription
- other: anything else
Respond with JSON: {"intent": "<label>"}"""


def classify_intent(message: str) -> str:
    response = client.chat.completions.create(
        model="gpt-5.6-luna",
        messages=[{"role": "system", "content": INTENT_PROMPT}, {"role": "user", "content": message}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)["intent"]
```

becomes a `Choice` with the same options and descriptions:

```python
from thinkless import Choice

INTENT = Choice(
    "What does the customer want?",
    name="intent",
    options={
        "refund": "the customer wants money back",
        "order_status": "the customer asks where an order is",
        "cancel": "the customer wants to cancel a subscription",
        "other": "anything else",
    },
)
```

A guardrail prompt becomes a `YesNo`, a 1 to 5 rating a `Score`, and an
extraction schema an `Extract`. See [questions](../concepts/questions.md).

## Step 3: build the engine

Put what is certain in rules, what is routine in small models, and keep the
LLM for the rest:

```python
from thinkless import Engine
from thinkless.llm import from_spec
from thinkless.providers import GLiNER, LLMDecider, Rules

rules = Rules()
rules.match("intent", r"\b(unsubscribe|cancel my (plan|subscription))\b", "cancel")

llm = from_spec("openai:gpt-5.6-luna")   # the model you already use
engine = Engine([rules, GLiNER(), LLMDecider(llm)], llm=llm, threshold=0.8)
```

Rules should only cover what is unambiguous. A rule that is right 95% of the
time belongs in a model, where a confidence can send the other 5% on.

## Step 4: shadow it

Measure before switching. Wrap the current function with
[shadow mode](shadow-mode.md):

```python
from thinkless.shadow import Shadow

shadow = Shadow(engine, log="shadow/intent.jsonl", sample=0.25, max_cost_usd=20)


@shadow.watch(INTENT, cost_usd=0.00042)   # today's cost per call, from your bill
def classify_intent(message: str) -> str:
    ...  # unchanged
```

After a few thousand calls:

```bash
thinkless shadow report shadow/intent.jsonl --volume 3000000
thinkless shadow export shadow/intent.jsonl --out disagreements.jsonl
```

Read the disagreements. Fix rules that fire too eagerly, sharpen option
descriptions that overlap, and recalibrate thresholds on your traffic
(`thinkless calibrate`, see [calibration](calibration.md)). Repeat until the
verdict is `ready`.

## Step 5: switch over, with the old code as the fallback

The lowest-risk switch keeps the current function as the path for anything
the fast planes are unsure about. Leave `LLMDecider` out of the engine for
this, so "unsure" means "do what we do today":

```python
fast = Engine([rules, GLiNER()], threshold=0.8)


def classify_intent(message: str) -> str:
    decision = fast.decide(message, INTENT)
    if decision.accepted:
        return decision.value
    return classify_intent_with_llm(message)   # the old function, renamed
```

Once that has run for a while, putting `LLMDecider` last in the engine does
the same thing with one batched LLM call for all the open questions of a
turn, instead of one call per question.

The same switch in a framework:

=== "LangGraph"

    ```python
    from thinkless.integrations.langgraph import router

    intent = router(fast, INTENT, {"order_status": "tracking", "cancel": "cancellations"},
                    default="agent")   # the LLM node you have today
    builder.add_conditional_edges(START, intent, intent.destinations)
    ```

=== "OpenAI Agents SDK"

    ```python
    from thinkless.integrations.openai_agents import route_agent

    agent = await route_agent(fast, message, INTENT,
                              {"order_status": tracking, "cancel": billing},
                              default=triage)   # the triage agent you have today
    result = await Runner.run(agent, message)
    ```

=== "Plain Python"

    ```python
    from thinkless.integrations import route

    handler = route(fast, message, INTENT, {"order_status": tracking_reply}, default=llm_agent)
    reply = handler(message)
    ```

To roll back, send everything to the default: remove the routes, or put the
switch behind the feature flag you already use.

## Step 6: keep measuring

- **Audit.** Keep a small shadow running the other way, the live engine as
  the primary and an LLM-only engine as the shadow, at one or two percent of
  traffic (see [audit a live engine](shadow-mode.md#audit-a-live-engine)).
- **Watch the escalation rate.** Traces record which plane settled each
  decision. A rising share reaching the LLM means traffic has changed.
- **Recalibrate** when either moves, on freshly labeled examples from the
  audit log.

## Checklist

- [ ] Each migrated decision is a named question with option descriptions.
- [ ] Rules cover only unambiguous cases.
- [ ] The shadow verdict was `ready` on at least a few hundred compared calls.
- [ ] Uncertain decisions reach the existing code path.
- [ ] Consequential actions (refunds, account changes) are also checked in
      code, not only by a model.
- [ ] An audit shadow runs in production, with a spend cap.
- [ ] Traces go somewhere you look at, with `capture_content=False` where the
      data requires it.
