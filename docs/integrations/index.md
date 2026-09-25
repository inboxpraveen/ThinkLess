# Use ThinkLess in your agent

ThinkLess does not replace your agent framework. It sits inside it, at the
points where the agent makes a decision, and answers those decisions with
rules and small models when they are confident and with your LLM when they
are not.

## Where it gets more powerful

A typical agent spends most of its LLM calls on judgments that are not
reasoning at all:

| In your agent today | What ThinkLess does there | Adapter |
|---|---|---|
| A triage agent or router prompt that picks the next step | Decides the intent in the router, sends routine requests straight to the right node or agent, and sends the unclear ones to the triage LLM you already have | [LangGraph `router`](langgraph.md#route-on-a-decision), [`route_agent`](openai-agents.md#route-to-a-specialist) |
| A guardrail that asks a second LLM "is this a prompt injection / off topic / abusive?" | Answers with a rule or a classifier in milliseconds, before the agent's own model call | [`input_guardrail`](openai-agents.md#input-guardrails) |
| A tool that must only run inside policy (refund limits, account ownership, allowed regions) | Checks the arguments with a deterministic rule or a calibrated model before the tool runs, and tells the model why when it refuses | [`gate`](any-framework.md#gate-a-tool-call), [`tool_input_guardrail`](openai-agents.md#tool-guardrails) |
| An extraction call that pulls an order number or a date out of the message | Extracts with a regex or GLiNER and asks the LLM only for what they miss | [`decision_node`](langgraph.md#answer-several-questions-in-one-node) |
| Several small LLM calls per turn, one per check | One batched call to each provider for all the open questions | every adapter |

Each row removes an LLM call from the common path without removing the LLM:
whatever the fast planes are unsure about goes to the same model as before.
On the [support benchmark](../benchmarks.md) this design matched or beat the
LLM-only agent on every model tested, at 40 to 44 percent lower billed cost.

What it does not do: write replies, plan multi-step work, or call tools on
its own. Those stay with your agent and its LLM.

## The pattern

```text
            user message
                 │
        ┌────────▼────────┐
        │  ThinkLess      │  rules → small models → LLM decider
        │  decisions      │  (only the unsure ones go further)
        └────────┬────────┘
     accepted    │    uncertain
   ┌─────────────┴──────────────┐
   ▼                            ▼
 the specialist node,       the LLM path you
 tool or agent              run today, unchanged
```

The uncertain branch is your existing code. That is what makes adoption safe:
a decision ThinkLess is not sure about behaves exactly as the agent did
before.

## Pick your framework

- [LangGraph](langgraph.md): a conditional-edge router, a decision node that
  writes typed results into graph state, and gated tools.
- [OpenAI Agents SDK](openai-agents.md): input guardrails, tool input
  guardrails and routing to a specialist agent.
- [Any framework](any-framework.md): `route` and `gate` for CrewAI,
  LlamaIndex, Pydantic AI, a hand-written loop, or a web service.

## Adopting it without risk

1. **Shadow first.** Run ThinkLess next to the current code with
   [shadow mode](../guides/shadow-mode.md). Nothing changes for users, and the
   report tells you, per decision, how often ThinkLess agrees with what you do
   today and what it would save.
2. **Switch one decision at a time.** Move a decision over once its shadow
   verdict is ready, with the current path as the router's default.
3. **Keep auditing.** Sample a few percent of production decisions against
   an LLM with the same shadow tools, so drift shows up in a report and not in
   a customer complaint.

The [migration guide](../guides/migration.md) walks through this on real code,
and the [FAQ](../faq.md) answers the common questions.
