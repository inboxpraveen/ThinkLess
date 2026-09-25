# The four planes

Most agents send every judgment to one generative model: what the user wants,
which tool to call, whether the tool result is good enough, whether the reply
is safe, and finally the reply itself. Each of those is a full LLM call with a
prompt, a completion, a parser and a few hundred milliseconds to several
seconds of latency.

Most of those judgments are not open-ended. "Which of these seven intents is
this?" has seven possible answers. "Is the customer asking for a human?" has
two. A small model trained to answer bounded questions with calibrated
probabilities answers them in tens of milliseconds, locally, for free, and
tells you how sure it is. Deterministic code answers some of them outright.

ThinkLess organizes an agent into four planes and makes it explicit which one
handled each step.

| Plane | What runs there | Typical latency | Examples |
|---|---|---|---|
| Decision | Rules, small decision models, and an LLM as the last resort | under 1 ms to 100 ms, seconds on escalation | intent, routing, yes/no checks, field extraction |
| Reasoning | Generative models | 1 to 30 s | replies, summaries, plans |
| Execution | Your tools and APIs | depends | refunds, lookups, tickets |
| Observability | Traces, summaries, cost | none on the hot path | JSONL audit files, OpenTelemetry, the viewer |

## The decision plane

The decision plane answers typed questions: a `Choice` among options, a
`Score` on an ordered scale, a `YesNo`, or an `Extract` of named fields. Every
answer is a `Decision` with a value, a normalized confidence, the provider that
answered, and the full list of providers that were tried.

Behind one call to `engine.decide()` sits a cascade of providers, cheapest
first:

```text
question ──► rules ──► GLiNER ──► Laya ──► Jev ──► LLM
               │          │         │        │       │
            answer if   answer if confidence ≥ threshold,
            it fires    otherwise pass to the next provider
```

Application code does not know or care which provider answered. It reads the
decision and acts:

```python
intent = engine.decide(ticket, INTENT)
if intent.is_("refund_duplicate_charge"):
    ...
elif not intent.accepted:
    escalate_to_a_person()
```

`is_()` only returns true for an accepted decision, so an uncertain answer
never triggers an action by accident.

## The reasoning plane

Generation stays with an LLM, and ThinkLess does not try to replace it. What
changes is how often the LLM is called and what it is asked to do: write the
reply, from facts the rest of the agent already established, instead of also
classifying, extracting and routing along the way.

`engine.generate()` records every generation as an `llm` span with the model,
token usage, latency and estimated cost.

## The execution plane

Tools are ordinary functions. Decorate them with `@thinkless.tool` and each
call inside a traced run becomes a `tool` span with its arguments and result.
Outside a run the decorator does nothing, so tools remain easy to test.

## The observability plane

Every decision, attempt, generation, tool call and rule check is a span in
one trace per run. The trace answers the questions that matter when an agent
misbehaves or costs too much: which plane made each decision, how confident it
was, what it cost, and where the time went. See [Tracing](tracing.md).

## Why keep the LLM in the decision cascade at all

Because small models are not always sure, and when they are not, the right
move is to ask something stronger rather than guess. The cascade uses the LLM
exactly where a small model reports low confidence. On the bundled support
benchmark this keeps accuracy at or above the LLM-only design while most
decisions never reach the LLM. The [benchmarks](../benchmarks.md) show the
numbers, including where the small models fall short.

A cascade without an LLM (the `models` mode in the benchmarks) is also a valid
design when latency or data residency rules out a hosted model: uncertain
decisions then go to a person instead.
