# The engine and the cascade

`Engine` owns the decision cascade, the reasoning model and the tracer.

```python
from thinkless import Engine, JSONLSink, Tracer
from thinkless.llm import TransformersLLM
from thinkless.providers import GLiNER, Laya, LLMDecider, Rules

llm = TransformersLLM("Qwen/Qwen3-1.7B")
engine = Engine(
    providers=[rules, GLiNER(), Laya(), LLMDecider(llm)],   # cheapest first
    llm=llm,                                                # for generate()
    threshold=0.8,
    thresholds={"urgency": 0.2},
    tracer=Tracer([JSONLSink(".thinkless/traces")]),
)
```

## How a batch moves through the cascade

For `engine.decide_many(state, questions)`:

1. Every question starts open.
2. For each provider, in order:
    - Select the open questions the provider supports (by kind, by its own
      `supports()` logic, and by each question's `providers` allowlist).
      Skip the provider if there are none.
    - Call it once with all of them.
    - For each answer, compute the normalized confidence and compare it with
      the question's threshold. Accepted answers close their question. The
      rest stay open, and the best one seen so far is remembered.
3. Questions still open after the last provider come back `uncertain` with
   their best answer, or `abstained` if no provider answered.

Each provider call is an `attempt` span inside the batch's `decide` span, so
the trace shows exactly which questions each provider saw and which it
settled.

## Statuses

| Status | Meaning | What `decision.value` holds |
|---|---|---|
| `accepted` | An answer met its threshold, or came from an uncalibrated provider the engine trusts | The answer |
| `uncertain` | Every provider answered below threshold | The most confident answer seen |
| `abstained` | No provider produced an answer | `None` |

`decision.accepted` and `decision.is_(value)` make routing code safe by
default: an uncertain answer never matches.

## Escalations

`decision.escalated` is true when an earlier provider answered below threshold
(or raised an error) and the next provider was asked. A provider that abstains,
such as a rule that does not match, passes the question on without counting as
an escalation: that is the cascade working as intended, at no cost.

## Which providers answer what

| Provider | Choice | Score | YesNo | Extract | Plane | Calibrated |
|---|:---:|:---:|:---:|:---:|---|:---:|
| `Rules` | yes | yes | yes | yes | rule | yes (confidence 1) |
| `GLiNER` | yes | | | yes | model | yes |
| `Laya` | yes | yes | yes | | model | yes |
| `SystemOne` (Jev, Kev, OpenJev) | yes | yes | yes | | model | yes |
| `LLMDecider` | yes | yes | yes | yes | llm | no |

A rules provider only claims the questions it has rules for.

## Errors

A provider that raises is recorded as a failed attempt (the span carries the
error) and the cascade continues with the next provider. Set
`Engine(on_error="raise")` to propagate instead, which is useful in tests.

## Restricting a question to specific providers

```python
APPROVE = YesNo("Should this refund be approved?", name="approve", providers=("rules", "jev"))
```

Use an allowlist when calibration shows a provider cannot answer a question
well. It is cheaper than letting the provider try and escalate every time.

## Latency accounting

`decision.latency_ms` is the time spent on that question across all of its
attempts. When several questions share a provider call, each of them is
charged the full call time, since each waited for it. Run-level numbers in
`TraceSummary` add up span durations and do not double count.

## Async

`adecide`, `adecide_many` and `agenerate` run the synchronous methods in a
worker thread with `asyncio.to_thread`, so an async web service can await
them without blocking its event loop. Tracing context carries across the
thread boundary.
