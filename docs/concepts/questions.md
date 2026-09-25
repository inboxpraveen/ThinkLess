# Questions

A question declares what the application wants to know and which answers are
allowed. Providers never receive free-form prompts from application code, only
these declarations. That is what lets the same question be answered by a
regular expression, a 200M parameter encoder or a frontier LLM without any
change to the code that reads the answer.

The four kinds follow the System One wire format that TypeSafe introduced for
Jev and that Kev, OpenJev and Laya also implement, plus `Extract`.

## Choice

Exactly one option out of a fixed set.

```python
from thinkless import Choice

INTENT = Choice(
    "What does the customer want?",
    name="intent",
    options={
        "refund_duplicate_charge": "was billed twice for one purchase and wants the extra charge back",
        "order_status": "asks where an order is, when it arrives, or about tracking",
        "other": "anything else",
    },
)
```

The decision value is the chosen label. `decision.probabilities` holds the
full distribution when the provider has one.

Options can also be a plain list of labels. Descriptions are worth writing:
every provider, from GLiNER to an LLM, uses them to tell options apart. Write
them the way you would brief a new teammate, and describe the boundary between
similar options explicitly.

## Score

A position on an ordered scale, lowest first.

```python
from thinkless import Score

URGENCY = Score("How urgent is the request?", name="urgency", levels=("low", "medium", "high"))
```

The value is the expected level index as a float (for example `1.7`), and
`decision.level` is the single most likely level.

## YesNo

A binary question. The value is a `bool`.

```python
from thinkless import YesNo

WANTS_HUMAN = YesNo(
    "Does the customer explicitly ask to talk to a human instead of an automated reply?",
    name="wants_human",
    yes_means="asks for a person, an agent, a callback or a manager",
    no_means="anything else, including frustration without a request for a person",
)
```

`yes_means` and `no_means` are optional. They help most on questions whose
boundary is subtle.

## Extract

Named fields pulled out of the input.

```python
from thinkless import Extract

ORDER = Extract(
    name="order",
    fields={"order_id": "the order number the customer refers to, four or five digits"},
    required=["order_id"],
)
```

The value is a dict of field values, with `None` for fields that were not
found. Fields listed in `required` must be present for the answer to count as
confident; a missing optional field is simply `None`. Extraction replaces one
of the most common LLM calls in agents: turning a message into tool arguments.

## Options shared by every question

| Option | Meaning |
|---|---|
| `name` | Stable identifier used in traces, per-question thresholds and rules. Derived from the instructions when omitted, but set it explicitly for anything you depend on. |
| `threshold` | Minimum normalized confidence to accept an answer. Overrides the engine default. |
| `providers` | Allowlist of provider names that may answer. Use it to keep a question away from a model that cannot answer it well. |

Questions are immutable. Define them once, at module level, and reuse them.

## Asking several questions at once

```python
decisions = engine.decide_many(ticket, [INTENT, URGENCY, WANTS_HUMAN, ORDER])
```

The questions travel through the cascade together. Each provider receives one
call with every open question it supports, which is how System One models are
designed to be used (one forward pass answers them all) and what keeps an
LLM-only baseline fair: one prompt answers the whole batch.

## Writing questions that small models answer well

- Ask about one thing. "Is the customer angry and asking for a refund?" is two
  questions.
- Name the population in the instructions: "the customer", "the message",
  "the help article".
- Prefer a `Choice` with an explicit `other` option over a set of yes/no
  questions that could all be true.
- Keep labels short and literal. The label is what your code switches on; the
  description is where the nuance goes.
- Measure, do not guess. [Calibration](../guides/calibration.md) tells you
  whether a model can answer a question and at what threshold.
