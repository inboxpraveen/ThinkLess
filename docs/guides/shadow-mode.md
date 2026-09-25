# Shadow mode

Shadow mode answers the question every team asks before adopting ThinkLess:
*on our traffic, how often would it agree with what we do today, and what
would it save?* It runs a ThinkLess engine next to your existing code on the
same inputs, records both answers, and changes nothing your code returns.

```text
request ──► your existing code ──► answer returned to the user (unchanged)
                  │
                  └──► ThinkLess engine (background thread) ──► shadow log
```

After a few days of traffic, `thinkless shadow report` reads the log and
states, per decision:

- how many calls were compared,
- the agreement rate, with a 95% confidence interval,
- the share ThinkLess settled without an LLM, and the agreement on that share,
- cost per 1,000 decisions and p50 latency, today and in the shadow,
- a threshold for each small model, fitted on your traffic,
- a verdict: `ready`, `not yet`, or `collect more`.

The example
[`examples/07_shadow_mode.py`](https://github.com/inboxpraveen/ThinkLess/blob/main/examples/07_shadow_mode.py)
runs the whole loop in a few seconds with no downloads.

## Wrap the existing decision

Most agents already have a function that makes the decision with an LLM.
Decorate it:

```python
from thinkless import Choice, Engine
from thinkless.providers import GLiNER, LLMDecider, Rules
from thinkless.shadow import Shadow

INTENT = Choice(
    "What does the customer want?",
    name="intent",
    options={"refund": "wants money back", "order_status": "asks where an order is", "other": "anything else"},
)

candidate = Engine([Rules(), GLiNER(), LLMDecider(llm)], llm=llm)
shadow = Shadow(candidate, log="shadow/intent.jsonl", sample=0.2)


@shadow.watch(
    INTENT,
    to_value=lambda response: response.intent,          # the label in your result
    cost_usd=lambda response: response.usage.cost_usd,  # what the call cost, if known
)
def classify(ticket: str) -> Classification:
    ...  # unchanged
```

The function returns and raises exactly as before. Each call is queued for
the candidate on a worker thread, so the shadow adds no latency to the
request.

- `state=` builds the candidate's input from the call's arguments. The
  default is the first positional argument.
- `to_value=` turns your result into an answer to the question. Labels
  compare case-insensitively, booleans accept `yes` and `no`, and a score can
  be a level name or a number.
- `cost_usd=` is a number or a function of the result. Without it the report
  shows the candidate's cost but projects no saving.

Async functions work the same way.

## Other entry points

When the decision cannot be wrapped, report it after the fact:

```python
label = existing_router(ticket)                    # anywhere in your code
shadow.observe(ticket, INTENT, label, cost_usd=0.0004, latency_ms=850)
```

`observe` also replays history: loop over logged decisions and pass each one
in, and the report is ready without waiting for new traffic.

`compare` runs a zero-argument callable, shadows it and returns its result:

```python
label = shadow.compare(ticket, INTENT, lambda: existing_router(ticket), cost_usd=0.0004)
```

## Read the report

```bash
thinkless shadow report shadow/intent.jsonl --volume 2000000
```

```text
question  compared  agreement (95% CI)   without an LLM  agreement there  cost / 1k: now  shadow   verdict
intent    4,812     97.9% (97.4-98.3%)   81%             98.4%            $0.3100         $0.0620  ready: agreement is at least 97.4% ...

intent@gliner: threshold 0.85 would settle 79.2% of the 4,120 calls it saw at 98.6% agreement.

The shadow costs $0.0620 per 1,000 decisions against $0.3100 now: $0.2480 less per 1,000 (80.0%).
At 2,000,000 decisions a month that is $496.00 less a month.
```

(Illustrative numbers.) `--target` sets the agreement the lower bound must
reach for `ready` (default 95%), `--min-calls` the number of compared calls
before any verdict (default 100), and `--json` prints everything for a
dashboard.

How to read it:

- **Agreement is not accuracy.** It measures how often ThinkLess matches the
  current system, mistakes included. A disagreement is either ThinkLess being
  wrong or the current system being wrong; the export below is how you find
  out which.
- **Use the lower bound.** A 98% agreement on 50 calls can be 90% on the
  next 5,000. The verdict uses the 95% lower bound for that reason, and so do
  the fitted thresholds.
- **"Without an LLM" is where the saving comes from.** The rest of the
  candidate's calls reach the same kind of LLM you use today, so the saving
  is roughly that share, less the cost of the fast planes.
- **Thresholds come from the calls each model actually saw.** A model later
  in the cascade only sees what earlier providers passed on.

## Label the disagreements

```bash
thinkless shadow export shadow/intent.jsonl --out disagreements.jsonl
```

Each row holds the input, both answers, the candidate's confidence and the
plane that answered, plus a `label` pre-filled with the current system's
answer. Reading fifty of them usually shows the pattern: a rule that fires
too eagerly, an option description that is ambiguous, or a case the current
system gets wrong.

To recalibrate on your own traffic, export every call, correct the labels on
the rows where `agree` is false, and calibrate on the whole file (calibrating
on the disagreements alone would fit the threshold to the hardest cases
only):

```bash
thinkless shadow export shadow/intent.jsonl --all --out intent_labeled.jsonl
# correct "label" where the current system was wrong, then:
thinkless calibrate intent_labeled.jsonl --provider gliner --kind choice \
    --question "What does the customer want?" --labels refund,order_status,other
```

`--label-from shadow` or `none` changes the pre-fill, and `--question`
exports a single question from a log that holds several.

## Safety and cost controls

| Setting | Default | Why |
|---|---|---|
| `sample` | 1.0 | shadow a share of calls; the candidate's LLM calls cost real money |
| `max_cost_usd` | none | stop shadowing once the candidate has spent this much |
| `max_pending` | 256 | drop and count calls when the queue is full, so a slow candidate never builds a backlog |
| `workers` | 2 | threads for background runs |
| `capture_content` | the candidate tracer's setting | when off, the log keeps a fingerprint of the input, never the input itself |
| `background` | True | `False` runs the candidate inline, for tests and batch replays |

A candidate that raises is logged and counted in `shadow.stats.errors`; the
exception never reaches your code. Shadow runs are separate traces marked
`shadow=True`, so they never add to the cost of the request they shadow.

Call `shadow.close()` at shutdown (or use it as a context manager) to finish
queued runs.

## Audit a live engine

The same class works in the other direction. Once ThinkLess makes a decision
in production, keep checking it against an LLM on a small sample:

```python
reference = Engine([LLMDecider(llm)])            # LLM only
audit = Shadow(reference, log="audit/intent.jsonl", sample=0.02, max_cost_usd=5)

decision = audit.compare(ticket, INTENT, engine)  # the engine is the primary here
```

The report then shows how often the live engine agrees with the LLM, broken
down by the plane that answered in the live engine
(`agreement by the plane that answered today`, or `primary_by_plane` in the
JSON). Run it weekly: a drop in agreement for the
model plane is the first sign that traffic has drifted away from what the
thresholds were calibrated on.

## The log format

One JSON object per line, schema version 1:

| Field | Meaning |
|---|---|
| `question`, `kind`, `spec` | the question and its definition |
| `state`, `state_sha` | the input (or `[redacted]`) and a 16-character fingerprint |
| `primary` | the current system's answer: `value`, `cost_usd`, `latency_ms`, and plane details when it is an engine |
| `shadow` | the candidate's decision: `value`, `status`, `plane`, `provider`, `confidence`, `cost_usd`, `latency_ms`, every `attempt`; or `error` |
| `agree` | whether they agree, `null` when either side has no answer |
| `fields` | per-field agreement for `Extract` questions |

The format is plain JSONL so that any analytics tool can read it.
