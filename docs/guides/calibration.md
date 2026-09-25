# Calibrating thresholds

A threshold is a claim: "when this provider is at least this confident about
this question, it is right often enough to act on." Calibration is how you
check the claim. It takes a few dozen to a few hundred labeled examples and a
few minutes, and it is the single most effective thing you can do before
putting a cascade in front of real traffic.

## The workflow

1. **Collect labeled examples** of the question as your agent will ask it.
   Real traffic is best. Keep them separate from whatever you use to report
   results, so the thresholds are not tuned to the test.
2. **Fix the target first.** Decide what accuracy an accepted answer must have
   before you look at any numbers, based on the cost of a wrong answer. Pick
   thresholds afterwards and you will end up fitting them to the data.
3. **Run `thinkless calibrate`** for each question and provider.
4. **Use the recommendation**, per provider when models differ, or restrict
   the question to stronger providers when no threshold reaches the target.
5. **Recalibrate** when you change a question's wording, a model, or when the
   traffic shifts.

## Running it

Rows are JSON Lines. For a choice question:

```json
{"text": "my card was charged twice for one order", "label": "refund_duplicate_charge"}
```

For a yes/no question, the label is a boolean:

```json
{"text": "can I talk to a real person?", "label": true}
```

```bash
thinkless calibrate intents.jsonl --provider gliner --question "What does the customer want?" --target 0.95
thinkless calibrate handoff.jsonl --provider laya --kind yes_no \
    --question "Does the customer ask to talk to a human?" --target 0.95
```

The output is a sweep: for every threshold, the share of inputs the provider
would answer alone and the accuracy of those answers. The recommendation is
the lowest threshold that meets the target, which leaves the most work to the
cheap provider:

```text
threshold   answers alone   accuracy of those answers
     0.45           89.6%                       90.7%
     0.50           89.6%                       90.7%
     ...
     0.80           56.2%                      100.0%
Recommended threshold 0.80: laya answers 56.2% of inputs on its own at 100.0% accuracy;
the rest go to the next provider.
```

When no threshold reaches the target, `calibrate` says so. That is a useful
result: the provider cannot answer this question reliably, and the question
should be restricted to providers that can:

```python
WANTS_HUMAN = YesNo("...", name="wants_human", providers=("rules", "llm"))
```

## Per-provider thresholds

Normalization makes confidences comparable across providers, but models still
calibrate differently. On the support demo's intent question, GLiNER needs
0.90 to reach 95 percent accuracy on the answers it accepts while Laya needs
0.95. Express that with `question@provider` keys:

```python
engine = Engine(
    providers,
    llm=llm,
    threshold=0.8,
    thresholds={"intent@gliner": 0.90, "intent@laya": 0.95, "urgency": 0.2},
)
```

Resolution order is `question@provider`, then `Question(threshold=...)`, then
`thresholds[question]`, then the engine default.

## What the support demo did

The demo ships a labeled calibration set, `data/calibration.jsonl`, written
separately from the 53 benchmark tickets. Its targets were fixed in advance:
100 percent for the security question, 95 percent for routing questions. The
results, reproducible with `python benchmarks/calibrate_support.py`:

| Key | Rows | Target | Threshold | Answers alone | Accuracy |
|---|---:|---:|---:|---:|---:|
| `intent@gliner` | 28 | 95% | 0.90 | 14% | 100% |
| `intent@laya` | 28 | 95% | 0.95 | 43% | 100% |
| `wants_human@laya` | 38 | 95% | 0.50 | 92% | 100% |
| `injection@laya` | 48 | 100% | 0.75 | 58% | 100% |

Three findings came out of this that no amount of reasoning about the models
would have produced:

- The first version of the `wants_human` rules caught none of the ten
  positive examples in the calibration set ("transfer me to an agent",
  "operator please", "escalate this to a supervisor"). Broadening the patterns
  took it to ten of ten with no false matches, and then held at three of
  three with none on the benchmark tickets.
- Laya's answers to "is this a prompt injection?" often sat close to a coin
  flip on ordinary messages. Only from a threshold of 0.75 were all of its
  accepted answers correct, and one ordinary benchmark ticket scored
  `P(yes) = 0.84`. A dedicated injection classifier is the better tool for
  production; see the [production guide](production.md).
- The same intent question needs a different threshold on each model.

Forty-eight rows is enough to find problems like these and not enough to
trust the exact numbers. Recalibrate on a few hundred examples of your own
traffic.

## Measuring from traces

Traces record every attempt's answer and confidence, so a trace directory
from a shadow deployment is a calibration set waiting for labels. Export the
`decide` spans of one question, label a sample, and feed it to `calibrate`.
