# Confidence and thresholds

Every accepted decision in a cascade depends on one comparison: is the
provider's confidence at least the question's threshold? That comparison is
only meaningful if confidence means the same thing no matter which provider
produced it. Out of the box, it does not.

## Providers disagree about "confidence"

Measured on the same ticket while building ThinkLess:

| Provider | Reported for a five-way choice with p_max = 0.9953 | Definition |
|---|---|---|
| Laya | 0.9782 | 1 minus normalized entropy |
| TypeSafe (documented) | 0.9941 | (k * p_max - 1) / (k - 1) |

For yes/no questions Laya reports `max(p, 1 - p)`, so a coin flip scores 0.5,
while the TypeSafe formula scores the same coin flip 0. A threshold of 0.8
would accept very different answers depending on which backend happened to
answer.

## One definition for every provider

ThinkLess ignores each provider's own confidence field for thresholding and
computes a single normalized confidence from the probability distribution:

```text
confidence = (k * p_max - 1) / (k - 1)
```

where `k` is the number of possible answers and `p_max` the probability of the
most likely one. It is the formula TypeSafe documents for Jev, applied
uniformly:

- 0 for a uniform distribution, 1 when all probability sits on one answer.
- For a yes/no question it reduces to `2 * max(p, 1 - p) - 1`. A threshold of
  0.8 therefore requires `p >= 0.9`, and 0.5 requires `p >= 0.75`.
- For a large choice it is close to `p_max`: with 77 options, `p_max = 0.8`
  gives a confidence of 0.797.

Providers that emit independent per-label scores (GLiNER's sigmoid outputs)
have their scores normalized into a distribution first. The provider's own
fields are preserved untouched in `decision.raw`.

When a provider returns no distribution at all, ThinkLess uses its reported
confidence if there is one. A prompted LLM has neither, so its decisions carry
`confidence=None`. By default such answers are accepted as the final word of a
cascade, since the LLM is what the agent would have used anyway; set
`Engine(trust_uncalibrated=False)` to have them come back `uncertain` instead.

## Normalized is not calibrated

Normalization makes confidences comparable. It does not make them correct. A
model can be confidently wrong on a question it was never trained for, and
the only way to know is to measure on labeled examples. That is what
`thinkless calibrate` is for; see the [calibration guide](../guides/calibration.md).

## Choosing thresholds

Set thresholds by the cost of a wrong answer, then verify them on data.

| Question | Cost of a wrong answer | Reasonable starting threshold |
|---|---|---|
| Ticket priority, sentiment for analytics | Low: a ticket sorts slightly wrong | 0.2 to 0.5 |
| Intent that routes to a workflow | Medium: the wrong workflow runs, usually recoverable | 0.8 |
| Anything that moves money or data, or a safety check | High | 0.9 and above, plus a deterministic guard |

Thresholds resolve in this order:

1. `threshold=` on the question itself.
2. `Engine(thresholds={"name": value})`.
3. `Engine(threshold=...)`, default 0.8.

## What happens below threshold

An answer below threshold is not discarded. The engine keeps the best answer
seen and asks the next provider. If every provider has been tried, the decision
comes back with `status="uncertain"` and that best answer. The application
decides what uncertainty means:

```python
injection = decisions["injection"]
if injection.is_(True):                 # confident yes
    route_to_security()
elif injection.value is True:           # unsure, leaning yes
    block_automatic_refunds_and_raise_priority()
```

The support demo does exactly this: only a confident answer routes to the
security queue, while an unsure answer that leans yes still fails closed by
blocking automatic refunds.
