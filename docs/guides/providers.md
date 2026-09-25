# Providers

A provider answers typed questions. ThinkLess ships five, and writing your own
takes one class.

## Rules

Deterministic Python. The cheapest and most predictable provider, and the one
that should come first in every cascade.

```python
from thinkless.providers import Rules

rules = Rules()
rules.match("wants_human", r"\b(real person|a human|operator|supervisor)\b", True, field="message")
rules.extract("order", field="message", order_id=r"#\s?(\d{4,5})\b")

@rules.rule("intent")
def empty_message(state):
    return "other" if not state["message"].strip() else None
```

A rule returns an answer to settle the question with confidence 1, or `None`
to pass it on. Rules only claim questions they have been registered for. A
rule that returns something invalid for the question (an unknown label, a
non-bool for a yes/no) raises, which the engine records as a failed attempt.

## GLiNER

[GLiNER 2.5](https://github.com/fastino-ai/GLiNER2) by Fastino is a family of
schema-driven encoders for classification and extraction, Apache 2.0.

```python
from thinkless.providers import GLiNER

gliner = GLiNER("fastino/gliner2.5-base-v1", device="auto")
```

| | |
|---|---|
| Install | `pip install "thinkless[gliner]"` |
| Answers | `Choice`, `Extract` |
| Checkpoints | `gliner2.5-small-v1` (74M), `gliner2.5-base-v1` (0.2B, default), `gliner2.5-multi-v1` (0.3B, multilingual) |
| Measured latency | 16 ms per classification and 38 ms per extraction on an RTX 5060 laptop GPU; about 85 ms each on CPU |

GLiNER scores every option independently; ThinkLess normalizes the scores
into a distribution. All choice questions of one call share a forward pass,
as do all extract questions. On Banking77 (77 intents) it was the most
accurate provider in our [benchmarks](../benchmarks.md), ahead of a 1.7B LLM.

## Laya

[Laya](https://github.com/NandhaKishorM/laya) by Convai Innovations is an
open-weight, non-autoregressive System One model, Apache 2.0.

```python
from thinkless.providers import Laya

laya = Laya("convaiinnovations/laya", device="auto")
```

| | |
|---|---|
| Install | `pip install "thinkless[laya]"` |
| Answers | `Choice`, `Score`, `YesNo` |
| Checkpoints | `convaiinnovations/laya` (English, 421M); see the Laya project for multilingual and fine-tuned checkpoints |
| Measured latency | 30 ms for three questions in one pass on an RTX 5060 laptop GPU; about 650 ms on CPU |

Laya answers every question of a call in one forward pass. It is the only
bundled local model for `Score` and `YesNo`. It is weaker on choices with many
options: 35 percent on Banking77's 77 intents, which matches the Laya
project's own report.

## SystemOne: Jev, Kev, OpenJev

Any server that implements TypeSafe's `POST /v1/systemone`.

```python
from thinkless.providers import SystemOne

jev = SystemOne.jev()                                         # reads TYPESAFE_API_KEY
kev = SystemOne.self_hosted("http://localhost:8009", name="kev", model="kev-latest")
```

| | |
|---|---|
| Install | nothing extra, it uses `httpx` |
| Answers | `Choice`, `Score`, `YesNo` |
| Cost | Jev is priced at $0.042 per million input tokens with free output (TypeSafe, September 2026); self-hosted servers are recorded as free |
| Retries | 429, 529, 5xx and connection errors, with exponential backoff that honors `Retry-After` |

TypeSafe reports 70 to 500 ms per call for Jev. Kev and OpenJev need a large
GPU to self-host (Kev-4B wants 32 GB).

## HFClassifier

Any Hugging Face `text-classification` model, answering the questions it was
trained for. A general decision model answers anything reasonably; a
classifier trained for one narrow question usually answers that question much
better.

```python
from thinkless.providers import HFClassifier

injection_guard = HFClassifier(
    "protectai/deberta-v3-base-prompt-injection-v2",
    answers={"injection": {"yes": "INJECTION"}},
    field="message",
)
sentiment = HFClassifier(
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    answers={"tone": {"upset": "negative", "happy": "positive", "unclear": "neutral"}},
)
```

| | |
|---|---|
| Install | `pip install "thinkless[local-llm]"` (transformers and torch) |
| Answers | `Choice` and `YesNo`, only for the question names in `answers` |
| Mapping | yes/no: the label that means yes; choice: option to model label |

Put it in the cascade before the general models so it answers its questions
first; it never claims any other question. Calibrate it on your own traffic
before relying on it: the injection model above is a good example of why. It
is widely used, and on customer support tickets it made confident mistakes
(see the [production guide](production.md)).

## LLMDecider

Any `LLM` answering questions through a prompt, the way most agents make
decisions today.

```python
from thinkless.llm import AnthropicLLM
from thinkless.providers import LLMDecider

decider = LLMDecider(AnthropicLLM("claude-haiku-4-5"))
```

It answers every kind, including `Extract`, in one prompt per batch, and
parses the reply leniently (code fences, trailing commas, label case and
spacing). Invalid answers become abstentions. Its answers carry no
probabilities, so the engine treats them as uncalibrated: accepted as the last
word of a cascade by default.

## Writing a provider

Subclass `DecisionProvider`, declare what you answer, and return a
`ProviderResult`:

```python
from typing import ClassVar

from thinkless import Answer, Kind, Plane
from thinkless.providers import DecisionProvider, ProviderResult, render_state


class SentimentProvider(DecisionProvider):
    name = "sentiment"
    plane: ClassVar[Plane] = Plane.MODEL
    kinds: ClassVar[frozenset[Kind]] = frozenset({Kind.CHOICE})

    def __init__(self, pipeline):
        self.pipeline = pipeline  # for example a transformers text-classification pipeline

    def supports(self, question):
        return super().supports(question) and question.key == "sentiment"

    def answer(self, state, questions):
        scores = {r["label"].lower(): r["score"] for r in self.pipeline(render_state(state), top_k=None)}
        best = max(scores, key=scores.get)
        return ProviderResult(answers={key: Answer(value=best, probabilities=scores) for key in questions})
```

Return probabilities whenever you have them: that is what lets the engine
threshold your answers consistently with every other provider. Return `None`
for a question you cannot answer. Set `calibrated = False` on the class if your
answers carry no meaningful confidence. Put slow setup in `warmup()` so the
first request does not pay for it.

Any fine-tuned Hugging Face classifier can become a provider this way, which is
often the best option for a narrow, high-volume question such as prompt
injection or toxicity.
