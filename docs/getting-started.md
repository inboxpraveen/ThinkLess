# Getting started

## Install

ThinkLess needs Python 3.10 or newer. The core package is small and pulls in
no model libraries; decision models and LLM backends are extras.

```bash
pip install thinkless                          # core: rules, tracing, the HTTP System One provider
pip install "thinkless[local]"                 # GLiNER, Laya and a local LLM (needs torch)
pip install "thinkless[anthropic]"             # Claude as the reasoning plane
pip install "thinkless[openai]"                # OpenAI, Ollama, vLLM and other compatible servers
pip install "thinkless[all]"                   # everything, including OpenTelemetry and benchmarks
```

With conda, `environment.yml` creates an environment with everything and the
right PyTorch build for recent NVIDIA GPUs:

```bash
conda env create -f environment.yml
conda activate thinkless
thinkless doctor
```

## Your first decision

```python
from thinkless import Choice, Engine
from thinkless.providers import GLiNER

engine = Engine([GLiNER()])

intent = engine.decide(
    "I was charged twice for order #4471, please refund one of them.",
    Choice(
        "What does the customer want?",
        name="intent",
        options={
            "refund": "wants money back",
            "order_status": "asks where an order is",
            "other": "anything else",
        },
    ),
)

print(intent.value)        # refund
print(intent.confidence)   # 0.99
print(intent.provider)     # gliner
print(intent.latency_ms)   # about 16 on a laptop GPU
```

## A cascade

Add rules in front and an LLM behind. Rules answer the obvious cases for free,
the small model answers most of the rest, and the LLM only sees what the small
model is unsure about.

```python
from thinkless import Choice, Engine, Extract, YesNo
from thinkless.llm import TransformersLLM
from thinkless.providers import GLiNER, Laya, LLMDecider, Rules

rules = Rules()
rules.match("wants_human", r"\b(real person|a human|operator)\b", True, field="message")
rules.extract("order", field="message", order_id=r"#\s?(\d{4,5})\b")

llm = TransformersLLM("Qwen/Qwen3-1.7B")
engine = Engine([rules, GLiNER(), Laya(), LLMDecider(llm)], llm=llm, threshold=0.8)

decisions = engine.decide_many(
    {"message": "Charged twice for #4471. Refund it or I'm cancelling."},
    [
        Choice("What does the customer want?", name="intent", options=["refund", "order_status", "other"]),
        YesNo("Is the customer asking for a person?", name="wants_human"),
        Extract(name="order", fields={"order_id": "the order number"}),
    ],
)
for name, d in decisions.items():
    print(name, d.value, d.status.value, d.provider, [a.provider for a in d.attempts])
```

## Generation

```python
reply = engine.generate(
    "Write two sentences confirming the refund of $49 for order 4471.",
    system="You are a concise support agent.",
    max_tokens=120,
)
print(reply.text, reply.usage)
```

## Traces

```python
from thinkless import JSONLSink, Tracer

engine = Engine([rules, GLiNER()], tracer=Tracer([JSONLSink(".thinkless/traces")]))

with engine.run("ticket", ticket_id="T-1", mode="hybrid") as run:
    engine.decide(ticket, INTENT)
print(run.summary())
```

```bash
thinkless trace ls
thinkless trace view         # opens the HTML viewer
```

## Try the demo agent

The package ships a complete support agent with a mock store and 53 labeled
tickets:

```bash
thinkless demo --ticket T-016
thinkless demo --ticket T-001 --mode hybrid --mode llm      # compare two decision planes
thinkless demo --message "Where is order 2290?" --customer C-1007
thinkless bench support                                      # every ticket, every mode
```

Add `--llm anthropic` (or `--llm ollama:qwen3:8b`, `--llm openai:<model>`) to
use a different reasoning model.

## Next

- [The four planes](concepts/planes.md) explains the architecture.
- [Questions](concepts/questions.md) covers the question types and how to
  write questions small models answer well.
- [Calibration](guides/calibration.md) shows how to set thresholds from data.
- [The support agent](guides/support-agent.md) walks through a realistic
  agent end to end.
