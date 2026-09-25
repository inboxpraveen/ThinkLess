<h1 align="center">ThinkLess</h1>

<p align="center"><b>Stop using an LLM for every decision.</b></p>

<p align="center">
  <a href="https://github.com/inboxpraveen/ThinkLess/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/inboxpraveen/ThinkLess/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <a href="docs/benchmarks.md"><img alt="Benchmarks" src="https://img.shields.io/badge/benchmarks-reproducible-brightgreen.svg"></a>
</p>

ThinkLess is an open-source decision plane for AI agents. The routine
judgments an agent makes (what the user wants, whether they asked for a
person, which order they mean, whether a message is trying to manipulate the
system) are answered by rules and small calibrated models in milliseconds. The
LLM is kept for the steps that need it: writing, planning, and the decisions
the small models are not sure about. Every step is traced with the plane that
handled it, its confidence, latency, tokens and cost.

It is model-neutral. Rules, [GLiNER 2.5](https://github.com/fastino-ai/GLiNER2),
[Laya](https://github.com/NandhaKishorM/laya), TypeSafe's
[Jev](https://typesafe.ai) (and any server speaking its System One API, such as
Kev and OpenJev), any Hugging Face classifier, and any LLM (local through
Transformers, Ollama or vLLM, or hosted through OpenRouter, Anthropic or
OpenAI) plug into the same cascade. The whole stack also runs offline on a
laptop GPU, with no API key.

<p align="center">
  <img src="docs/assets/trace-viewer.png" alt="The ThinkLess trace viewer comparing three decision planes on the support benchmark, with one ticket's waterfall: six decisions answered by rules, GLiNER and Laya in 126 ms, then a single LLM call for the reply" width="100%">
</p>

## Why

A typical agent sends every branch of its workflow to one generative model:
classify the request, extract the arguments, pick the tool, check the result,
check the reply, then write the reply. Most of those are bounded questions with
a handful of possible answers. A generative model answers them slowly, at full
token cost, and without telling you how sure it is.

ThinkLess treats each of them as a typed question. It asks the cheapest
provider first and escalates only when the answer's confidence is below a
threshold you set per question and per provider, from data. Your code reads a
typed decision and never needs to know which model produced it.

```python
intent = engine.decide(ticket, INTENT)          # rules, then GLiNER, then Laya, then the LLM
if intent.is_("refund_duplicate_charge"):       # only true when the answer is confident
    ...
```

## Results

Same agent, same 53 labeled support tickets, two decision planes: `llm` sends
every decision to the LLM, `hybrid` asks rules, GLiNER and Laya first. Hosted
models ran through OpenRouter, and the dollar figures are what OpenRouter
billed ([full results](docs/benchmarks.md)):

| Fallback LLM | Success, `llm` | Success, `hybrid` | Billed per 1k tickets, `llm` | Billed per 1k tickets, `hybrid` | Decision time |
|---|---:|---:|---:|---:|---:|
| Qwen 3.7 Flash | 92.5% | **98.1%** | $0.034 | **$0.019** | 1.27 s to **0.58 s** |
| Gemini 2.5 Flash Lite | 98.1% | **98.1%** | $0.104 | **$0.059** | 1.17 s to **0.76 s** |
| GPT-5.6 Luna | 98.1% | **100%** | $0.235 | **$0.141** | 2.50 s to **1.18 s** |
| Claude Haiku 4.5 | 98.1% | **98.1%** | $1.338 | **$0.788** | 2.21 s to **1.34 s** |
| Qwen3-1.7B, local | 86.8% | **94.3%** | free | free | 1.84 s to **0.43 s** |

- **Hybrid matched or beat the LLM-only design on every model, at 40 to 44
  percent lower billed cost** and 35 to 55 percent less time spent on
  decisions. 87 percent of hybrid decisions never reached the LLM.
- **Small models are the most reliable extractors.** Rules and GLiNER found
  every order number on every run. The larger hosted LLMs did too; Qwen 3.7
  Flash and the local 1.7B model missed ones written without the word
  "order".
- **On Banking77 (77 intents, 500 examples) the cascade matched or beat its
  LLM at a quarter of the cost.** GLiNER alone 71.8 percent; Qwen 3.7 Flash
  alone 73.8 percent, cascade 74.8 percent; Claude Haiku 4.5 alone 76.2
  percent, cascade 76.2 percent. Each time a quarter of the calls reached the
  LLM.
- **Nothing was tuned on the test tickets.** Thresholds and every fix were
  validated on a separate calibration set first. The baseline is the
  strongest form of the LLM design: all six triage questions in one prompt.
- The support set is 53 synthetic tickets: evidence of the mechanism and its
  failure modes, not a leaderboard. The [benchmarks page](docs/benchmarks.md)
  lists the weak areas found and what was done about each.

## Quick start

```bash
pip install "thinkless[local]"        # GLiNER, Laya and a local LLM
thinkless doctor                      # check the environment
```

```python
from thinkless import Choice, Engine, Extract, JSONLSink, Tracer, YesNo
from thinkless.llm import TransformersLLM
from thinkless.providers import GLiNER, Laya, LLMDecider, Rules

rules = Rules()
rules.match("wants_human", r"\b(real person|a human|an agent|operator)\b", True, field="message")
rules.extract("order", field="message", order_id=r"#\s?(\d{4,5})\b")

llm = TransformersLLM("Qwen/Qwen3-1.7B")
engine = Engine(
    [rules, GLiNER(), Laya(), LLMDecider(llm)],     # cheapest first
    llm=llm,
    threshold=0.8,
    tracer=Tracer([JSONLSink(".thinkless/traces")]),
)

with engine.run("ticket") as run:
    decisions = engine.decide_many(
        {"message": "I was charged twice for order #4471. Refund it or I'm cancelling."},
        [
            Choice("What does the customer want?", name="intent",
                   options={"refund": "wants money back", "order_status": "asks where an order is",
                            "cancel": "wants to cancel", "other": "anything else"}),
            YesNo("Is the customer asking for a person?", name="wants_human"),
            Extract(name="order", fields={"order_id": "the order number"}),
        ],
    )

for name, d in decisions.items():
    print(f"{name:12} {d.value!s:24} {d.status.value:9} via {d.provider}")
print(run.summary().decisions_by_plane)      # which plane answered what
```

Prefer a hosted model? Put `OPENROUTER_API_KEY=...` in `.env` and swap one
line; OpenRouter reaches Qwen, Gemini, GPT and Claude models with one key:

```python
from thinkless.llm import OpenRouterLLM

llm = OpenRouterLLM("qwen/qwen3.7-flash", reasoning={"enabled": False})
```

Then open the trace:

```bash
thinkless trace view
```

No GPU? `pip install thinkless` and run
[`examples/01_rules_only.py`](examples/01_rules_only.py): the API, the cascade
and the traces with no model downloads.

## Try the demo agent

A complete customer support agent ships with the package: a mock store with
orders, payments and a help center, 53 labeled tickets, and every hard case we
could think of (order numbers written in ways a regex misses, a customer
asking about someone else's order, prompt injections, requests for a human,
Spanish and German tickets).

```bash
thinkless demo --ticket T-016                          # one ticket, printed as a trace tree
thinkless demo --ticket T-051 --mode hybrid --mode llm # compare decision planes
thinkless bench support                                # every ticket in every mode
thinkless bench support --llm openrouter:qwen/qwen3.7-flash --reasoning off
thinkless bench intents --dataset banking77
```

## How it works

**Typed questions.** `Choice`, `Score`, `YesNo` and `Extract` declare what you
want to know and which answers are allowed. They follow the System One wire
format, so the same question works on every provider.
[Questions](docs/concepts/questions.md)

**A confidence cascade.** Providers are tried cheapest first. Each gets one
call with every open question it supports; answers that clear their threshold
close, the rest move on. Unresolved questions come back `uncertain` with the
best answer seen, and `decision.is_()` never matches an uncertain answer.
[The cascade](docs/concepts/cascade.md)

**One meaning of confidence.** Providers disagree about what "confidence"
means (Laya and TypeSafe compute it differently), so a threshold of 0.8 would
mean different things per backend. ThinkLess derives a single normalized
confidence from each provider's probabilities.
[Confidence](docs/concepts/confidence.md)

**Thresholds from data.** `thinkless calibrate` sweeps thresholds on labeled
examples and recommends the lowest one that meets your accuracy target, per
question and per provider (`thresholds={"intent@gliner": 0.9}`).
[Calibration](docs/guides/calibration.md)

**Traces for everything.** Decisions, provider attempts, generations, tool
calls and rule checks are spans with W3C-compatible ids. Write JSONL audit
files, export to any OpenTelemetry backend with GenAI semantic conventions, or
open the self-contained HTML viewer. Content capture can be switched off
without losing structure, timings or confidences.
[Tracing](docs/concepts/tracing.md)

## Providers and backends

| Decision providers | Answers | Runs |
|---|---|---|
| `Rules` | every kind | in-process, under 1 ms |
| `GLiNER` (GLiNER 2.5) | choice, extract | local, 16 to 38 ms on a laptop GPU, about 85 ms on CPU |
| `Laya` | choice, score, yes/no | local, about 30 ms per batch on a laptop GPU |
| `SystemOne` (Jev, Kev, OpenJev) | choice, score, yes/no | hosted or self-hosted HTTP |
| `HFClassifier` | choice, yes/no, for the questions it was trained on | any Hugging Face text classifier, local |
| `LLMDecider` | every kind | any LLM backend below |

| LLM backends | Covers |
|---|---|
| `TransformersLLM` | any Hugging Face chat model, in-process |
| `OpenRouterLLM` | hundreds of hosted models with one key, billed cost recorded per call |
| `OpenAICompatibleLLM` | OpenAI, Ollama, vLLM, LM Studio, llama.cpp, Groq |
| `AnthropicLLM` | Claude, with server-side refusal fallbacks |

Writing your own provider is one class. [Providers](docs/guides/providers.md),
[LLM backends](docs/guides/llm-backends.md)

## Documentation

- [Getting started](docs/getting-started.md)
- Concepts: [the four planes](docs/concepts/planes.md),
  [questions](docs/concepts/questions.md),
  [confidence](docs/concepts/confidence.md),
  [the cascade](docs/concepts/cascade.md),
  [tracing](docs/concepts/tracing.md)
- Guides: [the support agent](docs/guides/support-agent.md),
  [calibration](docs/guides/calibration.md),
  [providers](docs/guides/providers.md),
  [LLM backends](docs/guides/llm-backends.md),
  [production](docs/guides/production.md),
  [troubleshooting](docs/guides/troubleshooting.md)
- [Benchmarks](docs/benchmarks.md), [CLI reference](docs/reference/cli.md),
  [Python API](docs/reference/api.md), [roadmap](docs/roadmap.md)

## Status

ThinkLess is at 0.1: the core API (questions, the engine, decisions and
traces) is meant to stay stable, and providers and benchmarks will grow.
Next up are shadow mode, calibration from traces, calibrated LLM confidence
from log probabilities, and live Jev runs. See the [roadmap](docs/roadmap.md).

## Contributing

Issues, providers, benchmark runs on other hardware and models, and new demo
scenarios are all welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md). The
bar for changes that affect accuracy, latency or cost is evidence: a before
and after from `thinkless bench` or `thinkless calibrate`.

## Sponsoring

If ThinkLess saves your team time or money, consider
[sponsoring its development](https://github.com/sponsors/inboxpraveen).
Sponsorship funds frontier-model benchmark runs, more providers and more
demos.

## Citing

See [CITATION.cff](CITATION.cff).

## Acknowledgements

ThinkLess builds on the work of the teams behind
[Jev and the System One API](https://typesafe.ai) (TypeSafe AI),
[GLiNER2](https://github.com/fastino-ai/GLiNER2) (Fastino),
[Laya](https://github.com/NandhaKishorM/laya) (Convai Innovations),
[Qwen](https://github.com/QwenLM) and
[Banking77](https://github.com/PolyAI-LDN/task-specific-datasets) (PolyAI).

ThinkLess is not affiliated with the NeurIPS 2025 paper
[Thinkless: LLM Learns When to Think](https://github.com/VainF/Thinkless),
which studies adaptive reasoning inside a single model.

## License

[Apache License 2.0](LICENSE).
