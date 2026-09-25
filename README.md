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
Kev and OpenJev), and any LLM (local through Transformers, Ollama or vLLM, or
hosted through Anthropic or OpenAI) plug into the same cascade. The whole stack
runs offline on a laptop GPU, with no API key.

<p align="center">
  <img src="docs/assets/trace-viewer.png" alt="The ThinkLess trace viewer comparing three decision planes on the support benchmark, with one ticket's waterfall: six decisions answered by rules, GLiNER and Laya in 115 ms, then a single LLM call for the reply" width="100%">
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

Same agent, same 53 labeled support tickets, three decision planes. Local run
on an RTX 5060 laptop GPU with Qwen3-1.7B as the LLM
([full report](benchmarks/results/support-local/report.md),
[interactive traces](benchmarks/results/support-local/viewer.html)):

| | LLM decides everything | Hybrid (ThinkLess) | Small models only |
|---|---:|---:|---:|
| Task success (correct action) | 86.8% | **90.6%** | 73.6% |
| Order id extraction | 94.1% | **100%** | 100% |
| LLM calls for decisions, per ticket | 1.04 | **0.68** | 0 |
| Decision time per ticket | 1.71 s | **416 ms** | 88 ms |
| End-to-end latency p50 / p95 | 3.07 s / 4.24 s | **1.87 s / 2.57 s** | 1.45 s / 2.09 s |
| LLM tokens spent on decisions | 524 | **155** | 0 |
| Estimated cost per 1k tickets at Claude Sonnet 5 prices | $2.16 | **$1.16** | $0.77 |
| Decisions by plane | rule 33%, LLM 67% | rule 39%, model 48%, LLM 12% | rule 40%, model 60% |

What the numbers do and do not say:

- **Hybrid matched or beat the LLM-only baseline while taking the LLM out of
  88 percent of decisions.** With 53 tickets, a two-ticket gap is not a claim
  of higher accuracy; equal accuracy at a quarter of the decision latency is.
- **Every hybrid failure came from the fallback LLM**, a 1.7B model chosen so
  anyone can reproduce the run offline. None came from the small models.
  Frontier-model runs are the next addition. The hosted backends (Anthropic,
  OpenAI-compatible, Jev) are unit-tested against mocked clients but have not
  yet been run against the live APIs for these results.
- **Thresholds were calibrated on a separate labeled set**, not on these
  tickets, with accuracy targets fixed in advance. See
  [calibration](docs/guides/calibration.md).
- **The baseline is fair.** In LLM mode all six triage questions share one
  structured prompt per ticket, the strongest form of that design, and
  deterministic checks (authentication, duplicate detection, refund limits)
  run as code in every mode.
- Cost is an estimate: Qwen token counts priced at Anthropic's published rates.
  Compare the ratio between modes.

On public data, [Banking77](benchmarks/results/intents-banking77/report.md)
(500 examples, 77 intents) shows why calibration matters: GLiNER 2.5 reached
71.8 percent at 34 ms, while the same 1.7B LLM reached 51.4 percent at 405 ms.
Escalating to a weaker model cannot help a cascade, and the benchmark shows
exactly where that line is. More in [benchmarks](docs/benchmarks.md).

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
thinkless bench support --llm anthropic:claude-haiku-4-5
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
| `LLMDecider` | every kind | any LLM backend below |

| LLM backends | Covers |
|---|---|
| `TransformersLLM` | any Hugging Face chat model, in-process |
| `OpenAICompatibleLLM` | OpenAI, Ollama, vLLM, LM Studio, llama.cpp, OpenRouter, Groq |
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
Next up are shadow mode, a generic Hugging Face classifier provider,
calibration from traces, and frontier-model baselines. See the
[roadmap](docs/roadmap.md).

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
