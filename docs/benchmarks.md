# Benchmarks

Two benchmarks ship with ThinkLess. The **support benchmark** runs one
realistic agent end to end with three different decision planes. The
**intent benchmarks** measure providers on public datasets and simulate the
cascade at every threshold. Both are one command, and both write their full
results (every prediction, every trace) next to the summary.

All results below were produced on one machine: an NVIDIA RTX 5060 laptop GPU
(8 GB), Windows 11, Python 3.12, PyTorch 2.14 with CUDA 13.0. The reasoning
model is Qwen3-1.7B running in-process through Transformers, chosen so that
anyone can reproduce every number offline, without an API key.

## Support benchmark

```bash
thinkless bench support                       # llm, hybrid and models modes
thinkless bench support --llm anthropic:claude-haiku-4-5
```

### Setup

- **The agent** is the bundled [support agent](guides/support-agent.md): the
  same code in every mode.
- **The tickets** are 53 hand-written support messages with labeled expected
  outcomes, covering every path of the agent plus hard cases: order numbers
  without the word "order", a customer asking about someone else's order,
  prompt injections, requests for a person, signed-out users, and Spanish and
  German messages. They are synthetic and small; read the numbers as a
  demonstration of the mechanism, not as a general accuracy claim.
- **The modes** differ only in the decision cascade:

| Mode | Decision cascade |
|---|---|
| `llm` | The LLM answers every question: the common agent design |
| `hybrid` | Rules, then GLiNER, then Laya, then the LLM for anything they are unsure about |
| `models` | Rules, GLiNER and Laya only; unsure decisions go to a person |

- **The baseline is the strongest reasonable version of the LLM design.** All
  six triage questions (intent, urgency, churn risk, request for a person,
  prompt injection, order number) go to the LLM in one structured prompt per
  ticket, not one call per question. Deterministic checks (authentication,
  order ownership, duplicate detection, refund limits, reply grounding) are
  code in every mode.
- **Thresholds** for the routing questions were calibrated on a separate
  48-row labeled set with accuracy targets fixed in advance; see
  [calibration](guides/calibration.md). Priority-only questions (urgency,
  churn risk) use low, risk-based thresholds.
- **Task success** is the share of tickets where the agent took the expected
  action.
- **Cost** multiplies LLM token counts (from Qwen's tokenizer) by Claude Sonnet
  5's published price. Tokenizers differ, so compare modes by ratio.

### Results

[Full report](https://github.com/inboxpraveen/ThinkLess/blob/main/benchmarks/results/support-local/report.md) ·
[results.json](https://github.com/inboxpraveen/ThinkLess/blob/main/benchmarks/results/support-local/results.json) ·
[trace viewer](https://github.com/inboxpraveen/ThinkLess/blob/main/benchmarks/results/support-local/viewer.html)

| Metric | `llm` | `hybrid` | `models` |
|---|---:|---:|---:|
| Task success (correct action) | 86.8% | 90.6% | 73.6% |
| Intent accuracy | 87.2% | 91.5% | 72.3% |
| Order id accuracy | 94.1% | 100.0% | 100.0% |
| LLM calls per ticket | 1.93 | 1.60 | 0.93 |
| for decisions | 1.04 | 0.68 | 0.00 |
| for replies | 0.89 | 0.93 | 0.93 |
| Decision time per ticket | 1.71 s | 416 ms | 88 ms |
| Reply generation time per ticket | 1.30 s | 1.37 s | 1.34 s |
| End-to-end latency p50 | 3.07 s | 1.87 s | 1.45 s |
| End-to-end latency p95 | 4.24 s | 2.57 s | 2.09 s |
| LLM tokens per ticket (in / out) | 674 / 81 | 351 / 45 | 205 / 36 |
| LLM tokens spent on decisions | 524 | 155 | 0 |
| Estimated cost per 1k tickets | $2.16 | $1.16 | $0.77 |
| Replies passing the grounding check | 100% | 100% | 100% |
| Decisions by plane | rule 33%, llm 67% | rule 39%, model 48%, llm 12% | rule 40%, model 60% |

"Decisions" include the deterministic checks the agent records with
`engine.rule`, which is why the `llm` mode also shows a rule share.

### Reading the results

**Hybrid kept accuracy while moving most decisions off the LLM.** 88 percent
of hybrid decisions were made by rules or small models. Decision time fell
from 1.71 s to 416 ms per ticket, decision tokens by 70 percent, and p95
latency by 39 percent. With 53 tickets, the two-ticket accuracy gap in
hybrid's favor is within noise; the claim is equal accuracy for a fraction of
the latency and tokens.

**The remaining LLM work in hybrid is concentrated.** The intent question
escalated on 59 percent of tickets under its strict calibrated thresholds
(0.90 for GLiNER, 0.95 for Laya), and the injection question on 35 percent.
These are the two places a larger calibration set or a purpose-built
classifier would pay off first.

**Every hybrid failure came from the fallback LLM.** T-027 and T-048 were
wrongly flagged as prompt injections by the 1.7B model after Laya was unsure;
T-031, T-033 and T-039 got intent `other` from it. None came from rules,
GLiNER or Laya. A stronger fallback should close most of that gap, which is
the first thing the frontier runs will test.

**LLM-only failures were the LLM's.** It missed order numbers written without
the word "order" (T-004, T-023), classified two help-center questions as
`other` (T-031, T-033), read a promo code error and a request for a person as
return requests (T-039, T-046), and missed the subtlest prompt injection
(T-051), which hybrid's rules caught.

**Small models alone are fast but hand more to people.** The `models` mode
never calls an LLM for decisions and routes 12 tickets to human triage when
intent confidence is below threshold. That is the intended behavior of that
design, not a malfunction: it trades automation for zero LLM decision cost.

**Replies dominate the remaining latency and cost.** Generation takes about
1.3 s per ticket in every mode. Templates for simple outcomes, a faster
serving stack, or a smaller reply model are the next levers, and they are
orthogonal to the decision plane.

## Intent benchmarks

```bash
thinkless bench intents --dataset banking77
thinkless bench intents --dataset emotion
thinkless bench intents --dataset clinc150 --provider gliner --provider laya
```

Each run samples 500 test examples with a fixed seed, asks every provider the
same `Choice` question, and reports accuracy, expected calibration error
(ECE), latency and the threshold each provider needs to reach 95 percent
accuracy on the answers it accepts. It then simulates the cascade (GLiNER,
then Laya, then the LLM) at every threshold.

### Banking77

77 fine-grained banking intents ([PolyAI](https://github.com/PolyAI-LDN/task-specific-datasets)).
[Report](https://github.com/inboxpraveen/ThinkLess/blob/main/benchmarks/results/intents-banking77/report.md) ·
[predictions](https://github.com/inboxpraveen/ThinkLess/blob/main/benchmarks/results/intents-banking77/predictions.jsonl)

| Provider | Accuracy | ECE | p50 latency | Threshold for 95% |
|---|---:|---:|---:|---|
| GLiNER 2.5 base | 71.8% | 0.116 | 33 ms | 0.99, answers 25% alone at 96.8% |
| Laya | 35.0% | 0.184 | 51 ms | not reached |
| Qwen3-1.7B | 51.4% | n/a | 405 ms | uncalibrated |

| Cascade threshold | Accuracy | Calls reaching the LLM | Mean latency |
|---:|---:|---:|---:|
| 0.50 | 71.4% | 7.0% | 73 ms |
| 0.80 | 68.0% | 24.2% | 156 ms |
| 0.90 | 64.4% | 34.2% | 204 ms |
| LLM only | 51.4% | 100% | 506 ms |

GLiNER is 20 points more accurate than the local LLM on this task at a
twelfth of the latency, so escalating to this LLM buys nothing: the best
cascade setting (72.0 percent at 0.35) is within 0.2 points of GLiNER alone,
and accuracy falls steadily as more goes to the LLM. The lesson generalizes:
**a cascade only helps when the fallback is more accurate than the model it
replaces on the inputs it escalates**, and only measurement tells you whether
it is. GLiNER's confident answers are reliable (96.8 percent accurate on the
quarter it answers at 0.99), so a GLiNER-then-frontier cascade at that
threshold would keep a quarter of Banking77 local and send the rest to the
stronger model.

Laya's weak result on many-option questions matches its own project's report
(0.425 on Banking77 at default settings).

### Emotion

Six emotions on short social media text ([dair-ai](https://huggingface.co/datasets/dair-ai/emotion)).
[Report](https://github.com/inboxpraveen/ThinkLess/blob/main/benchmarks/results/intents-emotion/report.md)

| Provider | Accuracy | ECE | p50 latency |
|---|---:|---:|---:|
| GLiNER 2.5 base | 51.6% | 0.120 | 17 ms |
| Laya | 54.6% | 0.319 | 28 ms |
| Qwen3-1.7B | 47.0% | n/a | 269 ms |

Zero-shot, nobody does well: the labels are noisy and overlap (`joy` and
`love`, `fear` and `surprise`). The best cascade setting reaches 55.2 percent.
Tasks like this call for a fine-tuned classifier, which slots into the
cascade as a [custom provider](guides/providers.md#writing-a-provider).

## Not measured yet

Honest gaps, in the order we intend to close them:

- **Frontier LLMs.** No hosted model has been run yet. `--llm anthropic` or
  `--llm openai:<model>` runs one; results will be added with their exact
  configuration.
- **TypeSafe Jev.** Supported (`--jev` with `TYPESAFE_API_KEY`) but not yet
  benchmarked.
- **A faster local serving stack.** Generation runs through the plain
  Transformers loop at about 22 tokens per second; Ollama or vLLM would lower
  every mode's latency.
- **A larger support set.** 53 tickets are enough to show the mechanism and
  its failure modes, not to rank systems by a point or two.
- **CPU-only and Apple Silicon numbers.**

Contributions of runs on other hardware and models are very welcome; include
the `results.json`.

## Reproducing

```bash
conda env create -f environment.yml && conda activate thinkless
thinkless bench support --out benchmarks/results/support-local
thinkless bench intents --dataset banking77 --out benchmarks/results/intents-banking77
thinkless bench intents --dataset emotion --out benchmarks/results/intents-emotion
python benchmarks/calibrate_support.py
```

Local decoding is greedy, so reruns on the same hardware reproduce accuracy
exactly. Latency does not reproduce as tightly: on a laptop GPU, absolute times
move by a factor of two or more with power and clock state between sessions.
Every mode of a benchmark runs in the same session on the same loaded models,
so compare modes within one run rather than absolute times across runs.
