# Running ThinkLess in production

A checklist, roughly in the order problems show up.

## Before launch

**Calibrate every question on real traffic.** The bundled thresholds come
from small sets. Label a few hundred examples per question and run
`thinkless calibrate`; see the [calibration guide](calibration.md). Fix
the accuracy target per question before looking at results.

**Put a deterministic guard on every consequential action.** A decision model
choosing `refund` should never be the only thing between a message and a
refund. In the support demo, code checks that the order belongs to the
customer, that a duplicate charge actually exists in the payment records,
and that the amount is under the automatic limit. The models decide what the
customer wants; code decides whether it is allowed.

**Decide what uncertainty means for each question.** `decision.is_(value)`
never matches an uncertain answer. For safety questions, handle the unsure
case explicitly and fail closed: the demo blocks automatic refunds when the
injection check leans yes without reaching its threshold.

**Order the cascade by cost, then check it by trace.** Rules first, local
models next, hosted decision models after, the LLM last. Then read a day of
traces: a provider that escalates nearly every time it is asked is costing
latency without saving anything, and belongs off that question
(`providers=(...)` on the question).

**Use purpose-built models for narrow, high-stakes questions.** General
decision models are convenient, not specialized. Prompt injection detection is
the clearest case: a dedicated classifier such as
`protectai/deberta-v3-base-prompt-injection-v2` (Apache 2.0) wrapped as a
[custom provider](providers.md#writing-a-provider) will outperform a general
yes/no model. Mind GPU memory when adding models.

## Deployment

**Warm up at startup, with your own questions.** Call
`engine.warmup(QUESTIONS)` before accepting traffic. It loads every model (10
to 60 seconds for the local ones) and then has each non-LLM provider answer
your actual questions twice, so GPU kernels are specialized for the real input
shapes. On the support demo this took GLiNER's first request from 650 ms to
the same 130 ms as every later one. LLM providers are never called during
warmup.

**Share one engine per process.** Providers load weights once and serialize
GPU access with a lock. Build the engine at startup and reuse it across
requests. In async services use `adecide` and `agenerate`, which run on
worker threads.

**Size the GPU.** Measured peak allocation with Laya, GLiNER base and
Qwen3-1.7B loaded together: 6.7 GB. On CPU-only hosts, GLiNER stays fast
(about 85 ms) while Laya slows to about 650 ms per call.

**Pin versions.** GLiNER2 requires `transformers<5`. Model weights on the
Hugging Face Hub can change under a moving revision; pin revisions for
reproducible behavior.

## Observability

**Keep the JSONL audit trail.** One file per run, root span last. It records
which plane made every decision, with confidences and thresholds, which is
what you need when someone asks why the agent did something.

**Decide on content capture deliberately.** `THINKLESS_CAPTURE_CONTENT=false`
keeps structure, timings, labels and confidences while redacting inputs,
prompts, completions, extracted values and tool arguments. The OpenTelemetry
sink drops content by default regardless.

**Export to your tracing backend.** `OTelSink` maps spans onto OpenTelemetry
with GenAI semantic convention attributes. Alert on the numbers that drift:
escalation rate per question, share of decisions reaching the LLM, and
uncertain decisions per hour.

**Log through the `thinkless` logger.** The library never configures logging
itself. `thinkless.configure_logging(json_format=True)` emits one JSON object
per line for log shippers.

## Rolling out

**Start in shadow.** Run the hybrid engine next to your current agent on the
same inputs, compare decisions from the traces, and switch once agreement on
the questions that matter is where you need it. The `llm` mode of the support
benchmark is exactly this comparison, run offline.

**Watch the tail, not the mean.** Escalations are where latency hides: a
ticket that escalates one question pays for a full LLM call. Track p95
latency per mode, not only the average.

**Recalibrate on change.** New question wording, a new model version, or a
shift in traffic all move confidence distributions.

## Security

**Treat all input as untrusted.** Questions constrain what a provider can
answer, which removes whole classes of injection that free-form prompting
allows, but the LLM in the reasoning plane still reads user text. Keep facts
and instructions separate in generation prompts, and never let a generated
reply authorize an action.

**Check generated text before sending it.** The demo's grounding rule rejects a
reply that mentions a money amount absent from the facts, or an identifier
that appears in neither the facts nor the customer's message, and falls back
to a safe template.

**Keep secrets out of traces.** API keys never appear in spans, but tool
arguments and results do when content capture is on. Redact at the tool, or
turn capture off.
