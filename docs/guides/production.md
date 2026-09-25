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

**Keep escalation context on.** An escalated question reaches the LLM without
its siblings unless the engine passes the settled ones along, and that can
flip answers (see [escalation context](../concepts/cascade.md#escalation-context)).
It is on by default.

**Order the cascade by cost, then check it by trace.** Rules first, local
models next, hosted decision models after, the LLM last. Then read a day of
traces: a provider that escalates nearly every time it is asked is costing
latency without saving anything, and belongs off that question
(`providers=(...)` on the question).

**Try purpose-built models, and measure them on your traffic.** A classifier
trained for one narrow question can beat a general decision model, but a model
named for your problem is not automatically trained on your problem. In our
tests, `protectai/deberta-v3-base-prompt-injection-v2`, a widely used prompt
injection classifier, flagged ordinary support tickets (a card form bug, a
promo code complaint, messages in Spanish and German) as injections with full
confidence, and missed a policy-override attempt. It was trained on jailbreaks
aimed at LLM apps, not on customer messages. `HFClassifier` makes this kind of
comparison a few lines, and `thinkless calibrate` shows whether any threshold
makes the model safe to use. Mind GPU memory when adding models.

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

**Serve one engine to many agents.** `thinkless serve` puts an engine behind
HTTP, with a System One compatible endpoint, and `thinkless mcp` exposes its
questions as MCP tools; see [serving](serving.md).

**Pin versions.** GLiNER2 requires `transformers<5`. Model weights on the
Hugging Face Hub can change under a moving revision; pin revisions for
reproducible behavior.

## Limits

**Give every request a deadline.** `Engine(deadline_ms=...)`, or
`decide(..., deadline_ms=...)` per call, bounds the time a decision may take.
The deadline is checked before each provider: once it has passed, the rest of
the cascade is skipped and open questions come back `uncertain` with the best
answer so far. A provider call already running is not interrupted, so give
hosted clients their own timeout as well.

**Cap what the engine may spend.** `SpendLimit` is a dollar budget over the
engine's lifetime or a rolling window:

```python
from thinkless import Engine, SpendLimit

engine = Engine(providers, llm=llm, spend_limit=SpendLimit(50.0, window_s=86400))

with engine.run("ticket", max_cost_usd=0.02):   # and a cap for one run
    ...
```

Past a limit, paid providers are skipped (their attempts are recorded with
reason `spend_limit`), rules and local models keep answering, and
`engine.generate` raises `SpendLimitError`. The check runs before each paid
call, so the call that crosses the limit completes. `limit.spent_usd` and
`limit.remaining_usd` feed a dashboard.

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

**Check for drift on a schedule.** Traffic changes, and confidence
distributions move with it. `thinkless trace drift` compares two periods of
traces question by question: the share reaching the LLM, the share not
accepted, the answer distribution and the mean confidence. It exits with
status 1 when a question crosses a threshold, so a nightly job can page
someone:

```bash
thinkless trace drift --baseline traces/2026-09 --current traces/2026-10
```

**Turn traces into labels.** `thinkless trace export` writes traced decisions
as rows in the format `thinkless calibrate` reads. Export the uncertain and
escalated ones, label them, and recalibrate:

```bash
thinkless trace export traces/2026-10 --out to_label.jsonl --status uncertain --limit 500
```

Both need traces written with content capture on, at least for a sample.

**Log through the `thinkless` logger.** The library never configures logging
itself. `thinkless.configure_logging(json_format=True)` emits one JSON object
per line for log shippers.

## Rolling out

**Start in shadow.** [Shadow mode](shadow-mode.md) runs the engine next to
your current code on live traffic, changes nothing it returns, and reports
agreement, projected savings and a verdict per question. Switch a question
over once its verdict is ready, and keep a small audit shadow running
afterwards. The [migration guide](migration.md) walks through it.

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
