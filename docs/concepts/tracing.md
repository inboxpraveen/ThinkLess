# Tracing

A ThinkLess trace records one unit of work, such as a support ticket or an
API request, as a tree of spans. It is the audit record of what the agent
decided, which plane decided it, how confident it was, what it cost and where
the time went.

## Spans

| Kind | Created by | Plane | What it records |
|---|---|---|---|
| `run` | `engine.run(name)` | | The root. Attributes you pass, such as `mode` or `ticket_id`, plus the input when content capture is on |
| `step` | `engine.step(name)` | | A named phase of your workflow |
| `decide` | `decide()`, `decide_many()`, `extract()` | | The questions, their thresholds, and every resulting decision |
| `attempt` | the cascade | rule, model or llm | One provider call: questions asked, answers, confidences, which were accepted, model, tokens, cost |
| `llm` | `engine.generate()` | llm | Model, tokens, cost, stop reason, prompt and completion |
| `tool` | `@thinkless.tool` | tool | Arguments and result |
| `rule` | `engine.rule(name, value)` | rule | A deterministic check made in application code |

Span ids are W3C Trace Context compatible (128-bit trace ids, 64-bit span
ids), so they map directly onto OpenTelemetry.

## Setting up a tracer

```python
from thinkless import ConsoleSink, Engine, JSONLSink, Tracer

tracer = Tracer(
    sinks=[JSONLSink(".thinkless/traces"), ConsoleSink()],
    capture_content=True,
)
engine = Engine(providers, llm=llm, tracer=tracer)

with engine.run("support_ticket", ticket_id=ticket["id"], mode="hybrid") as run:
    ...
print(run.summary())
```

`engine.run()` returns a handle whose `summary()` gives a `TraceSummary` for
that run: LLM calls, decisions by plane, escalations, tokens, cost and time by
plane.

## Sinks

| Sink | Purpose |
|---|---|
| `JSONLSink(dir)` | One JSON Lines file per trace, named `<UTC time>_<run name>_<trace id>.jsonl`. The root span is always the last line. This is the audit log. |
| `MemorySink()` | Keeps spans in memory; used by tests and benchmarks. |
| `ConsoleSink()` | Prints each finished trace as a tree in the terminal. |
| `OTelSink(tracer_provider)` | Mirrors spans into OpenTelemetry. Requires the `otel` extra. |

A sink is any object with optional `on_start(span)`, `on_end(span)`, `flush()`
and `close()` methods. Sink failures are logged and never reach application
code: tracing must not take the agent down.

## Content capture and privacy

With `capture_content=True` (the default, `THINKLESS_CAPTURE_CONTENT`), traces
include the run input, decision states, prompts, completions, extracted values
and tool arguments and results. That is what makes a trace useful for
debugging, and also what makes it sensitive.

With `capture_content=False`, those fields are replaced by `[redacted]`, while
structure, timings, labels, confidences and token counts are kept. Choice
labels and yes/no values come from your question definitions, not from user
input, so they are always recorded; extracted field values are treated as user
content.

`OTelSink` drops content attributes by default even when the tracer captures
them, because telemetry backends are often shared more widely than local audit
files. Pass `include_content=True` to export them.

## OpenTelemetry

```python
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from thinkless import Tracer
from thinkless.tracing.otel import OTelSink

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
tracer = Tracer([OTelSink(provider)])
```

Spans are named `<kind> <name>`, carry their ThinkLess fields as
`thinkless.*` attributes, and generation spans carry the GenAI semantic
convention attributes (`gen_ai.operation.name`, `gen_ai.request.model`,
`gen_ai.provider.name`, `gen_ai.usage.input_tokens`,
`gen_ai.usage.output_tokens`). Any OTLP backend works: Jaeger, Grafana Tempo,
Honeycomb, Langfuse, Arize Phoenix and others.

## The viewer

```bash
thinkless trace view                  # every trace under THINKLESS_TRACE_DIR
thinkless trace view path/to/traces --out report.html --no-open
```

The viewer is a single HTML file with the data embedded and no external
requests. It shows a per-mode comparison when traces carry a `mode`
attribute, a filterable run list, and for each run a waterfall colored by
plane with the decisions of every `decide` span inline. Clicking a span shows
its attributes, prompt and completion.

From the terminal:

```bash
thinkless trace ls
thinkless trace show .thinkless/traces/demo/20260925T044421Z_support_ticket_1a2b3c4d.jsonl
```

## Cost

Attempt and generation spans carry `cost_usd`, computed from token usage and a
price table (`thinkless/data/pricing.toml`, overridable with
`THINKLESS_PRICING`). Local models cost 0. Hosted models without a price entry
are recorded with `cost_known=false` rather than a guessed number. Prices
change; the bundled table lists its source and date for every entry.
