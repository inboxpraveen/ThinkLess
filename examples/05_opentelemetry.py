"""Send ThinkLess traces to OpenTelemetry.

Prints spans to the console here; replace ConsoleSpanExporter with
OTLPSpanExporter to ship them to Jaeger, Tempo, Honeycomb, Langfuse, Phoenix
or any other OTLP backend.

    pip install "thinkless[otel]"
    python examples/05_opentelemetry.py
"""

from __future__ import annotations

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from thinkless import Engine, Tracer, YesNo
from thinkless.providers import Rules
from thinkless.tracing.otel import OTelSink

provider = TracerProvider(resource=Resource.create({"service.name": "support-agent"}))
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

rules = Rules()
rules.match("urgent", r"\b(outage|down|urgent|asap)\b", True)
rules.match("urgent", r".", False)

engine = Engine([rules], tracer=Tracer([OTelSink(provider)]))
with engine.run("alert_triage", team="sre"):
    decision = engine.decide(
        "The checkout service is down in eu-west", YesNo("Is it urgent?", name="urgent")
    )
print("urgent:", decision.value)
