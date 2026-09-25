"""ThinkLess inside an async web service.

One engine per process, models warmed up at startup, async calls that run on
worker threads, and a trace file per request.

    pip install "thinkless[gliner,laya]" fastapi uvicorn
    uvicorn examples.06_fastapi_service:app --port 8080
    curl -X POST localhost:8080/triage -H "content-type: application/json" \
         -d '{"message": "I was charged twice for order #4471"}'
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from thinkless import Choice, Engine, JSONLSink, Tracer, YesNo
from thinkless.providers import GLiNER, Laya, Rules

INTENT = Choice(
    "What does the customer want?",
    name="intent",
    options={
        "refund": "wants money back",
        "order_status": "asks about an order",
        "cancel": "wants to cancel a subscription",
        "other": "anything else",
    },
)
WANTS_HUMAN = YesNo("Does the customer ask for a person?", name="wants_human", threshold=0.5)

rules = Rules()
rules.match("wants_human", r"\b(real person|a human|agent|operator)\b", True, field="message")
engine = Engine(
    [rules, GLiNER(), Laya()],
    threshold=0.8,
    tracer=Tracer([JSONLSink(".thinkless/traces/service")], capture_content=False),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.warmup()
    yield
    engine.close()


app = FastAPI(lifespan=lifespan)


class Ticket(BaseModel):
    message: str


@app.post("/triage")
async def triage(ticket: Ticket) -> dict:
    with engine.run("triage") as run:
        decisions = await engine.adecide_many({"message": ticket.message}, [INTENT, WANTS_HUMAN])
    return {
        name: {
            "value": d.value,
            "status": d.status.value,
            "confidence": d.confidence,
            "provider": d.provider,
        }
        for name, d in decisions.items()
    } | {"trace_id": run.trace_id}
