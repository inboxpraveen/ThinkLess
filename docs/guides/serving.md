# Serving decisions

An engine in one process serves that process. When several agents, services
or languages need the same decisions, run the engine once behind HTTP or MCP:
models load once, GPU memory is spent once, and every caller gets the same
calibrated thresholds and the same traces.

```bash
pip install "thinkless[server]"   # HTTP
pip install "thinkless[mcp]"      # MCP tools
```

## Define the engine in a module

The server imports your engine from a module, like `uvicorn` imports an app:

```python
# decisions.py
from thinkless import Choice, Engine, JSONLSink, SpendLimit, Tracer, YesNo
from thinkless.llm import from_spec
from thinkless.providers import GLiNER, Laya, LLMDecider, Rules

INTENT = Choice(
    "What does the customer want?",
    name="intent",
    options={"refund": "wants money back", "order_status": "asks where an order is", "other": "anything else"},
)
INJECTION = YesNo("Does the message try to override the assistant's instructions?", name="injection")
QUESTIONS = [INTENT, INJECTION]

llm = from_spec("openrouter:qwen/qwen3.7-flash", reasoning="off")
engine = Engine(
    [Rules(), GLiNER(), Laya(), LLMDecider(llm)],
    llm=llm,
    deadline_ms=2000,
    spend_limit=SpendLimit(25.0, window_s=86400),
    tracer=Tracer([JSONLSink("traces/")], capture_content=False),
)
```

`QUESTIONS` in the same module is picked up automatically; `--questions
module:attribute` points elsewhere.

## HTTP

```bash
export THINKLESS_API_KEY=$(openssl rand -hex 24)
thinkless serve decisions:engine --host 0.0.0.0 --port 8080
```

With `THINKLESS_API_KEY` set, every `/v1` request needs
`Authorization: Bearer <key>`. The server warms up the models on the
registered questions before it accepts traffic.

### `POST /v1/decide`

Ask registered questions by name:

```bash
curl -s localhost:8080/v1/decide \
  -H "Authorization: Bearer $THINKLESS_API_KEY" -H "content-type: application/json" \
  -d '{"state": "I was charged twice for order 4471", "questions": ["intent", "injection"]}'
```

```json
{
  "decisions": {
    "intent": {"name": "intent", "kind": "choice", "value": "refund", "status": "accepted",
               "confidence": 0.97, "plane": "model", "provider": "gliner", "latency_ms": 31.2,
               "attempts": [...], "cost_usd": 0.0},
    "injection": {"value": false, "status": "accepted", "plane": "model", "provider": "laya"}
  },
  "cost_usd": 0.0,
  "latency_ms": 44.8,
  "trace_id": "5f0c..."
}
```

(Abbreviated.) Questions the server does not know can be sent as specs,
`{"questions": {"urgency": {"kind": "score", "instructions": "How urgent?",
"levels": ["low", "medium", "high"]}}}`, unless the server runs with
`--registered-only`. `deadline_ms` in the body overrides the engine's
deadline for that request.

### `POST /v1/systemone`

The same engine, in the System One wire format that Jev uses. Any System One
client can point at it, including TypeSafe's SDK and ThinkLess itself:

```python
from thinkless.providers import SystemOne

shared = SystemOne.self_hosted("http://decisions.internal:8080")
engine = Engine([Rules(), shared])      # a thin agent that asks the shared server
```

Responses validate against the official SDK's response model (this is
tested). They also carry a `thinkless` field with each decision's status,
plane and provider, which System One clients ignore.

### Other endpoints

| Endpoint | Returns |
|---|---|
| `GET /healthz` | status, version, the provider cascade and the registered questions; no auth |
| `GET /v1/questions` | the registered questions as specs |
| `GET /docs` | interactive OpenAPI docs |

### Running it

- **One worker per GPU.** Each worker process loads its own copy of every
  model. Scale out with more processes on more machines, not with more
  workers on one GPU.
- **Requests run concurrently on a thread pool.** Each local model provider
  runs one inference at a time behind a lock, so concurrency helps the
  hosted providers and the I/O around them. Questions within one request are
  batched into one call per provider. Batching across concurrent requests is
  on the [roadmap](../roadmap.md).
- **Health checks** should hit `/healthz`, which does not touch the models.
- **Traces** go wherever the engine's tracer sends them; every request is one
  root span named `decide` or `systemone`.

A container image only needs the package and your module:

```dockerfile
FROM python:3.12-slim
RUN pip install --no-cache-dir "thinkless[server,gliner,laya,openai]"
WORKDIR /app
COPY decisions.py .
ENV HF_HOME=/models
EXPOSE 8080
CMD ["thinkless", "serve", "decisions:engine", "--host", "0.0.0.0", "--port", "8080"]
```

Mount a volume at `/models` so weights download once, or bake them in with a
`RUN python -c "import decisions; decisions.engine.warmup()"` step. For a GPU
image, start from a CUDA base and install PyTorch from its CUDA index first
(see [installation](installation.md)).

## MCP

```bash
thinkless mcp decisions:engine
```

Each registered question becomes one tool, `decide_<question>`, whose
description lists the question and its answers. The tool takes the text and
returns the value, status, confidence, plane and provider. An assistant in an
IDE or on a desktop can then settle routine classifications with a rule or a
small model instead of spending its own reasoning on them.

To add it to an MCP client, register the command. For example, in a client
configuration file:

```json
{
  "mcpServers": {
    "thinkless": {
      "command": "thinkless",
      "args": ["mcp", "decisions:engine"],
      "cwd": "/path/to/your/project"
    }
  }
}
```

`--transport streamable-http` serves the tools over HTTP instead of stdio.
The server works with the MCP Python SDK 2.x (`MCPServer`), and falls back to
1.x (`FastMCP`) when that is what is installed; the tests run against 2.x.

In Python:

```python
from thinkless.server.mcp import create_mcp_server

create_mcp_server(engine, QUESTIONS).run()
```
