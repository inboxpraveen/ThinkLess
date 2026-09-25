# FAQ

## Adopting it

**Do I have to rewrite my agent?**
No. ThinkLess answers the decisions inside the agent and leaves the agent
itself alone. The adapters for [LangGraph](integrations/langgraph.md) and the
[OpenAI Agents SDK](integrations/openai-agents.md) return the framework's own
objects (edge functions, nodes, guardrails, agents), and
[`route` and `gate`](integrations/any-framework.md) work anywhere else. The
[migration guide](guides/migration.md) moves one decision at a time.

**Which framework adapters are there?**
LangGraph and the OpenAI Agents SDK, plus framework-neutral helpers that are
tested with LangChain tools and Pydantic AI. The helpers wrap plain Python
functions, so they work in CrewAI, LlamaIndex, AutoGen or a hand-written loop
as long as the framework builds tool schemas from the function signature.

**What if ThinkLess gets a decision wrong?**
Each decision carries a confidence and a status. Below the threshold, the
decision goes to the next provider and finally to the LLM or to your existing
code, and `Decision.is_()` is false for anything not accepted. The remaining
risk is a confident wrong answer; that is what calibration on your own
labeled data (the threshold is set so accepted answers reach a target
accuracy) and the [audit shadow](guides/shadow-mode.md#audit-a-live-engine)
are for. For actions with consequences, keep a deterministic check in code.

**How do I know it will save money on my traffic?**
Run [shadow mode](guides/shadow-mode.md) for a few days. The report gives the
share of decisions settled without an LLM, the agreement with your current
system with a confidence interval, and the cost per 1,000 decisions for both.
It uses your traffic and your bill, not our benchmark.

**How much does it save?**
It depends on how much of your traffic is routine. On the bundled support
benchmark, the hybrid design cost 40 to 44 percent less than the LLM-only
design on four hosted models and was as accurate or more
([benchmarks](benchmarks.md)). The saving is roughly the share of decisions
the fast planes settle; shadow mode measures that share for you.

**Does it add latency?**
Rules answer in microseconds. On the Banking77 benchmark, GLiNER's median
was 33 ms and Laya's 51 ms on a laptop GPU. A decision that escalates pays
that on top of the LLM call, which is why the thresholds matter. On the support benchmark, decision time per ticket fell by
35 to 55 percent, because most decisions never reached the LLM.

## Models and hosting

**Do I need a GPU?**
No. Rules, hosted decision models (Jev, any System One server) and hosted
LLMs need nothing local. Local models run on a CPU too: GLiNER answered a
two-question batch in about 60 ms on a laptop CPU after warmup. A GPU helps
at high volume.

**Which models can answer decisions?**
Rules, GLiNER 2.5, Laya, Jev and other System One models, any Hugging Face
text-classification model through `HFClassifier`, and any LLM through
`LLMDecider`: local, OpenRouter, OpenAI, Anthropic, Ollama or vLLM. Custom
providers are a class with one method; see [providers](guides/providers.md).

**Can I use it without any LLM?**
Yes. An engine of rules and small models returns `uncertain` for what it
cannot settle, and your code decides what happens next: ask a person, use a
default, or call the LLM path you already have.

**Does my data leave my machines?**
Only for the providers you configure. Rules and local models run in process.
Traces are written where you point them, and `Tracer(capture_content=False)`
keeps inputs out of traces and shadow logs.

## Shadow mode

**Does shadow mode slow down my requests?**
No. The candidate runs on background threads, capped by `max_pending`; when
the queue is full, calls are dropped and counted rather than waited on.

**What does shadow mode cost?**
Whatever the candidate's providers cost. Rules and local models are free; the
candidate's own LLM calls are billed. Use `sample` and `max_cost_usd` on high
volume traffic.

**Agreement is 92%. Is ThinkLess wrong 8% of the time?**
Not necessarily. Agreement compares two systems, and the current one makes
mistakes too. Export the disagreements and read them: some are ThinkLess
errors to fix, some are errors in the current system that ThinkLess got
right.

## Operations

**How is it traced?**
Every decision is a span with its plane, provider, confidence, latency,
tokens and cost, written to JSONL files, the console, or OpenTelemetry.
`thinkless trace view` opens a local viewer. See [tracing](concepts/tracing.md).

**Is it thread-safe? Does it work with async code?**
One engine can serve many threads; each local model provider runs one
inference at a time behind a lock. `adecide` and `adecide_many` run on worker
threads for async code, and the adapters have async forms.

**Which Python versions and platforms?**
Python 3.10 to 3.13 on Linux, macOS and Windows. See
[installation](guides/installation.md).
