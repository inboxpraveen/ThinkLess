# Roadmap

The core API (questions, the engine, decisions, traces) is intended to stay
stable; providers, adapters and benchmarks will grow. Plans change with what
people build, so open a discussion if something here matters to you, or if
something that is not here should be.

## Done in 0.3

- Shadow mode with agreement, projected savings, fitted thresholds and a
  verdict per question, and an audit mode for live engines.
- Adapters for LangGraph and the OpenAI Agents SDK, and framework-neutral
  `route` and `gate` helpers (tested with LangChain tools and Pydantic AI).
- Deadlines and spend limits on the engine, and per-run spend caps.
- A decision server with a System One compatible endpoint, and MCP tools.
- Traces to labeling rows, and a drift report between two periods.

## Next

- **A neutral decision-model benchmark.** A public, provider-neutral
  benchmark for decision models (Jev, Kev, OpenJev, Laya, GLiNER, Hugging
  Face classifiers, LLM deciders) on public multi-domain datasets, with
  multi-turn cases, that anyone can submit a model to.
- **Cross-request batching in the server.** Collect the questions of
  concurrent requests for a few milliseconds and send them to each local
  model as one batch, which needs a batch entry point on providers.
- **Calibrated LLM confidence.** Several hosted models expose token log
  probabilities; an `LLMDecider` that reads them would give LLM answers a real
  confidence, so they could be thresholded like every other provider.
- **Jev and the direct Anthropic SDK path in the published benchmarks.** Both
  are unit tested (Jev against TypeSafe's own SDK models) but have not been
  benchmarked live.

## Later

- A trace server with search across runs, for teams that outgrow single HTML
  files.
- Learned routing: pick the provider order per question from calibration data
  instead of by hand.
- More demos: document processing and an IT operations agent.

## Not planned

- A general agent framework. ThinkLess is a decision plane and a tracer that
  work inside whatever loop you already have.
- Hosting models. Providers call models; running them is left to the tools
  that do it well.
