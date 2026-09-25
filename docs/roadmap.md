# Roadmap

ThinkLess is at 0.1. The core API (questions, the engine, decisions, traces)
is intended to stay stable; providers and benchmarks will grow. Plans change
with what people build, so open a discussion if something here matters to
you, or if something that is not here should be.

## Next

- **Shadow mode.** Run a second provider on a sample of live decisions in the
  background and record agreement, so a team can measure a small model
  against its current LLM on real traffic before switching.
- **Calibration from traces.** Export the decisions of one question from a
  trace directory as a labeling file, and feed labeled traces back into
  `thinkless calibrate`.
- **Calibrated LLM confidence.** Several hosted models expose token log
  probabilities; an `LLMDecider` that reads them would give LLM answers a real
  confidence, so they could be thresholded like every other provider.
- **Jev and the direct Anthropic SDK path in the published benchmarks.** Both
  are unit tested (Jev against TypeSafe's own SDK models) but have not been
  benchmarked live.

## Later

- Framework adapters for LangGraph, Pydantic AI and the OpenAI Agents SDK, so
  existing agents can route their decisions through an engine without a
  rewrite.
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
