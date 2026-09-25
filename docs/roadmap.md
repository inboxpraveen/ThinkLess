# Roadmap

ThinkLess is at 0.1. The core API (questions, the engine, decisions, traces)
is intended to stay stable; providers and benchmarks will grow. Plans change
with what people build, so open a discussion if something here matters to
you, or if something that is not here should be.

## Next

- **Shadow mode.** Run a second provider on a sample of live decisions in the
  background and record agreement, so a team can measure a small model
  against its current LLM on real traffic before switching.
- **Hugging Face classifier provider.** Wrap any `text-classification`
  pipeline as a provider, mapping labels to choice options or yes/no, so a
  fine-tuned classifier (prompt injection, toxicity, a house intent model)
  drops into the cascade without custom code.
- **Calibration from traces.** Export the decisions of one question from a
  trace directory as a labeling file, and feed labeled traces back into
  `thinkless calibrate`.
- **Frontier baselines in the published benchmarks.** The current results use
  a local 1.7B reasoning model so anyone can reproduce them offline. Runs with
  hosted models, with their exact configuration, will be added next to them.

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
