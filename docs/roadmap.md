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

## Done in 0.4

- The [decision benchmark](decision-benchmark/index.md) 1.0: eight public
  tasks across the four question kinds, frozen rows with published hashes,
  result files with every prediction, recomputed metrics and open
  submissions.

## Done in 0.5

- [Habits](habits.md): one line under the OpenAI or Anthropic SDK (or
  an OpenAI-compatible proxy for any language) that records an agent's LLM
  calls, finds the decision sites, trains local students from the LLM's own
  answers, answers a site only after a lower confidence bound on agreement
  clears the target on past calls and again on a probation of new calls,
  audits itself and suspends a site whose agreement drops.
- `thinkless habits scan` reports which calls in an existing log are
  decisions and what they cost.

## Next

- **Pretrained students on day one.** GLiNER, Laya, Jev and any provider as
  Habits students, gated the same way, so a site can be answered before it
  has seen hundreds of calls.
- **Stronger students.** An optional embedding-based student for sites
  where hashed n-grams fall short, chosen per site by the same held-out
  measurement.
- **More of the wire.** Local answers for streamed Responses and Messages
  calls, the Gemini API, and Bedrock.
- **From a site to code.** Generate the engine and questions for a proven
  site, for teams that want rules and hosted decision models in the loop.

- **More models on the decision benchmark.** Jev and other hosted decision
  models, frontier LLMs as references, and Hugging Face classifiers.
  Version 1.0 has eight tasks and open submissions; later versions add
  multi-turn safety cases, more languages and harder out-of-scope sets.
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
