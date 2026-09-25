# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

## Unreleased

### Added

- `OpenRouterLLM` and the `openrouter:<slug>` spec: one key for models from
  Anthropic, OpenAI, Google, Qwen, DeepSeek and others.
- Billed cost: `Completion.cost_usd` and `ProviderResult.cost_usd` carry the
  cost a backend reports (OpenRouter does); the engine prefers it over the
  price table and records `cost_source` on spans. Benchmark reports add a
  billed cost row, and the intent benchmark reports cost per 1k per provider
  and per cascade threshold.
- `--reasoning` for every CLI command that builds an LLM, and
  `from_spec(..., reasoning=...)`, mapped to OpenRouter's `reasoning`,
  Anthropic's `effort` and local thinking switches.
- `HFClassifier`: any Hugging Face text-classification model as a provider for
  the questions it was trained for.
- `load_env()`, and the CLI reads `./.env` without overriding set variables.
- Escalation context: when a question escalates to the LLM decider, the
  engine includes the decisions already settled in the same batch. It removed
  false injection flags on requests for a human that appeared when an LLM saw
  the injection question alone. On by default (`Engine(escalation_context=...)`).
- The System One wire format is tested against TypeSafe's official SDK models.

### Fixed

- `LLMDecider` detects replies cut off at the token limit, logs why, and
  retries once with a larger budget instead of abstaining silently.
- The OpenAI-compatible client retries once without `response_format` when an
  endpoint rejects JSON mode.
- A redaction test that failed about one run in fifty when a random span id
  contained its marker.

## 0.1.0 - 2026-09-25

First public release.

### Added

- Typed questions: `Choice`, `Score`, `YesNo` and `Extract`, following the
  System One wire format.
- `Engine` with a confidence cascade across providers, batched provider calls,
  `accepted`, `uncertain` and `abstained` statuses, per-question and
  per-provider thresholds (`"question@provider"`), question-level provider
  allowlists, and async variants of every call.
- Normalized confidence, `(k * p_max - 1) / (k - 1)`, applied uniformly across
  providers.
- Decision providers: `Rules`, `GLiNER` (GLiNER 2.5), `Laya`, `SystemOne`
  (TypeSafe Jev and compatible servers such as Kev and OpenJev) and
  `LLMDecider`.
- LLM backends: `TransformersLLM`, `OpenAICompatibleLLM` (OpenAI, Ollama, vLLM
  and others) and `AnthropicLLM` with server-side refusal fallbacks.
- Tracing: spans for runs, steps, decisions, provider attempts, generations,
  tools and rules; JSONL, memory, console and OpenTelemetry sinks; run
  summaries with decisions by plane, escalations, tokens, cost and time by
  plane; content capture switch; a self-contained HTML viewer.
- A price table with sources, overridable with `THINKLESS_PRICING`.
- The support demo: an agent, a mock store, 53 labeled tickets and a separate
  48-row calibration set.
- Benchmarks: the support benchmark across `llm`, `hybrid` and `models` modes,
  and intent benchmarks on Banking77, CLINC150 and Emotion with a simulated
  cascade.
- `thinkless` CLI: `doctor`, `demo`, `bench support`, `bench intents`,
  `calibrate`, `trace ls`, `trace show` and `trace view`.
