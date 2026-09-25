# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

## Unreleased

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
