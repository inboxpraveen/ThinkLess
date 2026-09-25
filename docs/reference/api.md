# Python API

The public API is everything importable from `thinkless`,
`thinkless.providers`, `thinkless.llm` and `thinkless.tracing`. Anything with
a leading underscore is internal and may change.

## Engine

::: thinkless.Engine

::: thinkless.Run

## Questions

::: thinkless.Choice

::: thinkless.Score

::: thinkless.YesNo

::: thinkless.Extract

## Results

::: thinkless.Decision

::: thinkless.Attempt

::: thinkless.Status

::: thinkless.Plane

## Providers

::: thinkless.providers.DecisionProvider

::: thinkless.providers.ProviderResult

::: thinkless.providers.Rules

::: thinkless.providers.gliner.GLiNER

::: thinkless.providers.laya.Laya

::: thinkless.providers.SystemOne

::: thinkless.providers.LLMDecider

## LLM backends

::: thinkless.llm.LLM

::: thinkless.llm.Completion

::: thinkless.llm.local.TransformersLLM

::: thinkless.llm.openai_compat.OpenAICompatibleLLM

::: thinkless.llm.anthropic.AnthropicLLM

::: thinkless.llm.ScriptedLLM

::: thinkless.llm.from_spec

## Tracing

::: thinkless.Tracer

::: thinkless.tracing.Span

::: thinkless.tracing.TraceSummary

::: thinkless.tracing.summarize

::: thinkless.tracing.JSONLSink

::: thinkless.tracing.MemorySink

::: thinkless.tracing.ConsoleSink

::: thinkless.tracing.otel.OTelSink

::: thinkless.tool

## Confidence and pricing

::: thinkless.confidence

::: thinkless.pricing.PriceTable
