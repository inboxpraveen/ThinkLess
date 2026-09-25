# Python API

The public API is everything importable from `thinkless`,
`thinkless.providers`, `thinkless.llm`, `thinkless.tracing`,
`thinkless.shadow`, `thinkless.integrations` and `thinkless.server`. Anything
with a leading underscore is internal and may change.

## Engine

::: thinkless.Engine

::: thinkless.Run

## Questions

::: thinkless.Choice

::: thinkless.Score

::: thinkless.YesNo

::: thinkless.Extract

::: thinkless.questions.question_from_spec

## Results

::: thinkless.Decision

::: thinkless.Attempt

::: thinkless.Status

::: thinkless.Plane

## Limits

::: thinkless.SpendLimit

::: thinkless.SpendLimitError

## Shadow mode

::: thinkless.shadow.Shadow

::: thinkless.shadow.ShadowStats

::: thinkless.shadow.build_report

::: thinkless.shadow.ShadowReport

::: thinkless.shadow.QuestionReport

::: thinkless.shadow.export_labels

::: thinkless.shadow.agree

## Integrations

::: thinkless.integrations.Router

::: thinkless.integrations.route

::: thinkless.integrations.gate

::: thinkless.integrations.ToolBlockedError

::: thinkless.integrations.last_user_text

::: thinkless.integrations.langgraph.router

::: thinkless.integrations.langgraph.decision_node

::: thinkless.integrations.langgraph.adecision_node

::: thinkless.integrations.openai_agents.input_guardrail

::: thinkless.integrations.openai_agents.tool_input_guardrail

::: thinkless.integrations.openai_agents.route_agent

## Serving

::: thinkless.server.app.create_app

::: thinkless.server.mcp.create_mcp_server

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

::: thinkless.tracing.export.iter_decisions

::: thinkless.tracing.export.export_trace_labels

::: thinkless.tracing.export.drift_report

## Confidence and pricing

::: thinkless.confidence

::: thinkless.pricing.PriceTable
