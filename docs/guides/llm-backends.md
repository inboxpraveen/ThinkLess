# LLM backends

The reasoning plane is any object that implements `thinkless.llm.LLM`. Three
backends ship with ThinkLess, and together they cover nearly every way of
running a model.

| Backend | Class | Covers | Install |
|---|---|---|---|
| In-process | `TransformersLLM` | Any Hugging Face chat model, no server | `thinkless[local-llm]` |
| OpenRouter | `OpenRouterLLM` | Hundreds of hosted models (Anthropic, OpenAI, Google, Qwen, DeepSeek, Mistral, Meta) with one key, billed cost reported per call | `thinkless[openai]` |
| OpenAI-compatible | `OpenAICompatibleLLM` | OpenAI, Ollama, vLLM, LM Studio, llama.cpp server, Groq, Together | `thinkless[openai]` |
| Anthropic | `AnthropicLLM` | Claude models on the Claude API | `thinkless[anthropic]` |

The same backend is used for generation (`engine.generate`) and, wrapped in
`LLMDecider`, as the last provider of the decision cascade.

## Spec strings

The CLI and `thinkless.llm.from_spec()` accept `backend[:model]`:

| Spec | Result |
|---|---|
| `local` | `TransformersLLM("Qwen/Qwen3-1.7B")` |
| `local:Qwen/Qwen3-4B` | Any Hugging Face model id |
| `anthropic` | `AnthropicLLM("claude-opus-5")` |
| `anthropic:claude-haiku-4-5` | A specific Claude model |
| `openrouter:qwen/qwen3.7-flash` | Any OpenRouter model slug |
| `openrouter:anthropic/claude-haiku-4.5` | Claude through OpenRouter |
| `openai:<model>` | OpenAI's API |
| `ollama:qwen3:8b` | Ollama on `http://localhost:11434/v1` |
| `vllm:<model>` | vLLM on `http://localhost:8000/v1` |

## Reasoning

Many current models reason before they answer, and the reasoning tokens are
billed as output and count against `max_tokens`. For short decisions and
replies that is usually wasted, and on some models it is worse: in testing,
Qwen 3.7 Flash spent its entire budget reasoning and returned an empty reply.

`from_spec(..., reasoning=...)` and the CLI's `--reasoning` option set it per
backend: `off`, `minimal`, `low`, `medium`, `high`, or `default` for the model's
own default. They map to OpenRouter's `reasoning` option, Anthropic's
`effort`, and the thinking switch of local chat templates.

```bash
thinkless bench support --llm openrouter:qwen/qwen3.7-flash --reasoning off
thinkless bench support --llm openrouter:openai/gpt-5.6-luna --reasoning low
```

As a safety net, `LLMDecider` detects a reply cut off at the token limit with
answers missing, logs a warning naming the likely cause, and retries once with
three times the budget. The retry is recorded on the attempt span
(`truncated`, `retried`), so it shows up in traces rather than as silent
abstentions.

## In-process with Transformers

```python
from thinkless.llm import TransformersLLM

llm = TransformersLLM("Qwen/Qwen3-1.7B", device="auto", enable_thinking=False)
```

Decoding is greedy by default, so benchmark runs are reproducible.
Transformers normally swaps an explicit `do_sample=False` for the
checkpoint's own sampling defaults; ThinkLess passes `use_model_defaults=False`
to stop that. For Qwen3 and other models with a thinking switch, thinking is
off: decisions and short replies do not benefit from long reasoning traces,
and they cost latency.

This backend is the zero-setup option, and also the slowest: about 22 tokens
per second for Qwen3-1.7B on an RTX 5060 laptop GPU. For real throughput, serve
the same weights with Ollama, vLLM or llama.cpp and use the OpenAI-compatible
backend.

## OpenRouter

```python
from thinkless.llm import OpenRouterLLM

qwen = OpenRouterLLM("qwen/qwen3.7-flash", reasoning={"enabled": False})
luna = OpenRouterLLM("openai/gpt-5.6-luna", reasoning={"effort": "low"})
cheapest = OpenRouterLLM("google/gemini-2.5-flash-lite", provider_preferences={"sort": "price"})
```

- The key comes from `OPENROUTER_API_KEY`. The CLI reads `./.env`, so a
  line in that file is enough.
- OpenRouter reports the billed cost of every call. ThinkLess records it on
  the span with `cost_source="reported"` and uses it instead of the price
  table, so benchmark reports show what you actually paid.
- `reasoning` and `provider_preferences` pass through to OpenRouter's
  `reasoning` and `provider` request options.
- OpenRouter's key usage endpoint lags behind real spend by a few minutes;
  the per-call costs in traces are immediate.

## OpenAI-compatible servers

```python
from thinkless.llm import OpenAICompatibleLLM

ollama = OpenAICompatibleLLM("qwen3:8b", base_url="http://localhost:11434/v1", provider="ollama")
openai = OpenAICompatibleLLM("<model-id>")  # reads OPENAI_API_KEY
```

`provider` names the backend in traces and in the price table. Local servers
(`ollama`, `vllm`, `lmstudio`, `llamacpp`) are recorded at zero cost. When a
server reports cost in `usage.cost`, as OpenRouter does, that number is used.

When `LLMDecider` asks for JSON, the client sends
`response_format={"type": "json_object"}`. Some endpoints reject it; the
client then retries once without it, logs a warning, and stops sending it for
that instance, instead of failing every decision.
`token_param` picks between `max_completion_tokens` (OpenAI's current name) and
`max_tokens` (what most compatible servers accept); it is chosen from
`base_url` when omitted.

## Anthropic

```python
from thinkless.llm import AnthropicLLM

claude = AnthropicLLM("claude-sonnet-5", effort="low")
```

- Credentials resolve through the SDK: `ANTHROPIC_API_KEY`,
  `ANTHROPIC_AUTH_TOKEN`, or a profile from `ant auth login`.
- Sampling parameters are never sent. Current Claude models reject
  `temperature`, and control depth with `effort` instead. `low` suits short
  replies and decision questions. `effort` is supported on Opus 4.6 and later,
  Sonnet 5 and Fable; Haiku 4.5 and Sonnet 4.5 reject it, so leave it unset for
  those models.
- Server-side refusal fallbacks (`fallbacks="default"` with the
  `server-side-fallback-2026-07-01` beta) re-run a request that a safety
  classifier declines on the recommended fallback model, within the same call.
  They are on by default for the model families that document them (Opus 5 and
  later, Fable 5, Mythos 5) and off for others. Pass `fallbacks=False` when
  routing through Amazon Bedrock, Google Vertex AI or Microsoft Foundry.
- A response with `stop_reason == "refusal"` returns empty text, so it is never
  mistaken for an answer.

## Test status

| Backend | Unit tests | Live runs |
|---|---|---|
| `TransformersLLM` | yes | all local benchmarks |
| `OpenRouterLLM` | yes | support and Banking77 benchmarks with Qwen, Gemini, GPT and Claude models |
| `OpenAICompatibleLLM` | yes | through `OpenRouterLLM`, which is a thin subclass |
| `AnthropicLLM` (direct SDK) | yes, against a mocked client | not yet; Claude has been run through OpenRouter |
| `SystemOne` (Jev) | yes, and the wire format is checked against TypeSafe's own SDK models | not yet, no key; please report runs |

## Writing a backend

```python
from thinkless.llm import LLM, Completion
from thinkless import Usage


class MyLLM(LLM):
    provider = "mycompany"
    model = "house-model-v2"

    def complete(self, messages, *, system=None, max_tokens=512, temperature=None, json_mode=False):
        text, tokens_in, tokens_out = my_client.chat(messages, system=system, max_tokens=max_tokens)
        return Completion(text=text, model=self.model, usage=Usage(input_tokens=tokens_in, output_tokens=tokens_out))
```

Add a price entry with `provider = "mycompany"` to your pricing file to have
its cost show up in traces.
