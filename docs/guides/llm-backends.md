# LLM backends

The reasoning plane is any object that implements `thinkless.llm.LLM`. Three
backends ship with ThinkLess, and together they cover nearly every way of
running a model.

| Backend | Class | Covers | Install |
|---|---|---|---|
| In-process | `TransformersLLM` | Any Hugging Face chat model, no server | `thinkless[local-llm]` |
| OpenAI-compatible | `OpenAICompatibleLLM` | OpenAI, Ollama, vLLM, LM Studio, llama.cpp server, OpenRouter, Groq, Together | `thinkless[openai]` |
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
| `openai:<model>` | OpenAI's API |
| `ollama:qwen3:8b` | Ollama on `http://localhost:11434/v1` |
| `vllm:<model>` | vLLM on `http://localhost:8000/v1` |

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

## OpenAI-compatible servers

```python
from thinkless.llm import OpenAICompatibleLLM

ollama = OpenAICompatibleLLM("qwen3:8b", base_url="http://localhost:11434/v1", provider="ollama")
openai = OpenAICompatibleLLM("<model-id>")  # reads OPENAI_API_KEY
```

`provider` names the backend in traces and in the price table. Local servers
(`ollama`, `vllm`, `lmstudio`, `llamacpp`) are recorded at zero cost.
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

The hosted backends (`AnthropicLLM`, `OpenAICompatibleLLM` and the
`SystemOne` provider for Jev) are covered by unit tests against mocked clients
that check request shape, parsing, retries and error handling. The published
benchmarks were produced offline with the local backend; runs against the
live hosted APIs will be added with their configuration. Reports from your own
runs are welcome.

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
