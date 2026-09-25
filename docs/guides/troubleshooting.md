# Troubleshooting

Start with `thinkless doctor`. It prints the Python and torch versions, the
accelerator, which optional backends are installed, which API keys are set and
the active settings.

## Installation

**`ModuleNotFoundError` for `laya`, `gliner2`, `transformers`, `openai` or
`anthropic`.** Local models and hosted backends are optional extras. Install
what you use, for example `pip install "thinkless[gliner,laya,local-llm]"` or
`pip install "thinkless[all]"`.

**`pip` wants to install `transformers` 5.** GLiNER2 requires
`transformers<5`, and every ThinkLess local extra shares that bound. If
another package in your environment needs `transformers>=5`, give ThinkLess's
local models their own environment.

**CUDA is not available on an RTX 50 series card.** Blackwell GPUs need
PyTorch built for CUDA 12.8 or newer. Install from the matching index, for
example `pip install torch --index-url https://download.pytorch.org/whl/cu130`.
`environment.yml` does this for you.

**`requires the protobuf library` when loading GLiNER.** The DeBERTa tokenizer
needs `protobuf` and `sentencepiece`. The `gliner` extra installs both;
install them manually if you installed `gliner2` on its own.

## First model download

**`OSError: [WinError 1314] A required privilege is not held by the client`.**
The Hugging Face cache uses symbolic links, and on Windows a parallel download
can hit a race in the symlink check. Run the command again; the partial
download is reused. Downloading serially avoids the race:
`huggingface_hub.snapshot_download("<model>", max_workers=1)`. Enabling Windows
Developer Mode, which allows symlinks without administrator rights, avoids it
entirely.

**The first run is slow.** Models download on first use: about 0.9 GB for
Laya, 0.8 GB for GLiNER base and 3.4 GB for Qwen3-1.7B. Later runs load from
the cache in seconds.

## Running

**Out of memory on the GPU.** Laya, GLiNER base and Qwen3-1.7B peak at about
6.7 GB together. On smaller cards, move the decision models to the CPU
(`GLiNER(device="cpu")` stays fast), or serve the LLM from another machine
through an OpenAI-compatible server.

**Everything is suddenly ten times slower on Windows.** Another process is
probably holding GPU memory. On Windows the driver does not fail when video
memory runs out; it quietly spills into system memory, and every model slows
down by an order of magnitude. Laya going from 30 ms to over a second per call
is the typical sign. Check `nvidia-smi`, and run one GPU workload at a time
(two benchmark processes each load their own copy of the models).

**Generation is slow.** `TransformersLLM` runs the plain Transformers
generation loop, about 22 tokens per second for Qwen3-1.7B on a laptop GPU.
Serving the same model with Ollama, vLLM or llama.cpp is several times faster;
point `OpenAICompatibleLLM` at it.

**Local LLM answers change between runs.** They should not: decoding is
greedy. If you pass a custom generation config to your own model wrapper,
note that Transformers replaces values that equal its library defaults with
the checkpoint's defaults unless `use_model_defaults=False` is passed.

**A decision is always `uncertain`.** Every provider answered below threshold.
Check the trace: the `attempt` spans show each provider's confidence. Either
the threshold is higher than the providers can reach on that question (run
`thinkless calibrate`), or the question is ambiguous and needs clearer option
descriptions.

**A question always escalates to the LLM.** The small models reach it but
rarely clear its threshold. Calibrate, then either lower the threshold for
that provider (`thresholds={"question@provider": ...}`) or remove the provider
from the question (`providers=(...)`) so it stops adding latency.

**Warnings from Laya about "invalid temperatures".** Laya's checkpoint ships
calibration temperatures only for smaller option counts; for choices with more
than about ten options it warns and falls back. Treat its confidence on those
questions with care, and calibrate.

## Hosted models

**Empty replies or every LLM decision abstaining.** The model is probably
reasoning by default and spending the whole token budget before it answers.
Run with `--reasoning off` (or `low`). `LLMDecider` retries once with a larger
budget and logs a warning when this happens; the attempt span shows
`truncated: true`.

**`OpenRouter needs an API key`.** Put `OPENROUTER_API_KEY=...` in `.env` in
the directory you run the CLI from, or export it.

**OpenRouter's dashboard shows less spend than the traces.** Its key usage
figure lags by a few minutes. The costs on spans come from each response and
are immediate.

**A model rejects `response_format`.** The client retries without JSON mode
and logs a warning once; decisions keep working through the prompt.

## Traces

**No trace files appear.** Traces are written by a `JSONLSink`. The CLI
writes to `THINKLESS_TRACE_DIR` (default `.thinkless/traces`); in your own
code, pass `Tracer([JSONLSink(path)])` to the engine.

**Traces contain `[redacted]`.** Content capture is off
(`THINKLESS_CAPTURE_CONTENT=false` or `Tracer(capture_content=False)`).
