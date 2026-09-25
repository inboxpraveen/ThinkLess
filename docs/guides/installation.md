# Installation

ThinkLess needs Python 3.10 or newer and runs on Linux, macOS and Windows.

## Pick what you need

The core install is small: `pydantic`, `httpx`, `rich` and `typer`. Everything
that brings in heavy dependencies is an extra.

| You want | Install | Pulls in |
|---|---|---|
| Rules, tracing, the CLI, Jev or any System One server | `pip install thinkless` | nothing heavy |
| Hosted LLMs through OpenRouter, OpenAI, Ollama or vLLM | `pip install "thinkless[openai]"` | `openai` |
| Claude through the Anthropic SDK | `pip install "thinkless[anthropic]"` | `anthropic` |
| GLiNER 2.5 | `pip install "thinkless[gliner]"` | `gliner2`, `transformers<5`, `torch` |
| Laya | `pip install "thinkless[laya]"` | `laya`, `transformers<5`, `torch` |
| A local LLM or any Hugging Face classifier | `pip install "thinkless[local-llm]"` | `transformers<5`, `accelerate`, `torch` |
| All local models | `pip install "thinkless[local]"` | the three above |
| OpenTelemetry export | `pip install "thinkless[otel]"` | `opentelemetry-sdk` and the OTLP exporter |
| The public dataset benchmarks | `pip install "thinkless[bench]"` | `datasets` |
| The LangGraph adapter | `pip install "thinkless[langgraph]"` | `langgraph` |
| The OpenAI Agents SDK adapter | `pip install "thinkless[openai-agents]"` | `openai-agents` |
| The decision server (`thinkless serve`) | `pip install "thinkless[server]"` | `fastapi`, `uvicorn` |
| MCP tools (`thinkless mcp`) | `pip install "thinkless[mcp]"` | `mcp` |
| Everything | `pip install "thinkless[all]"` | all of the above |

Extras combine: `pip install "thinkless[gliner,laya,openai]"`.

## Local models: install PyTorch first

pip cannot choose the right PyTorch build for your GPU, so the local extras
take whatever `torch` it finds, and on Windows and macOS that is the CPU build.
Everything still works, only slower. For a GPU, install PyTorch first from
its own index, then ThinkLess:

**NVIDIA GPU (Linux or Windows).** CUDA 13.0 wheels cover current cards,
including the RTX 50 series:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install "thinkless[local]"
```

For older drivers, pick the matching CUDA version on
[pytorch.org](https://pytorch.org/get-started/locally/).

**Apple Silicon.** The default PyPI build includes Metal (MPS) support:

```bash
pip install "thinkless[local]"
```

**CPU only.**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "thinkless[local]"
```

Then check what you got:

```bash
thinkless doctor
```

`doctor` prints the accelerator torch can see. If an NVIDIA GPU is present but
torch is a CPU build, it says so and prints the command to fix it.

## Conda

`environment.yml` creates an environment with everything, including the CUDA
13.0 build of PyTorch:

```bash
conda env create -f environment.yml
conda activate thinkless
```

It installs ThinkLess from the checkout in editable mode, for development. To
use the released package in a conda environment, create the environment with
Python 3.12, install PyTorch as above, then `pip install "thinkless[local]"`.

## Where files go

| What | Where | Change it with |
|---|---|---|
| Model weights | the Hugging Face cache, `~/.cache/huggingface` | `HF_HOME` |
| Banking77 test file for the benchmarks | `~/.cache/thinkless/datasets` | |
| Traces written by the CLI | `./.thinkless/traces` | `THINKLESS_TRACE_DIR` |
| Shadow logs | wherever `Shadow(log=...)` points | |
| Benchmark output | `./.thinkless/bench/<run>` | `--out` |
| API keys | environment variables, or `./.env` for the CLI | |

Models download on first use: about 0.8 GB for GLiNER base, 0.9 GB for Laya
and 3.4 GB for Qwen3-1.7B. `thinkless doctor` shows which ones are already
cached.

## Offline and air-gapped machines

Download the weights once on a connected machine (running `thinkless demo`
does it), copy the Hugging Face cache directory, and set `HF_HUB_OFFLINE=1` on
the offline machine. Providers also accept a local directory in place of a
model id, for example `GLiNER("/models/gliner2.5-base-v1")`.

## Windows notes

- On Windows, ThinkLess downloads models one file at a time. That avoids a
  race in the Hugging Face cache's symlink handling (`WinError 1314`).
  Enabling Developer Mode lets the cache use symlinks and saves disk space.
- If a console shows garbled characters in tables, use Windows Terminal or
  set `PYTHONIOENCODING=utf-8`.

## Checking an install

```bash
thinkless --version
thinkless doctor
python -c "import thinkless; print(thinkless.__version__)"
```

A run that needs no downloads, for a quick end-to-end check:

```bash
python -m thinkless demo --help
```
