# Command line

Every command has `--help`. The reasoning model is chosen with
`--llm backend[:model]` (for example `openrouter:qwen/qwen3.7-flash`) and its
reasoning with `--reasoning default|off|minimal|low|medium|high`; see
[LLM backends](../guides/llm-backends.md#spec-strings).

Every command reads `./.env` first. Variables already set in the environment
win, and values are never printed.

## `thinkless doctor`

Checks the environment: Python, torch and the accelerator, which optional
backends are installed, which API keys are set, and the active settings.
Include its output in bug reports.

## `thinkless demo`

Runs the support agent on one ticket and prints the trace tree, the action
and the reply.

| Option | Default | |
|---|---|---|
| `--ticket` | `T-001` | A scenario id from the bundled set (`T-001` to `T-053`) |
| `--message`, `--customer` | | Your own ticket text instead of a scenario |
| `--mode` | `hybrid` | `hybrid`, `llm` or `models`; repeat to compare |
| `--llm` | `local` | Reasoning model |
| `--device` | `auto` | Device for local models |
| `--threshold` | `0.8` | Engine default threshold |
| `--view` | off | Write and open the HTML viewer |

## `thinkless bench support`

Runs every scenario in every mode and writes `results.json`, `report.md`,
per-ticket traces and `viewer.html` to the output directory.

| Option | Default | |
|---|---|---|
| `--mode` | `llm`, `hybrid`, `models` | Repeat to choose |
| `--llm` | `local` | Reasoning model, also the LLM decision provider |
| `--limit` | all | Only the first N tickets |
| `--out` | `.thinkless/bench/support-<time>` | Output directory |
| `--reference` | `anthropic:claude-sonnet-5` | Price list used to estimate cost |
| `--threshold` | `0.8` | Engine default threshold |
| `--jev` | off | Add TypeSafe Jev to the cascade (needs `TYPESAFE_API_KEY`) |

## `thinkless bench intents`

Accuracy, calibration and the simulated cascade on a public dataset.

| Option | Default | |
|---|---|---|
| `--dataset` | `banking77` | `banking77`, `clinc150` or `emotion` |
| `--provider` | `gliner`, `laya`, `llm` | Repeat to choose; `llm` is the cascade fallback |
| `--limit` | `500` | Examples sampled from the test split |
| `--seed` | `13` | Sampling seed |
| `--target` | `0.95` | Accuracy target for threshold recommendations |
| `--out` | `.thinkless/bench/intents-<dataset>-<time>` | Output directory |

## `thinkless calibrate DATA`

Sweeps thresholds for one question and provider on labeled JSON Lines and
recommends the lowest threshold that meets the target.

| Option | Default | |
|---|---|---|
| `--provider` | `gliner` | `gliner`, `laya`, `llm` or `jev` |
| `--kind` | `choice` | `choice` or `yes_no` |
| `--question` | | The instructions to ask |
| `--labels` | from the data | Choice options, comma-separated |
| `--demo-question` | | Use a question from the support demo by name |
| `--text-field`, `--label-field` | `text`, `label` | Field names in the data |
| `--target` | `0.95` | Accuracy the accepted answers must reach |

## `thinkless trace`

| Command | |
|---|---|
| `trace ls [DIR]` | Recent traces with their headline numbers |
| `trace show FILE` | One trace as a tree with its summary |
| `trace view [PATHS...] [--out FILE] [--no-open]` | Write the self-contained HTML viewer |
| `trace export PATHS... --out FILE` | Traced decisions as labeling rows for `calibrate`; filter with `--question`, `--status`, `--plane`, `--limit` |
| `trace drift --baseline PATHS --current PATHS` | Compare two periods question by question; exits with 1 when a question drifted |

## `thinkless shadow`

Reads the logs written by [shadow mode](../guides/shadow-mode.md).

| Command | |
|---|---|
| `shadow report LOGS...` | Agreement with a 95% interval, share settled without an LLM, cost and latency on both sides, fitted thresholds and a verdict per question |
| `shadow export LOGS... --out FILE` | Disagreements (or every call with `--all`) as labeling rows for `calibrate` |

`shadow report` options: `--target` (agreement the lower bound must reach,
default `0.95`), `--min-calls` (default `100`), `--volume` (decisions a month,
for the monthly projection) and `--json`. `shadow export` options:
`--question`, `--all` and `--label-from primary|shadow|none`.

## `thinkless serve ENGINE`

Serves an engine over HTTP; see [serving](../guides/serving.md). `ENGINE` is
`module:attribute`, an engine or a function that returns one.

| Option | Default | |
|---|---|---|
| `--questions` | `QUESTIONS` in the engine's module | Registered questions, as `module:attribute` |
| `--host`, `--port` | `127.0.0.1`, `8080` | Where to listen |
| `--api-key-env` | `THINKLESS_API_KEY` | Variable holding the bearer token clients must send |
| `--registered-only` | off | Refuse question specs that are not registered |

## `thinkless mcp ENGINE`

Serves the registered questions as MCP tools, one `decide_<question>` tool
each. `--transport stdio` (default) or `streamable-http`.

## Environment variables

| Variable | Default | |
|---|---|---|
| `THINKLESS_TRACE_DIR` | `.thinkless/traces` | Where the CLI writes traces |
| `THINKLESS_DEVICE` | `auto` | Default device for local models |
| `THINKLESS_CAPTURE_CONTENT` | `true` | Record inputs and outputs in traces |
| `THINKLESS_LOG_LEVEL` | `WARNING` | Log level for the CLI |
| `THINKLESS_PRICING` | bundled table | Path to a pricing TOML file |
| `TYPESAFE_API_KEY` | | For `SystemOne.jev()` |
| `OPENROUTER_API_KEY` | | For `openrouter:` models |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` | | For hosted reasoning models |
| `THINKLESS_API_KEY` | | Bearer token for `thinkless serve` |
