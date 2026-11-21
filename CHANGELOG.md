# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

## 0.6.0 - 2026-11-22

A review of the whole package. Habits gets fixes for streams cut short,
several processes sharing one directory, and the Responses API's stored
state; typed decisions get fixes for rules that never fired, spend caps,
cost reporting and the System One wire. A demo anyone can run right after
`pip install thinkless`, and CI that holds every lower bound to its promise.

### Added

- `thinkless habits demo`: a support agent's routing decision goes from
  recording to probation to being answered locally, offline, with a stand-in
  LLM, in about a second. It needs nothing beyond `pip install thinkless`,
  and the README now starts with it.
- A Responses API call that continues a locally answered response
  (`previous_response_id` naming a `resp_thinkless_` id) is sent on with the
  earlier input and answer written out, since the provider never stored the
  local answer. The process keeps the last 2,048 local answers for this.
- A logo, an architecture diagram of Habits and typed decisions (in the
  README, the docs home page and the four planes page), and a social preview
  image. The docs site uses the logo and gives every page a link preview.

### Fixed

- A rule registered for a question never fired when the question had no
  `name` and was asked through `decide_many({"intent": Choice(...)})`: rules
  and Hugging Face classifiers looked it up by a name derived from its
  instructions, so it went on to the next provider, often the paid LLM. An
  unnamed question now takes the name it is given in the mapping, which
  thresholds and traces already used. A LangGraph `router` also finds a
  decision a `decision_node` stored under that name instead of asking again.
- A nested `engine.run()` without `max_cost_usd` lifted the spend cap of the
  run around it. Every open run's cap now applies. `run(max_cost_usd=-1)`
  also left a trace sink behind.
- `/v1/decide` and `/v1/systemone` counted a batched call's cost and tokens
  once per question it answered, and left out paid attempts that were not
  accepted. Both now report the run's total.
- `/v1/systemone` sent uncertain answers from providers without
  probabilities at probability 1, so a ThinkLess client accepted answers the
  server was not sure of. They are left out, like abstentions.
- Shadow mode counted only the cost of the accepted attempt, which
  overstated savings and let spending pass `max_cost_usd`.
- One malformed entry in a System One reply, a `NaN` score from an LLM, or a
  negative `Retry-After` header failed the whole call; each now spoils only
  its own answer, or falls back to the normal backoff.
- `thinkless shadow report --min-calls 0` divided by zero when nothing had
  been compared.
- Habits recorded a streamed answer the caller stopped reading, or that
  broke off, as the LLM's complete answer. Truncated answers became training
  data and, in an audit, counted as disagreement. A Chat Completions stream
  now counts only with a finish reason or `[DONE]`, and a Messages stream
  only with `message_stop`.
- A site suspended by one process could be switched back on by a learning
  run in another process that had read the plan before the suspension. The
  run now keeps a suspension written while it was busy. A process also stops
  holding a suspension once another process has retrained the site and the
  new student passed its own checks.
- Two writers saving the plan at once could leave a half-written
  `plan.json`. Each write now uses its own temporary file, and the replace
  is retried while a reader on Windows holds the file.
- The gate could stop one step short of testing every held-out answer, so a
  student good enough to answer all of them kept up to about 9% of them
  going to the LLM.
- Responses API calls in a stored `conversation` were treated as
  answerable, although part of their context is on the provider's side.
- A tool-calling answer with malformed arguments raised inside Habits and
  the call was not recorded.
- A forked worker kept writing into its parent's call log; each process now
  writes its own file, as documented.
- `habits status` and `scan` counted audit and shadow agreement with a
  slightly different rule from the one that drives suspension; all three now
  share one.
- `Habits(target=1.0)` and other impossible settings were accepted and
  failed later in every learning run; they are now rejected up front.
- The CLI failed to start with typer 0.12, which the requirements allowed
  (it cannot read `str | None` options on Python 3.10). ThinkLess now needs
  typer 0.13, and a new CI job runs the tests with every direct dependency
  at its lowest allowed version.
- Older pydantic versions warned on import about the `model_input_tokens`
  field of `TraceSummary`.
- Tests that need an optional extra (the OpenAI SDK, OpenTelemetry) now skip
  when it is missing instead of failing.
- A Habits background learning test depended on thread timing and failed on
  Windows; a mypy error in `llm/factory.py` under older mypy versions.
- A Habits student test compared a tuple holding a dict with
  `pytest.approx`, which pytest 9.2 rejects; CI failed on every new run.

### Changed

- The `laya` extra needs Laya 0.4.17 or newer (below 0.5). Laya releases
  after 0.4.22 need transformers 5, which GLiNER2 does not support yet, so a
  shared environment resolves to Laya 0.4.22.
- Dependabot opens a pull request only when a release falls outside the
  supported range, instead of raising lower bounds, and CI runs weekly.

## 0.5.0 - 2026-10-17

Habits: the decisions an agent keeps making become local habits, once
proven, with one line under the LLM client it already uses.

### Added

- `thinkless.habits`: `wrap(client)` puts ThinkLess under an OpenAI or
  Anthropic SDK client, LangChain's `ChatOpenAI`, or any `httpx` or `httpx2`
  client, in place. It records every Chat Completions, Responses and
  Messages call, groups calls into sites by model, output contract and prompt
  template, and trains a local student per decision site from the LLM's own
  answers (a logistic regression over hashed words and character n-grams for
  closed answers, a learned value shape for values copied from the input).
- The gate: a site is answered locally only at confidence levels where
  cross-fitted students agreed with the LLM at the target rate as a one-sided
  Clopper-Pearson lower bound, with thresholds tested in fixed sequence.
- Probation: passing the gate only starts a trial. The student answers new
  calls in shadow while the LLM keeps answering them, and it answers nothing
  until it passes one more test, of a size fixed in advance, on those new
  calls (`confirm_calls`, 394 at the defaults; `0` skips probation).
- Audits and suspension: a random share of local answers still goes to the
  LLM; a site whose audited agreement falls below the target goes back to the
  LLM until calls made after the suspension clear the gate again.
- A student that passed the gate keeps serving until a newer one passes or
  audits suspend the site, so a site does not switch off and on as its
  confident calls stop reaching the LLM. Students are versioned files; old
  ones are removed.
- Predictability: the plan records how often a student matches the LLM's
  next answers on all held-out calls, and `learn` warns about sites where
  the LLM answers similar inputs differently.
- Fails open: an error inside ThinkLess sends the call to the LLM unchanged.
- Modes `auto`, `shadow`, `record` and `off`; site names through the
  `x-thinkless-site` header; learning in a background thread, at most once a
  minute, one process at a time per directory, over the most recent calls.
  Calls answered locally are logged as compact records, so the log grows
  with LLM calls only.
- Local answers keep the exact response shape: JSON schema and JSON mode
  answers, forced tool calls, plain text and streamed Chat Completions.
- `thinkless habits scan`, `learn`, `status`, `proxy` and `reset`. The
  scan reads Habits logs, chat fine-tuning files and request and response
  pairs, and reports each site's share of calls and cost and whether its
  answers are decisions.
- `thinkless habits proxy`: an OpenAI-compatible endpoint with Habits
  in front of any upstream, for clients in any language.
- Replays in `benchmarks/habits/` measure Habits against a real LLM's
  recorded answers.
- `examples/10_habits.py` runs the whole loop offline.
- Integration tests and examples for gated CrewAI and LlamaIndex tools.

### Changed

- NumPy is a dependency (for the students).

## 0.4.1 - 2026-09-25

Project and packaging updates; no change to the library's behavior.

### Changed

- Package metadata: development status Beta, more keywords, and links to
  Discussions, the decision benchmark leaderboard, release notes and funding.
- The README links to good first issues and Discussions and shows the docs,
  leaderboard and OpenSSF Scorecard badges.
- The submission guide says submitters regenerate the leaderboard page in
  their pull request; the release guide explains that release tags are
  permanent.

### Security

- Every GitHub Action in the workflows is pinned to a commit.
- An OpenSSF Scorecard workflow checks the repository's supply chain weekly.

## 0.4.0 - 2026-09-25

The decision benchmark.

### Added

- The decision benchmark 1.0 (`thinkless bench decisions`): eight public
  tasks across the four question kinds (banking77, clinc150 with out of
  scope, MASSIVE in five languages, MultiWOZ 2.2 conversations, jailbreaks,
  Civil Comments toxicity, HelpSteer2 helpfulness ratings and WNUT 2017
  entities), with frozen rows and published hashes, calibration rows from a
  separate split, result files that hold every prediction, `verify` that
  recomputes every metric, a generated leaderboard and open submissions.
  Baselines for GLiNER 2.5, Laya, Qwen3-1.7B and Qwen 3.7 Flash.

### Changed

- A list of chat messages now renders as `role: text` lines for providers
  that read text, instead of one `key: value` line per field.
- README links to the documentation point at the documentation site.

## 0.3.0 - 2026-09-25

Shadow mode, framework adapters, limits, a decision server and MCP tools.

### Added

- Shadow mode (`thinkless.shadow`): run a candidate engine next to existing
  decision code without changing what it returns, log both answers, and
  report agreement with a 95% interval, the share settled without an LLM,
  cost and latency on both sides, thresholds fitted on production traffic,
  and a verdict per question (`thinkless shadow report`). Disagreements
  export as labeling rows (`thinkless shadow export`). The same class audits
  a live engine against an LLM.
- Framework adapters (`thinkless.integrations`): a LangGraph router, decision
  node and async decision node; OpenAI Agents SDK input guardrails, tool input
  guardrails and `route_agent`; and framework-neutral `Router`, `route` and
  `gate`, tested with LangChain tools and Pydantic AI. Extras `langgraph` and
  `openai-agents`.
- Engine deadlines (`deadline_ms` on the engine and per call), spend limits
  (`SpendLimit`, lifetime or rolling window) and per-run caps
  (`engine.run(..., max_cost_usd=...)`). Skipped providers are recorded with
  reason `deadline` or `spend_limit`.
- A decision server (`thinkless serve`, `thinkless.server.app.create_app`)
  with `/v1/decide` and a System One compatible `/v1/systemone`, so another
  engine's `SystemOne` provider, or TypeSafe's SDK, can use it. Extra
  `server`.
- MCP tools (`thinkless mcp`, `thinkless.server.mcp.create_mcp_server`): one
  `decide_<question>` tool per registered question. Extra `mcp`.
- `thinkless trace export` turns traced decisions into labeling rows, and
  `thinkless trace drift` compares two periods and exits with 1 when a
  question drifted.
- `question_from_spec` rebuilds a question from its spec.
- Docs: an integrations section, a migration guide, shadow mode, serving,
  a pilot playbook and a FAQ, published to GitHub Pages by a new workflow.

## 0.2.0 - 2026-09-25

First release on PyPI.

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
