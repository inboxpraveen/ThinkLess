# Contributing to ThinkLess

Thank you for considering a contribution. Bug reports, new providers,
benchmark runs on hardware or models we have not tried, documentation fixes
and new demo scenarios are all welcome.

## Ground rules

- **Evidence over claims.** A change that affects accuracy, latency or cost
  should come with numbers, ideally a before and after from `thinkless bench`
  or `thinkless calibrate`. Benchmark tables in the docs are regenerated from
  result files, never edited by hand.
- **Small pull requests.** One change per pull request, with a test. Open an
  issue or a discussion first for anything large, such as a new provider
  family or a change to the public API.
- **Plain writing.** Docs, docstrings and comments should read like a careful
  engineer wrote them. No em dashes or en dashes (use a comma, a colon,
  parentheses or a new sentence), and no filler phrases. `scripts/check_prose.py`
  enforces the basics in CI.

## Setting up

```bash
git clone https://github.com/inboxpraveen/ThinkLess
cd ThinkLess

# conda, with the local model stack and the right PyTorch build
conda env create -f environment.yml
conda activate thinkless

# or plain pip, core and dev tools only
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,openai,anthropic,otel]"
```

## Checks

```bash
pytest                           # unit tests, no model downloads
pytest -m local                  # real inference with local models (downloads weights)
pytest -m network                # live OpenRouter calls, needs OPENROUTER_API_KEY (under a cent)
ruff check src tests scripts benchmarks examples
ruff format src tests scripts benchmarks examples
mypy
python scripts/check_prose.py
```

CI runs the unit tests on Linux, Windows and macOS for Python 3.10 to 3.13,
plus lint, types, prose and a build. Local model tests do not run in CI;
please run `pytest -m local` yourself when you touch a local provider.

## Adding a provider

1. Subclass `thinkless.providers.DecisionProvider` in
   `src/thinkless/providers/<name>.py`. Declare `plane`, `kinds` and
   `calibrated`, and return probabilities whenever the backend has them.
2. Import heavy dependencies lazily, inside `warmup()` or the first call, and
   add an optional extra in `pyproject.toml`.
3. Add unit tests that do not need the model (mock the backend), and a
   `@pytest.mark.local` test that does.
4. Document it in `docs/guides/providers.md` with measured latency and the
   hardware you measured on.
5. If you can, add a column to the intents benchmark and include the results.

## Adding demo scenarios

Scenarios live in `src/thinkless/demo/support/data/scenarios.jsonl`. Each
needs an expected `action`, and `intent` and `order_id` where they apply. The
oracle test in `tests/unit/test_support_demo.py` checks that perfect
decisions produce the expected action for every scenario, so a new scenario
either passes it or reveals a bug. Keep calibration examples
(`calibration.jsonl`) and benchmark scenarios separate: thresholds must never
be tuned on the tickets used to report results.

## Commit messages and releases

Write commit messages in the imperative ("Add Kev provider", "Fix escalation
count"). Add a line to `CHANGELOG.md` under "Unreleased" for anything a user
would notice. Releases are cut by tagging `vX.Y.Z`, which publishes to PyPI
through the release workflow; the steps are in
[docs/guides/releasing.md](docs/guides/releasing.md).

## Code of conduct

Everyone taking part is expected to follow the [code of conduct](CODE_OF_CONDUCT.md).
