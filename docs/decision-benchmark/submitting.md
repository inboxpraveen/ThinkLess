# Submitting a model

Anyone can put a model on the leaderboard: a vendor with a hosted decision
model, a lab with a new classifier, or a team that fine-tuned its own. The
steps are the same for all of them, and none of them needs the maintainers'
involvement until the pull request.

## 1. Make the model answerable

Pick the form that matches how your model runs.

**A System One server.** If your model serves the System One wire format
(`POST /v1/systemone`, as Jev, Kev, OpenJev and `thinkless serve` do), point
the benchmark at it. No code needed:

```bash
export SYSTEMONE_API_KEY=...   # if the server needs a bearer token
thinkless bench decisions run --provider systemone:https://api.example.com --name "Example Model 2"
```

The System One format carries no price, so such a run records a cost of $0.
State your list price per 1,000 calls with `--notes`.

**A Python provider.** Anything else becomes a
[`DecisionProvider`](../guides/providers.md): one class with an `answer`
method that takes the input and the questions and returns answers with
probabilities or a confidence. Put it in a module and pass `module:attribute`
(an instance, or a function that returns one):

```bash
thinkless bench decisions run --provider my_models.provider:build --name "My Classifier"
```

Answer only the question kinds your model supports and declare them in
`kinds`; tasks of other kinds are recorded as not supported, which is a
normal result.

**A prompted LLM.** Any model reachable through a ThinkLess LLM backend:

```bash
thinkless bench decisions run --provider llm --llm openrouter:vendor/model \
    --reasoning off --concurrency 8 --max-cost-usd 2 --name "Vendor Model (reasoning off)"
```

To use your own prompt instead of `LLMDecider`'s, wrap the model in a Python
provider.

## 2. Run it

Start with a smoke test that takes a minute and is never ranked:

```bash
thinkless bench decisions run --provider ... --name "..." --limit 20
```

Then the full run. Useful options:

| Option | |
|---|---|
| `--name` | the name on the leaderboard; include the version and any setting that matters |
| `--submitted-by` | your name or organization |
| `--notes` | settings a reader should know, such as quantization or context length |
| `--trained-on-task-data` | set it if the model was trained on the train splits of these datasets |
| `--concurrency` | parallel requests for hosted models |
| `--max-cost-usd` | the most the run may spend; a run stopped by it is partial and not ranked |
| `--device` | `cuda`, `cpu` or `mps` for local models |

## 3. Check it

```bash
thinkless bench decisions verify My-Classifier.json
```

This is the same check the pull request runs: the task fingerprints, the row
ids, and every metric recomputed from the predictions. Fix anything it
reports before opening the pull request.

## 4. Open a pull request

Add the file as `benchmarks/decisions/results/submitted/<name>.json`, and
regenerate the leaderboard page in the same pull request:

```bash
thinkless bench decisions leaderboard benchmarks/decisions/results \
    --out docs/decision-benchmark/leaderboard.md
```

In the description, say:

- what the model is, with a link to its model card or documentation;
- how to reproduce the run (the command line and any code);
- the hardware, for local models;
- whether the model saw any of these datasets in training.

CI verifies the file and checks that the leaderboard page matches the
results. After review the
result appears marked **submitted**. When the maintainers can rerun it
(public weights, or API access), they do, and the result is marked
**verified**.

## What gets a result rejected

- It fails `verify`: a different benchmark version, missing rows, or a task
  fingerprint that does not match.
- The model was tuned on test rows. Calibration rows are fine to look at;
  test rows are not.
- It cannot be reproduced, and the submitter cannot explain the gap when the
  maintainers' rerun disagrees beyond noise.

Results that are unflattering to anyone, including ThinkLess's own
components, are published like any other.
