# Decision benchmark

An open, provider-neutral benchmark for the models that make an agent's
routine decisions: intent classifiers, safety and moderation checks, raters
and extractors, whether they are small local encoders, hosted decision models
or prompted LLMs.

Most classification benchmarks ask one question: how accurate is the model?
An agent needs a second one answered: **on which inputs can the model be
trusted on its own?** A decision model earns its place in an agent by
answering the inputs it is sure about and passing the rest on, so this
benchmark measures calibration and selective accuracy next to accuracy, and
reports latency and billed cost for every result.

- [Leaderboard](leaderboard.md): current results.
- [Submitting a model](submitting.md): run it on your model, check the
  result, open a pull request.

## What version 1.0 shows

From the four [baseline runs](leaderboard.md), all done by the maintainers on
one laptop GPU and through OpenRouter:

- **No model is best at everything.** The small models are specialists. On
  jailbreaks, Laya scored 99.7% and could answer every row on its own; on
  toxic comments it matched Qwen 3.7 Flash (77.0% against 77.2%) and could
  answer half the rows on its own at 94.5% accuracy, in 48 ms instead of
  572 ms. GLiNER matched Qwen 3.7 Flash on Banking77 (75.8% against 75.2%) and
  could answer 35% of rows on its own at 94.9%.
- **The LLM is needed where the small models break down**: other languages
  (GLiNER fell from 50% in English to 5% in Hindi), conversations (72.2%
  against at most 40.6% on MultiWOZ), requests out of scope (87.4% against
  58.8% on CLINC150), and entity extraction.
- **A small local LLM is not a substitute for either.** Qwen3-1.7B scored
  below GLiNER on Banking77, below Laya on both safety tasks, and below the
  hosted model on every task.
- **Rating a reply is hard for everyone.** No model matched the human
  helpfulness rating on more than 31% of rows, or came within one level on
  more than 65%.
- **Confidence can rank answers well without being calibrated.** Laya's
  confidence on toxic comments is far from its accuracy (ECE 0.394), yet
  sorting by it works (AURC 0.080), which is why the threshold is fitted on
  calibration rows rather than read off the raw confidence.

Laya's near-perfect jailbreak score should be read with its training data in
mind, which is not published (see Limitations).

## Tasks

Eight tasks on public datasets cover the four kinds of question an agent
asks. Every task has 500 test rows (the jailbreak task has 399) and 300
calibration rows from a different split of the same source.

| Task | Question kind | What it tests | Source | License |
|---|---|---|---|---|
| `banking77` | choice, 77 options | fine-grained support intents | Banking77 (PolyAI) | CC BY 4.0 |
| `clinc150` | choice, 151 options | intents across 10 domains, including out of scope | CLINC150 plus | CC BY 3.0 |
| `massive` | choice, 60 options | the same assistant intents in English, German, Spanish, Hindi and Japanese | MASSIVE (Amazon) | CC BY 4.0 |
| `multiwoz` | choice, 10 options | the user's intent at a turn of a conversation, from the conversation so far | MultiWOZ 2.2 | Apache 2.0 |
| `jailbreak` | yes or no | prompts that try to make an assistant ignore its rules | jailbreak-classification | Apache 2.0 |
| `civil_comments` | yes or no | toxic comments | Civil Comments (Jigsaw) | CC0 1.0 |
| `helpsteer2` | score, 5 levels | how helpful an assistant's reply is, against human ratings | HelpSteer2 (NVIDIA) | CC BY 4.0 |
| `wnut17` | extract, 6 fields | people, places, organizations, products, creative works and groups in social media text | WNUT 2017 | CC BY 4.0 |

`thinkless bench decisions tasks` lists them, and each task's exact question
is in
[`tasks.json`](https://github.com/inboxpraveen/ThinkLess/blob/main/src/thinkless/bench/decisions/tasks.json).
The rows are frozen in
[`benchmarks/decisions/tasks`](https://github.com/inboxpraveen/ThinkLess/tree/main/benchmarks/decisions/tasks),
with the sources, revisions and changes in its
[NOTICE](https://github.com/inboxpraveen/ThinkLess/blob/main/benchmarks/decisions/tasks/NOTICE.md).

## Rules

These are what make results comparable, and what make them checkable.

1. **Every provider gets the same question.** The instructions, the options
   (label names only, no descriptions written for one model), the levels and
   the fields are fixed per task. Nothing is tuned per provider. A provider
   may translate the question into its own format internally, as it would in
   production.
2. **Every provider gets the same rows**, in the same order, with the same
   input: text, a mapping, or, for `multiwoz`, the conversation as a list of
   messages.
3. **Thresholds are fitted on calibration rows only.** The benchmark fits
   each provider's threshold on the calibration rows and reports it on the
   test rows, which come from a different split of the source. Inputs that
   appear in both were removed when the tasks were built.
4. **Results contain every prediction.** A result file holds the answer,
   confidence, latency and cost of every row, but no labels. Every metric is
   recomputed from the predictions and the published task files; the numbers
   a submitter writes in the file are never used.
5. **Training on task data is declared.** A model trained on the train
   splits of these sources is marked on the leaderboard. It measures a
   supervised model for that task, which is a legitimate result, but a
   different one.
6. **Versions are frozen.** The rows and questions of a benchmark version
   never change. Each result records a fingerprint of every task, and a result
   for another version is not ranked.

## Metrics

For every task the leaderboard reports:

| Metric | Meaning |
|---|---|
| Accuracy | share of test rows answered correctly, with a 95% Wilson interval; an abstention counts as wrong |
| Macro-F1 (choice) | F1 averaged over the classes present, so rare intents count as much as common ones |
| Balanced accuracy (yes or no) | the mean of the accuracy on each answer, so a model that always says no does not look good |
| MAE and within one (score) | mean absolute error in levels, and the share within one level of the human rating |
| Field F1 (extract) | precision and recall over the fields that have a value; a row is correct only when every field matches |
| ECE | expected calibration error: how far the reported confidence is from the actual accuracy |
| Coverage at 95% | the share of test rows the provider answers on its own at the threshold where its accepted calibration answers reach 95% accuracy |
| Accuracy there | the accuracy of those answers on the test rows, which shows whether the threshold held |
| AURC | area under the risk-coverage curve: the average error rate when answers are taken in order of confidence, from most to least confident (lower is better) |
| p50 latency | median wall time per row, on the hardware in the run table |
| Cost per 1k | billed cost when the backend reports it (OpenRouter does), otherwise the price table; local models count as $0 |

Coverage, accuracy there, ECE and AURC need a confidence. Prompted LLMs do not
report one, so for them only the accuracy metrics apply; they are the
reference an agent falls back to. Labels are compared after trimming and case
folding, and a score given as a number counts as the nearest level.

There is no single combined score. How much accuracy is worth a millisecond
or a dollar depends on the agent, so the table shows each number and leaves
the weighting to the reader.

## Running it

```bash
pip install "thinkless[gliner,laya]"

thinkless bench decisions tasks
thinkless bench decisions run --provider gliner --name "GLiNER 2.5 base"
thinkless bench decisions run --provider llm --llm openrouter:qwen/qwen3.7-flash \
    --reasoning off --concurrency 8 --max-cost-usd 1
thinkless bench decisions verify GLiNER-2.5-base.json
```

The task files are read from a checkout of the repository when there is one,
and otherwise downloaded once and checked against their published hashes. A
full run is 3,899 test rows, plus 2,400 calibration rows for providers that
report a confidence. `--limit 20` runs a quick smoke test, which is never
ranked. `--max-cost-usd` caps what a hosted run can spend.

## Limitations

- **Small samples.** 500 test rows per task give accuracy intervals of about
  plus or minus four points. Differences smaller than that are not
  meaningful, which is why the intervals are shown.
- **Option names only.** Options are the dataset's label names, some of them
  terse (`card arrival`, `iot hue lightoff`). A deployment would write
  descriptions, which usually helps prompted models most.
- **Default settings.** Maintainer runs use each provider's defaults. Some
  models expose settings that matter for a task (Laya, for example, documents
  a larger label budget for questions with 50 or more options). A vendor can
  submit a tuned configuration under its own name, with the settings in the
  notes.
- **One prompt format for LLMs.** Prompted LLMs are asked through ThinkLess's
  `LLMDecider`. A vendor with a better prompt for its own model can submit it
  as its own provider.
- **Public test rows.** Every source is public, so any model may have seen
  these rows in training, and most model cards do not say. The benchmark
  cannot rule this out. A suspiciously high score on one task is a reason to
  ask, and later versions will add sources released after the models they
  compare. Submitters declare training on the task sources; for models whose
  training data is not published, the run notes say so.
- **Human labels are imperfect.** HelpSteer2 ratings and Civil Comments
  toxicity scores are averages over annotators who disagree.
- **Hosted models vary between runs.** Two runs of Qwen 3.7 Flash a few
  minutes apart differed by up to 1.4 points on a task, well inside the
  intervals.
- **Latency depends on hardware.** It is comparable within one machine; the
  run table records the device for every result.

## Sponsoring runs

Hosted models cost money to benchmark, and a neutral benchmark cannot favor
the models its maintainers can afford to run. Model vendors and anyone else
can [sponsor runs](https://github.com/sponsors/inboxpraveen) with API credits
or funding. Sponsored runs are done by the maintainers, published as verified
whatever the result, and credited on the leaderboard. Sponsorship buys runs,
not rankings.

## Versions

| Version | Released | Tasks | Notes |
|---|---|---|---|
| 1.0 | 2026-09-25 | 8 | first release |

Future versions will add multi-turn safety cases, more languages and harder
out-of-scope sets. A new version never changes an old one's rows, so results
stay comparable within a version.
