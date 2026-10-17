# Habits replays

These replays measure [Habits](https://inboxpraveen.github.io/ThinkLess/habits/)
on real inputs and real LLM answers, without trusting the bookkeeping
Habits does about itself.

An agent asks an LLM for a decision on every input, through the OpenAI SDK
with Habits under it. Behind Habits, a stand-in LLM replies with the
answer a real model gave for that input, and reports the tokens and cost the
real model was billed. Because the real answer is known for every input,
each local answer can be checked against what the LLM would have said, and
every answer against the dataset's label.

| Path | |
|---|---|
| `replay.py` | runs a replay and writes a result file |
| `teach.py` | records a hosted model's answers on a long stream, once; rows already recorded are never asked again |
| `streams.py` | the long streams: Banking77 (13,083 queries) and Civil Comments (10,000 validation comments), at pinned revisions |
| `cache/` | the recorded answers of Qwen 3.7 Flash (through OpenRouter, reasoning off, temperature 0) |
| `results/` | result files: every setting, the timeline, the plan and the events |

Rerun every published replay from the recorded answers, for free, then print
the table used in the docs:

```bash
cd benchmarks/habits
python replay.py --source banking77 --learn-every 500 --teacher cache:cache/banking77-qwen_qwen3.7-flash.jsonl --out results/banking77-qwen3.7-flash.json
python replay.py --source banking77 --learn-every 500 --teacher labels --out results/banking77-labels.json
python replay.py --source civil_comments --learn-every 500 --teacher cache:cache/civil_comments-qwen_qwen3.7-flash.jsonl --out results/civil_comments-qwen3.7-flash.json
python replay.py --source civil_comments --learn-every 500 --teacher labels --out results/civil_comments-labels.json
python replay.py --source task:jailbreak --source task:civil_comments --source task:multiwoz --source task:banking77 --source task:clinc150 --source task:massive --source task:helpsteer2 --source task:wnut17 --teacher results:../decisions/results/verified/qwen3.7-flash.json --out results/decision-tasks-qwen3.7-flash.json
python table.py results/*.json
```

`--teacher labels` uses the dataset's labels as the LLM: a teacher that is
never wrong and never inconsistent, which shows the ceiling for a stream.

Record another model's answers (this spends money; `--max-cost-usd` stops
the run when it is reached):

```bash
python teach.py --stream banking77 --model vendor/model --max-cost-usd 0.50
```

Settings used for the published numbers: target agreement 97% as a 95%
lower bound, 5% audits, a learning round every 500 calls (every 100 for the
500-row benchmark tasks), seed 0. Inputs are shuffled once with the seed and
replayed in that order.
