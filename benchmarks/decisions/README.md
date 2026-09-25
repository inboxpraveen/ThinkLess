# Decision benchmark

A provider-neutral benchmark for the models that make an agent's routine
decisions. The method, the leaderboard and the submission guide are in the
documentation: https://inboxpraveen.github.io/ThinkLess/decision-benchmark/

| Path | |
|---|---|
| `tasks/<task>/test.jsonl`, `calibration.jsonl` | the frozen rows of benchmark 1.0 |
| `tasks/NOTICE.md` | sources, revisions, licenses and changes for every task |
| `build.py` | how the rows were sampled; maintainers only, and never rerun within a version |
| `results/verified/` | runs done by the maintainers |
| `results/submitted/` | runs submitted by others, verified by CI |

```bash
thinkless bench decisions run --provider gliner --name "GLiNER 2.5 base"
thinkless bench decisions verify GLiNER-2.5-base.json
```
