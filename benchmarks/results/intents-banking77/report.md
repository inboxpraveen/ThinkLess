# Intent benchmark: banking77

- Dataset: Banking77 (PolyAI), 77 banking intents ([source](https://github.com/PolyAI-LDN/task-specific-datasets))
- Sample: 500 test examples, seed 13, 77 labels
- LLM: `local`
- Environment: NVIDIA GeForce RTX 5060 Laptop GPU, torch 2.14.0+cu130
- ThinkLess 0.1.0, run 2026-09-25T04:54:02Z

## Providers

| Provider | Accuracy | ECE | p50 latency | p95 latency | Threshold for 95% accuracy |
|---|---:|---:|---:|---:|---|
| `gliner` | 71.8% | 0.116 | 33.5 ms | 38.8 ms | 0.99, answers 24.8% alone |
| `laya` | 35.0% | 0.184 | 51.0 ms | 71.0 ms | not reached |
| `llm` | 51.4% | n/a | 405.3 ms | 554.4 ms | not reached |

## Cascade

Small models are tried in order; an answer below the threshold goes to the next one, and finally to the LLM. Accuracy is for the whole cascade.

| Threshold | Accuracy | Calls reaching the LLM | Mean latency | Answered by |
|---:|---:|---:|---:|---|
| 0.00 | 71.8% | 0.0% | 36 ms | gliner 100.0% |
| 0.05 | 71.8% | 0.0% | 36 ms | gliner 100.0% |
| 0.10 | 71.8% | 0.0% | 36 ms | gliner 100.0% |
| 0.15 | 71.8% | 0.0% | 36 ms | gliner 100.0% |
| 0.20 | 71.8% | 0.0% | 36 ms | gliner 100.0% |
| 0.25 | 71.8% | 0.4% | 38 ms | gliner 99.6%, llm 0.4% |
| 0.30 | 71.8% | 1.0% | 41 ms | gliner 99.0%, llm 1.0% |
| 0.35 | 72.0% | 2.4% | 49 ms | gliner 97.2%, laya 0.4%, llm 2.4% |
| 0.40 | 71.4% | 3.2% | 54 ms | gliner 95.4%, laya 1.4%, llm 3.2% |
| 0.45 | 71.2% | 4.6% | 61 ms | gliner 93.0%, laya 2.4%, llm 4.6% |
| 0.50 | 71.4% | 7.0% | 73 ms | gliner 90.0%, laya 3.0%, llm 7.0% |
| 0.55 | 72.0% | 9.4% | 86 ms | gliner 86.2%, laya 4.4%, llm 9.4% |
| 0.60 | 71.4% | 13.2% | 104 ms | gliner 82.2%, laya 4.6%, llm 13.2% |
| 0.65 | 71.4% | 14.2% | 109 ms | gliner 80.0%, laya 5.8%, llm 14.2% |
| 0.70 | 70.0% | 17.0% | 122 ms | gliner 76.6%, laya 6.4%, llm 17.0% |
| 0.75 | 69.6% | 20.2% | 138 ms | gliner 73.2%, laya 6.6%, llm 20.2% |
| 0.80 | 68.0% | 24.2% | 156 ms | gliner 68.8%, laya 7.0%, llm 24.2% |
| 0.85 | 66.0% | 30.2% | 184 ms | gliner 62.2%, laya 7.6%, llm 30.2% |
| 0.90 | 64.4% | 34.2% | 204 ms | gliner 56.4%, laya 9.4%, llm 34.2% |
| 0.95 | 61.2% | 44.4% | 253 ms | gliner 46.4%, laya 9.2%, llm 44.4% |
| 0.97 | 59.2% | 51.8% | 289 ms | gliner 39.2%, laya 9.0%, llm 51.8% |
| 0.99 | 54.8% | 66.0% | 356 ms | gliner 24.8%, laya 9.2%, llm 66.0% |
| never accept | 51.4% | 100.0% | 506 ms | llm 100.0% |
