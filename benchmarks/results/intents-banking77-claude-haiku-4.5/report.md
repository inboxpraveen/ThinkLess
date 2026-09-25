# Intent benchmark: banking77

- Dataset: Banking77 (PolyAI), 77 banking intents ([source](https://github.com/PolyAI-LDN/task-specific-datasets))
- Sample: 500 test examples, seed 13, 77 labels
- LLM: `openrouter:anthropic/claude-haiku-4.5`
- Environment: NVIDIA GeForce RTX 5060 Laptop GPU, torch 2.14.0+cu130
- ThinkLess 0.1.0, run 2026-09-25T07:21:43Z

## Providers

| Provider | Accuracy | ECE | p50 latency | p95 latency | Cost per 1k | Threshold for 95% accuracy |
|---|---:|---:|---:|---:|---:|---|
| `gliner` | 71.8% | 0.116 | 32.7 ms | 37.3 ms | $0.0000 | 0.99, answers 24.8% alone |
| `laya` | 35.0% | 0.184 | 44.6 ms | 68.6 ms | $0.0000 | not reached |
| `llm` | 76.2% | n/a | 1411.6 ms | 3065.2 ms | $0.6807 | not reached |

## Cascade

Small models are tried in order; an answer below the threshold goes to the next one, and finally to the LLM. Accuracy is for the whole cascade.

| Threshold | Accuracy | Calls reaching the LLM | Mean latency | Cost per 1k | Answered by |
|---:|---:|---:|---:|---:|---|
| 0.00 | 71.8% | 0.0% | 33 ms | $0.0000 | gliner 100.0% |
| 0.05 | 71.8% | 0.0% | 33 ms | $0.0000 | gliner 100.0% |
| 0.10 | 71.8% | 0.0% | 33 ms | $0.0000 | gliner 100.0% |
| 0.15 | 71.8% | 0.0% | 33 ms | $0.0000 | gliner 100.0% |
| 0.20 | 71.8% | 0.0% | 33 ms | $0.0000 | gliner 100.0% |
| 0.25 | 72.0% | 0.4% | 40 ms | $0.0027 | gliner 99.6%, llm 0.4% |
| 0.30 | 72.6% | 1.0% | 53 ms | $0.0068 | gliner 99.0%, llm 1.0% |
| 0.35 | 73.0% | 2.4% | 77 ms | $0.0163 | gliner 97.2%, laya 0.4%, llm 2.4% |
| 0.40 | 72.8% | 3.2% | 88 ms | $0.0218 | gliner 95.4%, laya 1.4%, llm 3.2% |
| 0.45 | 73.0% | 4.6% | 110 ms | $0.0314 | gliner 93.0%, laya 2.4%, llm 4.6% |
| 0.50 | 74.4% | 7.0% | 150 ms | $0.0477 | gliner 90.0%, laya 3.0%, llm 7.0% |
| 0.55 | 75.2% | 9.4% | 191 ms | $0.0643 | gliner 86.2%, laya 4.4%, llm 9.4% |
| 0.60 | 77.0% | 13.2% | 256 ms | $0.0901 | gliner 82.2%, laya 4.6%, llm 13.2% |
| 0.65 | 77.2% | 14.2% | 275 ms | $0.0969 | gliner 80.0%, laya 5.8%, llm 14.2% |
| 0.70 | 76.2% | 17.0% | 319 ms | $0.1161 | gliner 76.6%, laya 6.4%, llm 17.0% |
| 0.75 | 76.4% | 20.2% | 372 ms | $0.1381 | gliner 73.2%, laya 6.6%, llm 20.2% |
| 0.80 | 76.2% | 24.2% | 432 ms | $0.1653 | gliner 68.8%, laya 7.0%, llm 24.2% |
| 0.85 | 76.6% | 30.2% | 533 ms | $0.2063 | gliner 62.2%, laya 7.6%, llm 30.2% |
| 0.90 | 76.2% | 34.2% | 599 ms | $0.2336 | gliner 56.4%, laya 9.4%, llm 34.2% |
| 0.95 | 75.6% | 44.4% | 766 ms | $0.3034 | gliner 46.4%, laya 9.2%, llm 44.4% |
| 0.97 | 75.4% | 51.8% | 909 ms | $0.3542 | gliner 39.2%, laya 9.0%, llm 51.8% |
| 0.99 | 75.4% | 66.0% | 1165 ms | $0.4498 | gliner 24.8%, laya 9.2%, llm 66.0% |
| never accept | 76.2% | 100.0% | 1803 ms | $0.6807 | llm 100.0% |
