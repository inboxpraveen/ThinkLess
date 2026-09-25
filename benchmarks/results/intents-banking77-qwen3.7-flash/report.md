# Intent benchmark: banking77

- Dataset: Banking77 (PolyAI), 77 banking intents ([source](https://github.com/PolyAI-LDN/task-specific-datasets))
- Sample: 500 test examples, seed 13, 77 labels
- LLM: `openrouter:qwen/qwen3.7-flash (reasoning off)`
- Environment: NVIDIA GeForce RTX 5060 Laptop GPU, torch 2.14.0+cu130
- ThinkLess 0.1.0, run 2026-09-25T06:46:14Z

## Providers

| Provider | Accuracy | ECE | p50 latency | p95 latency | Cost per 1k | Threshold for 95% accuracy |
|---|---:|---:|---:|---:|---:|---|
| `gliner` | 71.8% | 0.116 | 32.6 ms | 37.3 ms | $0.0000 | 0.99, answers 24.8% alone |
| `laya` | 35.0% | 0.184 | 48.3 ms | 69.6 ms | $0.0000 | not reached |
| `llm` | 73.8% | n/a | 575.7 ms | 778.2 ms | $0.0190 | not reached |

## Cascade

Small models are tried in order; an answer below the threshold goes to the next one, and finally to the LLM. Accuracy is for the whole cascade.

| Threshold | Accuracy | Calls reaching the LLM | Mean latency | Cost per 1k | Answered by |
|---:|---:|---:|---:|---:|---|
| 0.00 | 71.8% | 0.0% | 33 ms | $0.0000 | gliner 100.0% |
| 0.05 | 71.8% | 0.0% | 33 ms | $0.0000 | gliner 100.0% |
| 0.10 | 71.8% | 0.0% | 33 ms | $0.0000 | gliner 100.0% |
| 0.15 | 71.8% | 0.0% | 33 ms | $0.0000 | gliner 100.0% |
| 0.20 | 71.8% | 0.0% | 33 ms | $0.0000 | gliner 100.0% |
| 0.25 | 71.8% | 0.4% | 36 ms | $0.0001 | gliner 99.6%, llm 0.4% |
| 0.30 | 72.2% | 1.0% | 39 ms | $0.0002 | gliner 99.0%, llm 1.0% |
| 0.35 | 73.0% | 2.4% | 50 ms | $0.0004 | gliner 97.2%, laya 0.4%, llm 2.4% |
| 0.40 | 72.8% | 3.2% | 56 ms | $0.0006 | gliner 95.4%, laya 1.4%, llm 3.2% |
| 0.45 | 73.0% | 4.6% | 65 ms | $0.0009 | gliner 93.0%, laya 2.4%, llm 4.6% |
| 0.50 | 73.8% | 7.0% | 80 ms | $0.0013 | gliner 90.0%, laya 3.0%, llm 7.0% |
| 0.55 | 74.0% | 9.4% | 96 ms | $0.0018 | gliner 86.2%, laya 4.4%, llm 9.4% |
| 0.60 | 75.0% | 13.2% | 123 ms | $0.0025 | gliner 82.2%, laya 4.6%, llm 13.2% |
| 0.65 | 75.4% | 14.2% | 131 ms | $0.0027 | gliner 80.0%, laya 5.8%, llm 14.2% |
| 0.70 | 74.8% | 17.0% | 148 ms | $0.0032 | gliner 76.6%, laya 6.4%, llm 17.0% |
| 0.75 | 75.2% | 20.2% | 169 ms | $0.0038 | gliner 73.2%, laya 6.6%, llm 20.2% |
| 0.80 | 74.8% | 24.2% | 196 ms | $0.0046 | gliner 68.8%, laya 7.0%, llm 24.2% |
| 0.85 | 75.2% | 30.2% | 238 ms | $0.0057 | gliner 62.2%, laya 7.6%, llm 30.2% |
| 0.90 | 74.8% | 34.2% | 264 ms | $0.0065 | gliner 56.4%, laya 9.4%, llm 34.2% |
| 0.95 | 74.2% | 44.4% | 329 ms | $0.0084 | gliner 46.4%, laya 9.2%, llm 44.4% |
| 0.97 | 74.2% | 51.8% | 377 ms | $0.0099 | gliner 39.2%, laya 9.0%, llm 51.8% |
| 0.99 | 73.2% | 66.0% | 472 ms | $0.0126 | gliner 24.8%, laya 9.2%, llm 66.0% |
| never accept | 73.8% | 100.0% | 687 ms | $0.0190 | llm 100.0% |
