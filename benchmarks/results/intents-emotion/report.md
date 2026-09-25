# Intent benchmark: emotion

- Dataset: Emotion (dair-ai/emotion), 6 emotions ([source](https://huggingface.co/datasets/dair-ai/emotion))
- Sample: 500 test examples, seed 13, 6 labels
- LLM: `local:Qwen/Qwen3-1.7B`
- Environment: NVIDIA GeForce RTX 5060 Laptop GPU, torch 2.14.0+cu130
- ThinkLess 0.1.0, run 2026-09-25T05:09:39Z

## Providers

| Provider | Accuracy | ECE | p50 latency | p95 latency | Threshold for 95% accuracy |
|---|---:|---:|---:|---:|---|
| `gliner` | 51.6% | 0.120 | 16.7 ms | 19.2 ms | 0.99, answers 4.6% alone |
| `laya` | 54.6% | 0.319 | 28.2 ms | 32.9 ms | not reached |
| `llm` | 47.0% | n/a | 269.2 ms | 373.8 ms | not reached |

## Cascade

Small models are tried in order; an answer below the threshold goes to the next one, and finally to the LLM. Accuracy is for the whole cascade.

| Threshold | Accuracy | Calls reaching the LLM | Mean latency | Answered by |
|---:|---:|---:|---:|---|
| 0.00 | 51.6% | 0.0% | 18 ms | gliner 100.0% |
| 0.05 | 51.6% | 0.0% | 18 ms | gliner 100.0% |
| 0.10 | 51.6% | 0.0% | 18 ms | gliner 100.0% |
| 0.15 | 51.8% | 0.0% | 18 ms | gliner 99.6%, laya 0.4% |
| 0.20 | 51.8% | 0.0% | 18 ms | gliner 99.6%, laya 0.4% |
| 0.25 | 51.6% | 0.0% | 19 ms | gliner 96.8%, laya 3.2% |
| 0.30 | 51.4% | 0.0% | 20 ms | gliner 95.0%, laya 5.0% |
| 0.35 | 51.8% | 0.4% | 22 ms | gliner 92.2%, laya 7.4%, llm 0.4% |
| 0.40 | 53.4% | 0.8% | 26 ms | gliner 81.8%, laya 17.4%, llm 0.8% |
| 0.45 | 54.2% | 2.8% | 35 ms | gliner 66.8%, laya 30.4%, llm 2.8% |
| 0.50 | 54.4% | 5.8% | 47 ms | gliner 58.8%, laya 35.4%, llm 5.8% |
| 0.55 | 54.2% | 7.8% | 54 ms | gliner 53.8%, laya 38.4%, llm 7.8% |
| 0.60 | 54.6% | 10.0% | 61 ms | gliner 50.2%, laya 39.8%, llm 10.0% |
| 0.65 | 54.4% | 12.8% | 71 ms | gliner 44.0%, laya 43.2%, llm 12.8% |
| 0.70 | 54.2% | 14.0% | 75 ms | gliner 40.8%, laya 45.2%, llm 14.0% |
| 0.75 | 54.8% | 17.2% | 85 ms | gliner 36.8%, laya 46.0%, llm 17.2% |
| 0.80 | 55.0% | 20.6% | 96 ms | gliner 30.8%, laya 48.6%, llm 20.6% |
| 0.85 | 55.2% | 26.0% | 112 ms | gliner 25.8%, laya 48.2%, llm 26.0% |
| 0.90 | 54.2% | 31.2% | 129 ms | gliner 20.2%, laya 48.6%, llm 31.2% |
| 0.95 | 53.6% | 41.2% | 160 ms | gliner 12.8%, laya 46.0%, llm 41.2% |
| 0.97 | 52.6% | 48.4% | 182 ms | gliner 9.8%, laya 41.8%, llm 48.4% |
| 0.99 | 51.2% | 65.6% | 233 ms | gliner 4.6%, laya 29.8%, llm 65.6% |
| never accept | 47.0% | 100.0% | 330 ms | llm 100.0% |
