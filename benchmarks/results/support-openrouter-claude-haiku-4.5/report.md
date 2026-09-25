# Support benchmark

- ThinkLess 0.1.0, run 2026-09-25T06:39:57Z
- Reasoning model: `openrouter:anthropic/claude-haiku-4.5`
- Engine threshold: 0.8
- Reference price for cost estimates: `anthropic:claude-sonnet-5`
- Environment: NVIDIA GeForce RTX 5060 Laptop GPU, Python 3.12.14, torch 2.14.0+cu130
- Tickets: 53
- Billed cost is what the provider reported for each call (OpenRouter does). Reference cost prices the same LLM tokens at the reference model's published rates, so runs on different models and local runs can be compared; tokenizers differ, so compare it by ratio.

| Metric | `llm` | `hybrid` | `models` |
|---|---:|---:|---:|
| Task success (correct action) | 98.1% | 98.1% | 73.6% |
| Intent accuracy | 95.7% | 95.7% | 72.3% |
| Order id accuracy | 100.0% | 100.0% | 100.0% |
| LLM calls per ticket | 1.96 | 1.60 | 0.93 |
| for decisions | 1.06 | 0.70 | 0.00 |
| for replies | 0.91 | 0.91 | 0.93 |
| Decision time per ticket | 2.21 s | 1.33 s | 157 ms |
| Reply generation time per ticket | 2.34 s | 2.50 s | 2.88 s |
| End-to-end latency p50 | 4.10 s | 3.58 s | 2.25 s |
| End-to-end latency p95 | 8.98 s | 7.25 s | 5.28 s |
| LLM tokens per ticket (in / out) | 707 / 126 | 426 / 72 | 210 / 61 |
| LLM tokens spent on decisions | 570 | 236 | 0 |
| Billed cost per 1k tickets | $1.338 | $0.788 | $0.515 |
| Reference cost per 1k tickets | $2.68 | $1.58 | $1.03 |
| Replies passing the grounding check | 100.0% | 100.0% | 100.0% |
| Decisions by plane | rule 34%, llm 66% | rule 39%, model 48%, llm 12% | rule 40%, model 60% |

## Questions in `llm` mode

| Question | Answered by | Escalation rate | Accuracy (labeled) |
|---|---|---:|---:|
| `intent` | llm 51 | 0.0% | 95.7% of 47 |
| `urgency` | llm 51 | 0.0% | n/a |
| `churn_risk` | llm 51 | 0.0% | n/a |
| `wants_human` | llm 51 | 0.0% | 100.0% of 51 |
| `injection` | llm 51 | 0.0% | 100.0% of 51 |
| `order` | llm 51 | 0.0% | 100.0% of 51 |

Failures in `llm`:

- T-031: expected `kb_answer`, got `ask_order_id` (intent `order_status` from llm)

## Questions in `hybrid` mode

| Question | Answered by | Escalation rate | Accuracy (labeled) |
|---|---|---:|---:|
| `intent` | gliner 21, llm 16, laya 14 | 58.8% | 95.7% of 47 |
| `urgency` | laya 49, llm 2 | 3.9% | n/a |
| `churn_risk` | laya 39, llm 12 | 23.5% | n/a |
| `wants_human` | laya 42, llm 6, rules 3 | 11.8% | 100.0% of 51 |
| `injection` | laya 31, llm 18, rules 2 | 35.3% | 100.0% of 51 |
| `order` | gliner 31, rules 20 | 0.0% | 100.0% of 51 |

Failures in `hybrid`:

- T-031: expected `kb_answer`, got `ask_order_id` (intent `order_status` from llm)

## Questions in `models` mode

| Question | Answered by | Escalation rate | Accuracy (labeled) |
|---|---|---:|---:|
| `intent` | gliner 33, laya 18 | 58.8% | 89.4% of 47 |
| `urgency` | laya 51 | 0.0% | n/a |
| `churn_risk` | laya 51 | 0.0% | n/a |
| `wants_human` | laya 48, rules 3 | 0.0% | 98.0% of 51 |
| `injection` | laya 49, rules 2 | 0.0% | 92.2% of 51 |
| `order` | gliner 31, rules 20 | 0.0% | 100.0% of 51 |

Failures in `models`:

- T-010: expected `escalate_returns`, got `escalate_triage` (intent `refund_other` from gliner)
- T-011: expected `escalate_returns`, got `escalate_triage` (intent `refund_other` from laya)
- T-012: expected `escalate_returns`, got `escalate_triage` (intent `refund_other` from gliner)
- T-029: expected `kb_answer`, got `escalate_support` (intent `product_question` from gliner)
- T-030: expected `kb_answer`, got `escalate_triage` (intent `product_question` from gliner)
- T-031: expected `kb_answer`, got `escalate_triage` (intent `refund_other` from gliner)
- T-032: expected `kb_answer`, got `escalate_support` (intent `product_question` from gliner)
- T-033: expected `kb_answer`, got `escalate_triage` (intent `order_status` from gliner)
- T-034: expected `kb_answer`, got `escalate_triage` (intent `product_question` from laya)
- T-035: expected `escalate_support`, got `escalate_triage` (intent `product_question` from gliner)
- T-036: expected `escalate_support`, got `escalate_triage` (intent `other` from gliner)
- T-039: expected `escalate_technical`, got `escalate_triage` (intent `other` from gliner)
- T-042: expected `escalate_technical`, got `escalate_triage` (intent `technical_issue` from laya)
- T-044: expected `escalate_support`, got `escalate_triage` (intent `other` from gliner)
