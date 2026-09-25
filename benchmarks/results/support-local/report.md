# Support benchmark

- ThinkLess 0.1.0, run 2026-09-25T05:05:53Z
- Reasoning model: `local:Qwen/Qwen3-1.7B`
- Engine threshold: 0.8
- Reference price for cost estimates: `anthropic:claude-sonnet-5`
- Environment: NVIDIA GeForce RTX 5060 Laptop GPU, Python 3.12.14, torch 2.14.0+cu130
- Tickets: 53
- Cost is an estimate: LLM tokens are counted with the reasoning model's tokenizer and priced at the reference model's published rates. Tokenizers differ, so compare modes by ratio rather than reading the dollar figure as a quote.

| Metric | `llm` | `hybrid` | `models` |
|---|---:|---:|---:|
| Task success (correct action) | 86.8% | 90.6% | 73.6% |
| Intent accuracy | 87.2% | 91.5% | 72.3% |
| Order id accuracy | 94.1% | 100.0% | 100.0% |
| LLM calls per ticket | 1.93 | 1.60 | 0.93 |
| for decisions | 1.04 | 0.68 | 0.00 |
| for replies | 0.89 | 0.93 | 0.93 |
| Decision time per ticket | 1.71 s | 416 ms | 88 ms |
| Reply generation time per ticket | 1.30 s | 1.37 s | 1.34 s |
| End-to-end latency p50 | 3.07 s | 1.87 s | 1.45 s |
| End-to-end latency p95 | 4.24 s | 2.57 s | 2.09 s |
| LLM tokens per ticket (in / out) | 674 / 81 | 351 / 45 | 205 / 36 |
| LLM tokens spent on decisions | 524 | 155 | 0 |
| Reference cost per 1k tickets | $2.16 | $1.16 | $0.77 |
| Replies passing the grounding check | 100.0% | 100.0% | 100.0% |
| Decisions by plane | rule 33%, llm 67% | rule 39%, model 48%, llm 12% | rule 40%, model 60% |

## Questions in `llm` mode

| Question | Answered by | Escalation rate | Accuracy (labeled) |
|---|---|---:|---:|
| `intent` | llm 51 | 0.0% | 87.2% of 47 |
| `urgency` | llm 51 | 0.0% | n/a |
| `churn_risk` | llm 51 | 0.0% | n/a |
| `wants_human` | llm 51 | 0.0% | 98.0% of 51 |
| `injection` | llm 51 | 0.0% | 98.0% of 51 |
| `order` | llm 51 | 0.0% | 94.1% of 51 |

Failures in `llm`:

- T-004: expected `refund_issued`, got `ask_order_id` (intent `refund_duplicate_charge` from llm)
- T-023: expected `order_status`, got `ask_order_id` (intent `order_status` from llm)
- T-031: expected `kb_answer`, got `escalate_support` (intent `other` from llm)
- T-033: expected `kb_answer`, got `escalate_support` (intent `other` from llm)
- T-039: expected `escalate_technical`, got `escalate_returns` (intent `refund_other` from llm)
- T-046: expected `escalate_human`, got `escalate_returns` (intent `refund_other` from llm)
- T-051: expected `escalate_security`, got `escalate_returns` (intent `refund_other` from llm)

## Questions in `hybrid` mode

| Question | Answered by | Escalation rate | Accuracy (labeled) |
|---|---|---:|---:|
| `intent` | gliner 21, llm 16, laya 14 | 58.8% | 91.5% of 47 |
| `urgency` | laya 49, llm 2 | 3.9% | n/a |
| `churn_risk` | laya 39, llm 12 | 23.5% | n/a |
| `wants_human` | laya 42, llm 6, rules 3 | 11.8% | 100.0% of 51 |
| `injection` | laya 31, llm 18, rules 2 | 35.3% | 96.1% of 51 |
| `order` | gliner 31, rules 20 | 0.0% | 100.0% of 51 |

Failures in `hybrid`:

- T-027: expected `subscription_cancelled`, got `escalate_security` (intent `cancel_subscription` from gliner)
- T-031: expected `kb_answer`, got `escalate_support` (intent `other` from llm)
- T-033: expected `kb_answer`, got `escalate_support` (intent `other` from llm)
- T-039: expected `escalate_technical`, got `escalate_support` (intent `other` from llm)
- T-048: expected `escalate_human`, got `escalate_security` (intent `cancel_subscription` from laya)

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
