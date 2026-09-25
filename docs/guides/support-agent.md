# The support agent

`thinkless.demo.support` is a complete customer support agent for a fictional
store, Cobalt Market. It exists to show the pattern on something realistic
and to give the benchmarks a workload. Everything is in the package, so
`thinkless demo` and `thinkless bench support` work right after installing.

## What it handles

A ticket is a subject, a message and the signed-in customer id. The agent can
issue refunds for duplicate charges, report order status, cancel memberships,
answer questions from a small help center, and hand anything else to the
right human queue. 53 labeled tickets cover every path, including the hard
cases: order numbers written in ways a regex misses, a customer asking about
someone else's order, prompt injection attempts, explicit requests for a
person, signed-out users, and tickets in Spanish and German.

## The flow

```text
authenticate   rule        signed in?                        no: ask to sign in (template)
triage         decide      intent, urgency, churn_risk,      one batch through the cascade
                           wants_human, injection, order
act            rules/tools security, human handoff, then route by intent:
                             refund     order belongs to customer? duplicate in payments?
                                        under the automatic limit? then refund_payment
                             status     lookup_order
                             cancel     get_subscription, cancel_subscription
                             question   search_kb, then decide: does the article answer it?
                             other      create_ticket for the right queue
reply          generate    two to four sentences from the established facts
               rule        every amount and id in the reply is backed by the facts
```

The code is ordinary Python in `agent.py`. The decision plane is whatever
engine the agent is given, which is what lets the benchmark run the same agent
three ways.

## Decisions versus rules

The agent draws a hard line between what a model may decide and what code
decides:

| Models decide | Code decides |
|---|---|
| What the customer wants | Whether the order belongs to the customer |
| How urgent it is, whether they may leave | Whether a duplicate charge exists in the payment records |
| Whether they ask for a person | Whether the amount is under the automatic refund limit |
| Whether the message tries to instruct the system | Whether the reply mentions only amounts and ids backed by facts |
| Which order they mean | |
| Whether a help article answers the question | |

The models never authorize anything. A model saying `refund_duplicate_charge`
only selects the refund workflow; the refund itself happens only when the
payment records show a duplicate.

## Handling uncertainty

- An intent below threshold goes to a person (`escalate_triage`) rather than
  a guess.
- Only a confident injection answer routes to the security queue. An unsure
  answer that leans yes still fails closed: the ticket's priority rises, the
  trace records `injection_suspected`, and automatic refunds are blocked.
- Urgency and churn risk only set priority, so their thresholds are low (0.2
  and 0.5).

## Thresholds

Routing questions use per-provider thresholds calibrated on a separate labeled
set; see [calibration](calibration.md) for the method and the numbers.

## Running it

```bash
thinkless demo --ticket T-001                    # hybrid mode, local models
thinkless demo --ticket T-051 --mode hybrid --mode llm
thinkless demo --message "My parcel 7730 is late" --customer C-1002 --view
```

`--view` writes the HTML trace viewer and opens it.

From Python:

```python
from thinkless.demo.support import SupportAgent, SupportStack, World, load_scenarios
from thinkless.llm import from_spec

stack = SupportStack(from_spec("local"))
agent = SupportAgent(stack.engine("hybrid"))
ticket = load_scenarios()[0]
outcome, summary = agent.handle(ticket, World())
print(outcome.action, outcome.reply)
print(summary.llm_calls, summary.decisions_by_plane)
```

## Adapting it

To build your own agent from it, keep the structure and replace the parts:
the questions in `questions.py`, the tools in `world.py`, and the routing in
`agent.py`. Then write a calibration set for your questions before choosing
thresholds.
