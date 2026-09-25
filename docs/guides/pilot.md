# Running a pilot

A pilot answers one question with evidence: *on this team's traffic, can
ThinkLess make these decisions as well as the current system, for less?* It
takes about four weeks, touches no user-facing behavior until week three, and
ends with a report both sides can stand behind. This page is the playbook,
for a team trying ThinkLess on its own and for design partners working with
the maintainers.

## What makes a good pilot

- **One agent in production**, with real traffic: at least a few thousand
  decisions a week, so the confidence intervals close.
- **One to three decisions** that run on every request and have a fixed set
  of answers: intent routing, an injection or policy check, a field
  extraction. Not reply writing.
- **A known cost today**, from the LLM bill or the provider's usage
  dashboard, per call or per month.
- **Someone who can label** fifty to a few hundred examples per decision. A
  support lead or the engineer who owns the prompt is usually enough.

## Before week one: agree on success

Write these down before any data comes in, so the result cannot be argued
into a success afterwards:

| Criterion | Typical bar |
|---|---|
| Agreement with the current system, 95% lower bound | 95% on routing, 98% on safety checks |
| Accuracy on the labeled disagreements | ThinkLess right at least as often as the current system |
| Cost per 1,000 decisions | at least 30% lower |
| p95 decision latency | not worse than today |
| Decisions reaching a person | not more than today |

Also agree on data handling. Shadow logs and traces can run with
`capture_content=False`, which keeps only fingerprints of inputs; labeling
then needs a small sample exported with content, under whatever review the
team's data policy requires. Everything runs in the team's own
infrastructure; nothing is sent to the maintainers.

## Week 1: shadow

1. Write each decision as a question, using the option descriptions from the
   current prompt ([migration guide](migration.md), steps 1 and 2).
2. Build the candidate engine: rules for what is certain, a small model, and
   the team's current LLM as the last provider.
3. Wrap the current decision function with [shadow mode](shadow-mode.md),
   with a sample rate and a `max_cost_usd` that fit the traffic.
4. Deploy. Nothing user-facing changes.

Check after a day that `shadow.stats` shows records and no errors, and that
the log is growing at the expected rate.

## Week 2: read, label, fix

1. Run `thinkless shadow report` and look at agreement per question and the
   share settled without an LLM.
2. Export the disagreements and label them. For each one, note who was
   right: ThinkLess, the current system, or neither.
3. Fix what the labels show: a rule that fires too eagerly, overlapping
   option descriptions, a threshold that is too low. Recalibrate with
   `thinkless calibrate` on the labeled export.
4. Redeploy the candidate and let the shadow collect again.

Most of the improvement in a pilot happens here. Expect two rounds.

## Week 3: switch one decision

Pick the decision whose verdict is `ready` and whose disagreements were
mostly the current system's mistakes. Switch it with the current code as the
fallback for uncertain answers ([migration guide](migration.md), step 5),
behind the team's feature flag. Start an audit shadow at one or two percent
against the LLM.

## Week 4: report

The report is short and made of numbers the team produced itself:

- decisions compared, and the agreement with its 95% interval, per question;
- accuracy of each system on the labeled disagreements;
- cost per 1,000 decisions before and after, from the shadow report and the
  bill for the switched week;
- p50 and p95 latency before and after;
- the share of decisions settled without an LLM;
- what was changed in week two, and what did not work.

If the criteria from week zero are met, keep the decision switched and plan
the next one. If not, the report says which criterion failed and why, which
is as useful: switch the flag off, and the agent is exactly where it started.

## Reference deployments

With the team's permission, a finished pilot becomes a reference deployment:
a page in these docs with the use case, the decisions, the architecture
chosen, and the week-four numbers. Three architectures cover most teams:

| Architecture | When | How |
|---|---|---|
| In process | one service owns the decisions | `Engine` inside the agent, adapters for its framework |
| Sidecar | a service in another language, or several replicas that share a GPU | `thinkless serve` next to the agent, `POST /v1/decide` |
| Shared decision service | many agents across teams | one `thinkless serve` deployment, System One endpoint for other ThinkLess engines, MCP for assistants |

To run a pilot with the maintainers, open a discussion on GitHub with the
use case and the rough volume. The week-four report is shared publicly only
if you agree.
