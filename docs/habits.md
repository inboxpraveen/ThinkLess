# Habits

The decisions your agent keeps making become habits: answered locally, in
well under a millisecond, once a small model has shown that it gives the same
answers as your LLM.

ThinkLess goes under the LLM calls your agent already makes. You don't write
questions, pick models or set thresholds, and your agent's code doesn't
change. It records every call, finds the calls whose answers are decisions,
trains a small local student for each from the LLM's own answers, and
answers those calls itself only once the student has passed a statistical
test of agreement with your LLM, twice. A site that passes becomes a habit.
Audits keep checking it, and a habit that stops agreeing goes back to the LLM
on its own.

```python
from openai import OpenAI
from thinkless import habits

client = habits.wrap(OpenAI())      # the only change
```

Every response has the shape the LLM would have returned, so `parse()`,
Pydantic models, LangChain's `with_structured_output`, LangGraph routers and
the OpenAI Agents SDK see nothing different. A local answer costs nothing
and takes about half a millisecond inside ThinkLess, or about 4 ms for the
whole SDK call (see [Data, privacy and operations](#data-privacy-and-operations)).

## Why it works

Most agents make the same few decisions over and over, through an LLM: route
the request, grade the retrieved document, check the reply, detect the
language, decide whether a person is needed. Each of those call sites has a
fixed prompt template and a small set of possible answers. Every answer the
LLM gives is a labeled example. When the LLM answers a site consistently, a
few hundred to a few thousand calls teach a linear model to give the same
answer on the inputs it is sure about, and the rest go to the LLM as before.
When the LLM answers similar inputs differently, no student can match it;
ThinkLess leaves that site to the LLM and tells you it is hard to predict.

What makes it safe to switch on is that a student has to pass twice before
it answers anything. First on past calls it was not trained on: its answers
must match the LLM's at least 97% of the time, as a one-sided 95% lower
confidence bound. Then on probation, on the next few hundred live calls,
which the LLM still answers while the student answers them in shadow. Until a
site clears both, it behaves exactly as it did before.

## Get started

### 1. See what your calls are, with no risk

If you already log LLM calls, scan the logs first. Nothing is changed or sent
anywhere:

```bash
pip install thinkless
thinkless habits scan logs/                   # JSONL files or directories
```

The scan reads chat fine-tuning files (`{"messages": [...]}` ending with the
reply, the export format most observability tools offer), request and
response pairs as API gateways log them, and the logs Habits writes. It groups
the calls into sites and reports, for each, its share of calls and cost, what
its answers look like (`intent: 4 options`, `wants_human: yes/no`,
`order_id: copied from the input`, `reply: free text`) and a verdict:

| Verdict | Meaning |
|---|---|
| `decision` | Every field of the answer comes from a closed set or is copied from the input. ThinkLess can learn it. |
| `mixed` | A decision packed together with free text in one call. Splitting the call lets the decision part be answered locally. |
| `generation` | Free text. Only the LLM writes these. |

### 2. Put Habits under the client

=== "OpenAI SDK"

    ```python
    from openai import OpenAI
    from thinkless import habits

    client = habits.wrap(OpenAI())
    ```

    Works for anything built on the OpenAI SDK: OpenAI, Azure OpenAI,
    OpenRouter, Groq, Together, Fireworks, DeepSeek, Ollama, vLLM and LM
    Studio clients, and both the Chat Completions and Responses APIs.

=== "Anthropic SDK"

    ```python
    from anthropic import Anthropic
    from thinkless import habits

    client = habits.wrap(Anthropic())
    ```

=== "LangChain and LangGraph"

    ```python
    from langchain_openai import ChatOpenAI
    from thinkless import habits

    llm = habits.wrap(ChatOpenAI(model="gpt-5-mini"))
    router = llm.with_structured_output(Route)   # unchanged
    ```

=== "OpenAI Agents SDK"

    ```python
    from agents import set_default_openai_client
    from openai import AsyncOpenAI
    from thinkless import habits

    set_default_openai_client(habits.wrap(AsyncOpenAI()))
    ```

=== "Any language: the proxy"

    ```bash
    pip install "thinkless[server]"
    thinkless habits proxy --upstream https://api.openai.com/v1
    export OPENAI_BASE_URL=http://127.0.0.1:8787/v1
    ```

    Every request goes to the upstream with the caller's own API key, except
    the calls ThinkLess has proven. One proxy in front of many workers also
    gives them one shared call log and one set of students.

=== "Any httpx client"

    ```python
    from thinkless import habits

    http = habits.http_client()          # or async_http_client()
    ```

    Pass it wherever a library accepts an `http_client`.

### 3. Let it learn

ThinkLess learns in a background thread after every 200 new LLM answers, and
at most once a minute. To see where each site stands:

```bash
thinkless habits status
```

This is the status after [`examples/10_habits.py`](https://github.com/inboxpraveen/ThinkLess/blob/main/examples/10_habits.py),
which routes 1,050 support messages through the OpenAI SDK with a stand-in
LLM:

```text
1,050 calls, 1 site
+-------------------------------------------------------------------------------------------------+
| Site                    |  Calls | Status | Local | Audits agreed | Agreement before activation |
|-------------------------+--------+--------+-------+---------------+-----------------------------|
| Route                   |  1,050 | active |   276 |         19/19 |            held-out 297/297 |
| ed390a6d677b gpt-5-mini | 100.0% |        |       |               |           lower bound 99.0% |
|                         |        |        |       |               |           new calls 394/394 |
|                         |        |        |       |               |           lower bound 99.2% |
+-------------------------------------------------------------------------------------------------+

Answered locally: 276 of 1,050 calls (26.3%), LLM cost unknown; pass --price MODEL=INPUT,OUTPUT to
estimate it.
```

The first 300 calls went to the LLM and the first learning round put the
student on probation. The next 450 still went to the LLM while the student
answered them in shadow; it matched the first 394 of them, so the second
round switched the site on. Of the last 300 calls, 276 were answered locally,
and all 19 audits agreed. The cost is unknown because the stand-in reports no
price. OpenRouter reports the billed cost of every call and ThinkLess records
it; for other providers, `--price MODEL=INPUT,OUTPUT` with your rates in
dollars per million tokens turns the recorded token counts into dollars.

`thinkless habits learn` runs a learning round on demand and prints each
site's threshold and evidence.

## What happens to each call

1. **Match.** The call is matched to its site by model, output contract (the
   JSON schema, the forced tool, or plain text) and prompt template.
2. **Answer or forward.** If the site is active and its student's confidence
   clears the site's threshold, ThinkLess answers. Otherwise the call goes to
   the LLM unchanged. On a site that is learning or on probation, the
   student still answers every call in shadow, and whether it matched the LLM
   is recorded.
3. **Audit.** A random 5% of the calls ThinkLess could answer still go to the
   LLM, and the two answers are compared. Audits are the live check that the
   gate's bound still holds. They also keep easy calls in the training data:
   once a site is answered locally, its confident calls reach the LLM only as
   audits, so with `audit=0` the students would learn from hard cases alone.
   Keep audits on in production.
4. **Suspend.** When audits show a site's agreement has fallen below the
   target (the upper confidence bound on agreement drops below 97%), the site
   is suspended and every call goes to the LLM again. It comes back only when
   a student passes the gate on calls made after the suspension, and then
   probation.
5. **Record.** Every call is written to `.thinkless/habits/calls/`, one
   JSON line each. The LLM's answers are the training data for the next
   round; the calls it was unsure about teach it the most.

Transitions are written to `.thinkless/habits/events.jsonl` and logged as
warnings, so a suspension shows up in your logs.

## The gate, precisely

A student goes through three checks.

1. **The gate, on past calls.** Each learning round tests a new student.
   Candidate thresholds are tested in a fixed sequence on cross-fitted calls
   (each call scored by a student that never saw it), and the round keeps the
   least selective threshold at which the one-sided Clopper-Pearson lower
   bound on agreement stays at or above the target. Within one round, the
   chance of passing a threshold whose true agreement is below 97% is at most
   5%.
2. **Probation, on new calls.** Rounds repeat, and a site that fails many
   rounds on overlapping data gets many tries, so a marginal student can pass
   one by luck. Passing the gate therefore only starts a probation: the
   student answers new calls in shadow while the LLM keeps answering them,
   and the next round tests it once on the first 394 of those calls at the
   default target and confidence (the number is fixed in advance, and those
   calls were never used before). Only a student that passes this test
   answers anything.
3. **Audits, for as long as it serves.** A student that passed keeps serving
   until a newer one passes the gate or audits suspend the site. Audits
   compare a random share of its answers with the LLM on live traffic and
   suspend the site when the upper bound on agreement falls below the
   target. They catch a student whose traffic has changed, and one that
   should not have passed. A newer student that replaces a serving one
   passes only the gate, not a second probation: the site has already shown
   it can be learned, the newer student was trained on more calls, and the
   audit window starts again on it.

`confirm_calls` sets the size of the probation test; `confirm_calls=0` skips
probation, for decisions where a single gate is enough.

What that covers, and what it does not:

- **It is agreement with your LLM, not correctness.** When the LLM is wrong,
  a student that agrees with it is wrong too. On the replays below, accuracy
  against the true labels moved by the disagreement rate or less.
- **It holds for traffic like the traffic it was measured on.** When your
  inputs change, audits notice and the site is suspended; the bound itself
  says nothing about inputs it has never seen.
- **It is measured on students trained on four fifths of the calls.** The
  student that serves is trained on all of them, which usually makes it
  slightly better than the measurement.
- **The whole answer counts.** A call with three fields agrees only when all
  three match.

The target and confidence are settings:
`Habits(target=0.99, confidence=0.99)`, or
`thinkless habits learn --target 0.99 --confidence 0.99`. A stricter
target needs more calls before a site is switched on.

## What ThinkLess answers, and what it never does

It answers calls whose contract is a JSON schema, JSON mode, a forced tool
(`tool_choice` naming one function) or plain text, when every field of the
answer is a decision or a value copied from the input.

It always sends to the LLM:

- calls where the model may call tools on its own (an agent loop's next
  step), since a wrong local tool call would have side effects;
- inputs with images, audio or files;
- requests for several choices or for log probabilities;
- Responses API calls that continue a stored conversation
  (`previous_response_id`), since the conversation is not in the request;
- Anthropic calls with extended thinking;
- streamed Responses and Messages calls (recorded and learned from, but
  answered by the model; streamed Chat Completions calls are answered
  locally);
- sites whose answers are free text.

## Modes

| Mode | What it does |
|---|---|
| `auto` (default) | Answers active sites locally, audits, suspends when agreement drops. |
| `shadow` | Computes the local answer for every call and records whether it matched, but always uses the LLM. |
| `record` | Only records calls. |
| `off` | Passes everything through. |

```python
runtime = habits.Habits(".thinkless/habits", mode="shadow")
client = habits.wrap(OpenAI(), runtime)
```

The environment variables `THINKLESS_HABITS_DIR` and
`THINKLESS_HABITS_MODE` configure the process-wide `Habits` runtime that
`wrap(client)` uses when none is passed.

## Naming sites

ThinkLess finds sites by their prompt templates. Name a site yourself with a
header when the template changes on every call (a date or a user name in the
system prompt), or when two different prompts share a model and a JSON schema
and have no fixed instructions of their own, so nothing else tells them
apart. ThinkLess removes the header before the request is sent:

```python
client.chat.completions.create(..., extra_headers={"x-thinkless-site": "intent-router"})
```

## Data, privacy and operations

- **Everything stays on your disk.** Calls, students and the plan live in
  `.thinkless/habits/`. Nothing is sent anywhere except the calls that go
  to your LLM, as before. The call log holds prompts and answers, so treat
  the directory like any other log of user content.
- **Students are small.** A logistic regression over hashed words and
  character n-grams: no downloads, no GPU, NumPy only. A learning round runs
  in the background; in the Banking77 replay (77 answers, up to 10,000
  recorded calls, one laptop CPU core) a round took 25 seconds on average,
  and a few seconds while a site had a few hundred calls.
- **Local answers are fast.** Inside ThinkLess a local answer takes about
  half a millisecond (0.45 ms median, 1.3 ms at the 99th percentile on a
  laptop CPU, writing the log record included). Measured from the calling
  code, a whole OpenAI SDK call answered locally takes about 4 ms (4.3 ms
  median in the Banking77 replay with Qwen as teacher), because the SDK still
  builds the request and parses the response.
- **Several workers.** Each process writes its own call file. Point every
  worker at the proxy for one shared log and one set of students, or run
  `thinkless habits learn` on a schedule and set `learn_every=None` in the
  workers; they pick up a new plan within seconds.
- **Retention.** Learning reads the last 30 days of calls and at most the
  10,000 most recent answered calls per site. Delete old files in
  `calls/` whenever you like.
- **Starting over.** `thinkless habits reset` forgets the plan and the
  students; every call goes to the LLM until a site is proven again.

## Results

The replays in `benchmarks/habits/` send real inputs through the OpenAI
SDK with Habits under it. A stand-in replies with the answer a real model
gave for each input, so every local answer can be checked against what the
LLM would have said. The stand-in reports the tokens and cost the real model
was billed.

Each stream has two teachers. Qwen 3.7 Flash's recorded answers (through
OpenRouter, temperature 0, reasoning off, the options listed in the prompt)
are the production case. The dataset's own labels are a teacher that never
contradicts itself, which shows how far a stream can go. Every setting is the
default: 97% target at 95% confidence, probation on 394 new calls, 5% audits,
and a learning round every 500 calls.

| Stream | Teacher | Calls | Predictability | First local answer at call | Answered locally, whole stream | Answered locally, last 500 calls | Local answers matching the teacher | Accuracy vs labels, teacher alone to with Habits | Suspensions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Banking77, 77 intents | Qwen 3.7 Flash | 13,083 | 79.9% | 5,014 | **26.3%** | 43.6% | 98.3% | 73.2% to 73.2% | 0 |
| Banking77, 77 intents | dataset labels | 13,083 | 83.5% | 4,502 | **43.2%** | 66.8% | 98.2% | 100.0% to 99.2% | 0 |
| Civil Comments, is it toxic | Qwen 3.7 Flash | 10,000 | 80.2% | never | **0.0%** | 0.0% | n/a | 82.2% to 82.2% | 0 |
| Civil Comments, is it toxic | dataset labels | 10,000 | 91.7% | 6,001 | **18.7%** | 56.2% | 98.2% | 100.0% to 99.7% | 0 |

- **Routing 77 banking intents with Qwen.** After probation, the site started
  answering at call 5,014. It answered 26.3% of the 13,083 calls locally, and
  43.6% of the last 500; 98.3% of its answers matched what Qwen said.
  Accuracy against the dataset's labels stayed at 73.2%, and the billed LLM
  cost of the stream fell from $0.0975 to $0.0721, 26% less. 211 audits, no
  suspension.
- **Judging toxicity with Qwen, ThinkLess declined.** Qwen called 20.1% of
  the comments toxic, and a student trained on its answers matched its next
  ones 80.2% of the time: no better than always answering no (79.9%). One
  student passed the gate at call 7,500, at the edge of the bound, and never
  answered a call: when the stream ended it was still on probation, with 111
  of its first 116 new calls matching (95.7%), on track to fail the 97%
  target. Probation was slow because the student was confident on only 5.3%
  of calls, so collecting 394 confident answers takes about 7,500 calls.
  `thinkless habits learn` flags such a site as hard to predict.
- **The same streams with a teacher that never contradicts itself** reach
  43.2% and 18.7% of all calls (66.8% and 56.2% of the last 500), with 98.2%
  of local answers matching. The gap to the Qwen rows is the LLM's own
  inconsistency. How much ThinkLess can take over is bounded by how
  predictably your LLM answers a site, and ThinkLess measures that before it
  answers anything.
- **At a few hundred calls, nothing is answered yet.** On the 500-row tasks
  of the [decision benchmark](decision-benchmark/index.md), with the answers
  Qwen gave in its benchmark run, jailbreak detection passed the gate after
  200 calls and was on probation when the stream ended (88 of the first 89
  shadow answers matched); the other sites need more traffic.

| Task | Kind | Calls | Predictability | Status after the stream | Why |
|---|---|---:|---:|---|---|
| jailbreak | yes/no | 399 | 97.0% | probation | checking the student in shadow on new calls: 89 of 394 so far, 88 agreed with the LLM |
| civil_comments | yes/no | 500 | 65.2% | learning | agreement with the LLM has not reached 97% at 95% confidence yet |
| multiwoz | choice | 493 | 46.0% | learning | agreement with the LLM has not reached 97% at 95% confidence yet |
| banking77 | choice | 499 | 52.8% | learning | agreement with the LLM has not reached 97% at 95% confidence yet |
| clinc150 | choice | 500 | 39.6% | learning | agreement with the LLM has not reached 97% at 95% confidence yet |
| massive | choice | 500 | 25.8% | learning | agreement with the LLM has not reached 97% at 95% confidence yet |
| helpsteer2 | score | 500 | 51.6% | learning | agreement with the LLM has not reached 97% at 95% confidence yet |
| wnut17 | extract | 500 | n/a | llm | the answers are free text |

The replays, the recorded answers and the result files are in
[`benchmarks/habits/`](https://github.com/inboxpraveen/ThinkLess/tree/main/benchmarks/habits);
rerunning them costs nothing. Recording the Qwen answers cost $0.14 on
OpenRouter.

## Moving a proven site into your code

Each site's entry in `plan.json` lists the typed ThinkLess questions its
answers correspond to (a `Choice` with the options seen, a `YesNo`, an
`Extract`). When you want rules, GLiNER, Laya or a hosted decision model in
the loop as well, copy those questions into an
[engine](concepts/cascade.md) and replace the call with
`engine.decide(...)`.

## Compared with a cache, and prior work

A semantic cache returns a stored answer when a new prompt is close to an old
one, at a similarity threshold nobody measured. ThinkLess learns a model of
the answers, so it covers inputs it has never seen, and it states a bound on
its agreement with the LLM at the threshold it uses, and keeps checking it.

The idea is not new, and this page should not read as if it were:

- **Online distillation with a fallback.** Distilling an LLM into a small
  local model online, and calling the LLM only when the student is unsure,
  was studied in
  ["Cache & Distil: Optimising API Calls to Large Language Models"](https://arxiv.org/abs/2310.13561)
  (Ramírez et al., 2023). NVIDIA's
  ["Small Language Models are the Future of Agentic AI"](https://arxiv.org/abs/2506.02153)
  (2025) describes an algorithm for converting an agent's LLM calls into
  small models from its own logs.
- **The same gate, for one vendor.**
  [Jevstiller](https://github.com/tomerglick57/Jevstiller) does this for
  TypeSafe's Jev: a proxy that learns a local student from Jev's answers,
  sets its threshold with a Clopper-Pearson bound tested in a fixed
  sequence, and sends a share of traffic back to Jev as an audit. If you only
  call Jev, look at it.

What ThinkLess Habits adds is the place it runs and what it runs on: one
line under the OpenAI (Chat Completions and Responses) and Anthropic SDKs, or
an OpenAI-compatible proxy for any language; a student that needs only NumPy,
with no model download; an offline scan of the logs you already have; a
probation on new calls before a site first answers; and a site that declines
when your LLM's own answers are not predictable enough, as on Civil Comments
above.
