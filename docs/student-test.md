---
description: Type a banking question and watch a Habits student, trained on an LLM's own answers, decide in your browser whether it can answer without the LLM.
---

# Test a student in your browser

[Habits](habits.md) learns small local models, called students, from the
answers your LLM already gives, and lets one answer a call only once it has
shown it agrees with the LLM. This page runs one of those students in your
browser, with nothing to install.

It learned from 8,000 answers Qwen 3.7 Flash gave to Banking77 customer
messages (77 intents), recorded through OpenRouter. The gate then chose its
threshold on 2,500 further calls: the student answers a call only when its
confidence clears the lowest threshold whose agreement with the LLM is at
least 97 percent as a 95 percent lower bound. Below that, the call goes to
the LLM as it would without ThinkLess.

Type a message a bank customer might send. What you type stays in this page;
the student runs in JavaScript, on your machine.

<link rel="stylesheet" href="../assets/student/student.css">
<div id="student-test"><p>Loading the student (1.2 MB)...</p></div>
<script type="module" src="../assets/student/page.js"></script>
<noscript>This page needs JavaScript to run the student.</noscript>

## What the numbers mean

- **Agreement, not accuracy.** The student is measured against the LLM's
  answers, because that is what it replaces. Qwen 3.7 Flash itself matches
  the Banking77 labels on about 74 percent of these calls; where it is wrong,
  a student that agrees with it is wrong in the same way. On the
  [replays](https://github.com/inboxpraveen/ThinkLess/tree/main/benchmarks/habits),
  accuracy against the true labels moved by less than a point.
- **Messages it is unsure about go to the LLM.** Out-of-domain text ("what is
  the weather today") usually gets a low confidence and goes to the LLM,
  which is the point of the gate: the student only has to be right where it
  answers.
- **A warm start.** This student begins with 8,000 recorded answers. Habits
  in a new deployment starts from zero, records calls, and becomes active
  later: over the whole 13,083-call stream, starting cold, the
  [replay](habits.md#results) answered 26 percent of calls locally at 98 percent
  agreement.
- **The same student as in Python.** The page uses the same hashed features
  and the same weights as `thinkless.habits`, stored as 8-bit integers to
  keep the download small. On the test calls this copy takes the same route
  (local or LLM) and gives the same label as the full student 99.9 percent of
  the time, and a test in the repository checks the JavaScript against the
  Python features and predictions.

Inference here takes about 0.4 ms (median). In an agent, a call
answered locally takes about 4 ms end to end through the OpenAI SDK, because
the SDK still builds and parses the request.

## Run it on your own agent

```bash
pip install thinkless
thinkless habits demo        # offline: recording, probation, then local answers
```

```python
from openai import OpenAI
from thinkless import habits

client = habits.wrap(OpenAI())   # the only change
```

The [Habits guide](habits.md) covers what is recorded, how the gate and the
audits work, and how to turn it off. The student and the numbers on this page
are built by `benchmarks/habits/student_page.py` from the recorded answers, at
no cost.
