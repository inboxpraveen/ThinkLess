// The "Test a student" page: loads the student, answers what the visitor
// types, and replays held-out calls. Nothing typed here leaves the browser.
import { loadStudent, predict } from "./student.js";

const root = document.getElementById("student-test");
const base = new URL(".", import.meta.url);
const pct = (x, digits = 1) => `${(100 * x).toFixed(digits)}%`;
const usd = (x) => `$${x.toFixed(6)}`;

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else node.setAttribute(key, value);
  }
  for (const child of children) node.append(child);
  return node;
}

function routeBadge(local) {
  return el(
    "span",
    { class: `st-badge ${local ? "st-local" : "st-llm"}` },
    local ? "answered locally" : "sent to the LLM",
  );
}

async function start() {
  const [meta, buffer] = await Promise.all([
    fetch(new URL("banking77.json", base)).then((r) => r.json()),
    fetch(new URL(root.dataset.weights || "banking77.bin", base)).then((r) => r.arrayBuffer()),
  ]);
  const student = loadStudent(buffer, meta.labels);
  render(meta, student);
}

function render(meta, student) {
  const threshold = meta.threshold;
  const input = el("textarea", {
    class: "st-input",
    rows: "3",
    "aria-label": "A banking customer's message",
  });
  input.value = "I was charged twice for the same coffee, can you fix it?";
  const result = el("div", { class: "st-result", "aria-live": "polite" });

  const examples = el("div", { class: "st-examples" });
  for (const sample of meta.samples.slice(0, 6)) {
    const chip = el("button", { class: "st-chip", type: "button" }, sample.text);
    chip.addEventListener("click", () => {
      input.value = sample.text;
      answer();
    });
    examples.append(chip);
  }

  function answer() {
    const text = input.value;
    result.replaceChildren();
    if (!text.trim()) return;
    const { ranked, ms } = predict(student, text);
    const best = ranked[0];
    const local = best.probability >= threshold;
    const bars = el("div", { class: "st-bars" });
    for (const item of ranked.slice(0, 3)) {
      const bar = el("div", { class: "st-bar" });
      bar.style.width = pct(item.probability);
      bars.append(
        el(
          "div",
          { class: "st-bar-row" },
          el("span", { class: "st-bar-label" }, item.label),
          el("span", { class: "st-bar-track" }, bar),
          el("span", { class: "st-bar-value" }, pct(item.probability)),
        ),
      );
    }
    result.append(
      el(
        "div",
        { class: "st-head" },
        routeBadge(local),
        el("strong", {}, best.label),
        el(
          "span",
          { class: "st-muted" },
          `confidence ${pct(best.probability)}, threshold ${pct(threshold)}, ${ms.toFixed(2)} ms in your browser`,
        ),
      ),
      bars,
      el(
        "p",
        { class: "st-muted" },
        local
          ? "Habits would return this answer in the shape the LLM would have, without calling it."
          : "Below the threshold: Habits would pass this call to the LLM and record its answer to learn from.",
      ),
    );
  }

  input.addEventListener("input", answer);

  // Replay: held-out calls the student never trained on, with the LLM's real answer.
  const log = el("tbody");
  const counters = el("p", { class: "st-counters", "aria-live": "polite" });
  const button = el(
    "button",
    { class: "md-button md-button--primary", type: "button" },
    `Replay ${meta.samples.length} held-out calls`,
  );
  let timer = null;
  button.addEventListener("click", () => {
    if (timer) return;
    log.replaceChildren();
    let i = 0;
    let local = 0;
    let agreed = 0;
    let saved = 0;
    timer = setInterval(() => {
      const sample = meta.samples[i];
      const best = predict(student, sample.text).ranked[0];
      const isLocal = best.probability >= threshold;
      const same = best.label === sample.teacher;
      if (isLocal) {
        local += 1;
        agreed += same ? 1 : 0;
        saved += sample.cost_usd || 0;
      }
      log.prepend(
        el(
          "tr",
          { class: isLocal && !same ? "st-miss" : "" },
          el("td", {}, sample.text),
          el("td", {}, routeBadge(isLocal)),
          el("td", {}, isLocal ? best.label : "(the LLM answers)"),
          el("td", {}, sample.teacher),
          el("td", {}, isLocal ? (same ? "agrees" : "differs") : ""),
        ),
      );
      i += 1;
      counters.textContent =
        `${i} calls: ${local} answered locally, ${agreed} of them the same as the LLM; ` +
        `${usd(saved)} of LLM spend avoided.`;
      if (i >= meta.samples.length) {
        clearInterval(timer);
        timer = null;
      }
    }, 350);
  });

  const t = meta.test;
  const stats = el(
    "div",
    { class: "st-stats" },
    stat(pct(t.local / meta.calls.test), "of test calls answered locally"),
    stat(pct(t.agreed_local / t.local), `agreement with the LLM there (${t.agreed_local} of ${t.local})`),
    stat(pct(t.saved_cost_usd / t.teacher_cost_usd), "of the LLM spend on those calls avoided"),
    stat(pct(meta.compact.same_route), "same route as the full Python student"),
  );

  root.replaceChildren(
    input,
    el("p", { class: "st-muted" }, "Or try a held-out call:"),
    examples,
    result,
    el("h2", { id: "replay" }, "Replay"),
    el(
      "p",
      {},
      "Calls the student never saw, in order. The last column compares the student with the answer Qwen 3.7 Flash actually gave.",
    ),
    button,
    counters,
    el(
      "div",
      { class: "st-table" },
      el(
        "table",
        {},
        el(
          "thead",
          {},
          el("tr", {}, ...["Message", "Route", "Student", "LLM", ""].map((h) => el("th", {}, h))),
        ),
        log,
      ),
    ),
    el("h2", { id: "on-all-test-calls" }, `On all ${meta.calls.test.toLocaleString()} test calls`),
    stats,
  );
  answer();
}

function stat(value, label) {
  return el("div", { class: "st-stat" }, el("strong", {}, value), el("span", {}, label));
}

start().catch((error) => {
  root.textContent = `The student could not load: ${error.message}`;
});
