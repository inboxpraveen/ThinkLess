// A Habits student in the browser: the same features and the same linear
// model as thinkless.habits.features and thinkless.habits.student, so a text
// gets the same answer here as in Python. tests/unit/test_student_page.py
// checks both against each other.

const MAX_CHARS = 4000;
const BIAS = 0;
// Python's \w on str: letters, numbers (str.isalnum) and the underscore.
const WORD = /[\p{L}\p{N}_]+/gu;

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

const encoder = new TextEncoder();

// zlib.crc32 of the UTF-8 bytes, never 0 (0 is the bias feature).
export function hashToken(token) {
  let crc = 0xffffffff;
  for (const byte of encoder.encode(token)) crc = CRC_TABLE[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0 || 1;
}

// Sparse, L2-normalized features as a Map of feature id to weight.
export function featurize(text) {
  let chars = Array.from(text);
  if (chars.length > MAX_CHARS) {
    const half = MAX_CHARS / 2;
    text = chars.slice(0, half).join("") + "\n" + chars.slice(chars.length - half).join("");
  }
  // toLowerCase and str.lower() share the Unicode rules, final sigma included.
  const words = text.toLowerCase().match(WORD) || [];
  const counts = new Map();
  const add = (token) => counts.set(token, (counts.get(token) || 0) + 1);
  words.forEach((word, i) => {
    add("w" + word);
    if (i) add("b" + words[i - 1] + " " + word);
    const padded = Array.from("<" + word + ">");
    for (const n of [3, 4]) {
      for (let j = 0; j + n <= padded.length; j++) add("c" + padded.slice(j, j + n).join(""));
    }
  });
  const features = new Map();
  for (const [token, count] of counts) {
    const key = hashToken(token);
    features.set(key, (features.get(key) || 0) + 1 + Math.log(count));
  }
  let norm = 0;
  for (const value of features.values()) norm += value * value;
  norm = Math.sqrt(norm) || 1;
  const scaled = new Map();
  for (const [key, value] of features) scaled.set(key, value / norm);
  scaled.set(BIAS, 1);
  return scaled;
}

// Read the binary weights written by benchmarks/habits/student_page.py.
export function loadStudent(buffer, labels) {
  const view = new DataView(buffer);
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
  if (magic !== "TLS1") throw new Error("not a student weights file");
  const features = view.getUint32(4, true);
  const classes = view.getUint32(8, true);
  if (classes !== labels.length) throw new Error("labels do not match the weights");
  let offset = 12;
  const index = new Map();
  for (let i = 0; i < features; i++) index.set(view.getUint32(offset + 4 * i, true), i);
  offset += 4 * features;
  const scales = new Float32Array(features);
  for (let i = 0; i < features; i++) scales[i] = view.getFloat32(offset + 4 * i, true);
  offset += 4 * features;
  const weights = new Int8Array(buffer, offset, features * classes);
  return { labels, index, scales, weights, classes };
}

// Every label's probability, highest first, plus the time it took.
export function predict(student, text) {
  const started = performance.now();
  const logits = new Float64Array(student.classes);
  for (const [feature, value] of featurize(text)) {
    const row = student.index.get(feature);
    if (row === undefined) continue;
    const scale = student.scales[row] * value;
    const base = row * student.classes;
    for (let c = 0; c < student.classes; c++) logits[c] += student.weights[base + c] * scale;
  }
  let top = -Infinity;
  for (const value of logits) top = Math.max(top, value);
  let total = 0;
  const probs = Array.from(logits, (value) => {
    const p = Math.exp(value - top);
    total += p;
    return p;
  });
  const ranked = probs
    .map((p, i) => ({ label: student.labels[i], probability: p / total, index: i }))
    .sort((a, b) => b.probability - a.probability || a.index - b.index);
  return { ranked, ms: performance.now() - started };
}
