"""Build the frozen task files of the decision benchmark.

Maintainers run this once per benchmark version. It downloads every source at
a pinned revision, samples rows with a fixed seed, and writes

    benchmarks/decisions/tasks/<task>/calibration.jsonl
    benchmarks/decisions/tasks/<task>/test.jsonl
    src/thinkless/bench/decisions/tasks.json      (specs and file hashes)

Calibration rows come from each source's train or validation split and test
rows from its test split, so no provider can fit a threshold on test data.
Released files never change; a new sample means a new benchmark version.

    pip install "thinkless[bench]"
    python benchmarks/decisions/build.py
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

VERSION = "1.0"
SEED = 20260925
TEST_ROWS = 500
CALIBRATION_ROWS = 300
# Sampled with a margin, so rows removed as duplicates can be replaced.
CALIBRATION_POOL = CALIBRATION_ROWS + 20

ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = ROOT / "benchmarks" / "decisions" / "tasks"
SPEC_FILE = ROOT / "src" / "thinkless" / "bench" / "decisions" / "tasks.json"

Row = dict[str, Any]


def hub(repo: str, filename: str, revision: str) -> Path:
    return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))


def parquet_rows(repo: str, filename: str, revision: str) -> list[Row]:
    return pq.read_table(hub(repo, filename, revision)).to_pylist()


def class_names(repo: str, filename: str, revision: str, column: str) -> list[str]:
    schema = pq.read_schema(hub(repo, filename, revision))
    features = json.loads(schema.metadata[b"huggingface"])["info"]["features"]
    feature = features[column]
    if feature.get("_type") == "Sequence":
        feature = feature["feature"]
    return list(feature["names"])


def humanize(label: str) -> str:
    return label.replace("_", " ").strip()


def sample(rows: Sequence[Row], n: int, rng: random.Random) -> list[Row]:
    rows = list(rows)
    return rows if len(rows) <= n else rng.sample(rows, n)


def stratified(
    rows: Sequence[Row], n: int, rng: random.Random, key: Callable[[Row], Any]
) -> list[Row]:
    """Equal share per class, topped up at random when a class runs short."""
    groups: dict[Any, list[Row]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    per_class = n // len(groups)
    picked: list[Row] = []
    for label in sorted(groups, key=str):
        picked += sample(groups[label], per_class, rng)
    chosen = {id(r) for r in picked}
    rest = [r for r in rows if id(r) not in chosen]
    picked += sample(rest, n - len(picked), rng) if len(picked) < n else []
    rng.shuffle(picked)
    return picked[:n]


# ------------------------------------------------------------------ tasks

BANKING77_COMMIT = "57ec275d8078af65b7731c2a98be812d844a6d6b"


def banking77(rng: random.Random) -> dict[str, list[Row]]:
    base = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{BANKING77_COMMIT}/banking_data"

    def load(split: str) -> list[Row]:
        text = httpx.get(f"{base}/{split}.csv", timeout=60, follow_redirects=True).text
        return [
            {"state": r["text"], "label": humanize(r["category"]), "source": f"{split}:{i}"}
            for i, r in enumerate(csv.DictReader(io.StringIO(text)))
        ]

    test, train = load("test"), load("train")
    return {
        "test": sample(test, TEST_ROWS, rng),
        "calibration": sample(train, CALIBRATION_POOL, rng),
        "options": sorted({r["label"] for r in test + train}),
    }


CLINC_REV = "155b9c710419136e17307b80d0a13e68cd46b4ec"


def clinc150(rng: random.Random) -> dict[str, list[Row]]:
    names = class_names("clinc/clinc_oos", "plus/test-00000-of-00001.parquet", CLINC_REV, "intent")

    def load(split: str) -> list[Row]:
        rows = parquet_rows("clinc/clinc_oos", f"plus/{split}-00000-of-00001.parquet", CLINC_REV)
        return [
            {
                "state": r["text"],
                "label": "out of scope"
                if names[r["intent"]] == "oos"
                else humanize(names[r["intent"]]),
                "source": f"{split}:{i}",
            }
            for i, r in enumerate(rows)
        ]

    return {
        "test": sample(load("test"), TEST_ROWS, rng),
        "calibration": sample(load("validation"), CALIBRATION_POOL, rng),
        "options": sorted("out of scope" if n == "oos" else humanize(n) for n in names),
    }


MASSIVE_REV = "ed58ac423a2f4121720918bf5301577edce4ffd3"
MASSIVE_LOCALES = ("en-US", "de-DE", "es-ES", "hi-IN", "ja-JP")


def massive(rng: random.Random) -> dict[str, list[Row]]:
    names = class_names("AmazonScience/massive", "en-US/test/0000.parquet", MASSIVE_REV, "intent")

    def load(split: str, per_locale: int) -> list[Row]:
        picked: list[Row] = []
        for locale in MASSIVE_LOCALES:
            rows = parquet_rows(
                "AmazonScience/massive", f"{locale}/{split}/0000.parquet", MASSIVE_REV
            )
            converted = [
                {
                    "state": r["utt"],
                    "label": humanize(names[r["intent"]]),
                    "locale": locale,
                    "source": f"{locale}/{split}:{r['id']}",
                }
                for r in rows
            ]
            picked += sample(converted, per_locale, rng)
        rng.shuffle(picked)
        return picked

    return {
        "test": load("test", TEST_ROWS // len(MASSIVE_LOCALES)),
        "calibration": load("validation", CALIBRATION_POOL // len(MASSIVE_LOCALES)),
        "options": sorted(humanize(n) for n in names),
    }


MULTIWOZ_REV = "a65d2d3261c42a309b634179c186628bf35d9298"


def multiwoz(rng: random.Random) -> dict[str, list[Row]]:
    def load(split: str) -> list[Row]:
        rows = parquet_rows("pfb30/multi_woz_v22", f"v2.2/{split}/0000.parquet", MULTIWOZ_REV)
        turns: list[Row] = []
        for dialogue in rows:
            t = dialogue["turns"]
            history: list[dict[str, str]] = []
            for index, (speaker, text, frames) in enumerate(
                zip(t["speaker"], t["utterance"], t["frames"], strict=True)
            ):
                role = "user" if speaker == 0 else "assistant"
                history.append({"role": role, "content": text})
                if role != "user" or index < 2:
                    continue  # only turns with conversation before them
                active = {
                    state["active_intent"]
                    for state in frames["state"]
                    if state["active_intent"] != "NONE"
                }
                if len(active) > 1:
                    continue  # one intent per question
                label = humanize(active.pop()) if active else "none"
                turns.append(
                    {
                        "state": [dict(m) for m in history],
                        "label": label,
                        "source": f"{split}:{dialogue['dialogue_id']}#{index}",
                    }
                )
        return turns

    test, validation = load("test"), load("validation")
    return {
        "test": sample(test, TEST_ROWS, rng),
        "calibration": sample(validation, CALIBRATION_POOL, rng),
        "options": sorted({r["label"] for r in test + validation}),
    }


JAILBREAK_REV = "2f2ceeb39658696fd3f462403562b6eea5306287"


def jailbreak(rng: random.Random) -> dict[str, list[Row]]:
    def load(split: str) -> list[Row]:
        path = hub(
            "jackhhao/jailbreak-classification",
            f"default/jailbreak_dataset_{split}.csv",
            JAILBREAK_REV,
        )
        with path.open(encoding="utf-8") as handle:
            return [
                {"state": r["prompt"], "label": r["type"] == "jailbreak", "source": f"{split}:{i}"}
                for i, r in enumerate(csv.DictReader(handle))
            ]

    return {
        "test": sample(load("test"), TEST_ROWS, rng),
        "calibration": sample(load("train"), CALIBRATION_POOL, rng),
    }


CIVIL_REV = "f2970eb3a55777454c94069077cc8d9b5866312d"


def civil_comments(rng: random.Random) -> dict[str, list[Row]]:
    def load(split: str, n: int) -> list[Row]:
        rows = parquet_rows(
            "google/civil_comments", f"data/{split}-00000-of-00001.parquet", CIVIL_REV
        )
        converted = [
            {"state": r["text"], "label": r["toxicity"] >= 0.5, "source": f"{split}:{i}"}
            for i, r in enumerate(rows)
            if r["text"] and r["text"].strip()
        ]
        return stratified(converted, n, rng, key=lambda r: r["label"])

    return {"test": load("test", TEST_ROWS), "calibration": load("validation", CALIBRATION_POOL)}


HELPSTEER_REV = "990b2711a36180dd19d9c94b8627844866f8982a"
HELPSTEER_LEVELS = ("very unhelpful", "unhelpful", "somewhat helpful", "helpful", "very helpful")


def helpsteer2(rng: random.Random) -> dict[str, list[Row]]:
    def load(filename: str, n: int) -> list[Row]:
        path = hub("nvidia/HelpSteer2", filename, HELPSTEER_REV)
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        converted = [
            {
                "state": {"prompt": r["prompt"], "response": r["response"]},
                "label": HELPSTEER_LEVELS[int(r["helpfulness"])],
                "source": f"{filename}:{i}",
            }
            for i, r in enumerate(rows)
        ]
        return stratified(converted, n, rng, key=lambda r: r["label"])

    # HelpSteer2 has no test split: its validation split is the test set here.
    return {
        "test": load("validation.jsonl.gz", TEST_ROWS),
        "calibration": load("train.jsonl.gz", CALIBRATION_POOL),
    }


WNUT_REV = "1ac0b6d18c8a1a1d1606b24bef78b8af2f92d2d9"
WNUT_FIELDS = {
    "person": "person",
    "location": "location",
    "corporation": "corporation",
    "product": "product",
    "creative-work": "creative_work",
    "group": "group",
}


def wnut17(rng: random.Random) -> dict[str, list[Row]]:
    names = class_names("leondz/wnut_17", "wnut_17/test/0000.parquet", WNUT_REV, "ner_tags")

    def load(split: str, n: int) -> list[Row]:
        rows = parquet_rows("leondz/wnut_17", f"wnut_17/{split}/0000.parquet", WNUT_REV)
        converted: list[Row] = []
        for r in rows:
            spans: dict[str, list[str]] = defaultdict(list)
            current: list[str] = []
            kind = None
            for token, tag in [
                *zip(r["tokens"], (names[t] for t in r["ner_tags"]), strict=True),
                ("", "O"),
            ]:
                if tag.startswith("I-") and kind == tag[2:]:
                    current.append(token)
                    continue
                if kind is not None:
                    spans[kind].append(" ".join(current))
                kind, current = (tag[2:], [token]) if tag.startswith("B-") else (None, [])
            if any(len(values) > 1 for values in spans.values()):
                continue  # one value per field
            label = dict.fromkeys(WNUT_FIELDS.values())
            for kind_name, values in spans.items():
                label[WNUT_FIELDS[kind_name]] = values[0]
            converted.append(
                {"state": " ".join(r["tokens"]), "label": label, "source": f"{split}:{r['id']}"}
            )
        with_entities = [c for c in converted if any(v is not None for v in c["label"].values())]
        without = [c for c in converted if all(v is None for v in c["label"].values())]
        picked = sample(with_entities, n * 4 // 5, rng) + sample(without, n - n * 4 // 5, rng)
        rng.shuffle(picked)
        return picked

    return {"test": load("test", TEST_ROWS), "calibration": load("validation", CALIBRATION_POOL)}


TASKS: list[dict[str, Any]] = [
    {
        "name": "banking77",
        "build": banking77,
        "title": "Banking77: fine-grained banking support intents",
        "domain": "customer support",
        "question": {
            "kind": "choice",
            "instructions": "What is the customer's banking request about?",
        },
        "source": "https://github.com/PolyAI-LDN/task-specific-datasets",
        "revision": BANKING77_COMMIT,
        "license": "CC BY 4.0",
        "attribution": "Casanueva et al., Efficient Intent Detection with Dual Sentence Encoders, 2020 (PolyAI)",
        "notes": "77 intents. Test rows from the test split, calibration rows from the train split.",
    },
    {
        "name": "clinc150",
        "build": clinc150,
        "title": "CLINC150 plus: 150 intents across 10 domains, and out of scope",
        "domain": "virtual assistant",
        "question": {"kind": "choice", "instructions": "What does the user want?"},
        "source": "https://huggingface.co/datasets/clinc/clinc_oos",
        "revision": CLINC_REV,
        "license": "CC BY 3.0",
        "attribution": "Larson et al., An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction, 2019",
        "notes": "The 'out of scope' option covers requests outside every intent; about one row in five.",
    },
    {
        "name": "massive",
        "build": massive,
        "title": "MASSIVE: 60 assistant intents in five languages",
        "domain": "multilingual assistant",
        "question": {
            "kind": "choice",
            "instructions": "What does the user ask the assistant to do?",
        },
        "source": "https://huggingface.co/datasets/AmazonScience/massive",
        "revision": MASSIVE_REV,
        "license": "CC BY 4.0",
        "attribution": "FitzGerald et al., MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset, 2022 (Amazon)",
        "notes": "100 test rows each from en-US, de-DE, es-ES, hi-IN and ja-JP; options are in English.",
    },
    {
        "name": "multiwoz",
        "build": multiwoz,
        "title": "MultiWOZ 2.2: the user's intent at a turn of a conversation",
        "domain": "multi-turn dialogue",
        "question": {
            "kind": "choice",
            "instructions": "What does the user want at this point in the conversation?",
        },
        "source": "https://huggingface.co/datasets/pfb30/multi_woz_v22",
        "revision": MULTIWOZ_REV,
        "license": "Apache 2.0",
        "attribution": "Zang et al., MultiWOZ 2.2: A Dialogue Dataset with Additional Annotation Corrections and State Tracking Baselines, 2020",
        "notes": (
            "The input is the conversation so far as a list of messages, ending with a user turn. "
            "Only user turns with at least two earlier turns and at most one active intent are "
            "used; 'none' means no active intent, such as a thank-you."
        ),
    },
    {
        "name": "jailbreak",
        "build": jailbreak,
        "title": "Jailbreak classification: prompts that try to override an assistant's rules",
        "domain": "safety",
        "question": {
            "kind": "yes_no",
            "instructions": "Is this prompt a jailbreak attempt that tries to make the assistant ignore its rules?",
        },
        "source": "https://huggingface.co/datasets/jackhhao/jailbreak-classification",
        "revision": JAILBREAK_REV,
        "license": "Apache 2.0",
        "attribution": "jackhhao/jailbreak-classification on Hugging Face",
        "notes": "The whole test split less one repeated prompt (399 rows, 140 jailbreaks); calibration rows from the train split.",
    },
    {
        "name": "civil_comments",
        "build": civil_comments,
        "title": "Civil Comments: toxic comments",
        "domain": "moderation",
        "question": {
            "kind": "yes_no",
            "instructions": "Is this comment toxic: rude, disrespectful or unreasonable enough to make someone leave a discussion?",
        },
        "source": "https://huggingface.co/datasets/google/civil_comments",
        "revision": CIVIL_REV,
        "license": "CC0 1.0",
        "attribution": "Borkan et al., Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification, 2019 (Jigsaw)",
        "notes": "Toxic means a toxicity score of 0.5 or more. Sampled half toxic and half not. Contains offensive language.",
    },
    {
        "name": "helpsteer2",
        "build": helpsteer2,
        "title": "HelpSteer2: how helpful an assistant's reply is",
        "domain": "response quality",
        "question": {
            "kind": "score",
            "instructions": "How helpful is the assistant's response to the prompt?",
            "levels": list(HELPSTEER_LEVELS),
        },
        "source": "https://huggingface.co/datasets/nvidia/HelpSteer2",
        "revision": HELPSTEER_REV,
        "license": "CC BY 4.0",
        "attribution": "Wang et al., HelpSteer2: Open-source dataset for training top-performing reward models, 2024 (NVIDIA)",
        "notes": (
            "Human helpfulness ratings 0 to 4 mapped to five levels, sampled evenly per level. "
            "Test rows from the validation split, calibration rows from the train split. "
            "Human ratings are noisy, so mean absolute error and within-one accuracy are reported "
            "next to exact accuracy."
        ),
    },
    {
        "name": "wnut17",
        "build": wnut17,
        "title": "WNUT 2017: emerging entities in social media text",
        "domain": "extraction",
        "question": {
            "kind": "extract",
            "instructions": "Extract the named entities mentioned in the text.",
            "fields": {
                "person": "a person's name",
                "location": "a place",
                "corporation": "a company or organization",
                "product": "a product",
                "creative_work": "the title of a song, film, book, show or other creative work",
                "group": "a group, such as a band, sports team or community",
            },
        },
        "source": "https://huggingface.co/datasets/leondz/wnut_17",
        "revision": WNUT_REV,
        "license": "CC BY 4.0",
        "attribution": "Derczynski et al., Results of the WNUT2017 Shared Task on Novel and Emerging Entity Recognition, 2017",
        "notes": (
            "Rows with at most one entity of each type; four in five rows mention at least one "
            "entity. Tokens are joined with spaces, as in the source."
        ),
    },
]


def _key(row: Row) -> str:
    return json.dumps(row["state"], sort_keys=True, ensure_ascii=False).casefold()


def deduplicate(splits: dict[str, Any]) -> dict[str, Any]:
    """Drop repeated inputs within a split, and calibration inputs that are also test inputs.

    Some sources repeat a text across their own splits; a provider could then
    fit its threshold on a test row.
    """
    seen: set[str] = set()
    test = []
    for row in splits["test"]:
        if _key(row) not in seen:
            seen.add(_key(row))
            test.append(row)
    calibration = []
    for row in splits["calibration"]:
        if _key(row) not in seen:
            seen.add(_key(row))
            calibration.append(row)
    return {**splits, "test": test, "calibration": calibration[:CALIBRATION_ROWS]}


def write_notice(specs: Sequence[dict[str, Any]]) -> None:
    lines = [
        "# Sources and licenses of the task files",
        "",
        "The files in this directory are samples of public datasets, redistributed under",
        "their licenses. Each was changed in the same ways: rows were sampled with a fixed",
        "seed, labels were renamed to readable text (underscores become spaces), and",
        "records were reformatted as JSON Lines with an `id`, a `state`, a `label` and a",
        "`source` pointing back to the original row. Task-specific changes are listed",
        "below. The ThinkLess license does not apply to these files; each keeps its own.",
        "",
    ]
    for t in specs:
        lines += [
            f"## {t['name']}",
            "",
            f"- Source: {t['source']} (revision `{t['revision']}`)",
            f"- License: {t['license']}",
            f"- Citation: {t['attribution']}",
            f"- Changes: {t['notes']}",
            "",
        ]
    lines += [
        "Civil Comments contains offensive language, and the jailbreak prompts try to",
        "make assistants misbehave. Both are included because detecting them is the task.",
    ]
    (TASK_DIR / "NOTICE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    specs = []
    for task in TASKS:
        rng = random.Random(f"{SEED}:{task['name']}")
        splits = deduplicate(task["build"](rng))
        directory = TASK_DIR / task["name"]
        directory.mkdir(parents=True, exist_ok=True)
        files = {}
        for split in ("calibration", "test"):
            rows = splits[split]
            path = directory / f"{split}.jsonl"
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                for index, row in enumerate(rows):
                    record = {"id": f"{task['name']}/{split}/{index:04d}", **row}
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            files[split] = {
                "path": f"{task['name']}/{split}.jsonl",
                "rows": len(rows),
                "sha256": sha256(path),
            }
        question = dict(task["question"])
        if question["kind"] == "choice":
            question["options"] = splits["options"]
        spec = {k: v for k, v in task.items() if k not in ("build", "question")}
        spec["question"] = question
        spec["files"] = files
        specs.append(spec)
        print(
            f"{task['name']:15} test {files['test']['rows']:4}  calibration {files['calibration']['rows']:4}"
        )
    SPEC_FILE.write_text(
        json.dumps({"version": VERSION, "seed": SEED, "tasks": specs}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    write_notice(specs)
    print(f"wrote {SPEC_FILE.relative_to(ROOT)} and the NOTICE")


if __name__ == "__main__":
    main()
