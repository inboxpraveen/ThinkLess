"""Keep the writing in this repository plain.

Fails when a tracked text file contains an em dash or en dash, or one of a
short list of phrases that make technical writing read as filler. Run it
locally before opening a pull request:

    python scripts/check_prose.py

It runs in CI on every push. Benchmark results and the decision benchmark's
task rows and results are skipped: they hold model output and dataset text,
not writing from this project. The task NOTICE is checked.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CHECKED_SUFFIXES = {
    ".md",
    ".py",
    ".toml",
    ".yml",
    ".yaml",
    ".html",
    ".json",
    ".jsonl",
    ".txt",
    ".cfg",
    ".ini",
    ".cff",
}
SKIPPED = {"LICENSE"}
# Model output and third-party dataset text, not writing from this project.
DATA_DIRS = (
    "benchmarks/results/",
    "benchmarks/decisions/tasks/",
    "benchmarks/decisions/results/",
)

DASHES = {chr(0x2014): "em dash", chr(0x2013): "en dash"}

PHRASES = [
    r"\bdelve\b",
    r"\bdelves\b",
    r"\bseamless(ly)?\b",
    r"\bleverag(e|es|ed|ing)\b",
    r"\bcutting[- ]edge\b",
    r"\bgame[- ]chang(er|ing)\b",
    r"\bunleash\w*\b",
    r"\bsupercharg\w*\b",
    r"\brevolutioni[sz]\w*\b",
    r"\btapestry\b",
    r"\bembark\w*\b",
    r"\bin today's (fast-paced|digital)\b",
    r"\bit(?:'s| is) (important|worth) (to note|noting)\b",
    r"\bharness(ing)? the power\b",
    r"\bunlock(ing)? the (power|potential)\b",
    r"\bnavigat(e|ing) the (complexities|landscape)\b",
]
PHRASE_RE = re.compile("|".join(PHRASES), re.IGNORECASE)


def tracked_files() -> list[Path]:
    try:
        output = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        paths = [ROOT / line for line in output.splitlines() if line]
    except (OSError, subprocess.CalledProcessError):
        paths = [p for p in ROOT.rglob("*") if ".git" not in p.parts]
    return [
        p
        for p in paths
        if p.is_file()
        and p.suffix.lower() in CHECKED_SUFFIXES
        and p.name not in SKIPPED
        and "scripts/check_prose.py" not in p.as_posix()
        and not (p.suffix != ".md" and any(part in p.as_posix() for part in DATA_DIRS))
    ]


def main() -> int:
    problems = []
    for path in tracked_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            for char, name in DASHES.items():
                if char in line:
                    problems.append(f"{path.relative_to(ROOT)}:{number}: {name}")
            for match in PHRASE_RE.finditer(line):
                problems.append(
                    f"{path.relative_to(ROOT)}:{number}: filler phrase '{match.group(0)}'"
                )
    if problems:
        print("\n".join(problems))
        print(
            f"\n{len(problems)} prose issue(s). Use a comma, colon, parentheses or a new sentence instead of a dash."
        )
        return 1
    print("prose check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
