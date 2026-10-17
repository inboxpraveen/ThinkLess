"""Learning: from recorded calls to a plan of which sites ThinkLess may answer.

For every site whose answers are decisions, :func:`learn` trains the
students, measures them on calls they were not trained on (k-fold cross
fitting), runs the gate, and writes the students and a plan file:

    <dir>/plan.json            which sites are active, their thresholds and evidence
    <dir>/models/<site>.json   the students of each site
    <dir>/models/<site>.npz

Nothing is answered locally until a site's evidence clears the gate, so
running ``learn`` early is harmless: sites without enough calls stay on the
LLM and say how many more calls they need.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .._version import __version__
from ..errors import ThinkLessError
from .calls import Observation, iter_files, read_calls
from .features import featurize
from .gate import (
    GateResult,
    binomial_lower_bound,
    first_candidate,
    min_evidence,
    select_threshold,
)
from .sites import SiteProfile, discover
from .student import SiteModel
from .wire import MISSING

__all__ = [
    "PLAN_SCHEMA",
    "LearnOptions",
    "LearningInProgressError",
    "learn",
    "learn_sites",
    "load_plan",
    "save_plan",
]

PLAN_SCHEMA = "thinkless-habits-plan/1"
LOCK_STALE_S = 1800.0


class LearningInProgressError(ThinkLessError):
    """Another process is learning in the same Habits directory."""


@contextmanager
def _learning_lock(directory: Path) -> Iterator[None]:
    """One learning run per directory at a time, across processes."""
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".learning.lock"
    for _ in range(2):
        try:
            handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                stale = time.time() - lock.stat().st_mtime > LOCK_STALE_S
            except OSError:
                stale = True
            if not stale:
                raise LearningInProgressError(
                    f"another process is learning in {directory}"
                ) from None
            lock.unlink(missing_ok=True)
    else:
        raise LearningInProgressError(f"could not take the learning lock in {directory}")
    try:
        os.write(handle, str(os.getpid()).encode())
        yield
    finally:
        os.close(handle)
        lock.unlink(missing_ok=True)


@dataclass(frozen=True)
class LearnOptions:
    """Settings for :func:`learn`.

    Attributes:
        target: Agreement with the LLM a site must reach before it is answered locally.
        confidence: Level of the lower bound on that agreement.
        folds: Cross-fitting folds used to measure the students.
        min_calls: Answered calls a site needs before it is trained.
        max_calls: Most recent answered calls used per site.
        window_days: Only the daily call files Habits wrote in the last
            this many days are read. ``None`` reads them all.
        max_observations: At most this many of the most recent calls are
            read, newest files first.
        confirm_calls: New calls a student on probation is tested on before
            it answers any. ``None`` uses the size of the gate's first test
            (394 at the default target and confidence); ``0`` skips probation,
            so a site answers as soon as it passes on held-out calls.
        seed: Seed for fold assignment and training order.
    """

    target: float = 0.97
    confidence: float = 0.95
    folds: int = 5
    min_calls: int = 100
    max_calls: int = 10000
    window_days: int | None = 30
    max_observations: int = 50000
    confirm_calls: int | None = None
    seed: int = 0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _cross_fit(
    texts: Sequence[str],
    labels: Mapping[str, Sequence[str]],
    kinds: Mapping[str, str],
    features: Sequence[Mapping[int, float]],
    *,
    folds: int,
    seed: int,
) -> tuple[list[float], list[bool]] | None:
    """Joint confidence and agreement for every call, each from a student that never saw it."""
    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)
    fold_of = {index: position % folds for position, index in enumerate(order)}
    scores = [0.0] * len(texts)
    agreed = [False] * len(texts)
    for fold in range(folds):
        train = [i for i in range(len(texts)) if fold_of[i] != fold]
        held = [i for i in range(len(texts)) if fold_of[i] == fold]
        model = SiteModel.fit(
            [texts[i] for i in train],
            {name: [values[i] for i in train] for name, values in labels.items()},
            kinds,
            features=[features[i] for i in train],
            seed=seed,
        )
        if model is None:
            return None
        for i in held:
            answer, score = model.predict(texts[i], features[i])
            scores[i] = score
            agreed[i] = all(answer[name] == labels[name][i] for name in labels)
    return scores, agreed


def _field_specs(site: SiteProfile) -> dict[str, Any]:
    return {
        name: {
            "kind": profile.kind,
            "options": profile.options[:50],
            "unseen": round(profile.unseen, 4),
            "description": profile.description,
        }
        for name, profile in site.fields.items()
    }


def _entry(site: SiteProfile) -> dict[str, Any]:
    call = site.call
    return {
        "id": site.id,
        "name": site.name,
        "base": site.base,
        "template": sorted(site.template),
        "template_lines": site.template_lines[:20],
        "model": call.model,
        "contract": call.contract,
        "schema_name": call.schema_name,
        "calls": site.calls,
        "answered": len(site.answered),
        "verdict": site.verdict,
        "fields": _field_specs(site),
        "questions": [q.spec() for q in site.questions()],
        "response": site.response_template(),
    }


def _evidence(
    gate: GateResult | None, agreed: Sequence[bool], baseline: float | None = None
) -> dict[str, Any]:
    """The gate's evidence, plus how predictable the LLM is on the site.

    ``predictability`` is the share of all held-out calls, confident or not,
    on which the student gave the LLM's answer. ``baseline`` is the share a
    student would get by always giving the LLM's most common answer, so a
    predictability near the baseline means the student learned nothing the
    answer counts did not already say. A low value on a site with few possible
    answers usually means the LLM answers similar inputs differently, which is
    worth knowing whether or not ThinkLess answers.
    """
    found: dict[str, Any] = {"calls": len(agreed)}
    if agreed:
        found["predictability"] = round(sum(agreed) / len(agreed), 4)
    if baseline is not None:
        found["baseline"] = round(baseline, 4)
    if gate is None:
        return found
    found.update(
        accepted=gate.accepted,
        agreed=gate.agreed,
        coverage=round(gate.coverage, 4),
        agreement=round(gate.agreement, 4),
        lower_bound=round(gate.lower_bound, 4),
    )
    return found


def _serving(entry: Mapping[str, Any]) -> bool:
    """Whether a plan entry is an active site with a student on disk."""
    return (
        entry.get("status") == "active"
        and entry.get("threshold") is not None
        and bool(entry.get("student"))
    )


def _next_model(site_id: str, before: Mapping[str, Any]) -> str:
    stem = str(before.get("student") or "")
    try:
        version = int(stem.rsplit("-", 1)[1]) + 1 if stem.startswith(site_id) else 1
    except (IndexError, ValueError):
        version = 1
    return f"{site_id}-{version}"


def _train_and_gate(
    site: SiteProfile,
    entry: dict[str, Any],
    before: Mapping[str, Any],
    root: Path,
    options: LearnOptions,
) -> None:
    """Train a new student for a decision site and fill in its plan entry."""
    answered = site.answered[-options.max_calls :]
    suspended_at = before.get("suspended_at")
    needed = max(options.min_calls, min_evidence(options.target, options.confidence))
    if len(answered) < needed:
        entry["status"] = "suspended" if suspended_at else "learning"
        entry["reason"] = f"{len(answered)} of {needed} answered calls needed"
        entry["evidence"] = {"calls": len(answered)}
        if suspended_at:
            entry["suspended_at"] = suspended_at
        return
    texts = [_text(site, o) for o in answered]
    labels = {
        name: [o.output.fields.get(name, "~missing") for o in answered if o.output is not None]
        for name in site.fields
    }
    kinds = {name: profile.kind for name, profile in site.fields.items()}
    features = [featurize(t) for t in texts]
    measured = _cross_fit(texts, labels, kinds, features, folds=options.folds, seed=options.seed)
    model = SiteModel.fit(texts, labels, kinds, features=features, seed=options.seed)
    if measured is None or model is None:
        entry["status"] = "learning"
        entry["reason"] = "a field could not be learned from its answers"
        return
    scores, agreed = measured
    if suspended_at:
        # After a suspension, only calls made since then count as evidence.
        recent = [i for i, o in enumerate(answered) if (o.ts or "") > suspended_at]
        scores = [scores[i] for i in recent]
        agreed = [agreed[i] for i in recent]
    gate = select_threshold(scores, agreed, target=options.target, confidence=options.confidence)
    stem = _next_model(site.id, before)
    model.save(root / "models" / stem)
    entry["student"] = stem
    entry["trained_on"] = len(answered)
    entry["learned_at"] = _now()
    joint = Counter(zip(*labels.values(), strict=True))
    baseline = max(joint.values()) / len(answered) if joint else None
    entry["evidence"] = _evidence(gate, agreed, baseline)
    if gate is None:
        entry["status"] = "suspended" if suspended_at else "learning"
        if suspended_at:
            entry["suspended_at"] = suspended_at
        entry["reason"] = (
            f"agreement with the LLM has not reached {options.target:.0%} "
            f"at {options.confidence:.0%} confidence yet"
        )
        return
    entry["status"] = "active"
    entry["threshold"] = gate.threshold
    entry["activated_at"] = _now()
    entry["reason"] = (
        f"{gate.agreed} of {gate.accepted} held-out answers agreed with the LLM; "
        f"lower bound {gate.lower_bound:.1%}"
    )


def _confirm_size(options: LearnOptions) -> int:
    if options.confirm_calls is not None:
        return options.confirm_calls
    return first_candidate(options.target, options.confidence)


def _confirm(
    site: SiteProfile, entry: dict[str, Any], before: Mapping[str, Any], options: LearnOptions
) -> bool:
    """Test a student on probation on the new calls it was scored on in shadow.

    The test uses the first ``confirm_calls`` confident answers the student
    gave after its probation began, a fixed number decided in advance, so it
    is a single test on data the student never saw. Returns ``True`` when the
    entry is settled for this round (confirmed, or still collecting), and
    ``False`` when the student failed and a new one should be trained.
    """
    since = str(before.get("probation_since") or "")
    student = before.get("student")
    outcomes: list[bool] = []
    for observation in site.observations:
        local = observation.local or {}
        output = observation.output
        if not observation.teacher or output is None or (observation.ts or "") <= since:
            continue
        if local.get("student") != student or not local.get("confident"):
            continue
        fields = local.get("fields") or {}
        outcomes.append(
            all(output.fields.get(name, MISSING) == label for name, label in fields.items())
            and set(output.fields) <= set(fields)
        )
    needed = _confirm_size(options)
    kept = ("student", "threshold", "evidence", "probation_since", "trained_on", "learned_at")
    if len(outcomes) < needed:
        for key in kept:
            entry[key] = before.get(key)
        entry["status"] = "probation"
        entry["reason"] = (
            f"checking the student in shadow on new calls: {len(outcomes)} of {needed} "
            f"so far, {sum(outcomes)} agreed with the LLM"
        )
        return True
    agreed = sum(outcomes[:needed])
    bound = binomial_lower_bound(agreed, needed, options.confidence)
    confirmation = {"calls": needed, "agreed": agreed, "lower_bound": round(bound, 4)}
    if bound < options.target:
        entry["last_probation"] = {**confirmation, "student": student}
        return False
    for key in kept:
        entry[key] = before.get(key)
    entry["evidence"] = {**dict(before.get("evidence") or {}), "confirmation": confirmation}
    entry["status"] = "active"
    entry["activated_at"] = _now()
    entry.pop("probation_since", None)
    entry["reason"] = (
        f"{agreed} of {needed} new calls agreed with the LLM in shadow "
        f"(lower bound {bound:.1%}), after passing on held-out calls"
    )
    return True


def _prune_models(root: Path, entries: Mapping[str, Any]) -> None:
    """Delete students no plan entry points to."""
    folder = root / "models"
    if not folder.exists():
        return
    keep = {str(e.get("student")) for e in entries.values() if e.get("student")}
    for path in folder.iterdir():
        stem = path.name.split(".", 1)[0]
        if stem not in keep:
            path.unlink(missing_ok=True)


def learn_sites(
    sites: Iterable[SiteProfile],
    directory: str | Path,
    options: LearnOptions | None = None,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train and gate every decision site, save the students, return the plan."""
    options = options or LearnOptions()
    root = Path(directory)
    entries: dict[str, Any] = {}
    earlier = (previous or {}).get("sites", {})
    for site in sites:
        entry = _entry(site)
        entries[site.id] = entry
        entry.update(status="llm", threshold=None, evidence={"calls": 0})
        if site.verdict == "generation":
            entry["reason"] = "the answers are free text"
            continue
        if site.verdict == "mixed":
            open_fields = [n for n, f in site.fields.items() if not f.local]
            entry["reason"] = (
                f"the answer mixes decisions with free text ({', '.join(open_fields)}); "
                "split the call to answer the decisions locally"
            )
            continue
        before = earlier.get(site.id) or {}
        on_probation = before.get("status") == "probation" and before.get("student")
        if on_probation and _confirm(site, entry, before, options):
            continue
        _train_and_gate(site, entry, before, root, options)
        if entry["status"] == "active" and not _serving(before) and _confirm_size(options):
            # Passing on past calls earns a trial, not the site: the student is
            # scored in shadow on new calls, still answered by the LLM, and
            # answers only once it passes one more test on those.
            needed = _confirm_size(options)
            entry["status"] = "probation"
            entry["probation_since"] = _now()
            entry.pop("activated_at", None)
            entry["reason"] = (
                f"{entry['reason']}; now checked in shadow on {needed} new calls "
                "before it answers any"
            )
        elif entry["status"] != "active" and _serving(before):
            # The student that passed before keeps serving, with its threshold:
            # once a site is answered locally, its confident calls stop reaching
            # the LLM, so a fresh gate sees mostly the hard calls and can fail
            # for that reason alone. Audits measure the served calls directly,
            # and suspend the site when they fall below the target.
            waiting = entry.get("reason")
            for key in ("status", "threshold", "student", "evidence", "activated_at"):
                entry[key] = before.get(key)
            entry["reason"] = before.get("reason")
            entry["candidate"] = waiting
    _prune_models(root, entries)
    return {
        "schema": PLAN_SCHEMA,
        "created_at": _now(),
        "thinkless_version": __version__,
        "target": options.target,
        "confidence": options.confidence,
        "sites": entries,
    }


def _text(site: SiteProfile, observation: Observation) -> str:
    from .sites import variable_text

    return variable_text(observation.call, site.template)


def save_plan(plan: Mapping[str, Any], directory: str | Path) -> Path:
    path = Path(directory) / "plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def load_plan(directory: str | Path) -> dict[str, Any] | None:
    path = Path(directory) / "plan.json"
    if not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != PLAN_SCHEMA:
        return None
    return data


def _recent_files(directory: Path, window_days: int | None) -> list[Path]:
    """The daily call files Habits wrote, skipping days older than the window."""
    if not directory.exists():
        return []
    files = sorted(directory.glob("*.jsonl"))
    if window_days is None:
        return files
    cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).strftime("%Y-%m-%d")
    # Files are named YYYY-MM-DD-<pid>.jsonl, so the name sorts by day.
    return [f for f in files if f.name[:10] >= cutoff]


def learn(
    directory: str | Path = ".thinkless/habits",
    *,
    calls: Iterable[str | Path] | None = None,
    options: LearnOptions | None = None,
) -> dict[str, Any]:
    """Learn from recorded calls and write the plan.

    Args:
        directory: The Habits directory; the plan and students go here.
        calls: Log files or directories to learn from. Defaults to
            ``<directory>/calls``.
        options: Gate and training settings.
    """
    root = Path(directory)
    options = options or LearnOptions()
    if calls is not None:
        sources: list[Path] = [Path(c) for c in calls]
    else:
        sources = _recent_files(root / "calls", options.window_days)
    with _learning_lock(root):
        observations: list[Observation] = []
        # Newest files first, until enough calls are read; then back in time order.
        for path in reversed(list(iter_files(sources))):
            observations[:0] = list(read_calls([path]))
            if len(observations) >= options.max_observations:
                break
        observations = observations[-options.max_observations :]
        plan = learn_sites(discover(observations), root, options, previous=load_plan(root))
        save_plan(plan, root)
    return plan
