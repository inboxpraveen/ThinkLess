"""The Habits runtime: records calls, answers active sites, audits itself.

For each chat completion request that passes through it, :class:`Habits`:

1. finds the call's site in the current plan;
2. if the site is active and its student's answer clears the site's
   threshold, answers locally, except for a random ``audit`` share of those
   calls, which still go to the LLM so agreement keeps being measured;
3. otherwise forwards the call and records the LLM's answer, which becomes
   training data for the next round of learning;
4. suspends a site, sending all its calls back to the LLM, when audits show
   its agreement has fallen below the target.

Learning runs in a background thread every ``learn_every`` answered calls,
or on demand with :meth:`Habits.learn` and ``thinkless habits learn``.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..logs import get_logger
from .apis import APIS, CHAT, Api
from .calls import CallLog
from .gate import binomial_upper_bound
from .learn import LearningInProgressError, LearnOptions, learn, load_plan, save_plan
from .sites import base_key, call_lines, variable_text
from .student import SiteModel
from .wire import MISSING, Call

__all__ = ["MODES", "Habits", "Ticket", "default_habits"]

logger = get_logger("habits")

MODES = ("auto", "shadow", "record", "off")
AUDIT_WINDOW = 200
MIN_AUDITS = 20
PLAN_CHECK_S = 2.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class Ticket:
    """One call on its way through Habits.

    Attributes:
        call: The parsed request.
        site: The id of the site the call matched, if any.
        local: What the site's student answered: ``(fields, confidence)``.
        confident: Whether that answer cleared the site's threshold.
        audit: Whether the call was picked for an audit.
        response: The locally built response body, when ThinkLess answers.
    """

    call: Call
    body: Mapping[str, Any]
    host: str | None
    started: float
    site: str | None = None
    local: tuple[dict[str, str], float] | None = None
    confident: bool = False
    audit: bool = False
    response: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def answered_locally(self) -> bool:
        return self.response is not None


class Habits:
    """Learns the decisions an agent sends to its LLM and answers them locally once proven.

    Args:
        directory: Where calls, the plan and the students are kept.
        mode: ``auto`` answers active sites locally; ``shadow`` computes
            local answers and logs agreement but always uses the LLM;
            ``record`` only records calls; ``off`` passes everything through.
        target: Agreement with the LLM a site needs before it is answered
            locally, as a lower confidence bound.
        confidence: Level of that bound.
        audit: Share of confident local answers still sent to the LLM to
            keep measuring agreement.
        learn_every: Learn again after this many new LLM answers. ``None``
            leaves learning to :meth:`learn` or the CLI.
        min_calls: Answered calls a site needs before it is trained.
        confirm_calls: New calls a student on probation is checked on, in
            shadow, before it answers any. ``None`` uses the size of the
            gate's first test (394 at the default target and confidence);
            ``0`` switches a site on as soon as it passes on held-out calls.
        record_local_calls: Also keep the full request and response of calls
            ThinkLess answered. Off by default: they teach the students
            nothing, and a compact record (site, answer, confidence, latency)
            is enough for reports, so the log grows with LLM calls only.
        min_learn_interval_s: Shortest time between two background learning
            runs, so busy agents do not learn continuously.
        seed: Seed for audit sampling.
    """

    def __init__(
        self,
        directory: str | Path = ".thinkless/habits",
        *,
        mode: str = "auto",
        target: float = 0.97,
        confidence: float = 0.95,
        audit: float = 0.05,
        learn_every: int | None = 200,
        min_calls: int = 100,
        confirm_calls: int | None = None,
        record_local_calls: bool = False,
        min_learn_interval_s: float = 60.0,
        seed: int | None = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose one of {', '.join(MODES)}")
        if not 0.0 <= audit <= 1.0:
            raise ValueError("audit must be between 0 and 1")
        self.directory = Path(directory)
        self.mode = mode
        self.options = LearnOptions(
            target=target, confidence=confidence, min_calls=min_calls, confirm_calls=confirm_calls
        )
        self.audit = audit
        self.learn_every = learn_every
        self.record_local_calls = record_local_calls
        self.min_learn_interval_s = min_learn_interval_s
        self._last_learn: float | None = None
        self._rng = random.Random(seed)
        self._log = CallLog(self.directory / "calls")
        self._lock = threading.RLock()
        self._plan: dict[str, Any] | None = None
        self._plan_mtime: float | None = None
        self._plan_checked = 0.0
        self._models: dict[str, SiteModel | None] = {}
        self._by_base: dict[str, list[dict[str, Any]]] = {}
        self._audits: dict[str, deque[bool]] = {}
        self._suspended: dict[str, str] = {}
        self._serving: dict[str, Any] = {}
        self._since_learn = 0
        self._learning: threading.Thread | None = None
        self.counts: dict[str, int] = {"local": 0, "llm": 0, "audit": 0, "passthrough": 0}

    # Plan and students

    @property
    def plan(self) -> dict[str, Any] | None:
        self._refresh_plan()
        return self._plan

    def _refresh_plan(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._plan_checked < PLAN_CHECK_S:
            return
        self._plan_checked = now
        path = self.directory / "plan.json"
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return
        if not force and mtime == self._plan_mtime:
            return
        plan = load_plan(self.directory)
        with self._lock:
            self._plan = plan
            self._plan_mtime = mtime
            self._models = {}
            self._by_base = {}
            for entry in (plan or {}).get("sites", {}).values():
                self._by_base.setdefault(entry["base"], []).append(entry)
                # Audits measure one student; a new one starts a new window.
                if self._serving.get(entry["id"]) != entry.get("student"):
                    self._audits.pop(entry["id"], None)
                    self._serving[entry["id"]] = entry.get("student")

    def _model(self, entry: Mapping[str, Any]) -> SiteModel | None:
        stem = entry.get("student")
        if not stem:
            return None
        with self._lock:
            if stem not in self._models:
                path = self.directory / "models" / str(stem)
                try:
                    self._models[stem] = SiteModel.load(path)
                except (OSError, ValueError, KeyError) as exc:
                    logger.warning("could not load the student %s: %s", stem, exc)
                    self._models[stem] = None
            return self._models[stem]

    def _match(self, call: Call) -> dict[str, Any] | None:
        candidates = self._by_base.get(base_key(call))
        if not candidates:
            return None
        ids = {line_id for line_id, _, _ in call_lines(call)}
        best: dict[str, Any] | None = None
        for entry in candidates:
            template = set(entry["template"])
            if template <= ids and (best is None or len(template) > len(best["template"])):
                best = entry
        return best

    # The request path

    def begin(
        self,
        body: Any,
        headers: Mapping[str, str] | None = None,
        host: str | None = None,
        api: Api = CHAT,
    ) -> Ticket | None:
        """Look at a request before it is sent. ``None`` when ThinkLess ignores it.

        Args:
            body: The decoded JSON request body.
            headers: The request headers (for ``x-thinkless-site``).
            host: The upstream host, recorded for cost estimates.
            api: Which API the request speaks.
        """
        if self.mode == "off":
            return None
        call = api.parse_request(body, headers)
        if call is None:
            return None
        ticket = Ticket(call=call, body=body, host=host, started=time.perf_counter())
        if self.mode == "record":
            return ticket
        self._refresh_plan()
        entry = self._match(call)
        if entry is None:
            return ticket
        ticket.site = entry["id"]
        scored = ("active", "probation", "learning", "suspended")
        if entry.get("status") not in scored or not call.answerable:
            return ticket
        model = self._model(entry)
        if model is None:
            return ticket
        text = variable_text(call, frozenset(entry["template"]))
        fields, score = model.predict(text)
        ticket.local = (fields, score)
        threshold = entry.get("threshold")
        ticket.confident = threshold is not None and score >= threshold
        ticket.extra["student"] = entry.get("student")
        # Only an active site answers; on probation the answer is only recorded.
        serving = entry.get("status") == "active" and entry["id"] not in self._suspended
        can_serve = not call.stream or api.streams_locally
        if self.mode == "auto" and serving and ticket.confident and can_serve:
            if self._rng.random() < self.audit:
                ticket.audit = True
            else:
                ticket.response = api.build_response(call, fields, entry.get("response") or {})
                ticket.extra["confidence"] = score
        return ticket

    def served(self, ticket: Ticket) -> None:
        """Record a call ThinkLess answered itself."""
        assert ticket.response is not None
        with self._lock:
            self.counts["local"] += 1
        local = {"fields": ticket.local[0], "confidence": ticket.local[1]} if ticket.local else None
        full = self.record_local_calls
        self._log.write(
            ticket.body if full else None,
            ticket.response if full else None,
            answered_by="habit",
            site=ticket.site,
            latency_ms=(time.perf_counter() - ticket.started) * 1000,
            host=ticket.host,
            local=local,
            site_name=ticket.call.site_name,
            api=ticket.call.api,
        )

    def finish(self, ticket: Ticket, response: Any, *, status_code: int = 200) -> None:
        """Record what the LLM answered for a forwarded call."""
        latency_ms = (time.perf_counter() - ticket.started) * 1000
        if status_code != 200 or not isinstance(response, Mapping):
            with self._lock:
                self.counts["passthrough"] += 1
            return
        output = APIS[ticket.call.api].parse_response(ticket.call, response)
        agreed: bool | None = None
        if ticket.local is not None and output is not None:
            fields = ticket.local[0]
            agreed = all(
                output.fields.get(name, MISSING) == label for name, label in fields.items()
            )
            agreed = agreed and set(output.fields) <= set(fields)
        local = (
            {
                "fields": ticket.local[0],
                "confidence": ticket.local[1],
                "confident": ticket.confident,
                "student": ticket.extra.get("student"),
            }
            if ticket.local
            else None
        )
        self._log.write(
            ticket.body,
            response,
            answered_by="audit" if ticket.audit else "llm",
            site=ticket.site,
            latency_ms=latency_ms,
            host=ticket.host,
            local=local,
            agreed=agreed,
            site_name=ticket.call.site_name,
            api=ticket.call.api,
        )
        with self._lock:
            self.counts["audit" if ticket.audit else "llm"] += 1
            if output is not None:
                self._since_learn += 1
        if ticket.audit and agreed is not None and ticket.site is not None:
            self._record_audit(ticket.site, agreed)
        self._maybe_learn()

    # Audits and suspension

    def _record_audit(self, site_id: str, agreed: bool) -> None:
        with self._lock:
            window = self._audits.setdefault(site_id, deque(maxlen=AUDIT_WINDOW))
            window.append(agreed)
            n, k = len(window), sum(window)
            if n < MIN_AUDITS or site_id in self._suspended:
                return
            upper = binomial_upper_bound(k, n, self.options.confidence)
            if upper >= self.options.target:
                return
            self._suspend(site_id, n, k, upper)

    def _suspend(self, site_id: str, audits: int, agreed: int, upper: float) -> None:
        stamp = _now()
        self._suspended[site_id] = stamp
        self._audits.pop(site_id, None)
        logger.warning(
            "suspended site %s: %d of %d audits agreed with the LLM "
            "(upper bound %.1f%%, target %.1f%%)",
            site_id,
            agreed,
            audits,
            upper * 100,
            self.options.target * 100,
        )
        self._event(
            {
                "event": "suspended",
                "site": site_id,
                "audits": audits,
                "agreed": agreed,
                "upper_bound": round(upper, 4),
            }
        )
        plan = load_plan(self.directory)
        if plan and site_id in plan.get("sites", {}):
            entry = plan["sites"][site_id]
            entry["status"] = "suspended"
            entry["suspended_at"] = stamp
            entry["reason"] = (
                f"audits fell below the target: {agreed} of {audits} agreed with the LLM"
            )
            save_plan(plan, self.directory)
            self._refresh_plan(force=True)

    def _event(self, event: Mapping[str, Any]) -> None:
        path = self.directory / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"ts": _now(), **event}, ensure_ascii=False) + "\n")

    # Learning

    def _maybe_learn(self) -> None:
        if self.learn_every is None or self.mode in ("off", "record"):
            return
        with self._lock:
            if self._since_learn < self.learn_every:
                return
            if self._learning is not None and self._learning.is_alive():
                return
            now = time.monotonic()
            if self._last_learn is not None and now - self._last_learn < self.min_learn_interval_s:
                return
            self._last_learn = now
            self._since_learn = 0
            self._learning = threading.Thread(
                target=self._learn_safely, name="thinkless-habits-learn", daemon=True
            )
            self._learning.start()

    def _learn_safely(self) -> None:
        try:
            self.learn()
        except LearningInProgressError:
            logger.debug("another process is learning in %s; skipped", self.directory)
        except Exception:  # pragma: no cover - logged, never raised into the request path
            logger.exception("background learning failed")

    def learn(self) -> dict[str, Any]:
        """Learn from every recorded call now, and switch to the new plan."""
        before = self.plan or {}
        started = _now()
        plan = learn(self.directory, options=self.options)
        with self._lock:
            # A suspension that happened while learning ran wins over the new plan.
            changed = False
            for site_id, stamp in list(self._suspended.items()):
                entry = plan["sites"].get(site_id)
                if entry is None:
                    continue
                if stamp >= started and entry.get("status") == "active":
                    entry["status"] = "suspended"
                    entry["suspended_at"] = stamp
                    changed = True
                elif entry.get("status") == "active":
                    del self._suspended[site_id]
            if changed:
                save_plan(plan, self.directory)
        for site_id, entry in plan["sites"].items():
            was = (before.get("sites", {}).get(site_id) or {}).get("status")
            if entry.get("status") == "probation" and was != "probation":
                self._event(
                    {
                        "event": "probation",
                        "site": site_id,
                        "name": entry.get("name"),
                        "threshold": entry.get("threshold"),
                        "evidence": entry.get("evidence"),
                    }
                )
            if entry.get("status") == "active" and was != "active":
                self._event(
                    {
                        "event": "activated",
                        "site": site_id,
                        "name": entry.get("name"),
                        "threshold": entry.get("threshold"),
                        "evidence": entry.get("evidence"),
                    }
                )
        self._refresh_plan(force=True)
        return plan

    def wait(self, timeout: float | None = None) -> None:
        """Wait for a background learning run to finish."""
        thread = self._learning
        if thread is not None:
            thread.join(timeout)

    def close(self) -> None:
        self.wait()
        self._log.close()

    def __repr__(self) -> str:
        return f"Habits(directory={str(self.directory)!r}, mode={self.mode!r})"


_default: Habits | None = None
_default_lock = threading.Lock()


def default_habits() -> Habits:
    """The process-wide Habits, configured from the environment.

    ``THINKLESS_HABITS_DIR`` sets the directory (default
    ``.thinkless/habits``) and ``THINKLESS_HABITS_MODE`` the mode
    (default ``auto``).
    """
    global _default
    with _default_lock:
        if _default is None:
            _default = Habits(
                os.environ.get("THINKLESS_HABITS_DIR", ".thinkless/habits"),
                mode=os.environ.get("THINKLESS_HABITS_MODE", "auto"),
            )
        return _default
