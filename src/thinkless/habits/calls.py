"""Call logs: what ThinkLess records, and the log formats it can read.

ThinkLess writes one JSON line per LLM call to ``<dir>/calls/``. The same
readers accept logs from elsewhere, so ``thinkless habits scan`` works on
traffic recorded before ThinkLess was installed:

* The logs Habits writes itself (``"schema": "thinkless-call/1"``).
* Chat fine-tuning files (``{"messages": [...]}`` ending with the assistant's
  reply, optionally with ``tools``, ``response_format`` and ``model``), which
  most observability tools can export.
* Request and response pairs (``{"request": {...}, "response": {...}}``) with
  a Chat Completions request body and response body, as API gateways log them.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..logs import get_logger
from .apis import APIS, CHAT, api_for_record
from .wire import Call, Output, output_from_message

__all__ = ["CALL_SCHEMA", "CallLog", "Observation", "iter_files", "local_answers", "read_calls"]

logger = get_logger("habits.calls")

CALL_SCHEMA = "thinkless-call/1"
ANSWERED_BY = ("llm", "habit", "audit")


@dataclass(frozen=True)
class Observation:
    """One recorded call, read back.

    Attributes:
        call: The request.
        output: The model's answer, or ``None`` when it had none that fits
            the contract (an error, a refusal, malformed JSON).
        answered_by: ``llm`` when the model answered, ``habit`` when
            ThinkLess did, ``audit`` when ThinkLess had an answer but sent the
            call to the model to check it.
        local: For audits and shadowed calls, what ThinkLess would have
            answered: ``{"fields": {...}, "confidence": ...}``.
    """

    call: Call
    output: Output | None
    ts: str | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None
    host: str | None = None
    answered_by: str = "llm"
    site: str | None = None
    local: Mapping[str, Any] | None = None

    @property
    def teacher(self) -> bool:
        """Whether the answer came from the model, so it can be learned from."""
        return self.answered_by in ("llm", "audit") and self.output is not None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class CallLog:
    """Appends call records to daily JSONL files, one file per process.

    One file per process keeps several workers from interleaving lines.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self._lock = threading.Lock()
        self._handle: Any = None
        self._day: str | None = None

    def _file(self) -> Any:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._handle is None or day != self._day:
            if self._handle is not None:
                self._handle.close()
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"{day}-{os.getpid()}.jsonl"
            self._handle = path.open("a", encoding="utf-8")
            self._day = day
        return self._handle

    def write(
        self,
        request: Mapping[str, Any] | None,
        response: Mapping[str, Any] | None,
        *,
        answered_by: str,
        site: str | None,
        latency_ms: float,
        host: str | None,
        local: Mapping[str, Any] | None = None,
        agreed: bool | None = None,
        site_name: str | None = None,
        api: str = "chat",
    ) -> None:
        record = {
            "schema": CALL_SCHEMA,
            "api": api,
            "ts": _now(),
            "host": host,
            "site": site,
            "site_name": site_name,
            "answered_by": answered_by,
            "latency_ms": round(latency_ms, 2),
            "request": dict(request) if request is not None else None,
            "response": dict(response) if response is not None else None,
            "local": dict(local) if local is not None else None,
            "agreed": agreed,
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            handle = self._file()
            handle.write(line + "\n")
            handle.flush()

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None


def _from_record(record: Mapping[str, Any]) -> Observation | None:
    name = record.get("site_name")
    api = APIS.get(str(record.get("api") or "chat"), CHAT)
    call = api.parse_request(record.get("request"), {"x-thinkless-site": name} if name else {})
    if call is None:
        return None
    site = record.get("site")
    response = record.get("response")
    output = api.parse_response(call, response) if isinstance(response, Mapping) else None
    cost = None
    if output is not None:
        cost = output.cost_usd
    answered_by = str(record.get("answered_by") or "llm")
    if answered_by not in ANSWERED_BY:
        answered_by = "llm"
    return Observation(
        call=call,
        output=output,
        ts=record.get("ts"),
        latency_ms=record.get("latency_ms"),
        cost_usd=cost,
        host=record.get("host"),
        answered_by=answered_by,
        site=site,
        local=record.get("local"),
    )


def _from_finetune(record: Mapping[str, Any]) -> Observation | None:
    messages = list(record.get("messages") or [])
    if len(messages) < 2 or not isinstance(messages[-1], Mapping):
        return None
    reply = messages[-1]
    if reply.get("role") != "assistant":
        return None
    body = {key: value for key, value in record.items() if key != "messages"}
    body["messages"] = messages[:-1]
    call = CHAT.parse_request(body, {})
    if call is None:
        return None
    output = output_from_message(call, reply, model=record.get("model"))
    return Observation(call=call, output=output, ts=record.get("ts"))


def _from_pair(record: Mapping[str, Any]) -> Observation | None:
    request = record.get("request")
    if isinstance(request, Mapping) and isinstance(request.get("body"), Mapping):
        request = request["body"]
    response = record.get("response")
    if isinstance(response, Mapping) and isinstance(response.get("body"), Mapping):
        response = response["body"]
    api = api_for_record(request, response)
    call = api.parse_request(request, {})
    if call is None:
        return None
    output = api.parse_response(call, response)
    return Observation(
        call=call,
        output=output,
        ts=record.get("ts") or record.get("timestamp"),
        latency_ms=record.get("latency_ms"),
        cost_usd=output.cost_usd if output is not None else None,
    )


def observation_from(record: Any) -> Observation | None:
    """Read one log line in any supported format. ``None`` if it is not a chat call."""
    if not isinstance(record, Mapping):
        return None
    if record.get("schema") == CALL_SCHEMA:
        return _from_record(record)
    if "messages" in record:
        return _from_finetune(record)
    if "request" in record:
        return _from_pair(record)
    return None


def local_answers(paths: Iterable[str | Path]) -> dict[str, int]:
    """How many calls ThinkLess answered per site, from the compact records it keeps for them."""
    counts: dict[str, int] = {}
    for path in iter_files(paths):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if '"answered_by": "habit"' not in line or '"request": null' not in line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                site = record.get("site")
                if isinstance(site, str):
                    counts[site] = counts.get(site, 0) + 1
    return counts


def iter_files(paths: Iterable[str | Path]) -> Iterator[Path]:
    """JSONL files under ``paths``, oldest name first within each directory."""
    for path in paths:
        target = Path(path)
        if target.is_dir():
            yield from sorted(target.rglob("*.jsonl"))
        elif target.exists():
            yield target


def read_calls(paths: Iterable[str | Path]) -> Iterator[Observation]:
    """Every chat call in the given files and directories.

    Lines that are not JSON, or not a chat call, are skipped and counted in
    the debug log.
    """
    for path in iter_files(paths):
        skipped = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    observation = observation_from(json.loads(line))
                except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                    observation = None
                if observation is None:
                    skipped += 1
                    continue
                yield observation
        if skipped:
            logger.debug("skipped %d lines in %s that are not chat calls", skipped, path)
