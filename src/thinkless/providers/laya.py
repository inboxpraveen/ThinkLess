"""Laya: an open-weight System One decision model that runs locally."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from .._hub import ensure_downloaded
from ..decision import Plane, Usage
from ..logs import get_logger
from ..questions import Kind, Question
from ..settings import resolve_device
from .base import BatchItem, DecisionProvider, ProviderResult, State, group_by_questions
from .wire import from_wire, to_wire

__all__ = ["Laya"]

logger = get_logger("providers.laya")


class Laya(DecisionProvider):
    """Answers choice, score and yes/no questions with Laya.

    Laya is a non-autoregressive encoder (about 421M parameters for the
    English checkpoint) that answers every question of a call in one forward
    pass. Measured on an RTX 5060 laptop GPU: about 30 ms for three questions;
    about 650 ms on CPU.

    Args:
        model: Hugging Face repo of the checkpoint. ``convaiinnovations/laya``
            is the English model; see the Laya project for the multilingual
            and fine-tuned checkpoints.
        device: ``auto``, ``cpu``, ``cuda`` or ``mps``.
        name: Provider name in traces.
        max_len: Token budget for the state. Longer inputs are truncated by
            the model.

    Requires the ``laya`` extra. Weights download on first use.
    """

    plane: ClassVar[Plane] = Plane.MODEL
    kinds: ClassVar[frozenset[Kind]] = frozenset({Kind.CHOICE, Kind.SCORE, Kind.YES_NO})

    def __init__(
        self,
        model: str = "convaiinnovations/laya",
        *,
        device: str = "auto",
        name: str = "laya",
        max_len: int | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self._device_request = device
        self._max_len = max_len
        self._agent: Any = None
        self._lock = threading.Lock()
        self.device: str | None = None

    def warmup(self) -> None:
        """Load the weights and run one tiny inference, so the first request is not slow."""
        agent = self._load()
        with self._lock:
            agent.predict(
                "Hello, I placed order 4471 last Tuesday and the tracking page has not changed since. "
                "Could you tell me when it will arrive, or refund it if it is lost?",
                {
                    "a": {
                        "type": "choice",
                        "instructions": "Topic?",
                        "criteria": {f"t{i}": None for i in range(7)},
                    },
                    "b": {
                        "type": "score",
                        "instructions": "Urgency?",
                        "criteria": ["low", "medium", "high"],
                    },
                    "c": {"type": "noul", "instructions": "Is this a test?"},
                },
            )

    def _load(self) -> Any:
        if self._agent is not None:
            return self._agent
        with self._lock:
            if self._agent is None:
                try:
                    import laya
                except ImportError as exc:  # pragma: no cover - depends on the extra
                    raise ImportError(
                        'The Laya provider needs the laya extra: pip install "thinkless[laya]"'
                    ) from exc
                device = resolve_device(self._device_request)
                started = time.perf_counter()
                ensure_downloaded(self.model)
                self._agent = laya.load(self.model, device=device)
                self.device = device
                logger.info(
                    "loaded %s on %s in %.1f s", self.model, device, time.perf_counter() - started
                )
        return self._agent

    def answer(self, state: State, questions: Mapping[str, Question]) -> ProviderResult:
        agent = self._load()
        kwargs: dict[str, Any] = {}
        if self._max_len is not None:
            kwargs["max_len"] = self._max_len
        with self._lock:
            result = agent.predict(state, to_wire(questions), **kwargs)
        return self._result(result, questions)

    def answer_batch(self, items: Sequence[BatchItem]) -> list[ProviderResult]:
        """Answer many states through Laya's ``predict_batch``, which shares forward passes."""
        agent = self._load()
        if not hasattr(agent, "predict_batch"):
            return super().answer_batch(items)
        kwargs: dict[str, Any] = {}
        if self._max_len is not None:
            kwargs["max_len"] = self._max_len
        out: list[ProviderResult | None] = [None] * len(items)
        for group in group_by_questions(items):
            questions = items[group[0]][1]
            states = [items[i][0] for i in group]
            with self._lock:
                results = agent.predict_batch(states, to_wire(questions), **kwargs)
            if len(results) != len(group):
                raise RuntimeError(f"Laya returned {len(results)} results for {len(group)} states")
            for index, result in zip(group, results, strict=True):
                out[index] = self._result(result, questions, batch=len(group))
        return [r for r in out if r is not None]

    def _result(
        self, result: Mapping[str, Any], questions: Mapping[str, Question], batch: int = 1
    ) -> ProviderResult:
        usage = result.get("usage") or {}
        meta: dict[str, Any] = {"device": self.device, "checkpoint": self.model}
        if batch > 1:
            meta["batch"] = batch
        return ProviderResult(
            answers=from_wire(result.get("answers") or {}, questions),
            model=result.get("model") or self.model,
            usage=Usage(input_tokens=int(usage.get("input_tokens") or 0)),
            meta=meta,
        )

    def close(self) -> None:
        self._agent = None
