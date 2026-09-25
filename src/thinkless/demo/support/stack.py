"""Engines for each benchmark mode, sharing one set of loaded models."""

from __future__ import annotations

from ...engine import Engine
from ...llm.base import LLM
from ...providers import DecisionProvider, LLMDecider, SystemOne
from ...tracing import Tracer
from .questions import CALIBRATED_THRESHOLDS, KB_MATCH, TRIAGE, support_rules

__all__ = ["MODES", "SupportStack"]

MODES = {
    "hybrid": "rules and small local models first, the LLM only when they are not confident",
    "llm": "every decision made by the LLM, the way most agents are built today",
    "models": "rules and small local models only; decisions they are unsure about go to a person",
}


class SupportStack:
    """Holds the providers once and builds an engine per mode.

    Args:
        llm: Reasoning plane, also used as the LLM decision provider.
        device: Device for the local decision models.
        gliner: Include GLiNER in the cascade.
        laya: Include Laya in the cascade.
        jev: Optional hosted System One provider, placed after the local models.
    """

    def __init__(
        self,
        llm: LLM,
        *,
        device: str = "auto",
        gliner: bool = True,
        laya: bool = True,
        jev: SystemOne | None = None,
        extra_providers: list[DecisionProvider] | None = None,
    ) -> None:
        self.llm = llm
        self.decider = LLMDecider(llm)
        self.rules = support_rules()
        self.models: list[DecisionProvider] = []
        if gliner:
            from ...providers.gliner import GLiNER

            self.models.append(GLiNER(device=device))
        if laya:
            from ...providers.laya import Laya

            self.models.append(Laya(device=device))
        if jev is not None:
            self.models.append(jev)
        self.models.extend(extra_providers or [])

    def providers(self, mode: str) -> list[DecisionProvider]:
        if mode == "llm":
            return [self.decider]
        if mode == "hybrid":
            return [self.rules, *self.models, self.decider]
        if mode == "models":
            return [self.rules, *self.models]
        raise ValueError(f"unknown mode {mode!r}; choose one of {', '.join(MODES)}")

    def engine(
        self,
        mode: str,
        *,
        tracer: Tracer | None = None,
        threshold: float = 0.8,
        calibrated: bool = True,
    ) -> Engine:
        """The engine for ``mode``.

        Args:
            calibrated: Apply the per-provider thresholds measured on the
                calibration set. With ``False`` every question uses
                ``threshold`` (or its own), which is useful for comparison.
        """
        thresholds = dict(CALIBRATED_THRESHOLDS) if calibrated else {}
        return Engine(
            self.providers(mode),
            llm=self.llm,
            threshold=threshold,
            thresholds=thresholds,
            tracer=tracer,
        )

    def warmup(self, modes: tuple[str, ...] = tuple(MODES)) -> None:
        """Load every model the given modes need and warm them on the agent's own questions."""
        needed = {id(p): p for mode in modes for p in self.providers(mode)}
        Engine(list(needed.values()), llm=self.llm).warmup([*TRIAGE, KB_MATCH])
