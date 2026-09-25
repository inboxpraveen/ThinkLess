"""Decision providers.

Local model providers are imported lazily, so the core package does not pull
in torch or model libraries until one is used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import DecisionProvider, ProviderResult, State, render_state
from .llm import LLMDecider
from .rules import Rules
from .systemone import SystemOne, SystemOneError

if TYPE_CHECKING:
    from .gliner import GLiNER
    from .laya import Laya

__all__ = [
    "DecisionProvider",
    "GLiNER",
    "LLMDecider",
    "Laya",
    "ProviderResult",
    "Rules",
    "State",
    "SystemOne",
    "SystemOneError",
    "render_state",
]

_LAZY = {"GLiNER": ".gliner", "Laya": ".laya"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from importlib import import_module

        return getattr(import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
