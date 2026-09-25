"""Cost estimation from a price table."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .decision import Usage

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

__all__ = ["Price", "PriceTable", "default_prices"]

LOCAL_PROVIDERS = frozenset(
    {
        "local",
        "transformers",
        "rules",
        "laya",
        "gliner",
        "scripted",
        "ollama",
        "vllm",
        "lmstudio",
        "llamacpp",
    }
)


class Price(BaseModel):
    provider: str
    model: str
    input: float
    output: float
    match: str = "exact"
    source: str | None = None

    def matches(self, provider: str, model: str) -> bool:
        if provider != self.provider:
            return False
        if self.match == "prefix":
            return model.startswith(self.model)
        return model == self.model


class PriceTable:
    """Looks up per-token prices and turns usage into dollars."""

    def __init__(self, prices: list[Price], as_of: str | None = None) -> None:
        # Longest model id first so a specific entry wins over a prefix.
        self.prices = sorted(prices, key=lambda p: len(p.model), reverse=True)
        self.as_of = as_of

    @classmethod
    def from_toml(cls, text: str) -> PriceTable:
        data: dict[str, Any] = tomllib.loads(text)
        return cls([Price(**entry) for entry in data.get("price", [])], data.get("as_of"))

    @classmethod
    def load(cls, path: str | Path | None = None) -> PriceTable:
        """Load a price table from ``path``, ``$THINKLESS_PRICING`` or the bundled file."""
        target = path or os.environ.get("THINKLESS_PRICING")
        if target:
            return cls.from_toml(Path(target).read_text(encoding="utf-8"))
        bundled = resources.files("thinkless").joinpath("data/pricing.toml")
        return cls.from_toml(bundled.read_text(encoding="utf-8"))

    def find(self, provider: str, model: str) -> Price | None:
        for price in self.prices:
            if price.matches(provider, model):
                return price
        return None

    def cost(self, provider: str, model: str | None, usage: Usage) -> tuple[float, bool]:
        """Dollar cost of ``usage`` and whether the price was known.

        Local providers always cost 0 and count as known.
        """
        if provider in LOCAL_PROVIDERS:
            return 0.0, True
        price = self.find(provider, model or "")
        if price is None:
            return 0.0, False
        dollars = (usage.input_tokens * price.input + usage.output_tokens * price.output) / 1e6
        return dollars, True


@lru_cache(maxsize=1)
def default_prices() -> PriceTable:
    """The process-wide price table (cached)."""
    return PriceTable.load()
