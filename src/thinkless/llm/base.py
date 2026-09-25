"""The reasoning plane interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..decision import Usage

__all__ = ["LLM", "Completion", "Message", "as_messages"]

Message = dict[str, str]


def as_messages(prompt: str | Sequence[Message]) -> list[Message]:
    """Accept a plain prompt or a list of ``{"role", "content"}`` messages."""
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return [dict(m) for m in prompt]


class Completion(BaseModel):
    """The result of one generation.

    ``cost_usd`` is the cost the backend reported for this call (OpenRouter
    does). When it is ``None`` the engine estimates cost from the price table.
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    model: str
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
    stop_reason: str | None = None
    cost_usd: float | None = None
    raw: dict[str, Any] | None = None


class LLM(ABC):
    """A generative model.

    Implementations wrap one backend. ``provider`` identifies the backend in
    traces and in the price table (``anthropic``, ``openai``, ``local``, ...).

    Args of :meth:`complete`:
        messages: Conversation turns, oldest first.
        system: Optional system prompt.
        max_tokens: Upper bound on generated tokens.
        temperature: Sampling temperature. ``None`` leaves the backend
            default; some hosted models reject the parameter entirely, and
            their implementations ignore it.
        json_mode: Ask the backend for a JSON object when it supports that.
    """

    provider: str = "llm"
    model: str = "unknown"

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    @abstractmethod
    def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> Completion:
        raise NotImplementedError

    def warmup(self) -> None:  # noqa: B027 - optional hook
        """Load weights or open connections ahead of the first call."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources."""
