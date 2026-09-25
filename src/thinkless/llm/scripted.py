"""A deterministic LLM for tests, examples and dry runs."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from ..decision import Usage
from .base import LLM, Completion, Message

__all__ = ["ScriptedLLM"]

Responder = Callable[[Sequence[Message], "str | None"], str]


def _approx_tokens(text: str) -> int:
    # Roughly four characters per token for English text.
    return max(1, len(text) // 4)


class ScriptedLLM(LLM):
    """Returns canned responses.

    Args:
        responses: Either a list of strings returned in order (the last one
            repeats), or a callable ``(messages, system) -> str``.
        model: Name reported in traces.
        latency_ms: Artificial delay, to make dry-run traces look realistic.

    Token usage is approximated from character counts.
    """

    provider = "scripted"

    def __init__(
        self,
        responses: Sequence[str] | Responder,
        *,
        model: str = "scripted",
        latency_ms: float = 0.0,
    ) -> None:
        self.model = model
        self._responses = responses
        self._index = 0
        self._latency_ms = latency_ms
        self.calls: list[tuple[list[Message], str | None]] = []

    def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> Completion:
        started = time.perf_counter()
        self.calls.append((list(messages), system))
        if callable(self._responses):
            text = self._responses(messages, system)
        else:
            if not self._responses:
                raise ValueError("ScriptedLLM has no responses")
            text = self._responses[min(self._index, len(self._responses) - 1)]
            self._index += 1
        if self._latency_ms:
            time.sleep(self._latency_ms / 1000.0)
        prompt = (system or "") + "".join(m.get("content", "") for m in messages)
        return Completion(
            text=text,
            model=self.model,
            usage=Usage(input_tokens=_approx_tokens(prompt), output_tokens=_approx_tokens(text)),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            stop_reason="end_turn",
        )
