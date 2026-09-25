"""Claude through the official Anthropic SDK."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from ..decision import Usage
from .base import LLM, Completion, Message

__all__ = ["AnthropicLLM"]

FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Model families for which server-side refusal fallbacks are documented.
FALLBACK_FAMILIES = ("claude-opus-5", "claude-fable-5", "claude-mythos-5")


class AnthropicLLM(LLM):
    """Claude models via the Messages API.

    Args:
        model: Claude model id. Defaults to ``claude-opus-5``. For
            high-volume reply generation, ``claude-sonnet-5`` or
            ``claude-haiku-4-5`` are the cheaper options.
        api_key: Defaults to the SDK's credential resolution
            (``ANTHROPIC_API_KEY``, ``ANTHROPIC_AUTH_TOKEN`` or an
            ``ant auth login`` profile).
        effort: Optional ``output_config.effort`` (``low`` to ``max``).
            ``low`` suits short replies and decision questions. Supported on
            Claude Opus 4.6 and later, Sonnet 5 and Fable; Haiku 4.5 and
            Sonnet 4.5 reject it, so leave it unset for those.
        fallbacks: Server-side refusal fallbacks (``fallbacks="default"``):
            if a safety classifier declines the request, the API re-runs it
            on the recommended fallback model in the same call. ``None`` (the
            default) turns them on for the model families that document them
            (Opus 5 and later, Fable 5, Mythos 5) and off otherwise. Available
            on the Claude API and Claude Platform on AWS; pass ``False`` when
            routing through Bedrock, Vertex AI or Foundry.
        client: A preconfigured ``anthropic.Anthropic`` client (tests, proxies,
            custom base URL).

    Sampling parameters are not sent: current Claude models reject
    ``temperature`` and control depth through ``effort`` instead.

    Requires the ``anthropic`` extra.
    """

    provider = "anthropic"

    def __init__(
        self,
        model: str = "claude-opus-5",
        *,
        api_key: str | None = None,
        effort: str | None = None,
        fallbacks: bool | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self._effort = effort
        self._fallbacks = model.startswith(FALLBACK_FAMILIES) if fallbacks is None else fallbacks
        if client is not None:
            self._client = client
            return
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise ImportError(
                'AnthropicLLM needs the anthropic extra: pip install "thinkless[anthropic]"'
            ) from exc
        kwargs: dict[str, Any] = {"timeout": timeout, "max_retries": max_retries}
        if api_key:
            kwargs["api_key"] = api_key
        self._client = anthropic.Anthropic(**kwargs)

    def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> Completion:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [dict(m) for m in messages],
        }
        if system:
            kwargs["system"] = system
        if self._effort:
            kwargs["output_config"] = {"effort": self._effort}
        started = time.perf_counter()
        if self._fallbacks:
            response = self._client.beta.messages.create(
                betas=[FALLBACK_BETA], fallbacks="default", **kwargs
            )
        else:
            response = self._client.messages.create(**kwargs)
        latency = (time.perf_counter() - started) * 1000.0

        # A refusal arrives as HTTP 200 with stop_reason "refusal"; content may
        # be empty or partial, so it is never treated as an answer.
        if response.stop_reason == "refusal":
            text = ""
        else:
            text = "".join(
                block.text for block in response.content if getattr(block, "type", None) == "text"
            )
        usage = response.usage
        return Completion(
            text=text,
            model=getattr(response, "model", None) or self.model,
            usage=Usage(
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
            ),
            latency_ms=latency,
            stop_reason=response.stop_reason,
            raw={"request_id": getattr(response, "_request_id", None)},
        )
