"""Any server that speaks the OpenAI Chat Completions API."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import Any

from ..decision import Usage
from .base import LLM, Completion, Message

__all__ = ["OpenAICompatibleLLM"]


class OpenAICompatibleLLM(LLM):
    """Chat Completions client for OpenAI and compatible servers.

    Covers OpenAI itself plus Ollama, vLLM, LM Studio, llama.cpp server,
    OpenRouter, Groq, Together and any other server that implements
    ``/v1/chat/completions``.

    Args:
        model: Model id as the server knows it.
        base_url: Server URL, for example ``http://localhost:11434/v1`` for
            Ollama. ``None`` targets api.openai.com.
        api_key: Defaults to ``OPENAI_API_KEY``. Local servers usually accept
            any value.
        provider: Name used in traces and for price lookup. Use ``ollama``,
            ``vllm`` or ``lmstudio`` for local servers so their cost is 0.
        token_param: ``max_completion_tokens`` (OpenAI's current name) or
            ``max_tokens`` (what most compatible servers accept). Chosen from
            ``base_url`` when omitted.
        supports_json_mode: Whether the server accepts
            ``response_format={"type": "json_object"}``.

    Requires the ``openai`` extra.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        provider: str = "openai",
        token_param: str | None = None,
        supports_json_mode: bool = True,
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise ImportError(
                'OpenAICompatibleLLM needs the openai extra: pip install "thinkless[openai]"'
            ) from exc
        self.model = model
        self.provider = provider
        self._token_param = token_param or ("max_tokens" if base_url else "max_completion_tokens")
        self._json_mode = supports_json_mode
        key = api_key or os.environ.get("OPENAI_API_KEY") or ("unused" if base_url else None)
        self._client = OpenAI(
            base_url=base_url, api_key=key, timeout=timeout, max_retries=max_retries
        )

    def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> Completion:
        chat: list[dict[str, Any]] = []
        if system:
            chat.append({"role": "system", "content": system})
        chat.extend(dict(m) for m in messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": chat,
            self._token_param: max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if json_mode and self._json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        started = time.perf_counter()
        response = self._client.chat.completions.create(**kwargs)
        latency = (time.perf_counter() - started) * 1000.0
        choice = response.choices[0]
        usage = response.usage
        return Completion(
            text=choice.message.content or "",
            model=response.model or self.model,
            usage=Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            ),
            latency_ms=latency,
            stop_reason=choice.finish_reason,
        )
