"""Any server that speaks the OpenAI Chat Completions API."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from typing import Any

from ..decision import Usage
from ..logs import get_logger
from .base import LLM, Completion, Message

__all__ = ["OpenAICompatibleLLM", "reported_cost"]

logger = get_logger("llm.openai")


def reported_cost(usage: Any) -> float | None:
    """The dollar cost a server reports in ``usage``, if it reports one.

    OpenRouter adds ``usage.cost`` to every response. The OpenAI SDK keeps
    fields it does not know in ``model_extra``, so both places are checked.
    """
    if usage is None:
        return None
    value = getattr(usage, "cost", None)
    if value is None:
        extra = getattr(usage, "model_extra", None) or {}
        value = extra.get("cost")
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


class OpenAICompatibleLLM(LLM):
    """Chat Completions client for OpenAI and compatible servers.

    Covers OpenAI itself plus Ollama, vLLM, LM Studio, llama.cpp server,
    Groq, Together and any other server that implements
    ``/v1/chat/completions``. For OpenRouter, use
    :class:`~thinkless.llm.OpenRouterLLM`, which sets the endpoint, key and
    reported cost for you.

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
        supports_json_mode: Whether to send ``response_format`` when a caller
            asks for JSON. If the server rejects it, the client retries once
            without it and stops sending it.
        default_headers: Extra HTTP headers sent with every request.
        extra_body: Extra fields merged into every request body, for
            server-specific options.

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
        default_headers: Mapping[str, str] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.provider = provider
        self._token_param = token_param or ("max_tokens" if base_url else "max_completion_tokens")
        self._json_mode = supports_json_mode
        self._extra_body = dict(extra_body or {})
        if client is not None:
            self._client = client
            return
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise ImportError(
                'OpenAICompatibleLLM needs the openai extra: pip install "thinkless[openai]"'
            ) from exc
        key = api_key or os.environ.get("OPENAI_API_KEY") or ("unused" if base_url else None)
        self._client = OpenAI(
            base_url=base_url,
            api_key=key,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=dict(default_headers) if default_headers else None,
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
        if self._extra_body:
            kwargs["extra_body"] = dict(self._extra_body)
        use_json = json_mode and self._json_mode
        if use_json:
            kwargs["response_format"] = {"type": "json_object"}
        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            if not (use_json and _rejects_response_format(exc)):
                raise
            logger.warning(
                "%s rejected response_format; retrying without JSON mode for this client",
                self.model,
            )
            self._json_mode = False
            kwargs.pop("response_format", None)
            response = self._client.chat.completions.create(**kwargs)
        latency = (time.perf_counter() - started) * 1000.0
        choice = response.choices[0]
        usage = response.usage
        return Completion(
            text=choice.message.content or "",
            model=getattr(response, "model", None) or self.model,
            usage=Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            ),
            latency_ms=latency,
            stop_reason=choice.finish_reason,
            cost_usd=reported_cost(usage),
        )


def _rejects_response_format(exc: Exception) -> bool:
    """True for a client error that names response_format (or JSON mode)."""
    status = getattr(exc, "status_code", None)
    if status is not None and not 400 <= int(status) < 500:
        return False
    text = str(exc).lower()
    return "response_format" in text or "json_object" in text or "json mode" in text
