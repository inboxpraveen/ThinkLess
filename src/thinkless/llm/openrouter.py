"""OpenRouter: hundreds of models from every major lab behind one API key."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from .openai_compat import OpenAICompatibleLLM

__all__ = ["OPENROUTER_URL", "OpenRouterLLM"]

OPENROUTER_URL = "https://openrouter.ai/api/v1"
PROJECT_URL = "https://github.com/inboxpraveen/ThinkLess"


class OpenRouterLLM(OpenAICompatibleLLM):
    """Models served through OpenRouter.

    OpenRouter routes one OpenAI-compatible API to models from Anthropic,
    OpenAI, Google, Qwen, DeepSeek, Mistral, Meta and others, so a single key
    is enough to compare reasoning models. It reports the billed cost of every
    call, which ThinkLess records in traces instead of an estimate.

    Args:
        model: OpenRouter model slug, for example ``qwen/qwen3.7-flash`` or
            ``anthropic/claude-haiku-4.5``.
        api_key: Defaults to ``OPENROUTER_API_KEY``.
        reasoning: OpenRouter's ``reasoning`` options, for example
            ``{"effort": "low"}`` or ``{"enabled": False}``. Left unset, each
            model uses its own default. Short decisions and replies rarely
            need long reasoning, and reasoning tokens are billed as output.
        provider_preferences: OpenRouter's ``provider`` routing options, for
            example ``{"sort": "price"}`` or ``{"require_parameters": True}``.
        app_name, app_url: Attribution headers OpenRouter shows in its
            dashboards. Set either to ``None`` to omit it.
        **kwargs: Passed to :class:`OpenAICompatibleLLM`.

    Requires the ``openai`` extra.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        reasoning: Mapping[str, Any] | None = None,
        provider_preferences: Mapping[str, Any] | None = None,
        app_name: str | None = "ThinkLess",
        app_url: str | None = PROJECT_URL,
        **kwargs: Any,
    ) -> None:
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key and kwargs.get("client") is None:
            raise ValueError(
                "OpenRouter needs an API key: set OPENROUTER_API_KEY (a .env file works "
                "with the CLI) or pass api_key="
            )
        headers: dict[str, str] = {}
        if app_url:
            headers["HTTP-Referer"] = app_url
        if app_name:
            headers["X-Title"] = app_name
        extra: dict[str, Any] = dict(kwargs.pop("extra_body", None) or {})
        if reasoning is not None:
            extra["reasoning"] = dict(reasoning)
        if provider_preferences is not None:
            extra["provider"] = dict(provider_preferences)
        super().__init__(
            model,
            base_url=OPENROUTER_URL,
            api_key=key,
            provider="openrouter",
            token_param=kwargs.pop("token_param", "max_tokens"),
            default_headers=headers,
            extra_body=extra,
            **kwargs,
        )
