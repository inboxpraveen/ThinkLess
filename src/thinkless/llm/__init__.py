"""Reasoning plane backends.

Heavy backends are imported lazily so ``import thinkless.llm`` works with only
the core dependencies installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import LLM, Completion, Message, as_messages
from .factory import from_spec
from .scripted import ScriptedLLM

if TYPE_CHECKING:
    from .anthropic import AnthropicLLM
    from .local import TransformersLLM
    from .openai_compat import OpenAICompatibleLLM
    from .openrouter import OpenRouterLLM

__all__ = [
    "LLM",
    "AnthropicLLM",
    "Completion",
    "Message",
    "OpenAICompatibleLLM",
    "OpenRouterLLM",
    "ScriptedLLM",
    "TransformersLLM",
    "as_messages",
    "from_spec",
]

_LAZY = {
    "AnthropicLLM": ".anthropic",
    "OpenAICompatibleLLM": ".openai_compat",
    "OpenRouterLLM": ".openrouter",
    "TransformersLLM": ".local",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from importlib import import_module

        return getattr(import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
