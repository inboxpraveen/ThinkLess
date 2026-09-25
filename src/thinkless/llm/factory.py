"""Build an LLM from a short spec string, as used by the CLI."""

from __future__ import annotations

from .base import LLM

__all__ = ["DEFAULT_LOCAL_MODEL", "from_spec"]

DEFAULT_LOCAL_MODEL = "Qwen/Qwen3-1.7B"

BACKENDS = ("local", "anthropic", "openrouter", "openai", "ollama", "vllm")


REASONING = ("default", "off", "minimal", "low", "medium", "high")


def from_spec(
    spec: str,
    *,
    device: str = "auto",
    base_url: str | None = None,
    reasoning: str | None = None,
) -> LLM:
    """Create an LLM from ``backend[:model]``.

    Examples:
        ``local`` (Qwen3-1.7B in-process), ``local:Qwen/Qwen3-4B``,
        ``anthropic`` (Claude Opus 5), ``anthropic:claude-haiku-4-5``,
        ``openrouter:qwen/qwen3.7-flash``, ``openrouter:anthropic/claude-haiku-4.5``,
        ``openai:<model>``, ``ollama:qwen3:8b``, ``vllm:<model>``.

    Args:
        spec: Backend name, optionally followed by a colon and a model id.
        device: Device for the ``local`` backend.
        base_url: Server URL for ``openai``, ``ollama`` and ``vllm``.
        reasoning: ``off``, ``minimal``, ``low``, ``medium`` or ``high``;
            ``None`` or ``default`` keeps the model's own default. Mapped to
            OpenRouter's ``reasoning`` option, Anthropic's ``effort``, and the
            thinking switch of local chat templates. Short decisions rarely
            need reasoning, and hidden reasoning tokens are billed and count
            against the output limit.
    """
    if reasoning not in (None, *REASONING):
        raise ValueError(f"reasoning must be one of {', '.join(REASONING)}")
    level = None if reasoning in (None, "default") else reasoning
    backend, _, model = spec.partition(":")
    backend = backend.strip().lower()
    model = model.strip()
    if backend == "local":
        from .local import TransformersLLM

        return TransformersLLM(
            model or DEFAULT_LOCAL_MODEL, device=device, enable_thinking=level not in (None, "off")
        )
    if backend == "anthropic":
        from .anthropic import AnthropicLLM

        effort = {"minimal": "low", "off": None}.get(level or "", level)
        return AnthropicLLM(model or "claude-opus-5", effort=effort)
    if backend == "openrouter":
        from .openrouter import OpenRouterLLM

        if not model:
            raise ValueError(
                "the openrouter backend needs a model slug, for example "
                "openrouter:qwen/qwen3.7-flash"
            )
        options = (
            None if level is None else {"enabled": False} if level == "off" else {"effort": level}
        )
        return OpenRouterLLM(model, reasoning=options)
    if backend in ("openai", "ollama", "vllm"):
        from .openai_compat import OpenAICompatibleLLM

        if backend == "openai":
            if not model:
                raise ValueError("the openai backend needs a model, for example openai:<model-id>")
            return OpenAICompatibleLLM(model, base_url=base_url)
        defaults = {
            "ollama": ("qwen3:4b", "http://localhost:11434/v1"),
            "vllm": ("", "http://localhost:8000/v1"),
        }
        default_model, default_url = defaults[backend]
        if not (model or default_model):
            raise ValueError(
                f"the {backend} backend needs a model, for example {backend}:<model-id>"
            )
        return OpenAICompatibleLLM(
            model or default_model, base_url=base_url or default_url, provider=backend
        )
    raise ValueError(f"unknown LLM backend {backend!r}; choose one of {', '.join(BACKENDS)}")
