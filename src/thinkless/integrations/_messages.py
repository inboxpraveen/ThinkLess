"""Read the latest user message out of the shapes agent frameworks pass around."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["content_text", "last_user_text"]

_USER_ROLES = ("user", "human")


def content_text(content: Any) -> str:
    """Flatten message content (a string or a list of content parts) into text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, Mapping):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(p for p in parts if p)
    return str(content)


def _role(message: Any) -> str | None:
    if isinstance(message, Mapping):
        role = message.get("role") or message.get("type")
        return None if role is None else str(role)
    role = getattr(message, "role", None) or getattr(message, "type", None)
    return None if role is None else str(role)


def _content(message: Any) -> Any:
    if isinstance(message, Mapping):
        return message.get("content")
    return getattr(message, "content", None)


def last_user_text(value: Any) -> str:
    """The text of the most recent user message.

    Accepts a plain string, a list of LangChain messages, OpenAI chat or
    Responses API input items, or a mapping with a ``messages`` list (a
    LangGraph state). Falls back to the last message of any role when none is
    marked as the user's.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping) and "messages" in value:
        return last_user_text(value["messages"])
    messages = getattr(value, "messages", None)
    if messages is not None and not isinstance(value, Sequence):
        return last_user_text(messages)
    if isinstance(value, Sequence):
        items = list(value)
        for message in reversed(items):
            if _role(message) in _USER_ROLES:
                return content_text(_content(message))
        if items:
            return content_text(_content(items[-1]))
        return ""
    return str(value)
