"""Exceptions raised by ThinkLess."""

from __future__ import annotations

__all__ = ["ConfigurationError", "ThinkLessError"]


class ThinkLessError(Exception):
    """Base class for ThinkLess errors."""


class ConfigurationError(ThinkLessError, ValueError):
    """The engine or a provider was set up in a way that cannot work."""
