"""Runtime settings read from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Settings", "resolve_device"]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Process settings. Every field has an environment variable.

    Attributes:
        trace_dir: ``THINKLESS_TRACE_DIR``. Where JSONL traces are written.
        device: ``THINKLESS_DEVICE``. ``auto``, ``cpu``, ``cuda``, ``cuda:1``
            or ``mps`` for local models.
        capture_content: ``THINKLESS_CAPTURE_CONTENT``. Record inputs and
            outputs in traces.
        log_level: ``THINKLESS_LOG_LEVEL``.
    """

    trace_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("THINKLESS_TRACE_DIR", ".thinkless/traces"))
    )
    device: str = field(default_factory=lambda: os.environ.get("THINKLESS_DEVICE", "auto"))
    capture_content: bool = field(
        default_factory=lambda: _env_bool("THINKLESS_CAPTURE_CONTENT", True)
    )
    log_level: str = field(default_factory=lambda: os.environ.get("THINKLESS_LOG_LEVEL", "WARNING"))


def resolve_device(device: str | None = None) -> str:
    """Turn ``auto`` into a concrete torch device string.

    Prefers CUDA, then Apple MPS, then CPU. Imports torch lazily so the core
    package works without it.
    """
    requested = (device or Settings().device or "auto").lower()
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
