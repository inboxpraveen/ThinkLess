"""Runtime settings read from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Settings", "load_env", "resolve_device"]


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


def load_env(path: str | Path = ".env", *, override: bool = False) -> list[str]:
    """Load ``KEY=value`` lines from a dotenv file into ``os.environ``.

    Blank lines and ``#`` comments are skipped, an ``export`` prefix and
    matching quotes around values are removed, and empty values are ignored.
    Variables already set in the environment win unless ``override`` is true.
    The CLI calls this for ``./.env``; library code calls it explicitly.

    Returns:
        The names that were set. Values are never logged or returned.
    """
    target = Path(path)
    if not target.is_file():
        return []
    loaded = []
    for raw in target.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if not name or not value:
            continue
        if override or not os.environ.get(name):
            os.environ[name] = value
            loaded.append(name)
    return loaded


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
