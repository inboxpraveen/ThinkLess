"""Logging setup.

The library only ever logs through the ``thinkless`` logger and installs a
``NullHandler``, so it stays silent unless the application configures logging.
:func:`configure_logging` is a convenience for scripts and the CLI.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

__all__ = ["JSONFormatter", "configure_logging", "get_logger"]

logging.getLogger("thinkless").addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """A child of the ``thinkless`` logger."""
    return logging.getLogger(name if name.startswith("thinkless") else f"thinkless.{name}")


class JSONFormatter(logging.Formatter):
    """One JSON object per line, for log shippers."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key in ("trace_id", "span_id", "provider", "question"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str | int = "INFO", *, json_format: bool = False) -> None:
    """Attach a handler to the ``thinkless`` logger.

    Args:
        level: Log level name or number.
        json_format: Emit JSON lines instead of Rich-formatted text.
    """
    logger = logging.getLogger("thinkless")
    for existing in list(logger.handlers):
        if not isinstance(existing, logging.NullHandler):
            logger.removeHandler(existing)
    handler: logging.Handler
    if json_format:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JSONFormatter())
    else:
        from rich.logging import RichHandler

        handler = RichHandler(show_path=False, rich_tracebacks=True, markup=False)
        handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level if isinstance(level, int) else level.upper())
    logger.propagate = False
