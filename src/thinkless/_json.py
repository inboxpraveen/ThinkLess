"""JSON-safe conversion used by traces and reports."""

from __future__ import annotations

import dataclasses
import enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

__all__ = ["jsonable"]


def jsonable(value: Any, *, depth: int = 0) -> Any:
    """Best-effort conversion of arbitrary values into JSON-safe data."""
    if depth > 8:
        return repr(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, BaseModel):
        return jsonable(value.model_dump(mode="json"), depth=depth + 1)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return jsonable(dataclasses.asdict(value), depth=depth + 1)
    if isinstance(value, dict):
        return {str(k): jsonable(v, depth=depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v, depth=depth + 1) for v in value]
    if isinstance(value, Path):
        return str(value)
    return repr(value)
