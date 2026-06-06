"""Serialization helpers for runtime-free intervention contracts."""

from __future__ import annotations

from dataclasses import is_dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence, TypeVar

T = TypeVar("T")


def utc_now_iso() -> str:
    """Return an ISO timestamp with explicit UTC timezone."""
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    """Convert tuples, mappings, and dataclass-like values into JSON-safe data."""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return {key: jsonable(item) for key, item in value.__dict__.items()}
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    return value


def mapping_from(value: Any, *, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return {str(key): jsonable(item) for key, item in value.items()}


def required_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return value


def tuple_from(value: Any, *, cast: Callable[[Any], T], field: str) -> tuple[T, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raise TypeError(f"{field} must be a sequence, not a string")
    if not isinstance(value, Sequence):
        raise TypeError(f"{field} must be a sequence")
    return tuple(cast(item) for item in value)


def tuple_of_mappings(value: Any, *, field: str) -> tuple[dict[str, Any], ...]:
    return tuple(mapping_from(item, field=f"{field} item") for item in value or ())


def optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return None if text == "" else text


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def require_nonempty(value: str | None, *, field: str) -> str:
    text = optional_str(value)
    if text is None:
        raise ValueError(f"{field} is required")
    return text


def require_literal(value: str, allowed: set[str], *, field: str) -> None:
    if value not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise ValueError(f"{field} must be one of: {allowed_text}")
