"""Strict, deterministic I/O helpers for immutable research contracts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode


class StrictResearchDataError(ValueError):
    """Raised when research data is ambiguous or cannot be hashed canonically."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses ambiguous duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_research_mapping(path: str | Path) -> Mapping[str, Any]:
    """Load strict JSON or YAML and require canonical, string-keyed data."""

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        payload = json.loads(text, object_pairs_hook=_unique_json_object)
    else:
        payload = yaml.load(text, Loader=_UniqueKeyLoader)
    validate_canonical_research_value(payload)
    if not isinstance(payload, Mapping):
        raise StrictResearchDataError("Research data must be a mapping at the root")
    return payload


def canonical_research_fingerprint(payload: Mapping[str, Any]) -> str:
    """Return a stable content ID after enforcing the canonical value subset."""

    validate_canonical_research_value(payload)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def file_sha256(path: str | Path) -> str:
    """Hash exact file bytes for an external artifact reference."""

    return f"sha256:{hashlib.sha256(Path(path).read_bytes()).hexdigest()}"


def write_bytes_create_only(path: str | Path, content: bytes) -> bool:
    """Atomically create immutable evidence, accepting only an identical existing file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_file() and destination.read_bytes() == content:
            return False
        raise FileExistsError(f"Refusing to replace existing evidence: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_file() and destination.read_bytes() == content:
                return False
            raise FileExistsError(
                f"Refusing to replace concurrently created evidence: {destination}"
            ) from None
        return True
    finally:
        temporary.unlink(missing_ok=True)


def validate_canonical_research_value(value: Any, *, path: str = "$") -> None:
    """Reject values whose YAML/JSON interpretation or hash would be ambiguous."""

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StrictResearchDataError(f"{path} must not contain NaN or infinity")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise StrictResearchDataError(f"{path} has a non-string mapping key")
            validate_canonical_research_value(child, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            validate_canonical_research_value(child, path=f"{path}[{index}]")
        return
    raise StrictResearchDataError(
        f"{path} contains unsupported {type(value).__name__}; use explicit strings"
    )


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise StrictResearchDataError(f"JSON contains duplicate key {key!r}")
        payload[key] = value
    return payload


__all__ = [
    "StrictResearchDataError",
    "canonical_research_fingerprint",
    "file_sha256",
    "load_research_mapping",
    "validate_canonical_research_value",
    "write_bytes_create_only",
]
