"""Probe workflow spec normalization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from vla_lens.probes.workflow_types import DEFAULT_PROBE_SPEC


def normalize_probe_spec(spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return a complete probe spec with conservative defaults."""
    merged = _deep_merge(DEFAULT_PROBE_SPEC, dict(spec or {}))
    target = merged.get("target")
    if isinstance(target, str):
        merged["target"] = {"kind": target}
    features = merged.setdefault("features", {})
    if "reduce_tokens" in features and "reduction" not in features:
        features["reduction"] = features.pop("reduce_tokens")
    if "policy_calls" not in features:
        features["policy_calls"] = "all"
    if "dtype" not in features:
        features["dtype"] = "float32"
    split = merged.get("split")
    if isinstance(split, str):
        merged["split"] = {"kind": split}
    merged.setdefault("probe", {"models": ["linear", "mlp"]})
    if isinstance(merged.get("probe"), str):
        merged["probe"] = {"models": [merged["probe"]]}
    if isinstance(merged.get("probe"), Mapping):
        probe = merged["probe"]
        if isinstance(probe.get("models"), str):
            probe["models"] = [probe["models"]]
    merged.setdefault("baseline", [])
    return merged


def load_probe_spec(path: str | Path) -> dict[str, Any]:
    """Load a probe spec from YAML. Use ``-`` for stdin."""
    if str(path) == "-":
        import sys

        payload = yaml.safe_load(sys.stdin.read())
    else:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload is None:
        raise ValueError("Probe spec is empty")
    if not isinstance(payload, Mapping):
        raise TypeError("Probe spec must be a mapping")
    return normalize_probe_spec(payload)


def dump_probe_spec(spec: Mapping[str, Any]) -> str:
    return yaml.safe_dump(normalize_probe_spec(spec), sort_keys=False)


def baseline_columns(items: Sequence[Any]) -> list[str]:
    columns: list[str] = []
    aliases = {
        "majority_class": None,
        "majority": None,
        "benchmark": "benchmark",
        "benchmark_only": "benchmark",
        "task": "task_id",
        "task_id": "task_id",
        "target_object": "target_object",
        "object": "target_object",
        "env": "env_id",
        "env_id": "env_id",
    }
    for item in items:
        value = str(item).strip()
        column = aliases.get(value, value)
        if column and column not in columns:
            columns.append(column)
    return columns


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
