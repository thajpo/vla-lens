"""Minimal experiment-run manifest helper."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExperimentRun:
    name: str
    out_dir: Path
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "config": self.config,
        }

    def write_manifest(self) -> Path:
        path = self.out_dir / "manifest.json"
        path.write_text(json.dumps(self.manifest(), indent=2), encoding="utf-8")
        return path

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.out_dir / name
        if hasattr(payload, "__dataclass_fields__"):
            payload = asdict(payload)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip() or None
