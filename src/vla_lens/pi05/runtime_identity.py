"""Canonical PI0.5 runtime identities and immutable checkpoint receipts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping

IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def require_immutable_revision(revision: str) -> str:
    normalized = str(revision or "").strip().lower()
    if not IMMUTABLE_REVISION_RE.fullmatch(normalized):
        raise ValueError("model_revision must be an exact 40-character hexadecimal commit revision")
    return normalized


def canonical_component_identities(
    *,
    obs_size: int,
    model_id: str,
    model_revision: str | None,
    device: str,
    dtype: str,
) -> dict[str, Any]:
    components = {
        "camera": {
            "camera_name": "agentview_image,robot0_eye_in_hand_image",
            "observation_height": int(obs_size),
            "observation_width": int(obs_size),
        },
        "controller": {"control_mode": "relative"},
        "preprocessor": {
            "implementation": "lerobot.make_pre_post_processors",
            "model_id": str(model_id),
            "model_revision": str(model_revision or ""),
            "device_processor": {"device": str(device)},
            "rename_observations_processor": {"rename_map": {}},
        },
        "postprocessor": {
            "implementation": "lerobot.make_pre_post_processors+make_env_pre_post_processors",
            "model_id": str(model_id),
            "model_revision": str(model_revision or ""),
            "dtype": str(dtype),
        },
    }
    return {
        f"{name}_config": value
        for name, value in components.items()
    } | {
        f"{name}_config_sha256": canonical_sha256(value)
        for name, value in components.items()
    }


def resolve_immutable_checkpoint(
    model_id: str,
    revision: str,
    *,
    snapshot_download: Callable[..., str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    exact_revision = require_immutable_revision(revision)
    if snapshot_download is None:
        from huggingface_hub import snapshot_download as hf_snapshot_download

        snapshot_download = hf_snapshot_download
    snapshot = Path(snapshot_download(repo_id=model_id, revision=exact_revision)).resolve()
    resolved_revision = snapshot.name.lower()
    if not IMMUTABLE_REVISION_RE.fullmatch(resolved_revision):
        raise ValueError(
            f"checkpoint snapshot path does not expose an immutable revision: {snapshot}"
        )
    if resolved_revision != exact_revision:
        raise ValueError(
            f"checkpoint resolved to {resolved_revision}, expected exact revision {exact_revision}"
        )
    manifest = checkpoint_snapshot_manifest(snapshot)
    receipt = {
        "repo_id": str(model_id),
        "requested_revision": exact_revision,
        "resolved_revision": resolved_revision,
        "snapshot_path": str(snapshot),
        "snapshot_manifest_sha256": canonical_sha256(manifest),
        "files": manifest,
    }
    return snapshot, receipt


def checkpoint_snapshot_manifest(snapshot: Path) -> list[dict[str, Any]]:
    if not snapshot.is_dir():
        raise ValueError(f"checkpoint snapshot is not a directory: {snapshot}")
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in snapshot.rglob("*") if item.is_file()):
        relative = path.relative_to(snapshot).as_posix()
        target = path.resolve()
        files.append(
            {
                "path": relative,
                "size": target.stat().st_size,
                "sha256": _file_identity(path),
            }
        )
    if not files:
        raise ValueError(f"checkpoint snapshot contains no files: {snapshot}")
    return files


def declared_runtime_identity(args: Any) -> dict[str, Any]:
    revision = str(getattr(args, "model_revision", None) or "")
    payload = {
        "schema_version": 1,
        "kind": "vla_lens.pi05_runtime_identity",
        "model": {"repo_id": str(args.model_id), "revision": revision},
        "components": canonical_component_identities(
            obs_size=int(args.obs_size),
            model_id=str(args.model_id),
            model_revision=revision,
            device=str(args.device),
            dtype=str(args.dtype),
        ),
    }
    environment = load_environment_receipt()
    if environment is not None:
        payload["capture_environment"] = environment
        payload["capture_environment_sha256"] = canonical_sha256(environment)
    return payload


def load_environment_receipt() -> dict[str, Any] | None:
    value = os.environ.get("VLA_LENS_CAPTURE_ENV_RECEIPT", "").strip()
    if not value:
        return None
    path = Path(value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"capture environment receipt must be a JSON object: {path}")
    return payload


def persist_runtime_identity(output_root: Path, identity: Mapping[str, Any]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "runtime_identity.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    environment_path = os.environ.get("VLA_LENS_CAPTURE_ENV_RECEIPT", "").strip()
    if environment_path:
        shutil.copyfile(environment_path, output_root / "capture_environment_receipt.json")
    return path


def _file_identity(path: Path) -> str:
    if path.is_symlink():
        target_name = path.resolve().name
        if re.fullmatch(r"[0-9a-f]{64}", target_name):
            return f"sha256:{target_name}"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"
