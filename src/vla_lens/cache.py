"""Managed, disposable caches for repeatable VLA Lens research.

Cache entries are addressed by a scientific recipe, but freshness is checked
separately against the source data.  This keeps an experiment's identity stable
when a dataset is moved while still rebuilding when its inputs change.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import time
import uuid
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

CACHE_MANIFEST = "manifest.json"
CACHE_SCHEMA_VERSION = 1


def fingerprint_payload(payload: Any) -> str:
    """Return a stable SHA-256 fingerprint for a JSON-compatible value."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True, slots=True)
class CacheBuildMetadata:
    """Optional scientific shape information returned by an entry builder."""

    shape: tuple[int, ...] = ()
    dtype: str | None = None
    axes: tuple[str, ...] = ()
    row_count: int | None = None
    rebuild: Mapping[str, Any] = field(default_factory=dict)
    content_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class CacheEntryManifest:
    """Small durable record describing one disposable cache entry."""

    namespace: str
    key: str
    scientific_fingerprint: str
    source_fingerprint: str
    content_fingerprint: str
    recipe: Mapping[str, Any]
    size_bytes: int
    created_utc: str
    last_accessed_utc: str
    pinned: bool = False
    complete: bool = True
    shape: tuple[int, ...] = ()
    dtype: str | None = None
    axes: tuple[str, ...] = ()
    row_count: int | None = None
    rebuild: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = CACHE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["shape"] = list(self.shape)
        payload["axes"] = list(self.axes)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CacheEntryManifest:
        values = dict(payload)
        values["shape"] = tuple(int(item) for item in values.get("shape") or ())
        values["axes"] = tuple(str(item) for item in values.get("axes") or ())
        values["recipe"] = dict(values.get("recipe") or {})
        values["rebuild"] = dict(values.get("rebuild") or {})
        return cls(**values)


@dataclass(frozen=True, slots=True)
class CachePruneResult:
    """A dry-run or applied cache cleanup decision."""

    apply: bool
    before_bytes: int
    after_bytes: int
    reclaimed_bytes: int
    entries: tuple[str, ...]
    blocked_by_pins: bool


class InterProcessFileLock(AbstractContextManager["InterProcessFileLock"]):
    """Advisory process lock backed by ``flock`` on a small lock file."""

    def __init__(self, path: Path, *, timeout_s: float = 300.0):
        self.path = Path(path)
        self.timeout_s = timeout_s
        self._handle: Any = None

    def __enter__(self) -> InterProcessFileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+b")
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    self._handle.close()
                    self._handle = None
                    raise TimeoutError(f"Timed out waiting for lock {self.path}") from None
                time.sleep(0.05)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


class CacheManager:
    """Build, inspect, pin, and safely remove entries below ``.vla_cache``."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def entry_path(self, namespace: str, key: str) -> Path:
        namespace = _safe_component(namespace, label="cache namespace")
        key = _safe_component(key, label="cache key")
        return self.root / namespace / key

    def lock_for(self, namespace: str, key: str, *, timeout_s: float = 300.0) -> Any:
        namespace = _safe_component(namespace, label="cache namespace")
        key = _safe_component(key, label="cache key")
        return InterProcessFileLock(
            self.root / ".locks" / namespace / f"{key}.lock", timeout_s=timeout_s
        )

    def get_or_build(
        self,
        *,
        namespace: str,
        key: str,
        recipe: Mapping[str, Any],
        source_fingerprint: str,
        builder: Callable[[Path], CacheBuildMetadata | None],
        validator: Callable[[Path], bool] | None = None,
        legacy_metadata: Callable[[Path], CacheBuildMetadata] | None = None,
        timeout_s: float = 300.0,
    ) -> tuple[Path, CacheEntryManifest, bool]:
        """Return a valid entry, building it exactly once across processes.

        The builder writes only to its supplied temporary directory.  A
        completed entry becomes visible after validation and an atomic rename.
        The returned boolean is true when this call performed the build.
        """
        entry = self.entry_path(namespace, key)
        scientific_fingerprint = fingerprint_payload(recipe)
        with self.lock_for(namespace, key, timeout_s=timeout_s):
            self._recover_interrupted_commit(entry)
            previous_manifest = self._read_manifest(entry)
            manifest = self._valid_manifest(
                entry,
                scientific_fingerprint=scientific_fingerprint,
                source_fingerprint=source_fingerprint,
                validator=validator,
            )
            if manifest is not None:
                touched = self._touch(entry, manifest)
                return entry, touched, False

            # Older cache writers used the same namespace/key directories but
            # did not write a CacheManager manifest. Adopt a complete legacy
            # entry under the shared lock instead of rebuilding expensive data.
            if (
                previous_manifest is None
                and legacy_metadata is not None
                and entry.exists()
                and (validator is None or validator(entry))
            ):
                metadata = legacy_metadata(entry)
                now = _utc_now()
                manifest = CacheEntryManifest(
                    namespace=namespace,
                    key=key,
                    scientific_fingerprint=scientific_fingerprint,
                    source_fingerprint=source_fingerprint,
                    content_fingerprint=metadata.content_fingerprint
                    or _directory_fingerprint(entry),
                    recipe=dict(recipe),
                    size_bytes=_directory_size(entry),
                    created_utc=now,
                    last_accessed_utc=now,
                    shape=metadata.shape,
                    dtype=metadata.dtype,
                    axes=metadata.axes,
                    row_count=metadata.row_count,
                    rebuild=dict(metadata.rebuild),
                )
                _atomic_write_json(entry / CACHE_MANIFEST, manifest.to_dict())
                return entry, manifest, False

            entry.parent.mkdir(parents=True, exist_ok=True)
            temporary = entry.parent / f".{entry.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
            temporary.mkdir()
            try:
                metadata = builder(temporary) or CacheBuildMetadata()
                if validator is not None and not validator(temporary):
                    raise ValueError(
                        f"Cache builder produced an invalid entry for {namespace}/{key}"
                    )
                now = _utc_now()
                manifest = CacheEntryManifest(
                    namespace=namespace,
                    key=key,
                    scientific_fingerprint=scientific_fingerprint,
                    source_fingerprint=source_fingerprint,
                    content_fingerprint=metadata.content_fingerprint
                    or _directory_fingerprint(temporary),
                    recipe=dict(recipe),
                    size_bytes=_directory_size(temporary),
                    created_utc=now,
                    last_accessed_utc=now,
                    pinned=previous_manifest.pinned if previous_manifest is not None else False,
                    shape=metadata.shape,
                    dtype=metadata.dtype,
                    axes=metadata.axes,
                    row_count=metadata.row_count,
                    rebuild=dict(metadata.rebuild),
                )
                _atomic_write_json(temporary / CACHE_MANIFEST, manifest.to_dict())
                self._commit_directory(temporary, entry)
                return entry, manifest, True
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)

    def manifest(self, namespace: str, key: str) -> CacheEntryManifest:
        path = self.entry_path(namespace, key) / CACHE_MANIFEST
        return CacheEntryManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def entries(self) -> list[CacheEntryManifest]:
        """Read recorded cache metadata without traversing capture files."""
        if not self.root.exists():
            return []
        records: list[CacheEntryManifest] = []
        for path in sorted(self.root.glob(f"*/*/{CACHE_MANIFEST}")):
            try:
                manifest = CacheEntryManifest.from_dict(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            records.append(manifest)
        return records

    def set_pinned(self, namespace: str, key: str, *, pinned: bool) -> CacheEntryManifest:
        entry = self.entry_path(namespace, key)
        with self.lock_for(namespace, key):
            manifest = self.manifest(namespace, key)
            updated = CacheEntryManifest.from_dict({**manifest.to_dict(), "pinned": pinned})
            _atomic_write_json(entry / CACHE_MANIFEST, updated.to_dict())
            return updated

    def prune(
        self,
        *,
        max_bytes: int,
        min_free_bytes: int,
        apply: bool = False,
    ) -> CachePruneResult:
        """Plan or apply LRU cleanup while preserving pinned entries.

        Cleanup is constrained to manifest-backed directories immediately below
        this manager's root.  It never follows a user-supplied path.
        """
        if max_bytes < 0 or min_free_bytes < 0:
            raise ValueError("Cache budget and free-space floor must be non-negative")
        if self.root.name != ".vla_cache":
            raise ValueError("Prune is only allowed for a directory named .vla_cache")
        manifests = self.entries()
        before = sum(item.size_bytes for item in manifests if item.complete)
        free = shutil.disk_usage(self.root if self.root.exists() else self.root.parent).free
        need = max(before - max_bytes, min_free_bytes - free, 0)
        candidates = sorted(
            (item for item in manifests if item.complete and not item.pinned),
            key=lambda item: (item.last_accessed_utc, item.created_utc, item.namespace, item.key),
        )
        pinned_blocked = sum(item.size_bytes for item in candidates) < need
        chosen: list[CacheEntryManifest] = []
        reclaimed = 0
        for item in candidates:
            if reclaimed >= need:
                break
            chosen.append(item)
            reclaimed += item.size_bytes

        applied: list[CacheEntryManifest] = []
        skipped_pinned = False
        if apply:
            for item in chosen:
                with self.lock_for(item.namespace, item.key):
                    entry = self.entry_path(item.namespace, item.key)
                    _assert_within(entry, self.root)
                    if not entry.exists():
                        continue
                    # The plan may have been made before another worker
                    # rebuilt this key. Never delete a replacement that was
                    # not the manifest selected by this prune pass.
                    current = self._read_manifest(entry)
                    if current is None or current.to_dict() != item.to_dict():
                        continue
                    if current.pinned:
                        skipped_pinned = True
                        continue
                    shutil.rmtree(entry)
                    applied.append(item)

        removed = applied if apply else chosen
        actual_reclaimed = sum(item.size_bytes for item in removed)
        return CachePruneResult(
            apply=apply,
            before_bytes=before,
            after_bytes=max(0, before - actual_reclaimed),
            reclaimed_bytes=actual_reclaimed,
            entries=tuple(f"{item.namespace}/{item.key}" for item in removed),
            blocked_by_pins=actual_reclaimed < need and (pinned_blocked or skipped_pinned),
        )

    def _valid_manifest(
        self,
        entry: Path,
        *,
        scientific_fingerprint: str,
        source_fingerprint: str,
        validator: Callable[[Path], bool] | None,
    ) -> CacheEntryManifest | None:
        manifest = self._read_manifest(entry)
        if manifest is None:
            return None
        if not manifest.complete or manifest.schema_version != CACHE_SCHEMA_VERSION:
            return None
        if manifest.scientific_fingerprint != scientific_fingerprint:
            return None
        if manifest.source_fingerprint != source_fingerprint:
            return None
        if validator is not None and not validator(entry):
            return None
        return manifest

    def _read_manifest(self, entry: Path) -> CacheEntryManifest | None:
        try:
            return CacheEntryManifest.from_dict(
                json.loads((entry / CACHE_MANIFEST).read_text(encoding="utf-8"))
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _touch(self, entry: Path, manifest: CacheEntryManifest) -> CacheEntryManifest:
        now = datetime.now(UTC)
        previous = datetime.fromisoformat(manifest.last_accessed_utc.replace("Z", "+00:00"))
        if (now - previous).total_seconds() < 60:
            return manifest
        updated = CacheEntryManifest.from_dict(
            {**manifest.to_dict(), "last_accessed_utc": now.isoformat().replace("+00:00", "Z")}
        )
        _atomic_write_json(entry / CACHE_MANIFEST, updated.to_dict())
        return updated

    def _recover_interrupted_commit(self, entry: Path) -> None:
        entry.parent.mkdir(parents=True, exist_ok=True)
        temporary = list(entry.parent.glob(f".{entry.name}.tmp-*"))
        previous = sorted(entry.parent.glob(f".{entry.name}.old-*"))
        for path in temporary:
            shutil.rmtree(path, ignore_errors=True)
        if not entry.exists() and previous:
            os.replace(previous[-1], entry)
            previous = previous[:-1]
        for path in previous:
            shutil.rmtree(path, ignore_errors=True)

    def _commit_directory(self, temporary: Path, entry: Path) -> None:
        previous = entry.parent / f".{entry.name}.old-{uuid.uuid4().hex}"
        if entry.exists():
            os.replace(entry, previous)
        try:
            os.replace(temporary, entry)
        except BaseException:
            if previous.exists() and not entry.exists():
                os.replace(previous, entry)
            raise
        if previous.exists():
            shutil.rmtree(previous)


def atomic_replace_file(path: str | Path, writer: Callable[[Path], None]) -> None:
    """Write a file through a same-directory temporary and atomically replace it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        writer(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_replace_file(
        path,
        lambda temporary: temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8"
        ),
    )


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _directory_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        with item.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _safe_component(value: str, *, label: str) -> str:
    value = str(value)
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def _assert_within(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        raise ValueError(f"Refusing to operate outside cache root: {path}") from None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


__all__: Sequence[str] = (
    "CACHE_MANIFEST",
    "CacheBuildMetadata",
    "CacheEntryManifest",
    "CacheManager",
    "CachePruneResult",
    "InterProcessFileLock",
    "atomic_replace_file",
    "fingerprint_payload",
)
