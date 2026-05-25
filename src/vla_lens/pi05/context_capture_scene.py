"""Scene snapshot extraction for PI0.5/LIBERO captures."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from vla_lens.pi05.context_capture_common import (
    _body_name_from_value,
    _env_candidates,
    _first_existing_attr,
    _indexed_data_matrix,
    _indexed_data_vector,
    _mat_to_quat_xyzw,
    _mujoco_quat_to_xyzw,
    _named_data_matrix,
    _named_data_vector,
    _numeric_vector,
    _optional_int,
    _resolve_body_id,
    _resolve_site_id,
)


def capture_scene_snapshot(env: Any | None) -> dict[str, Any]:
    """Capture cheap scene object state directly from a nested LIBERO / MuJoCo env."""

    candidates = _env_candidates(env)
    sim = _first_existing_attr(candidates, ("sim", "_sim"))
    if sim is None:
        return {"objects": [], "source": "", "reason": "env.sim unavailable"}

    objects: list[dict[str, Any]] = []
    for descriptor in _scene_object_descriptors(candidates, sim):
        pose = _sample_scene_object_pose(sim, descriptor)
        if pose is None:
            continue
        objects.append({**descriptor, **pose})

    reason = "" if objects else "no MuJoCo object bodies or sites were resolved"
    return {"objects": objects, "source": "libero.mujoco", "reason": reason}


def _object_records_from_scene_snapshots(
    scene_snapshots: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for snapshot in scene_snapshots:
        for raw in _snapshot_objects(snapshot):
            name = raw.get("object_name")
            if name is None:
                continue
            key = str(name)
            if key in records:
                continue
            records[key] = {
                "object_index": len(records),
                "object_name": key,
                "object_kind": str(raw.get("object_kind") or ""),
                "source": str(raw.get("source") or "libero.mujoco_snapshot"),
                "body_id": _optional_int(raw.get("body_id")),
                "body_name": str(raw.get("body_name") or ""),
                "site_name": str(raw.get("site_name") or ""),
            }
    return list(records.values())


def _snapshot_objects(snapshot: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    objects = snapshot.get("objects")
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)):
        return []
    return [item for item in objects if isinstance(item, Mapping)]


def _object_pose_from_scene_snapshots(
    scene_snapshots: Sequence[Mapping[str, Any]],
    names: Sequence[str],
) -> dict[str, np.ndarray]:
    if not scene_snapshots or not names:
        return {}
    positions = np.full((len(scene_snapshots), len(names), 3), np.nan, dtype=np.float32)
    quats = np.full((len(scene_snapshots), len(names), 4), np.nan, dtype=np.float32)
    geom_centers = np.full((len(scene_snapshots), len(names), 3), np.nan, dtype=np.float32)
    bboxes = np.full((len(scene_snapshots), len(names), 2, 3), np.nan, dtype=np.float32)
    geom_counts = np.full((len(scene_snapshots), len(names)), np.nan, dtype=np.float32)
    has_pos = False
    has_quat = False
    has_geom_center = False
    has_bbox = False
    has_geom_count = False
    name_to_index = {str(name): index for index, name in enumerate(names)}
    for timestep, snapshot in enumerate(scene_snapshots):
        for raw in _snapshot_objects(snapshot):
            index = name_to_index.get(str(raw.get("object_name")))
            if index is None:
                continue
            pos = _numeric_vector(raw.get("pos"), 3)
            if pos is not None:
                positions[timestep, index] = pos
                has_pos = True
            quat = _numeric_vector(raw.get("quat"), 4)
            if quat is not None:
                quats[timestep, index] = quat
                has_quat = True
            geom_center = _numeric_vector(raw.get("geom_center"), 3)
            if geom_center is not None:
                geom_centers[timestep, index] = geom_center
                has_geom_center = True
            bbox_min = _numeric_vector(raw.get("bbox_min"), 3)
            bbox_max = _numeric_vector(raw.get("bbox_max"), 3)
            if bbox_min is not None and bbox_max is not None:
                bboxes[timestep, index, 0] = bbox_min
                bboxes[timestep, index, 1] = bbox_max
                has_bbox = True
            geom_count = _optional_int(raw.get("geom_count"))
            if geom_count is not None:
                geom_counts[timestep, index] = float(geom_count)
                has_geom_count = True
    out: dict[str, np.ndarray] = {}
    if has_pos:
        out["pos"] = positions
    if has_quat:
        out["quat"] = quats
    if has_geom_center:
        out["geom_center"] = geom_centers
    if has_bbox:
        out["bbox_world"] = bboxes
    if has_geom_count:
        out["geom_count"] = geom_counts
    return out


def _scene_object_descriptors(
    candidates: Sequence[Any],
    sim: Any,
) -> list[dict[str, Any]]:
    descriptors: dict[str, dict[str, Any]] = {}

    body_ids = _first_existing_attr(candidates, ("obj_body_id", "object_body_ids"))
    if isinstance(body_ids, Mapping):
        for name, body_id in body_ids.items():
            key = str(name)
            descriptors[key] = {
                "object_name": key,
                "object_kind": _object_kind_for_name(candidates, key),
                "source": "libero.obj_body_id",
                "body_id": _resolve_body_id(sim, body_id),
                "body_name": _body_name_from_value(sim, body_id),
                "site_name": "",
            }

    for attr_name, object_kind in (
        ("objects_dict", "object"),
        ("fixtures_dict", "fixture"),
    ):
        mapping = _first_existing_attr(candidates, (attr_name,))
        if isinstance(mapping, Mapping):
            for name, item in mapping.items():
                key = str(name)
                body_name = str(
                    getattr(item, "root_body", None)
                    or getattr(item, "body_name", None)
                    or getattr(item, "name", None)
                    or key
                )
                current = descriptors.setdefault(
                    key,
                    {
                        "object_name": key,
                        "object_kind": object_kind,
                        "source": f"libero.{attr_name}",
                        "body_id": None,
                        "body_name": body_name,
                        "site_name": "",
                    },
                )
                current["object_kind"] = object_kind
                current["body_name"] = current.get("body_name") or body_name
                current["body_id"] = current.get("body_id") or _resolve_body_id(sim, body_name)

    for attr_name, object_kind in (("objects", "object"), ("fixtures", "fixture")):
        values = _first_existing_attr(candidates, (attr_name,))
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue
        for item in values:
            name = getattr(item, "name", None) or getattr(item, "object_name", None)
            if name is None:
                continue
            key = str(name)
            body_name = str(getattr(item, "root_body", None) or key)
            descriptors.setdefault(
                key,
                {
                    "object_name": key,
                    "object_kind": object_kind,
                    "source": f"libero.{attr_name}",
                    "body_id": _resolve_body_id(sim, body_name),
                    "body_name": body_name,
                    "site_name": "",
                },
            )

    sites = _first_existing_attr(candidates, ("object_sites_dict", "sites_dict"))
    if isinstance(sites, Mapping):
        for name, item in sites.items():
            key = str(name)
            site_name = str(getattr(item, "name", None) or key)
            descriptors.setdefault(
                key,
                {
                    "object_name": key,
                    "object_kind": "site",
                    "source": "libero.object_sites_dict",
                    "body_id": None,
                    "body_name": "",
                    "site_name": site_name,
                },
            )

    return list(descriptors.values())


def _object_kind_for_name(candidates: Sequence[Any], name: str) -> str:
    for attr_name, object_kind in (
        ("objects_dict", "object"),
        ("fixtures_dict", "fixture"),
        ("object_sites_dict", "site"),
    ):
        value = _first_existing_attr(candidates, (attr_name,))
        if isinstance(value, Mapping) and name in value:
            return object_kind
    return "object"


def _sample_scene_object_pose(
    sim: Any,
    descriptor: Mapping[str, Any],
) -> dict[str, Any] | None:
    if descriptor.get("site_name"):
        pose = _sample_site_pose(sim, str(descriptor["site_name"]))
    else:
        pose = _sample_body_pose(
            sim,
            body_id=_optional_int(descriptor.get("body_id")),
            body_name=str(descriptor.get("body_name") or ""),
        )
    if pose is None:
        return None
    geometry = _sample_object_geometry(sim, descriptor)
    if geometry:
        pose.update(geometry)
    return pose


def _sample_body_pose(
    sim: Any,
    *,
    body_id: int | None,
    body_name: str,
) -> dict[str, Any] | None:
    data = getattr(sim, "data", None)
    if data is None:
        return None
    pos = _named_data_vector(data, "get_body_xpos", body_name, 3)
    if pos is None and body_id is not None:
        pos = _indexed_data_vector(data, "body_xpos", body_id, 3)

    quat = _named_data_vector(data, "get_body_xquat", body_name, 4)
    if quat is None and body_id is not None:
        quat = _indexed_data_vector(data, "body_xquat", body_id, 4)
    if quat is not None:
        quat = _mujoco_quat_to_xyzw(quat)

    if quat is None:
        mat = _named_data_matrix(data, "get_body_xmat", body_name)
        if mat is None and body_id is not None:
            mat = _indexed_data_matrix(data, "body_xmat", body_id)
        if mat is not None:
            quat = _mat_to_quat_xyzw(mat)

    if pos is None and quat is None:
        return None
    return {
        "pos": pos if pos is not None else np.full(3, np.nan, dtype=np.float32),
        "quat": quat if quat is not None else np.full(4, np.nan, dtype=np.float32),
    }


def _sample_site_pose(sim: Any, site_name: str) -> dict[str, Any] | None:
    data = getattr(sim, "data", None)
    model = getattr(sim, "model", None)
    if data is None:
        return None
    site_id = _resolve_site_id(model, site_name)
    pos = _named_data_vector(data, "get_site_xpos", site_name, 3)
    if pos is None and site_id is not None:
        pos = _indexed_data_vector(data, "site_xpos", site_id, 3)

    mat = _named_data_matrix(data, "get_site_xmat", site_name)
    if mat is None and site_id is not None:
        mat = _indexed_data_matrix(data, "site_xmat", site_id)
    quat = _mat_to_quat_xyzw(mat) if mat is not None else None

    if pos is None and quat is None:
        return None
    return {
        "pos": pos if pos is not None else np.full(3, np.nan, dtype=np.float32),
        "quat": quat if quat is not None else np.full(4, np.nan, dtype=np.float32),
    }


def _sample_object_geometry(
    sim: Any,
    descriptor: Mapping[str, Any],
) -> dict[str, Any] | None:
    if descriptor.get("site_name"):
        return None
    model = getattr(sim, "model", None)
    data = getattr(sim, "data", None)
    if model is None or data is None:
        return None
    body_id = _optional_int(descriptor.get("body_id"))
    if body_id is None:
        body_id = _resolve_body_id(sim, descriptor.get("body_name"))
    if body_id is None:
        return None

    points: list[np.ndarray] = []
    geom_count = 0
    for geom_id in _geom_indices_for_body_tree(model, body_id):
        geom_points = _geom_world_points(model, data, geom_id)
        if geom_points is None:
            continue
        points.append(geom_points)
        geom_count += 1
    if not points:
        return None

    all_points = np.concatenate(points, axis=0)
    finite = np.all(np.isfinite(all_points), axis=1)
    if not np.any(finite):
        return None
    all_points = all_points[finite]
    bbox_min = np.min(all_points, axis=0).astype(np.float32)
    bbox_max = np.max(all_points, axis=0).astype(np.float32)
    return {
        "geom_center": ((bbox_min + bbox_max) * 0.5).astype(np.float32),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "geom_count": int(geom_count),
    }


def _geom_indices_for_body_tree(model: Any, body_id: int) -> list[int]:
    geom_bodyid = getattr(model, "geom_bodyid", None)
    if geom_bodyid is None:
        return []
    body_ids = set(_body_tree_ids(model, body_id))
    geoms: list[int] = []
    try:
        geom_total = int(getattr(model, "ngeom", len(geom_bodyid)))
    except TypeError:
        geom_total = len(geom_bodyid)
    for geom_id in range(geom_total):
        try:
            if int(geom_bodyid[geom_id]) in body_ids:
                geoms.append(geom_id)
        except Exception:
            continue
    return geoms


def _body_tree_ids(model: Any, body_id: int) -> list[int]:
    parent_ids = getattr(model, "body_parentid", None)
    if parent_ids is None:
        return [body_id]
    try:
        body_total = int(getattr(model, "nbody", len(parent_ids)))
    except TypeError:
        body_total = len(parent_ids)
    descendants: list[int] = []
    for candidate in range(body_total):
        current = candidate
        seen: set[int] = set()
        while current not in seen and current >= 0:
            if current == body_id:
                descendants.append(candidate)
                break
            seen.add(current)
            try:
                current = int(parent_ids[current])
            except Exception:
                break
    return descendants or [body_id]


def _geom_world_points(model: Any, data: Any, geom_id: int) -> np.ndarray | None:
    center = _indexed_data_vector(data, "geom_xpos", geom_id, 3)
    if center is None:
        return None
    points = [center.astype(np.float32)]
    size = _indexed_data_vector(model, "geom_size", geom_id, 3)
    mat = _indexed_data_matrix(data, "geom_xmat", geom_id)
    if (
        size is not None
        and mat is not None
        and np.all(np.isfinite(size))
        and np.all(np.isfinite(mat))
    ):
        size = np.maximum(np.abs(size.astype(np.float32)), 0.0)
        offsets = np.asarray(
            [
                [sx, sy, sz]
                for sx in (-size[0], size[0])
                for sy in (-size[1], size[1])
                for sz in (-size[2], size[2])
            ],
            dtype=np.float32,
        )
        corners = center.astype(np.float32) + offsets @ mat.astype(np.float32).T
        points.extend(corners)
    return np.asarray(points, dtype=np.float32)
