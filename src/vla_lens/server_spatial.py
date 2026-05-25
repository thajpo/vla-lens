"""Spatial dashboard server helpers."""


from __future__ import annotations

from typing import Any

import numpy as np

from vla_lens.server_common import (
    _json_parse,
    _json_scalar,
    _jsonable,
    _optional_array,
    _optional_int,
    _query_one,
    _round,
)
from vla_lens.traces import TraceBundle


def _object_camera_overlay_payload(
    bundle: TraceBundle,
    query: dict[str, list[str]],
) -> dict[str, Any]:
    camera = _query_one(query, "camera")
    timestep = int(query.get("timestep", ["0"])[0] or 0)
    timestep = max(0, min(timestep, max(0, bundle.manifest.length - 1)))
    include_sites = str(query.get("include_sites", ["false"])[0]).lower() in {
        "1",
        "true",
        "yes",
    }
    try:
        object_pos = bundle.array("scene_object_pos", mmap=True)
    except KeyError:
        return {
            "available": False,
            "reason": "object_positions_unavailable",
            "detail": "scene_object_pos is not captured in this trace.",
            "camera": camera,
            "timestep": timestep,
            "objects": [],
        }

    object_quat = None
    try:
        object_quat = bundle.array("scene_object_quat", mmap=True)
    except KeyError:
        pass
    object_geom_center = _optional_array(bundle, "scene_object_geom_center")
    object_bbox = _optional_array(bundle, "scene_object_bbox_world")
    camera_object_bbox = _optional_array(bundle, "camera_object_bbox")
    camera_object_visible = _optional_array(bundle, "camera_object_visible")

    height, width = _camera_frame_size(bundle, camera)
    calibration = _camera_projection_calibration(bundle, camera, timestep)
    object_rows = _scene_object_rows(bundle, include_sites=include_sites)
    if not object_rows:
        return {
            "available": False,
            "reason": "object_metadata_unavailable",
            "detail": "scene_state has no object rows to map onto camera frames.",
            "camera": camera,
            "timestep": timestep,
            "objects": [],
        }
    if calibration is None:
        return {
            "available": False,
            "reason": "camera_calibration_unavailable",
            "detail": "camera_intrinsics/camera_extrinsics are required for projection.",
            "camera": camera,
            "timestep": timestep,
            "objects": [_jsonable(row) for row in object_rows],
        }

    intrinsic, extrinsic, calibration_camera = calibration
    camera_to_pixel = _camera_to_pixel_transform(intrinsic, extrinsic)
    objects: list[dict[str, Any]] = []
    for row in object_rows:
        object_index = row.get("object_index")
        if object_index is None:
            continue
        index = int(object_index)
        pos = _object_position_at(object_pos, index, timestep)
        if pos is None:
            continue
        geom_center = (
            _object_position_at(object_geom_center, index, timestep)
            if object_geom_center is not None
            else None
        )
        bbox = _object_bbox_at(object_bbox, index, timestep) if object_bbox is not None else None
        projection_kind = "object_pose_center"
        projection = _project_world_point(pos, camera_to_pixel, width=width, height=height)
        bbox_projection = None
        if geom_center is not None:
            projection = _project_world_point(
                geom_center,
                camera_to_pixel,
                width=width,
                height=height,
            )
            projection_kind = "object_geometry_center"
        if bbox is not None:
            bbox_projection = _project_world_bbox(
                bbox,
                camera_to_pixel,
                width=width,
                height=height,
            )
            if bbox_projection is not None:
                projection = {
                    **projection,
                    "pixel_x": bbox_projection["center_pixel_x"],
                    "pixel_y": bbox_projection["center_pixel_y"],
                    "x": bbox_projection["center_x"],
                    "y": bbox_projection["center_y"],
                    "in_frame": bbox_projection["in_frame"],
                }
                projection_kind = "object_geometry_bbox"
        camera_bbox_projection = _camera_object_bbox_projection(
            bundle,
            camera_object_bbox,
            camera_object_visible,
            camera=camera,
            object_name=str(row.get("object_name") or ""),
            object_index=index,
            timestep=timestep,
            width=width,
            height=height,
        )
        if camera_bbox_projection is not None:
            bbox_projection = camera_bbox_projection
            projection = {
                **projection,
                "pixel_x": bbox_projection["center_pixel_x"],
                "pixel_y": bbox_projection["center_pixel_y"],
                "x": bbox_projection["center_x"],
                "y": bbox_projection["center_y"],
                "in_frame": bbox_projection["in_frame"],
            }
            projection_kind = "camera_segmentation_bbox"
        quat = _object_quat_at(object_quat, index, timestep) if object_quat is not None else None
        objects.append(
            {
                **row,
                "position_world": _round(pos),
                "geometry_center_world": _round(geom_center) if geom_center is not None else None,
                "quaternion_xyzw": _round(quat) if quat is not None else None,
                "bbox": bbox_projection,
                "pixel_x": projection.get("pixel_x"),
                "pixel_y": projection.get("pixel_y"),
                "x": projection.get("x"),
                "y": projection.get("y"),
                "depth": projection.get("depth"),
                "in_frame": projection.get("in_frame"),
                "approximate": True,
                "projection_kind": projection_kind,
            }
        )

    visible = [item for item in objects if item.get("in_frame")]
    return {
        "available": bool(objects),
        "camera": camera,
        "calibration_camera": calibration_camera,
        "timestep": timestep,
        "width": width,
        "height": height,
        "include_sites": include_sites,
        "approximate": True,
        "projection_kind": (
            "camera_segmentation_bbox"
            if camera_object_bbox is not None
            else "object_geometry_bbox"
            if object_bbox is not None
            else "object_pose_center"
        ),
        "visible_count": len(visible),
        "objects": _jsonable(objects),
        "note": (
            "Object labels use captured world-frame object geometry bounds when available, "
            "falling back to object pose centers."
        ),
    }

def _scene_object_rows(bundle: TraceBundle, *, include_sites: bool) -> list[dict[str, Any]]:
    table = bundle.scene_state
    if table.empty:
        return []
    if "context_kind" in table:
        table = table.loc[table["context_kind"].astype(str) == "object"]
    if "object_index" not in table or "object_name" not in table:
        return []
    rows: list[dict[str, Any]] = []
    for raw in table.sort_values("object_index").to_dict("records"):
        object_index = _optional_int(raw.get("object_index"))
        object_name = raw.get("object_name")
        if object_index is None or object_name is None or str(object_name) == "nan":
            continue
        object_kind = str(raw.get("object_kind") or "object")
        if object_kind == "site" and not include_sites:
            continue
        rows.append(
            {
                "object_index": object_index,
                "object_name": str(object_name),
                "object_kind": object_kind,
                "body_id": _optional_int(raw.get("body_id")),
                "body_name": str(raw.get("body_name") or ""),
                "site_name": str(raw.get("site_name") or ""),
                "source": str(raw.get("source") or ""),
            }
        )
    return rows

def _camera_frame_size(bundle: TraceBundle, camera: str) -> tuple[int, int]:
    try:
        frames = bundle.frames(camera, mmap=True)
        if frames.ndim >= 3:
            return int(frames.shape[1]), int(frames.shape[2])
    except KeyError:
        pass
    resolution = _camera_resolution_from_context(bundle, camera)
    if resolution is not None:
        return resolution
    return 1, 1

def _camera_resolution_from_context(bundle: TraceBundle, camera: str) -> tuple[int, int] | None:
    try:
        resolution = bundle.array("camera_resolution", mmap=True)
    except KeyError:
        resolution = None
    camera_index = _camera_index_for_array(bundle, "camera_resolution", camera)
    if resolution is not None and camera_index is not None and resolution.ndim == 2:
        pair = np.asarray(resolution[camera_index]).astype(int)
        if pair.size >= 2 and pair[0] > 0 and pair[1] > 0:
            return int(pair[0]), int(pair[1])
    table = bundle.camera_state
    if table.empty:
        return None
    names = _camera_aliases(camera)
    for row in table.to_dict("records"):
        row_name = str(row.get("camera_name") or row.get("name") or row.get("camera_id") or "")
        if row_name in names:
            height = _optional_int(row.get("height"))
            width = _optional_int(row.get("width"))
            if height and width:
                return height, width
    return None

def _camera_projection_calibration(
    bundle: TraceBundle,
    camera: str,
    timestep: int,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    try:
        intrinsics = bundle.array("camera_intrinsics", mmap=True)
        extrinsics = bundle.array("camera_extrinsics", mmap=True)
    except KeyError:
        return None
    camera_index = _camera_index_for_array(bundle, "camera_intrinsics", camera)
    if camera_index is None:
        camera_index = _camera_index_for_array(bundle, "camera_extrinsics", camera)
    if camera_index is None:
        return None
    intrinsic = np.asarray(intrinsics[camera_index], dtype=np.float32)
    extrinsic = _camera_extrinsic_at(extrinsics, camera_index, timestep)
    if intrinsic.shape != (3, 3) or extrinsic is None or extrinsic.shape != (4, 4):
        return None
    camera_names = _camera_names_for_array(bundle, "camera_intrinsics")
    calibration_camera = camera_names[camera_index] if camera_index < len(camera_names) else camera
    return intrinsic, extrinsic, calibration_camera

def _camera_extrinsic_at(
    extrinsics: np.ndarray,
    camera_index: int,
    timestep: int,
) -> np.ndarray | None:
    value = np.asarray(extrinsics)
    if value.ndim == 3:
        extrinsic = np.asarray(value[camera_index], dtype=np.float32)
        return extrinsic if np.all(np.isfinite(extrinsic)) else None
    if value.ndim == 4:
        step = max(0, min(timestep, value.shape[0] - 1))
        extrinsic = np.asarray(value[step, camera_index], dtype=np.float32)
        return extrinsic if np.all(np.isfinite(extrinsic)) else None
    return None

def _camera_index_for_array(bundle: TraceBundle, array_name: str, camera: str) -> int | None:
    names = _camera_names_for_array(bundle, array_name)
    aliases = _camera_aliases(camera)
    for index, name in enumerate(names):
        if name in aliases:
            return index
    return None

def _camera_names_for_array(bundle: TraceBundle, array_name: str) -> list[str]:
    return _metadata_list_for_array(bundle, array_name, "camera_names")

def _object_names_for_array(bundle: TraceBundle, array_name: str) -> list[str]:
    return _metadata_list_for_array(bundle, array_name, "object_names")

def _metadata_list_for_array(bundle: TraceBundle, array_name: str, key: str) -> list[str]:
    if bundle.array_index.empty or "name" not in bundle.array_index:
        return []
    rows = bundle.array_index.loc[bundle.array_index["name"].astype(str) == array_name]
    if rows.empty:
        return []
    metadata = _json_parse(rows.iloc[0].get("metadata"))
    if isinstance(metadata, dict):
        names = metadata.get(key)
        if isinstance(names, list):
            return [str(name) for name in names]
    return []

def _camera_aliases(camera: str) -> set[str]:
    aliases = {camera, camera.removesuffix("_image")}
    if camera == "main":
        aliases.update({"agentview", "agentview_image", "image"})
    if camera == "wrist":
        aliases.update({"robot0_eye_in_hand", "robot0_eye_in_hand_image", "image2"})
    if camera == "agentview":
        aliases.update({"main", "agentview_image", "image"})
    if camera == "robot0_eye_in_hand":
        aliases.update({"wrist", "robot0_eye_in_hand_image", "image2"})
    return aliases

def _camera_to_pixel_transform(intrinsic: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    expanded = np.eye(4, dtype=np.float32)
    expanded[:3, :3] = np.asarray(intrinsic, dtype=np.float32)
    return expanded @ np.linalg.inv(np.asarray(extrinsic, dtype=np.float32))

def _object_position_at(
    object_pos: np.ndarray, object_index: int, timestep: int
) -> np.ndarray | None:
    value = np.asarray(object_pos)
    try:
        if value.ndim == 3:
            step = max(0, min(timestep, value.shape[0] - 1))
            pos = np.asarray(value[step, object_index], dtype=np.float32)
        elif value.ndim == 2:
            pos = np.asarray(value[object_index], dtype=np.float32)
        else:
            return None
    except IndexError:
        return None
    if pos.shape[-1] < 3 or not np.all(np.isfinite(pos[:3])):
        return None
    return pos[:3]

def _object_quat_at(
    object_quat: np.ndarray | None,
    object_index: int,
    timestep: int,
) -> np.ndarray | None:
    if object_quat is None:
        return None
    value = np.asarray(object_quat)
    try:
        if value.ndim == 3:
            step = max(0, min(timestep, value.shape[0] - 1))
            quat = np.asarray(value[step, object_index], dtype=np.float32)
        elif value.ndim == 2:
            quat = np.asarray(value[object_index], dtype=np.float32)
        else:
            return None
    except IndexError:
        return None
    if quat.shape[-1] < 4 or not np.all(np.isfinite(quat[:4])):
        return None
    return quat[:4]

def _object_bbox_at(
    object_bbox: np.ndarray,
    object_index: int,
    timestep: int,
) -> np.ndarray | None:
    value = np.asarray(object_bbox)
    try:
        if value.ndim == 4:
            step = max(0, min(timestep, value.shape[0] - 1))
            bbox = np.asarray(value[step, object_index], dtype=np.float32)
        elif value.ndim == 3:
            bbox = np.asarray(value[object_index], dtype=np.float32)
        else:
            return None
    except IndexError:
        return None
    if bbox.shape != (2, 3) or not np.all(np.isfinite(bbox)):
        return None
    return bbox

def _project_world_point(
    point: np.ndarray,
    camera_to_pixel: np.ndarray,
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    homogeneous = np.asarray([point[0], point[1], point[2], 1.0], dtype=np.float32)
    projected = camera_to_pixel @ homogeneous
    depth = float(projected[2])
    if not np.isfinite(depth) or abs(depth) <= 1e-8:
        return {"in_frame": False, "depth": _json_scalar(depth)}
    pixel_x = float(projected[0] / depth)
    pixel_y = float(projected[1] / depth)
    x = pixel_x / max(1, width)
    y = pixel_y / max(1, height)
    in_frame = bool(depth > 0 and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)
    return {
        "pixel_x": _json_scalar(pixel_x),
        "pixel_y": _json_scalar(pixel_y),
        "x": _json_scalar(x),
        "y": _json_scalar(y),
        "depth": _json_scalar(depth),
        "in_frame": in_frame,
    }

def _project_world_bbox(
    bbox: np.ndarray,
    camera_to_pixel: np.ndarray,
    *,
    width: int,
    height: int,
) -> dict[str, Any] | None:
    bounds = np.asarray(bbox, dtype=np.float32)
    mins = np.minimum(bounds[0], bounds[1])
    maxs = np.maximum(bounds[0], bounds[1])
    corners = np.asarray(
        [
            [x, y, z]
            for x in (mins[0], maxs[0])
            for y in (mins[1], maxs[1])
            for z in (mins[2], maxs[2])
        ],
        dtype=np.float32,
    )
    projections = [
        _project_world_point(corner, camera_to_pixel, width=width, height=height)
        for corner in corners
    ]
    visible_points = [
        item
        for item in projections
        if item.get("pixel_x") is not None
        and item.get("pixel_y") is not None
        and item.get("depth") is not None
        and float(item["depth"]) > 0.0
    ]
    if not visible_points:
        return None
    xs = np.asarray([float(item["x"]) for item in visible_points], dtype=np.float32)
    ys = np.asarray([float(item["y"]) for item in visible_points], dtype=np.float32)
    x0_raw = float(np.nanmin(xs))
    x1_raw = float(np.nanmax(xs))
    y0_raw = float(np.nanmin(ys))
    y1_raw = float(np.nanmax(ys))
    in_frame = bool(x1_raw >= 0.0 and x0_raw <= 1.0 and y1_raw >= 0.0 and y0_raw <= 1.0)
    x0 = min(1.0, max(0.0, x0_raw))
    x1 = min(1.0, max(0.0, x1_raw))
    y0 = min(1.0, max(0.0, y0_raw))
    y1 = min(1.0, max(0.0, y1_raw))
    center_x = (x0 + x1) * 0.5
    center_y = (y0 + y1) * 0.5
    return {
        "x0": _json_scalar(x0),
        "y0": _json_scalar(y0),
        "x1": _json_scalar(x1),
        "y1": _json_scalar(y1),
        "raw_x0": _json_scalar(x0_raw),
        "raw_y0": _json_scalar(y0_raw),
        "raw_x1": _json_scalar(x1_raw),
        "raw_y1": _json_scalar(y1_raw),
        "center_x": _json_scalar(center_x),
        "center_y": _json_scalar(center_y),
        "center_pixel_x": _json_scalar(center_x * max(1, width)),
        "center_pixel_y": _json_scalar(center_y * max(1, height)),
        "in_frame": in_frame,
    }

def _camera_object_bbox_projection(
    bundle: TraceBundle,
    camera_object_bbox: np.ndarray | None,
    camera_object_visible: np.ndarray | None,
    *,
    camera: str,
    object_name: str,
    object_index: int,
    timestep: int,
    width: int,
    height: int,
) -> dict[str, Any] | None:
    if camera_object_bbox is None:
        return None
    camera_index = _camera_index_for_array(bundle, "camera_object_bbox", camera)
    if camera_index is None:
        return None
    object_names = _object_names_for_array(bundle, "camera_object_bbox")
    if object_name in object_names:
        array_object_index = object_names.index(object_name)
    else:
        array_object_index = object_index
    value = np.asarray(camera_object_bbox)
    try:
        if value.ndim == 4:
            step = max(0, min(timestep, value.shape[0] - 1))
            bbox = np.asarray(value[step, camera_index, array_object_index], dtype=np.float32)
        elif value.ndim == 3:
            bbox = np.asarray(value[camera_index, array_object_index], dtype=np.float32)
        else:
            return None
    except IndexError:
        return None
    if bbox.shape[-1] < 4 or not np.all(np.isfinite(bbox[:4])):
        return None
    visible = True
    if camera_object_visible is not None:
        visible_value = np.asarray(camera_object_visible)
        try:
            if visible_value.ndim == 3:
                step = max(0, min(timestep, visible_value.shape[0] - 1))
                visible = bool(visible_value[step, camera_index, array_object_index])
            elif visible_value.ndim == 2:
                visible = bool(visible_value[camera_index, array_object_index])
        except IndexError:
            visible = False
    x0_raw = float(bbox[0]) / max(1, width)
    y0_raw = float(bbox[1]) / max(1, height)
    x1_raw = float(bbox[2]) / max(1, width)
    y1_raw = float(bbox[3]) / max(1, height)
    in_frame = bool(visible and x1_raw >= 0.0 and x0_raw <= 1.0 and y1_raw >= 0.0 and y0_raw <= 1.0)
    x0 = min(1.0, max(0.0, min(x0_raw, x1_raw)))
    x1 = min(1.0, max(0.0, max(x0_raw, x1_raw)))
    y0 = min(1.0, max(0.0, min(y0_raw, y1_raw)))
    y1 = min(1.0, max(0.0, max(y0_raw, y1_raw)))
    center_x = (x0 + x1) * 0.5
    center_y = (y0 + y1) * 0.5
    return {
        "x0": _json_scalar(x0),
        "y0": _json_scalar(y0),
        "x1": _json_scalar(x1),
        "y1": _json_scalar(y1),
        "raw_x0": _json_scalar(x0_raw),
        "raw_y0": _json_scalar(y0_raw),
        "raw_x1": _json_scalar(x1_raw),
        "raw_y1": _json_scalar(y1_raw),
        "center_x": _json_scalar(center_x),
        "center_y": _json_scalar(center_y),
        "center_pixel_x": _json_scalar(center_x * max(1, width)),
        "center_pixel_y": _json_scalar(center_y * max(1, height)),
        "in_frame": in_frame,
        "source": "camera_segmentation",
    }
