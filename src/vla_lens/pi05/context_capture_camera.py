"""Camera context and calibration extraction for PI0.5 captures."""

from __future__ import annotations

import importlib
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from vla_lens.pi05.context_capture_common import (
    _camera_snapshot_sequence,
    _env_candidates,
    _first_existing_attr,
    _flatten_mapping_keys,
    _lookup_mapping_path,
    _names_from_value,
    _numeric_array,
    _numeric_matrix,
    _numeric_vector,
    _observation_sequence,
    _optional_int,
    _resolve_body_id,
    _squeeze_single_env,
    _Status,
)
from vla_lens.pi05.context_capture_scene import (
    _geom_indices_for_body_tree,
    _scene_object_descriptors,
)
from vla_lens.traces import ArraySpec


def capture_camera_snapshot(
    env: Any | None,
    observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture camera calibration at the current simulator timestep.

    Wrist / eye-in-hand cameras move with the robot, so camera extrinsics must be
    sampled at the same pre-action time as frames and object state.
    """

    observations = _observation_sequence(observation)
    names = _camera_names(env, observations)
    if not names:
        return {"cameras": [], "source": "", "reason": "no cameras were exposed"}

    candidates = _env_candidates(env)
    sim = _first_existing_attr(candidates, ("sim", "_sim"))
    if sim is None:
        return {"cameras": [], "source": "", "reason": "env.sim unavailable"}

    camera_utils = _robosuite_camera_utils()
    object_descriptors = _scene_object_descriptors(candidates, sim)
    cameras: list[dict[str, Any]] = []
    for name in names:
        height, width = _camera_resolution(name, observations, env)
        intrinsic, intrinsic_reason = _camera_intrinsic(
            camera_utils,
            sim,
            name,
            height=height,
            width=width,
        )
        extrinsic, extrinsic_reason = _camera_extrinsic(camera_utils, sim, name)
        object_bboxes = _camera_object_bboxes_from_segmentation(
            sim,
            name,
            height=height,
            width=width,
            object_descriptors=object_descriptors,
        )
        cameras.append(
            {
                "camera_name": name,
                "height": height,
                "width": width,
                "intrinsic": intrinsic,
                "intrinsic_reason": intrinsic_reason,
                "extrinsic": extrinsic,
                "extrinsic_reason": extrinsic_reason,
                "object_bboxes": object_bboxes,
            }
        )

    return {"cameras": cameras, "source": "robosuite.camera_utils", "reason": ""}


def extract_camera_context(
    observations: Sequence[Mapping[str, Any]],
    env: Any | None = None,
    *,
    camera_snapshots: Sequence[Mapping[str, Any]] | None = None,
    status: "_Status | None" = None,
) -> tuple[pd.DataFrame, dict[str, ArraySpec]]:
    """Extract camera labels, image sizes, and robosuite calibration when present."""

    status = status or _Status()
    snapshot_sequence = _camera_snapshot_sequence(camera_snapshots)
    if snapshot_sequence:
        return _extract_camera_context_from_snapshots(snapshot_sequence, status=status)

    arrays: dict[str, ArraySpec] = {}
    names = _camera_names(env, observations)
    resolutions = [_camera_resolution(name, observations, env) for name in names]
    camera_utils = _robosuite_camera_utils()
    sim = _first_existing_attr(_env_candidates(env), ("sim", "_sim")) if env is not None else None

    intrinsics: list[np.ndarray] = []
    extrinsics: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        height, width = resolutions[index]
        intrinsic, intrinsic_reason = _camera_intrinsic(
            camera_utils,
            sim,
            name,
            height=height,
            width=width,
        )
        extrinsic, extrinsic_reason = _camera_extrinsic(camera_utils, sim, name)
        records.append(
            {
                "camera_index": index,
                "camera_name": name,
                "height": height,
                "width": width,
                "intrinsics_available": intrinsic is not None,
                "intrinsics_reason": intrinsic_reason,
                "extrinsics_available": extrinsic is not None,
                "extrinsics_reason": extrinsic_reason,
            }
        )
        if intrinsic is not None:
            intrinsics.append(intrinsic)
        if extrinsic is not None:
            extrinsics.append(extrinsic)

    if names:
        resolution_array = np.asarray(
            [[-1 if item is None else int(item) for item in pair] for pair in resolutions],
            dtype=np.int32,
        )
        arrays["camera_resolution"] = ArraySpec(
            resolution_array,
            ["camera", "height_width"],
            metadata={"camera_names": list(names)},
        )
        status.available("camera", "names", "env_or_observation", shape=(len(names),))
        status.available("camera", "resolution", "env_or_observation", shape=resolution_array.shape)
    else:
        status.missing("camera", "names", "no camera names or image observations were exposed")
        status.missing("camera", "resolution", "no cameras were exposed")

    if len(intrinsics) == len(names) and intrinsics:
        array = np.stack(intrinsics).astype(np.float32)
        arrays["camera_intrinsics"] = ArraySpec(
            array,
            ["camera", "row", "col"],
            metadata={"camera_names": list(names)},
        )
        status.available("camera", "intrinsics", "robosuite.camera_utils", shape=array.shape)
    else:
        reason = "robosuite camera_utils unavailable or env.sim missing"
        status.missing("camera", "intrinsics", reason)

    if len(extrinsics) == len(names) and extrinsics:
        array = np.stack(extrinsics).astype(np.float32)
        arrays["camera_extrinsics"] = ArraySpec(
            array,
            ["camera", "row", "col"],
            metadata={"camera_names": list(names)},
        )
        status.available("camera", "extrinsics", "robosuite.camera_utils", shape=array.shape)
    else:
        reason = "robosuite camera_utils unavailable or env.sim missing"
        status.missing("camera", "extrinsics", reason)

    return pd.DataFrame.from_records(records), arrays


def _extract_camera_context_from_snapshots(
    camera_snapshots: Sequence[Mapping[str, Any]],
    *,
    status: "_Status",
) -> tuple[pd.DataFrame, dict[str, ArraySpec]]:
    arrays: dict[str, ArraySpec] = {}
    names: list[str] = []
    object_names: list[str] = []
    for snapshot in camera_snapshots:
        for raw in _snapshot_cameras(snapshot):
            name = raw.get("camera_name")
            if name is None:
                continue
            key = str(name)
            if key not in names:
                names.append(key)
            for bbox in _snapshot_object_bboxes(raw):
                object_name = bbox.get("object_name")
                if object_name is None:
                    continue
                object_key = str(object_name)
                if object_key not in object_names:
                    object_names.append(object_key)

    if not names:
        status.missing("camera", "names", "camera snapshots had no camera rows")
        status.missing("camera", "resolution", "camera snapshots had no camera rows")
        status.missing("camera", "intrinsics", "camera snapshots had no camera rows")
        status.missing("camera", "extrinsics", "camera snapshots had no camera rows")
        return pd.DataFrame(), arrays

    name_to_index = {name: index for index, name in enumerate(names)}
    resolution = np.full((len(names), 2), -1, dtype=np.int32)
    intrinsics = np.full((len(names), 3, 3), np.nan, dtype=np.float32)
    extrinsics = np.full((len(camera_snapshots), len(names), 4, 4), np.nan, dtype=np.float32)
    intrinsic_reasons: dict[str, str] = {name: "" for name in names}
    extrinsic_reasons: dict[str, str] = {name: "" for name in names}
    has_intrinsic = {name: False for name in names}
    has_extrinsic = {name: False for name in names}
    object_bbox = np.full(
        (len(camera_snapshots), len(names), len(object_names), 4),
        np.nan,
        dtype=np.float32,
    )
    object_visible = np.zeros(
        (len(camera_snapshots), len(names), len(object_names)),
        dtype=np.uint8,
    )
    object_name_to_index = {name: index for index, name in enumerate(object_names)}

    for timestep, snapshot in enumerate(camera_snapshots):
        for raw in _snapshot_cameras(snapshot):
            name = raw.get("camera_name")
            if name is None:
                continue
            key = str(name)
            index = name_to_index.get(key)
            if index is None:
                continue
            height = _optional_int(raw.get("height"))
            width = _optional_int(raw.get("width"))
            if height is not None and width is not None:
                resolution[index] = [height, width]
            intrinsic = _numeric_matrix(raw.get("intrinsic"), 3, 3)
            if intrinsic is not None:
                intrinsics[index] = intrinsic
                has_intrinsic[key] = True
            elif not intrinsic_reasons[key]:
                intrinsic_reasons[key] = str(raw.get("intrinsic_reason") or "")
            extrinsic = _numeric_matrix(raw.get("extrinsic"), 4, 4)
            if extrinsic is not None:
                extrinsics[timestep, index] = extrinsic
                has_extrinsic[key] = True
            elif not extrinsic_reasons[key]:
                extrinsic_reasons[key] = str(raw.get("extrinsic_reason") or "")
            for bbox in _snapshot_object_bboxes(raw):
                object_name = bbox.get("object_name")
                object_index = object_name_to_index.get(str(object_name))
                if object_index is None:
                    continue
                pixel_bbox = _numeric_vector(bbox.get("bbox_pixel_xyxy"), 4)
                if pixel_bbox is None:
                    continue
                object_bbox[timestep, index, object_index] = pixel_bbox
                object_visible[timestep, index, object_index] = 1

    records = []
    for index, name in enumerate(names):
        records.append(
            {
                "camera_index": index,
                "camera_name": name,
                "height": int(resolution[index, 0]),
                "width": int(resolution[index, 1]),
                "intrinsics_available": bool(has_intrinsic[name]),
                "intrinsics_reason": intrinsic_reasons[name],
                "extrinsics_available": bool(has_extrinsic[name]),
                "extrinsics_reason": extrinsic_reasons[name],
                "extrinsics_time_varying": True,
            }
        )

    arrays["camera_resolution"] = ArraySpec(
        resolution,
        ["camera", "height_width"],
        metadata={"camera_names": list(names)},
    )
    status.available("camera", "names", "camera_snapshots", shape=(len(names),))
    status.available("camera", "resolution", "camera_snapshots", shape=resolution.shape)

    if all(has_intrinsic.values()):
        arrays["camera_intrinsics"] = ArraySpec(
            intrinsics,
            ["camera", "row", "col"],
            metadata={"camera_names": list(names)},
        )
        status.available("camera", "intrinsics", "camera_snapshots", shape=intrinsics.shape)
    else:
        missing = [name for name, available in has_intrinsic.items() if not available]
        status.missing("camera", "intrinsics", f"missing intrinsics for cameras: {missing}")

    if all(has_extrinsic.values()):
        arrays["camera_extrinsics"] = ArraySpec(
            extrinsics,
            ["timestep", "camera", "row", "col"],
            metadata={"camera_names": list(names), "time_aligned": True},
        )
        status.available("camera", "extrinsics", "camera_snapshots", shape=extrinsics.shape)
    else:
        missing = [name for name, available in has_extrinsic.items() if not available]
        status.missing("camera", "extrinsics", f"missing extrinsics for cameras: {missing}")

    if object_names and bool(np.any(object_visible)):
        arrays["camera_object_bbox"] = ArraySpec(
            object_bbox,
            ["timestep", "camera", "object", "bbox_xyxy"],
            metadata={
                "camera_names": list(names),
                "object_names": list(object_names),
                "bbox_format": "pixel_xyxy_exclusive",
                "source": "robosuite.segmentation",
            },
        )
        arrays["camera_object_visible"] = ArraySpec(
            object_visible,
            ["timestep", "camera", "object"],
            metadata={
                "camera_names": list(names),
                "object_names": list(object_names),
                "source": "robosuite.segmentation",
            },
        )
        status.available(
            "camera",
            "object_bbox",
            "robosuite.segmentation",
            shape=object_bbox.shape,
        )
    else:
        status.missing("camera", "object_bbox", "camera segmentation bboxes unavailable")

    return pd.DataFrame.from_records(records), arrays


def _snapshot_cameras(snapshot: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    cameras = snapshot.get("cameras")
    if not isinstance(cameras, Sequence) or isinstance(cameras, (str, bytes)):
        return []
    return [item for item in cameras if isinstance(item, Mapping)]


def _snapshot_object_bboxes(camera_snapshot: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    bboxes = camera_snapshot.get("object_bboxes")
    if not isinstance(bboxes, Sequence) or isinstance(bboxes, (str, bytes)):
        return []
    return [item for item in bboxes if isinstance(item, Mapping)]


def _camera_object_bboxes_from_segmentation(
    sim: Any,
    camera_name: str,
    *,
    height: int | None,
    width: int | None,
    object_descriptors: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if height is None or width is None or height <= 0 or width <= 0:
        return []
    segmentation = _render_camera_segmentation(
        sim,
        camera_name,
        height=int(height),
        width=int(width),
    )
    if segmentation is None:
        return []
    try:
        mujoco = importlib.import_module("mujoco")
        geom_obj_type = int(mujoco.mjtObj.mjOBJ_GEOM)
    except Exception:
        geom_obj_type = 5

    model = getattr(sim, "model", None)
    if model is None:
        return []

    rows: list[dict[str, Any]] = []
    for descriptor in object_descriptors:
        object_name = descriptor.get("object_name")
        if object_name is None or descriptor.get("site_name"):
            continue
        body_id = _optional_int(descriptor.get("body_id"))
        if body_id is None:
            body_id = _resolve_body_id(sim, descriptor.get("body_name"))
        if body_id is None:
            continue
        geom_ids = _geom_indices_for_body_tree(model, body_id)
        if not geom_ids:
            continue
        mask = (segmentation[:, :, 0] == geom_obj_type) & np.isin(segmentation[:, :, 1], geom_ids)
        if not bool(np.any(mask)):
            continue
        ys, xs = np.where(mask)
        rows.append(
            {
                "object_name": str(object_name),
                "bbox_pixel_xyxy": np.asarray(
                    [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
                    dtype=np.float32,
                ),
                "pixel_area": int(mask.sum()),
                "source": "robosuite.segmentation",
            }
        )
    return rows


def _render_camera_segmentation(
    sim: Any,
    camera_name: str,
    *,
    height: int,
    width: int,
) -> np.ndarray | None:
    """Render robust segmentation IDs, avoiding robosuite's uint8 overflow path."""

    try:
        mujoco = importlib.import_module("mujoco")
        binding_utils = importlib.import_module("robosuite.utils.binding_utils")
        context = getattr(sim, "_render_context_offscreen", None)
        model = getattr(sim, "model", None)
        if context is None or model is None:
            return None
        camera_id = model.camera_name2id(camera_name)
        lock = binding_utils._MjSim_render_lock
        with lock:
            context.render(
                width=width,
                height=height,
                camera_id=camera_id,
                segmentation=True,
            )
            viewport = mujoco.MjrRect(0, 0, width, height)
            rgb = np.empty((height, width, 3), dtype=np.uint8)
            mujoco.mjr_readPixels(rgb=rgb, depth=None, viewport=viewport, con=context.con)
            seg_img = (
                rgb[:, :, 0].astype(np.int32)
                + rgb[:, :, 1].astype(np.int32) * (2**8)
                + rgb[:, :, 2].astype(np.int32) * (2**16)
            )
            seg_img[seg_img >= (context.scn.ngeom + 1)] = 0
            seg_ids = np.full((context.scn.ngeom + 1, 2), fill_value=-1, dtype=np.int32)
            for index in range(context.scn.ngeom):
                geom = context.scn.geoms[index]
                if geom.segid != -1:
                    seg_ids[geom.segid + 1, 0] = geom.objtype
                    seg_ids[geom.segid + 1, 1] = geom.objid
            return seg_ids[seg_img]
    except Exception:
        return None


def _camera_names(env: Any | None, observations: Sequence[Mapping[str, Any]]) -> list[str]:
    candidates = _env_candidates(env)
    values = _first_existing_attr(candidates, ("camera_names", "_camera_names", "cameras"))
    names = _names_from_value(values)
    if names:
        return [_normalize_camera_name(name) for name in names]
    if not observations:
        return []
    names = []
    for key in _flatten_mapping_keys(observations[0]):
        short = key.rsplit(".", 1)[-1]
        if short.endswith("_image"):
            names.append(short[: -len("_image")])
        elif short == "image":
            names.append("image")
    return sorted(set(names))


def _normalize_camera_name(name: str) -> str:
    text = str(name)
    return text[: -len("_image")] if text.endswith("_image") else text


def _camera_resolution(
    name: str,
    observations: Sequence[Mapping[str, Any]],
    env: Any | None,
) -> tuple[int | None, int | None]:
    image = _camera_image(name, observations)
    if image is not None and image.ndim >= 2:
        return int(image.shape[0]), int(image.shape[1])
    candidates = _env_candidates(env)
    height = _camera_dimension(candidates, name, ("camera_heights", "_camera_heights", "height"))
    width = _camera_dimension(candidates, name, ("camera_widths", "_camera_widths", "width"))
    return height, width


def _camera_image(name: str, observations: Sequence[Mapping[str, Any]]) -> np.ndarray | None:
    if not observations:
        return None
    obs = observations[0]
    for key in (f"{name}_image", f"observation.images.{name}", "image" if name == "image" else ""):
        if not key:
            continue
        value = _lookup_mapping_path(obs, key)
        array = _numeric_array(value)
        if array is not None:
            return _squeeze_single_env(array)
    return None


def _camera_dimension(
    candidates: Sequence[Any],
    name: str,
    attrs: Sequence[str],
) -> int | None:
    for attr in attrs:
        value = _first_existing_attr(candidates, (attr,))
        if isinstance(value, Mapping):
            item = value.get(name)
            if item is not None:
                return int(item)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            camera_name_attr = _first_existing_attr(candidates, ("camera_names", "_camera_names"))
            names = _names_from_value(camera_name_attr)
            if name in names:
                return int(value[names.index(name)])
        elif value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return None


def _robosuite_camera_utils() -> Any | None:
    try:
        return importlib.import_module("robosuite.utils.camera_utils")
    except Exception:
        return None


def _camera_intrinsic(
    camera_utils: Any | None,
    sim: Any | None,
    name: str,
    *,
    height: int | None,
    width: int | None,
) -> tuple[np.ndarray | None, str]:
    if camera_utils is None:
        return None, "robosuite camera_utils unavailable"
    if sim is None:
        return None, "env.sim unavailable"
    if height is None or width is None:
        return None, "camera resolution unavailable"
    func = getattr(camera_utils, "get_camera_intrinsic_matrix", None)
    if not callable(func):
        return None, "get_camera_intrinsic_matrix unavailable"
    try:
        return np.asarray(func(sim, name, int(height), int(width)), dtype=np.float32), ""
    except Exception as exc:
        return None, f"camera_utils intrinsic failed: {exc}"


def _camera_extrinsic(
    camera_utils: Any | None,
    sim: Any | None,
    name: str,
) -> tuple[np.ndarray | None, str]:
    if camera_utils is None:
        return None, "robosuite camera_utils unavailable"
    if sim is None:
        return None, "env.sim unavailable"
    func = getattr(camera_utils, "get_camera_extrinsic_matrix", None)
    if not callable(func):
        return None, "get_camera_extrinsic_matrix unavailable"
    try:
        return np.asarray(func(sim, name), dtype=np.float32), ""
    except Exception as exc:
        return None, f"camera_utils extrinsic failed: {exc}"
