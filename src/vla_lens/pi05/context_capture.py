"""Cheap LIBERO / LeRobot context extraction helpers.

These helpers intentionally depend only on NumPy / pandas plus the local
``ArraySpec`` type.  Real LIBERO runs expose more state through robosuite and
MuJoCo, but tests and lightweight smoke runs often do not have those packages
or simulators available.  Missing optional fields are therefore represented as
status rows instead of exceptions.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from vla_lens.pi05.context_capture_camera import capture_camera_snapshot, extract_camera_context
from vla_lens.pi05.context_capture_common import (
    _observation_sequence,
    _scene_snapshot_sequence,
    _Status,
)
from vla_lens.pi05.context_capture_objects import extract_object_context
from vla_lens.pi05.context_capture_robot import extract_env_metadata, extract_robot_arrays
from vla_lens.pi05.context_capture_scene import capture_scene_snapshot
from vla_lens.pi05.context_capture_types import ContextCaptureResult
from vla_lens.traces import ArraySpec


def capture_libero_context(
    observations: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    env: Any | None = None,
    scene_snapshots: Sequence[Mapping[str, Any]] | None = None,
    camera_snapshots: Sequence[Mapping[str, Any]] | None = None,
) -> ContextCaptureResult:
    """Extract cheap environment context from observations and an optional env.

    The returned ``arrays`` mapping can be passed to ``TraceBundle.create`` as
    episode arrays.  The returned tables are plain pandas frames intended for
    richer capture manifests or future .vlatrace table slots.
    """

    obs_sequence = _observation_sequence(observations)
    scene_snapshot_sequence = _scene_snapshot_sequence(scene_snapshots)
    arrays: dict[str, ArraySpec] = {}
    tables: dict[str, pd.DataFrame] = {}
    status = _Status()

    arrays.update(extract_robot_arrays(obs_sequence, status=status))
    env_metadata, env_arrays = extract_env_metadata(env, status=status)
    tables["episode_context"] = env_metadata
    arrays.update(env_arrays)

    object_table, object_arrays = extract_object_context(
        obs_sequence,
        env,
        scene_snapshots=scene_snapshot_sequence,
        status=status,
    )
    tables["objects"] = object_table
    arrays.update(object_arrays)

    camera_table, camera_arrays = extract_camera_context(
        obs_sequence,
        env,
        camera_snapshots=camera_snapshots,
        status=status,
    )
    tables["cameras"] = camera_table
    arrays.update(camera_arrays)

    tables["context_availability"] = status.frame()
    return ContextCaptureResult(arrays=arrays, tables=tables)


__all__ = [
    "ContextCaptureResult",
    "capture_camera_snapshot",
    "capture_libero_context",
    "capture_scene_snapshot",
    "extract_camera_context",
    "extract_env_metadata",
    "extract_object_context",
    "extract_robot_arrays",
]
