"""Robot and environment metadata extraction for PI0.5 captures."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from vla_lens.pi05.context_capture_common import (
    _env_candidates,
    _first_attr,
    _generic_axes,
    _metadata_row,
    _numeric_array,
    _quat_array_to_matrix,
    _robot_axes,
    _stack_observation_field,
    _Status,
)
from vla_lens.pi05.context_capture_types import ENV_METADATA_FIELDS, ROBOT_FIELD_CANDIDATES
from vla_lens.traces import ArraySpec


def extract_robot_arrays(
    observations: Sequence[Mapping[str, Any]],
    *,
    status: "_Status | None" = None,
) -> dict[str, ArraySpec]:
    """Extract robot proprioception arrays from formatted observations."""

    status = status or _Status()
    arrays: dict[str, ArraySpec] = {}
    if not observations:
        for field in ROBOT_FIELD_CANDIDATES:
            status.missing("robot", field, "no observations were provided")
        return arrays

    for field, candidates in ROBOT_FIELD_CANDIDATES.items():
        values, source = _stack_observation_field(observations, candidates)
        if values is None:
            if field == "eef_mat" and "eef_quat" in arrays:
                mat = _quat_array_to_matrix(arrays["eef_quat"].array)
                arrays["eef_mat"] = ArraySpec(
                    mat.astype(np.float32),
                    ["timestep", "row", "col"],
                    metadata={"source": "derived:eef_quat"},
                )
                status.available("robot", "eef_mat", "derived:eef_quat", shape=mat.shape)
                continue
            status.missing(
                "robot",
                field,
                f"none of these observation keys were present: {', '.join(candidates)}",
            )
            continue
        axes = _robot_axes(field, values)
        arrays[field] = ArraySpec(
            values.astype(np.float32, copy=False),
            axes,
            metadata={"source": str(source)},
        )
        status.available("robot", field, str(source), shape=values.shape)

    return arrays


def extract_env_metadata(
    env: Any | None,
    *,
    status: "_Status | None" = None,
) -> tuple[pd.DataFrame, dict[str, ArraySpec]]:
    """Extract reset/init state and task/layout metadata from wrapper objects."""

    status = status or _Status()
    arrays: dict[str, ArraySpec] = {}
    records: list[dict[str, Any]] = []
    if env is None:
        for field in ENV_METADATA_FIELDS:
            status.missing("env", field, "env is not available")
            records.append(_metadata_row(field, available=False, reason="env is not available"))
        return pd.DataFrame.from_records(records), arrays

    candidates = _env_candidates(env)
    for field, attr_names in ENV_METADATA_FIELDS.items():
        found = _first_attr(candidates, attr_names)
        if found is None:
            reason = f"no wrapper attribute found among: {', '.join(attr_names)}"
            status.missing("env", field, reason)
            records.append(
                _metadata_row(
                    field,
                    available=False,
                    reason=reason,
                )
            )
            continue

        value, source = found
        array = _numeric_array(value)
        if field in {"reset_state", "init_state"} and array is not None:
            name = f"scene_{field}"
            arrays[name] = ArraySpec(
                array.astype(np.float32, copy=False),
                _generic_axes(array, trailing_prefix="state"),
                metadata={"source": source},
            )
            records.append(
                _metadata_row(
                    field,
                    available=True,
                    source=source,
                    value=None,
                    array_name=name,
                    shape=array.shape,
                )
            )
        else:
            records.append(_metadata_row(field, available=True, source=source, value=value))
        status.available("env", field, source, shape=None if array is None else array.shape)

    return pd.DataFrame.from_records(records), arrays
