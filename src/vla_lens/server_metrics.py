"""Metrics dashboard server helpers."""


from __future__ import annotations

from typing import Any

import numpy as np

from vla_lens.server_common import (
    _domain_x_label,
    _json_parse,
    _jsonable,
    _label_from_metric_name,
    _round,
)
from vla_lens.traces import TraceBundle


def _manifest_payload(bundle: TraceBundle) -> dict[str, Any]:
    manifest = bundle.manifest.to_dict()
    return _jsonable(manifest)

def _action_norm_payload(bundle: TraceBundle) -> dict[str, Any]:
    actions = np.asarray(bundle.actions(mmap=True), dtype=np.float32)
    return {"values": _round(np.linalg.norm(actions, axis=-1))}

def _policy_calls_payload(bundle: TraceBundle) -> dict[str, Any]:
    calls = _policy_calls(bundle)
    return {
        "calls": calls,
        "count": len(calls),
        "env_length": int(bundle.manifest.length),
    }

def _policy_calls(bundle: TraceBundle) -> list[dict[str, Any]]:
    call_rows = bundle.policy_calls.copy()
    if call_rows.empty:
        return []
    calls: list[dict[str, Any]] = []
    call_rows = call_rows.sort_values("policy_call_index").reset_index(drop=True)
    for index, row in enumerate(call_rows.to_dict("records")):
        call_index = int(row.get("policy_call_index", index))
        timestep = int(row.get("observation_timestep", row.get("env_timestep_start", 0)))
        segment_start = int(row.get("env_timestep_start", timestep))
        segment_end = int(
            row.get(
                "env_timestep_end",
                call_rows.iloc[index + 1]["env_timestep_start"] - 1
                if index + 1 < len(call_rows)
                else bundle.manifest.length - 1,
            )
        )
        calls.append(
            {
                "index": index,
                "model_call_index": call_index,
                "env_timestep": timestep,
                "segment_start": segment_start,
                "segment_end": max(segment_start, segment_end),
                "segment_length": max(1, segment_end - segment_start + 1),
            }
        )
    return calls

def _generation_commitment_payload(bundle: TraceBundle) -> dict[str, Any]:
    generation = np.asarray(bundle.generation_actions(mmap=True), dtype=np.float32)
    final = generation[:, -1:, :, :]
    commitment = np.linalg.norm(generation - final, axis=(-1, -2))
    return {"values": _round(commitment)}

def _action_metric_metadata(bundle: TraceBundle) -> dict[int, dict[str, str]]:
    table = bundle.action_normalization
    if table.empty:
        return {}
    row = table.iloc[0]
    names = _json_parse(row.get("action_dim_names"))
    metadata = _json_parse(row.get("metadata")) or {}
    labels = metadata.get("action_labels") if isinstance(metadata, dict) else None
    units = metadata.get("action_units") if isinstance(metadata, dict) else None
    if not isinstance(names, list):
        names = metadata.get("action_names") if isinstance(metadata, dict) else None
    if not isinstance(names, list):
        return {}
    result: dict[int, dict[str, str]] = {}
    for index, name in enumerate(names):
        label = (
            labels[index]
            if isinstance(labels, list) and index < len(labels) and labels[index]
            else _label_from_metric_name(str(name))
        )
        unit = (
            units[index]
            if isinstance(units, list) and index < len(units) and units[index]
            else "normalized controller units"
        )
        result[index] = {"name": str(name), "label": str(label), "unit": str(unit)}
    return result

def _policy_call_x_values(bundle: TraceBundle, count: int) -> np.ndarray:
    calls = bundle.policy_calls
    if calls.empty or "observation_timestep" not in calls:
        return np.arange(count, dtype=np.float32)
    values = (
        calls.sort_values("policy_call_index")["observation_timestep"]
        .to_numpy(dtype=np.float32)
        .reshape(-1)
    )
    if values.size < count:
        return np.arange(count, dtype=np.float32)
    return values[:count]

def _episode_metrics_payload(bundle: TraceBundle) -> dict[str, Any]:
    """Return plot-ready episode metrics from logged state/action arrays."""
    metrics: list[dict[str, Any]] = []
    action_metadata = _action_metric_metadata(bundle)

    def add_metric(
        key: str,
        label: str,
        values: Any,
        *,
        domain: str = "time",
        kind: str = "line",
        description: str = "",
        x_values: Any | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        y_unit: str | None = None,
    ) -> None:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 1 or array.size == 0:
            return
        if x_values is None:
            x_array = np.arange(array.size, dtype=np.float32)
        else:
            x_array = np.asarray(x_values, dtype=np.float32)
            if x_array.ndim != 1 or x_array.size != array.size:
                x_array = np.arange(array.size, dtype=np.float32)
        metrics.append(
            {
                "key": key,
                "label": label,
                "domain": domain,
                "kind": kind,
                "description": description,
                "values": _round(array),
                "x_values": _round(x_array),
                "x_label": x_label or _domain_x_label(domain),
                "y_label": y_label or label,
                "y_unit": y_unit,
            }
        )

    try:
        actions = np.asarray(bundle.actions(mmap=True), dtype=np.float32)
        add_metric(
            "action_norm",
            "Action norm",
            np.linalg.norm(actions, axis=-1),
            description="Executed action magnitude by environment timestep.",
            y_label="Action norm",
            y_unit="normalized controller units",
        )
        for dim in range(min(actions.shape[-1], 8)):
            dim_info = action_metadata.get(dim, {})
            dim_label = str(dim_info.get("label") or f"Action dim {dim}")
            dim_name = str(dim_info.get("name") or f"dim_{dim}")
            dim_unit = str(dim_info.get("unit") or "normalized controller units")
            add_metric(
                f"action_dim_{dim}",
                dim_label,
                actions[:, dim],
                description=(
                    f"Executed action dimension {dim} ({dim_name}) by environment timestep."
                ),
                y_label=dim_label,
                y_unit=dim_unit,
            )
    except KeyError:
        pass

    for name, label, description in [
        ("gripper_open_signal", "Gripper open", "Logged gripper open/close signal."),
        ("rewards", "Reward", "Environment reward by timestep."),
    ]:
        try:
            add_metric(name, label, bundle.array(name, mmap=True), description=description)
        except KeyError:
            pass

    try:
        eef = np.asarray(bundle.array("eef_pos", mmap=True), dtype=np.float32)
        add_metric("eef_x", "EEF x", eef[:, 0], description="End-effector x position.")
        add_metric("eef_y", "EEF y", eef[:, 1], description="End-effector y position.")
        add_metric("eef_z", "EEF z", eef[:, 2], description="End-effector z position.")
        if eef.shape[0] > 1:
            speed = np.concatenate([[0.0], np.linalg.norm(np.diff(eef, axis=0), axis=-1)])
            add_metric(
                "eef_speed",
                "EEF speed",
                speed,
                description="End-effector step-to-step movement.",
                y_label="EEF speed",
                y_unit="position units / timestep",
            )
    except KeyError:
        pass

    try:
        gripper = np.asarray(bundle.array("gripper_qpos", mmap=True), dtype=np.float32)
        add_metric(
            "gripper_qpos_mean",
            "Gripper qpos mean",
            gripper.mean(axis=-1),
            description="Mean gripper joint position.",
        )
    except KeyError:
        pass

    try:
        generation = np.asarray(bundle.generation_actions(mmap=True), dtype=np.float32)
        final = generation[:, -1:, :, :]
        commitment = np.linalg.norm(generation - final, axis=(-1, -2))
        if commitment.ndim == 2:
            call_x = _policy_call_x_values(bundle, commitment.shape[0])
            add_metric(
                "generation_start",
                "Generation start delta",
                commitment[:, 0],
                domain="call",
                description="First generation-step distance from final sampled action.",
                x_values=call_x,
                y_label="Generation delta",
                y_unit="action L2",
            )
            add_metric(
                "generation_end",
                "Generation end delta",
                commitment[:, -1],
                domain="call",
                description="Final generation-step distance from sampled action.",
                x_values=call_x,
                y_label="Generation delta",
                y_unit="action L2",
            )
            add_metric(
                "generation_delta",
                "Generation convergence",
                commitment[:, 0] - commitment[:, -1],
                domain="call",
                description="Start-to-end generation commitment change per timestep.",
                x_values=call_x,
                y_label="Generation convergence",
                y_unit="action L2",
            )
    except KeyError:
        pass

    return {
        "domains": [
            {"key": "time", "label": "Time"},
            {"key": "call", "label": "Policy call"},
            {"key": "generation", "label": "Generation step"},
        ],
        "metrics": metrics,
    }
