"""Target-object encoding workflow for axis-native VLA-lens workbenches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from vla_lens.artifacts import LensArtifact
from vla_lens.probes.suite import run_probe_suite
from vla_lens.probes.workflow_artifacts import _value_counts
from vla_lens.probes.workflow_prepare import _attach_episode_metadata, _ensure_split
from vla_lens.selectors import ActivationQuery
from vla_lens.traces import TraceDataset
from vla_lens.workbench import AnalysisRunSpec, save_analysis_run


@dataclass(frozen=True, slots=True)
class SavedTargetObjectEncoding:
    artifact: LensArtifact
    metric_cube: np.ndarray
    baseline_cube: np.ndarray
    delta_cube: np.ndarray
    layers: tuple[int, ...]
    timesteps: tuple[int, ...]
    token_kinds: tuple[str, ...]


def save_target_object_encoding_artifact(
    dataset: TraceDataset,
    *,
    name: str = "Target-object encoding",
    module: str | None = None,
    tensor_type: str | None = None,
    token_kinds: Sequence[str] | None = None,
    max_timesteps: int | None = 64,
) -> SavedTargetObjectEncoding:
    """Train episode-safe probes over layer x timestep x token-kind axes."""
    layers = _available_layers(dataset)
    timesteps = _policy_timesteps(dataset, max_timesteps=max_timesteps)
    kinds = tuple(token_kinds or _available_token_kinds(dataset))
    if not layers or not timesteps or not kinds:
        raise ValueError("Target-object encoding requires layers, timesteps, and token kinds")

    metric_cube = np.full((len(layers), len(timesteps), len(kinds)), np.nan, dtype=np.float32)
    baseline_cube = np.full_like(metric_cube, np.nan)
    records: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    confusion: list[dict[str, Any]] = []
    cell_details: list[dict[str, Any]] = []
    split_summary: dict[str, Any] | None = None

    for layer_index, layer in enumerate(layers):
        for timestep_index, timestep in enumerate(timesteps):
            for kind_index, token_kind in enumerate(kinds):
                result = _probe_cell(
                    dataset,
                    layer=layer,
                    timestep=timestep,
                    token_kind=token_kind,
                    module=module,
                    tensor_type=tensor_type,
                )
                if result is None:
                    continue
                row = result["result"]
                metric_cube[layer_index, timestep_index, kind_index] = row["score"]
                baseline_cube[layer_index, timestep_index, kind_index] = row["baseline_score"]
                records.append(
                    {
                        "layer": layer,
                        "timestep": timestep,
                        "token_kind": token_kind,
                        "score": row["score"],
                        "baseline_score": row["baseline_score"],
                        "delta": row["score"] - row["baseline_score"],
                        "n_train": row["n_train"],
                        "n_test": row["n_test"],
                        "metadata_baseline": row["metadata_baseline"],
                    }
                )
                details = row.get("details") or {}
                cell_examples = _linked_examples(
                    details.get("test_predictions") or (),
                    timestep=timestep,
                )
                cell_confusion = list(details.get("confusion_matrix") or ())
                cell_details.append(
                    {
                        "layer": layer,
                        "timestep": timestep,
                        "token_kind": token_kind,
                        "score": row["score"],
                        "baseline_score": row["baseline_score"],
                        "delta": row["score"] - row["baseline_score"],
                        "n_train": row["n_train"],
                        "n_test": row["n_test"],
                        "split_summary": result["split_summary"],
                        "confusion_matrix": cell_confusion,
                        "linked_examples": cell_examples,
                    }
                )
                if not examples and details.get("test_predictions"):
                    examples = cell_examples
                if not confusion and details.get("confusion_matrix"):
                    confusion = cell_confusion
                split_summary = split_summary or result["split_summary"]

    if not records:
        raise ValueError("No target-object probe cells could be trained")

    delta_cube = metric_cube - baseline_cube
    best = max(records, key=lambda item: float(item["delta"]))
    artifact = LensArtifact.create(
        artifact_type="target_object_encoding",
        name=name,
        group_id="target_object_encoding",
        scope="dataset",
        selector={
            "activation_query": {
                "module": module or "*",
                "tensor_type": tensor_type,
                "layers": list(layers),
                "timesteps": list(timesteps),
                "token_kinds": list(kinds),
                "reduction": "mean",
            },
            "target": "target_object",
        },
        method={
            "workflow": "target_object_encoding",
            "split": {"unit": "episode", "kind": "random_episode"},
            "probe": {"type": "linear", "standardize": True},
            "metrics": ["accuracy", "baseline_accuracy", "delta"],
        },
        metrics={
            "best_score": float(best["score"]),
            "best_baseline": float(best["baseline_score"]),
            "best_delta": float(best["delta"]),
            "best_layer": int(best["layer"]),
            "best_timestep": int(best["timestep"]),
            "best_token_kind": str(best["token_kind"]),
            "trained_cells": int(len(records)),
            "layer_count": int(len(layers)),
            "timestep_count": int(len(timesteps)),
            "token_kind_count": int(len(kinds)),
        },
        display={
            "kind": "target_object_encoding",
            "axes": {
                "layer": list(layers),
                "timestep": list(timesteps),
                "token_kind": list(kinds),
            },
            "primary_array": "metric_cube",
            "baseline_array": "baseline_cube",
            "delta_array": "delta_cube",
            "best_cell": best,
            "records": records,
            "cell_details": cell_details,
            "split_summary": split_summary or {},
            "confusion_matrix": confusion,
            "linked_examples": examples,
            "interpretation_notes": [
                "Probe accuracy is diagnostic and correlational, not causal evidence.",
                "Train/test split is by episode, not random timestep.",
                "Compare metric_cube against baseline_cube before interpreting a cell.",
            ],
        },
        tags=("probe", "target_object", "encoding"),
        source_trace_ids=tuple(dataset.episode_index["trace_id"].astype(str).tolist()),
    )
    saved = dataset.save_artifact(
        artifact,
        arrays={
            "metric_cube": metric_cube,
            "baseline_cube": baseline_cube,
            "delta_cube": delta_cube,
        },
    )
    save_analysis_run(
        dataset,
        AnalysisRunSpec(
            run_id=saved.artifact_id,
            workflow="target_object_encoding",
            inputs=saved.selector,
            outputs=tuple(saved.arrays),
            provenance={"artifact_id": saved.artifact_id},
        ),
    )
    return SavedTargetObjectEncoding(
        artifact=saved,
        metric_cube=metric_cube,
        baseline_cube=baseline_cube,
        delta_cube=delta_cube,
        layers=layers,
        timesteps=timesteps,
        token_kinds=kinds,
    )


def _linked_examples(
    predictions: Sequence[Mapping[str, Any]],
    *,
    timestep: int,
    limit: int = 24,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(predictions):
        trace_id = row.get("trace_id")
        actual = row.get("actual") or row.get("target_object")
        predicted = row.get("predicted")
        margin = row.get("margin")
        status = "unknown"
        if actual is not None and predicted is not None:
            status = "correct" if str(actual) == str(predicted) else "false_positive"
        rows.append(
            {
                **dict(row),
                "example_id": str(row.get("example_id") or f"{trace_id}:{timestep}:{index}"),
                "trace_id": trace_id,
                "episode_id": row.get("episode_id"),
                "timestep": timestep,
                "target_object": actual,
                "actual": actual,
                "predicted": predicted,
                "prediction_status": row.get("prediction_status") or status,
                "margin": margin,
            }
        )
    return rows[:limit]


def _probe_cell(
    dataset: TraceDataset,
    *,
    layer: int,
    timestep: int,
    token_kind: str,
    module: str | None,
    tensor_type: str | None,
) -> dict[str, Any] | None:
    selector = ActivationQuery(
        module=module or "*",
        layers=[layer],
        tensor_type=tensor_type,
        token_kind=token_kind,
        timesteps=[timestep],
        reduce_tokens="mean",
    )
    X, rows = dataset.select_model_sites(selector).to_matrix(cache=True)
    if X.shape[0] == 0 or rows.empty:
        return None
    rows = _attach_episode_metadata(rows, dataset)
    if "target_object" not in rows or rows["target_object"].astype(str).nunique() < 2:
        return None
    rows = _ensure_split(
        rows,
        "split",
        train_value="train",
        test_value="test",
        split_kind="random_episode",
    )
    result = run_probe_suite(
        rows,
        {f"layer {layer} timestep {timestep} token {token_kind}": X},
        ["target_object"],
        split_column="split",
        train_value="train",
        test_value="test",
        metadata_baseline_columns=[column for column in ["benchmark", "task_id"] if column in rows],
    )
    if result.empty:
        return None
    row = result.iloc[0].to_dict()
    return {
        "result": {
            **row,
            "score": float(row["score"]),
            "baseline_score": float(row["baseline_score"]),
            "n_train": int(row["n_train"]),
            "n_test": int(row["n_test"]),
        },
        "split_summary": {
            "column": "split",
            "values": _value_counts(rows["split"]),
            "episodes": _value_counts(rows[["split", "trace_id"]].drop_duplicates()["split"]),
            "class_counts": _value_counts(rows["target_object"]),
        },
    }


def _available_layers(dataset: TraceDataset) -> tuple[int, ...]:
    index = dataset.model_site_index
    if index.empty or "layer" not in index:
        return ()
    return tuple(sorted(int(value) for value in index["layer"].dropna().unique()))


def _available_token_kinds(dataset: TraceDataset) -> tuple[str, ...]:
    index = dataset.model_site_index
    if index.empty or "token_kind" not in index:
        return ()
    return tuple(sorted(str(value) for value in index["token_kind"].dropna().unique()))


def _policy_timesteps(dataset: TraceDataset, *, max_timesteps: int | None) -> tuple[int, ...]:
    values: list[int] = []
    for bundle in dataset.bundles:
        calls = bundle.policy_calls
        if calls.empty:
            continue
        column = "observation_timestep" if "observation_timestep" in calls else "env_timestep_start"
        if column in calls:
            values.extend(int(value) for value in calls[column].dropna().unique())
    values = sorted(set(values))
    if max_timesteps is not None and len(values) > max_timesteps:
        positions = np.linspace(0, len(values) - 1, max_timesteps).round().astype(int)
        values = [values[int(index)] for index in positions]
    return tuple(values)
