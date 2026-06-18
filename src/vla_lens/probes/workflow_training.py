"""Probe training workflow orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn

from vla_lens.artifacts import LensArtifact, make_artifact_id
from vla_lens.dataset import build_dataset_index
from vla_lens.probes.suite import run_probe_suite
from vla_lens.probes.workflow_artifacts import (
    _array_fingerprint,
    _artifact_dir,
    _best_model_arrays,
    _best_result_details,
    _best_result_index,
    _json_scalar,
    _metric_definitions,
    _null_metrics,
    _per_group_metrics,
    _per_split_metrics,
    _prediction_frame,
    _primary_metric,
    _primary_probe_type,
    _probe_data_quality,
    _probe_examples,
    _probe_hyperparams,
    _probe_input,
    _probe_lineage,
    _probe_metrics,
    _probe_notes,
    _probe_source,
    _probe_split,
    _probe_target,
    _records,
    _split_summary,
    _value_counts,
)
from vla_lens.probes.workflow_prepare import (
    _apply_missing_policy,
    _apply_row_expansion,
    _apply_row_filters,
    _attach_episode_metadata,
    _ensure_split,
)
from vla_lens.probes.workflow_spec import baseline_columns, normalize_probe_spec
from vla_lens.probes.workflow_targets import (
    _normalize_target_spec,
    _resolve_probe_target,
    _target_name,
)
from vla_lens.probes.workflow_types import PROBE_ARTIFACT_SCHEMA_VERSION, SavedProbeSuite
from vla_lens.selectors import ActivationQuery
from vla_lens.traces import TraceDataset
from vla_lens.workbench import AnalysisRunSpec, save_analysis_run


def train_probe_artifact(
    dataset: TraceDataset,
    *,
    name: str,
    selector: ActivationQuery,
    target: str | Mapping[str, Any] = "outcome",
    split_kind: str = "random_episode",
    split_column: str = "split",
    train_value: str = "train",
    test_value: str = "test",
    metadata_baseline_columns: Sequence[str] = (),
    sweep: str | Sequence[str] = "layer",
    row_filter: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    eval_values: Sequence[str] | None = None,
    selection_value: str | None = None,
    probe_models: Sequence[str] = ("linear",),
    research: Mapping[str, Any] | None = None,
    row_expand: Mapping[str, Any] | None = None,
) -> SavedProbeSuite:
    """Train simple probes from an activation selector and save a ``LensArtifact``."""
    feature_matrix = dataset.select_model_sites(selector).materialize(cache=True)
    X, rows = feature_matrix.X, feature_matrix.rows
    if rows.empty or X.shape[0] == 0:
        raise ValueError(f"Probe selector matched no activation rows: {selector.to_dict()}")
    rows = _attach_episode_metadata(rows, dataset)
    X, rows, expansion_summary = _apply_row_expansion(X, rows, dataset, row_expand)
    target_spec = _normalize_target_spec(target)
    target_name = _target_name(target_spec)
    rows = _resolve_probe_target(dataset, rows, target_spec)
    X, rows, filter_summary = _apply_row_filters(X, rows, row_filter)
    X, rows, missing_summary = _apply_missing_policy(
        X,
        rows,
        target_name,
        policy=str(target_spec.get("missing_policy") or "error"),
    )
    rows = _ensure_split(
        rows,
        split_column,
        train_value=train_value,
        test_value=test_value,
        split_kind=split_kind,
    )
    if target_name not in rows:
        raise KeyError(f"Probe target '{target_name}' is not present in selected rows")
    if X.shape[0] != len(rows):
        raise ValueError(
            f"Feature rows mismatch: X has {X.shape[0]} rows, metadata has {len(rows)}"
        )

    results = _run_sweep(
        X=X,
        rows=rows,
        target=target_name,
        split_column=split_column,
        train_value=train_value,
        test_value=test_value,
        metadata_baseline_columns=[
            column for column in metadata_baseline_columns if column in rows.columns
        ],
        sweep=sweep,
        target_kind=str(_probe_target(target_name, rows, target_spec=target_spec)["kind"]),
        eval_values=list(eval_values or [test_value]),
        probe_models=list(probe_models),
    )
    if results.empty:
        raise ValueError(
            "No probe result could be trained. Check that train/test rows exist "
            "and the training split has at least two target values."
        )

    artifact_id = make_artifact_id(name, "probe_suite")
    selected_result_index = _best_result_index(
        results,
        selection_value=selection_value or test_value,
        prefer_model="linear",
    )
    selected_results = results.loc[[selected_result_index]]
    selected_eval_results = _selected_eval_results(results, selected_result_index)
    prediction_records = _prediction_frame(selected_eval_results)
    scored_prediction_records = _prediction_frame(
        selected_results,
        record_column="all_prediction_records",
    )
    if scored_prediction_records.empty:
        scored_prediction_records = prediction_records
    model_arrays, model_state_summary = _best_model_arrays(
        results,
        selection_value=selection_value or test_value,
    )
    output_dir = _dataset_output_dir(dataset, artifact_id)
    outputs = {
        "metrics": str(output_dir / "metrics.json"),
        "predictions": str(output_dir / "predictions.parquet"),
        "scored_predictions": str(output_dir / "scored_predictions.parquet"),
        "per_split_metrics": str(output_dir / "per_split_metrics.parquet"),
        "per_group_metrics": str(output_dir / "per_group_metrics.parquet"),
        "null_metrics": str(output_dir / "null_metrics.parquet"),
        "weights": str(output_dir / "weights.zarr")
        if "weights" in model_arrays
        else None,
        "bias": str(output_dir / "bias.zarr")
        if "bias" in model_arrays
        else None,
        "normalizer_feature_mean": str(output_dir / "feature_mean.zarr")
        if "feature_mean" in model_arrays
        else None,
        "normalizer_feature_scale": str(output_dir / "feature_scale.zarr")
        if "feature_scale" in model_arrays
        else None,
    }
    research_framing = _probe_research_framing(research)
    method = {
        "workflow": "train_probe_artifact",
        "probe_artifact_schema_version": PROBE_ARTIFACT_SCHEMA_VERSION,
        "research": research_framing,
        "lineage": _probe_lineage(random_seed=None),
        "source": _probe_source(dataset, rows),
        "input": _probe_input(selector, rows, X, feature_matrix.cache_key),
        "target": _probe_target(target_name, rows, target_spec=target_spec),
        "examples": _probe_examples(rows, target=target_name, split_column=split_column),
        "split": _probe_split(
            rows,
            split_kind=split_kind,
            split_column=split_column,
            train_value=train_value,
            test_value=test_value,
            eval_values=list(eval_values or [test_value]),
            selection_value=selection_value or test_value,
        ),
        "normalization": {
            "method": "standardize",
            "feature_centering": True,
            "feature_scaling": True,
            "target_centering": False,
            "target_scaling": False,
            "fit_split": train_value,
            "weights_space": "normalized_feature_space",
        },
        "probe": {
            "type": _primary_probe_type(results),
            "library": "sklearn",
            "library_version": sklearn.__version__,
            "hyperparams": _probe_hyperparams(results),
            "models": list(probe_models),
            "primary_model": "linear",
            "secondary_models": [model for model in probe_models if model != "linear"],
            "trained_on_split": train_value,
            "weights_space": "normalized_feature_space",
            "best_model_state": model_state_summary,
        },
        "evaluation": {
            "primary_split": test_value,
            "selection_split": selection_value or test_value,
            "eval_splits": list(eval_values or [test_value]),
            "primary_metric": _primary_metric(results),
            "grain": "row",
            "aggregation": "over_rows",
            "metric_definitions": _metric_definitions(results),
        },
        "prediction_retention": {
            "mode": "row_level_eval_plus_all_source_scoring",
            "eval_splits": list(eval_values or [test_value]),
            "eval_row_count": int(len(prediction_records)),
            "scored_split_values": sorted(
                str(value)
                for value in scored_prediction_records.get("split", pd.Series(dtype=object))
                .dropna()
                .unique()
            )
            if "split" in scored_prediction_records
            else [],
            "scored_row_count": int(len(scored_prediction_records)),
        },
        "outputs": {key: value for key, value in outputs.items() if value is not None},
        "split_kind": split_kind,
        "split_column": split_column,
        "train_value": train_value,
        "test_value": test_value,
        "eval_values": list(eval_values or [test_value]),
        "selection_value": selection_value or test_value,
        "metadata_baseline_columns": [
            column for column in metadata_baseline_columns if column in rows.columns
        ],
        "sweep": sweep,
        "row_expansion": expansion_summary,
        "row_filter": filter_summary,
        "missing_target": missing_summary,
    }
    metrics = _probe_metrics(
        results,
        rows,
        target=target_name,
        selection_value=selection_value or test_value,
    )
    metrics["probe_artifact_schema_version"] = PROBE_ARTIFACT_SCHEMA_VERSION
    metrics["prediction_row_count"] = int(len(prediction_records))
    metrics["scored_prediction_row_count"] = int(len(scored_prediction_records))
    metrics["feature_matrix_fingerprint"] = _array_fingerprint(X)
    per_split_metrics = _per_split_metrics(prediction_records)
    per_group_metrics = _per_group_metrics(
        prediction_records,
        rows,
        group_columns=["benchmark", "task_id", "scene_family", "target_parse_status"],
    )
    null_metrics = _null_metrics(prediction_records)
    if not null_metrics.empty:
        metrics["null_score_mean"] = float(null_metrics["score"].mean())
        metrics["null_score_std"] = float(null_metrics["score"].std(ddof=0))
        best_score = metrics.get("best_score")
        if best_score is not None:
            metrics["null_p_value"] = float(
                (1 + (null_metrics["score"] >= float(best_score)).sum())
                / (len(null_metrics) + 1)
            )
    artifact = LensArtifact(
        artifact_id=artifact_id,
        artifact_type="probe_suite",
        name=name,
        group_id="probe_suites",
        scope="dataset",
        selector=selector.to_dict(),
        method=method,
        metrics=metrics,
        display={
            "kind": "probe_suite",
            "research": research_framing,
            "results": _records(results),
            "best_result_details": _best_result_details(
                results,
                selection_value=selection_value or test_value,
            ),
            "target": target_name,
            "split_summary": _split_summary(rows, split_column),
            "target_distribution": _value_counts(rows[target_name]),
            "baseline_columns": [
                column for column in metadata_baseline_columns if column in rows.columns
            ],
            "interpretation_notes": _probe_notes(
                rows,
                target_name,
                [column for column in metadata_baseline_columns if column in rows.columns],
            ),
            "row_count": int(len(rows)),
            "sample_count": int(X.shape[0]),
            "feature_dim": int(X.shape[1]) if X.ndim == 2 else None,
            "source_columns": sorted(str(column) for column in rows.columns),
            "data_quality": _probe_data_quality(
                rows,
                target=target_name,
                split_column=split_column,
                metadata_baseline_columns=[
                    column for column in metadata_baseline_columns if column in rows.columns
                ],
            ),
            "row_expansion": expansion_summary,
            "row_filter": filter_summary,
            "missing_target": missing_summary,
        },
        tags=("probe", target_name),
        source_trace_ids=tuple(sorted(str(value) for value in rows["trace_id"].dropna().unique())),
    )
    saved = dataset.save_artifact(artifact, arrays=model_arrays)
    artifact_dir = _artifact_dir(dataset, saved)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    prediction_records.to_parquet(artifact_dir / "predictions.parquet", index=False)
    scored_prediction_records.to_parquet(
        artifact_dir / "scored_predictions.parquet",
        index=False,
    )
    per_split_metrics.to_parquet(artifact_dir / "per_split_metrics.parquet", index=False)
    per_group_metrics.to_parquet(artifact_dir / "per_group_metrics.parquet", index=False)
    null_metrics.to_parquet(artifact_dir / "null_metrics.parquet", index=False)
    (artifact_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    save_analysis_run(
        dataset,
        AnalysisRunSpec(
            run_id=saved.artifact_id,
            workflow="probe_suite",
            inputs=saved.selector,
            outputs=tuple(saved.arrays),
            provenance={"artifact_id": saved.artifact_id},
        ),
    )
    build_dataset_index(dataset.root, overwrite=True)
    return SavedProbeSuite(artifact=saved, results=results, rows=rows)


def _selected_eval_results(results: pd.DataFrame, selected_result_index: int) -> pd.DataFrame:
    selected = results.loc[selected_result_index]
    mask = pd.Series(True, index=results.index)
    for column in ["feature", "target", "probe_type", "model", "primary_metric"]:
        if column in results:
            mask &= results[column].astype(str) == str(selected.get(column))
    matched = results.loc[mask]
    if matched.empty:
        return results.loc[[selected_result_index]]
    return matched


def train_probe_artifact_from_spec(
    dataset: TraceDataset,
    spec: Mapping[str, Any],
) -> SavedProbeSuite:
    """Train a probe artifact from a YAML/JSON-compatible spec."""
    normalized = normalize_probe_spec(spec)
    features = normalized["features"]
    selector = ActivationQuery(
        episodes=dict(features.get("episodes") or {}),
        name=features.get("name"),
        module=features.get("module"),
        layers=features.get("layers"),
        tensor_type=features.get("tensor_type"),
        token_kind=features.get("token_kind"),
        timesteps=features.get("timesteps", "all"),
        policy_calls=features.get("policy_calls", "all"),
        generation_step=features.get("generation_step"),
        reduce_tokens=features.get("reduction", "mean"),
        dtype=str(features.get("dtype", "float32")),
    )
    split = normalized["split"]
    return train_probe_artifact(
        dataset,
        name=str(normalized["name"]),
        selector=selector,
        target=normalized["target"],
        split_kind=str(split["kind"]),
        split_column=str(split.get("column", "split")),
        train_value=str(split.get("train_value", "train")),
        test_value=str(split.get("test_value", "test")),
        metadata_baseline_columns=baseline_columns(normalized.get("baseline", [])),
        sweep=normalized.get("sweep", "layer"),
        row_filter=normalized.get("row_filter"),
        row_expand=normalized.get("row_expand"),
        eval_values=[
            str(value)
            for value in split.get("eval_values", [split.get("test_value", "test")])
        ],
        selection_value=str(split.get("selection_value", split.get("test_value", "test"))),
        probe_models=[
            str(value) for value in normalized.get("probe", {}).get("models", ["linear"])
        ],
        research=normalized,
    )


def _probe_research_framing(spec: Mapping[str, Any] | None) -> dict[str, Any]:
    if not spec:
        return {}
    keys = ["question", "hypothesis_family", "intended_claim"]
    return {key: str(spec[key]) for key in keys if spec.get(key) not in {None, ""}}


def _dataset_output_dir(dataset: TraceDataset, artifact_id: str) -> Path:
    artifact_root = dataset._dataset_artifact_root()
    if artifact_root == dataset.root:
        return Path("artifacts") / artifact_id
    return artifact_root.relative_to(dataset.root) / "artifacts" / artifact_id


def _run_sweep(
    *,
    X: np.ndarray,
    rows: pd.DataFrame,
    target: str,
    split_column: str,
    train_value: str,
    test_value: str,
    metadata_baseline_columns: list[str],
    sweep: str | Sequence[str],
    target_kind: str,
    eval_values: list[str],
    probe_models: list[str],
) -> pd.DataFrame:
    sweep_columns = _normalize_sweep_columns(sweep)
    if not sweep_columns:
        return run_probe_suite(
            rows,
            {"selected model_sites": X},
            [target],
            split_column=split_column,
            train_value=train_value,
            test_value=test_value,
            eval_values=eval_values,
            metadata_baseline_columns=metadata_baseline_columns,
            target_kinds={target: target_kind},
            probe_models=probe_models,
        )

    missing = [column for column in sweep_columns if column not in rows]
    if missing:
        raise KeyError(f"Sweep column(s) {missing!r} are not present in selected rows")
    frames: list[pd.DataFrame] = []
    group_key = sweep_columns[0] if len(sweep_columns) == 1 else sweep_columns
    for values, group in rows.groupby(group_key, dropna=False, sort=True):
        value_tuple = values if isinstance(values, tuple) else (values,)
        sweep_value = (
            _json_scalar(value_tuple[0])
            if len(sweep_columns) == 1
            else {
                column: _json_scalar(value)
                for column, value in zip(sweep_columns, value_tuple, strict=False)
            }
        )
        index = group.index.to_numpy()
        result = run_probe_suite(
            group.reset_index(drop=True),
            {_feature_name_for_sweep(sweep_columns, value_tuple): X[index]},
            [target],
            split_column=split_column,
            train_value=train_value,
            test_value=test_value,
            eval_values=eval_values,
            metadata_baseline_columns=metadata_baseline_columns,
            target_kinds={target: target_kind},
            probe_models=probe_models,
        )
        if result.empty:
            continue
        result.insert(0, "sweep", ",".join(sweep_columns))
        result.insert(1, "sweep_value", sweep_value)
        for column, value in zip(sweep_columns, value_tuple, strict=False):
            result[f"sweep_{column}"] = _json_scalar(value)
        frames.append(result)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _normalize_sweep_columns(sweep: str | Sequence[str]) -> list[str]:
    if isinstance(sweep, str):
        if sweep in {"", "none", "null"}:
            return []
        return [sweep]
    return [str(column) for column in sweep if str(column) not in {"", "none", "null"}]


def _feature_name_for_sweep(columns: Sequence[str], values: Sequence[Any]) -> str:
    if len(columns) == 1:
        return f"{columns[0]} {values[0]}"
    parts = [
        f"{column}={_json_scalar(value)}"
        for column, value in zip(columns, values, strict=False)
    ]
    return ", ".join(parts)
