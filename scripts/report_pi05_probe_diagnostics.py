"""Report PI0.5 object-centric probe diagnostics from saved activations."""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from vla_lens.probes.workflow_prepare import (
    _apply_missing_policy,
    _apply_row_filters,
    _artifact_output_path,
    _attach_episode_metadata,
    _ensure_split,
    _latest_artifact,
)
from vla_lens.probes.workflow_spec import load_probe_spec, normalize_probe_spec
from vla_lens.probes.workflow_targets import (
    _normalize_target_spec,
    _resolve_probe_target,
    _target_name,
)
from vla_lens.probes.workflow_types import OBJECT_FLOW_ARTIFACT_TYPE
from vla_lens.selectors import ActivationQuery
from vla_lens.traces import TraceDataset

DEFAULT_ROOT = Path("/mnt/new-volume/vla-lens/pi05-broad-1000-mech-light-lerobot-v3")
DEFAULT_SPEC = Path(
    "configs/probes/pi05_broad_1000_next_manipulated_pre_contact_expert_action_hidden.yaml"
)
DEFAULT_BATTERY_TARGETS = [
    "next_manipulated_object",
    "active_manipulated_object",
    "active_receptacle_object",
    "task_phase",
    "next_action_type",
]
POLICY_CALL_KEYS = ["trace_id", "policy_call_index"]


@dataclass(frozen=True, slots=True)
class PreparedProbeData:
    dataset: TraceDataset
    X: np.ndarray
    rows: pd.DataFrame
    target: str
    train_value: str
    selection_value: str
    test_value: str
    eval_values: list[str]
    split_column: str
    filter_summary: dict[str, Any]
    missing_summary: dict[str, Any]
    cache_key: str


@dataclass(frozen=True, slots=True)
class LayerReadout:
    layer: str
    model: Any
    classes: np.ndarray
    row_index: np.ndarray
    rows: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=DEFAULT_ROOT,
        help="Trace dataset root.",
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC, help="Probe YAML spec.")
    parser.add_argument(
        "--artifact-id",
        default=None,
        help="Probe artifact to attach diagnostics to. Defaults to latest probe_suite.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for report parquet/json files.",
    )
    parser.add_argument("--selected-layer", default="12", help="Layer to emphasize.")
    parser.add_argument(
        "--battery-target",
        action="append",
        default=[],
        help="Additional readout target. Defaults to the first object-centric battery.",
    )
    parser.add_argument(
        "--skip-battery",
        action="store_true",
        help="Only run diagnostics for the primary target.",
    )
    parser.add_argument(
        "--bootstrap-runs",
        type=int,
        default=1000,
        help="Grouped bootstrap runs for selected-layer eval metrics.",
    )
    parser.add_argument(
        "--null-shuffles",
        type=int,
        default=0,
        help="Selection-aware label shuffles. Use 1000 for claim-bearing null.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Max iterations for logistic readouts.",
    )
    parser.add_argument(
        "--model",
        choices=["sgd", "ridge", "logistic", "logistic-lbfgs"],
        default="sgd",
        help=(
            "Linear model for diagnostics. sgd/ridge are fast screening readouts; "
            "logistic-lbfgs matches the original probe family more closely."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = load_probe_spec(args.spec)
    prepared = prepare_probe_data(args.root, spec)
    output_dir = args.output_dir or _default_output_dir(prepared.dataset, args.artifact_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_layer = _normalize_layer(args.selected_layer)
    readouts = train_layer_readouts(
        prepared.X,
        prepared.rows,
        target=prepared.target,
        split_column=prepared.split_column,
        train_value=prepared.train_value,
        max_iter=args.max_iter,
        seed=args.seed,
        model_name=args.model,
    )
    if not readouts:
        raise SystemExit("No layer readouts could be trained.")
    if selected_layer not in readouts:
        selected_layer = _select_best_layer(
            readouts,
            prepared.target,
            prepared.selection_value,
            prepared.split_column,
            top_k=args.top_k,
        )

    predictions = score_readouts_with_X(
        prepared.X,
        readouts,
        target=prepared.target,
        split_column=prepared.split_column,
        top_k=args.top_k,
    )
    predictions = attach_error_browser_context(prepared.dataset, predictions)

    layer_metrics = layer_split_metrics(
        predictions,
        target=prepared.target,
        split_column=prepared.split_column,
    )
    per_class = per_class_metrics(
        predictions,
        target=prepared.target,
        split_column=prepared.split_column,
    )
    confusion = confusion_matrix_records(
        predictions,
        target=prepared.target,
        split_column=prepared.split_column,
    )
    supports = support_tables(
        prepared.rows,
        target=prepared.target,
        split_column=prepared.split_column,
    )
    lead_time = lead_time_metrics(
        predictions,
        target=prepared.target,
        split_column=prepared.split_column,
    )
    bootstrap = bootstrap_intervals(
        predictions.loc[predictions["layer"] == selected_layer],
        split_values=prepared.eval_values,
        group_columns=["episode_id", "task_id"],
        runs=args.bootstrap_runs,
        seed=args.seed,
    )
    battery = pd.DataFrame()
    if not args.skip_battery:
        battery = readout_battery_from_prepared(
            prepared,
            target_names=args.battery_target or DEFAULT_BATTERY_TARGETS,
            max_iter=args.max_iter,
            seed=args.seed,
            top_k=args.top_k,
            model_name=args.model,
        )

    null_frame = pd.DataFrame()
    if args.null_shuffles > 0:
        null_frame = selection_aware_null(
            prepared,
            shuffles=args.null_shuffles,
            max_iter=args.max_iter,
            seed=args.seed,
            top_k=args.top_k,
            model_name=args.model,
        )

    summary = build_summary(
        prepared,
        layer_metrics,
        supports,
        selected_layer=selected_layer,
        null_frame=null_frame,
    )
    write_reports(
        output_dir,
        summary=summary,
        predictions=predictions,
        layer_metrics=layer_metrics,
        per_class=per_class,
        confusion=confusion,
        supports=supports,
        lead_time=lead_time,
        bootstrap=bootstrap,
        battery=battery,
        null_frame=null_frame,
    )
    _print_report(output_dir, summary)


def prepare_probe_data(
    root: Path,
    spec: Mapping[str, Any],
    *,
    target_override: str | None = None,
) -> PreparedProbeData:
    normalized = normalize_probe_spec(spec)
    dataset = TraceDataset.open(root)
    selector = _selector_from_spec(normalized)
    feature_matrix = dataset.select_model_sites(selector).materialize(cache=True)
    X = feature_matrix.X
    rows = _attach_episode_metadata(feature_matrix.rows.copy(), dataset)
    target_spec = _target_spec(normalized, target_override)
    target_name = _target_name(target_spec)
    rows = _resolve_probe_target(dataset, rows, target_spec)
    X, rows, filter_summary = _apply_row_filters(X, rows, normalized.get("row_filter"))
    X, rows, missing_summary = _apply_missing_policy(
        X,
        rows,
        target_name,
        policy=str(target_spec.get("missing_policy") or "error"),
    )
    split = normalized["split"]
    split_column = str(split.get("column", "split"))
    train_value = str(split.get("train_value", "train"))
    test_value = str(split.get("test_value", "test"))
    rows = _ensure_split(
        rows,
        split_column,
        train_value=train_value,
        test_value=test_value,
        split_kind=str(split.get("kind", "random_episode")),
    )
    eval_values = [str(value) for value in split.get("eval_values", [test_value])]
    return PreparedProbeData(
        dataset=dataset,
        X=X,
        rows=rows.reset_index(drop=True),
        target=target_name,
        train_value=train_value,
        selection_value=str(split.get("selection_value", test_value)),
        test_value=test_value,
        eval_values=eval_values,
        split_column=split_column,
        filter_summary=filter_summary,
        missing_summary=missing_summary,
        cache_key=feature_matrix.cache_key,
    )


def _selector_from_spec(spec: Mapping[str, Any]) -> ActivationQuery:
    features = spec["features"]
    return ActivationQuery(
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


def _target_spec(spec: Mapping[str, Any], target_override: str | None) -> dict[str, Any]:
    if target_override is None:
        return _normalize_target_spec(spec["target"])
    return _normalize_target_spec(
        {
            "name": target_override,
            "kind": "classification",
            "source": "row",
            "column": target_override,
            "missing_policy": "drop",
        }
    )


def train_layer_readouts(
    X: np.ndarray,
    rows: pd.DataFrame,
    *,
    target: str,
    split_column: str,
    train_value: str,
    max_iter: int,
    seed: int,
    model_name: str,
) -> dict[str, LayerReadout]:
    readouts: dict[str, LayerReadout] = {}
    for layer, group in rows.groupby("layer", dropna=False, sort=True):
        layer_name = _normalize_layer(layer)
        row_index = group.index.to_numpy()
        layer_rows = group.reset_index(drop=True)
        y = layer_rows[target].astype(str).to_numpy()
        train_mask = layer_rows[split_column].astype(str).to_numpy() == train_value
        if int(train_mask.sum()) == 0 or len(np.unique(y[train_mask])) < 2:
            continue
        model = _classifier(max_iter=max_iter, seed=seed, model_name=model_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X[row_index][train_mask], y[train_mask])
        readouts[layer_name] = LayerReadout(
            layer=layer_name,
            model=model,
            classes=np.asarray(model.classes_, dtype=object),
            row_index=row_index,
            rows=layer_rows,
        )
    return readouts


def _classifier(*, max_iter: int, seed: int, model_name: str) -> Any:
    if model_name == "sgd":
        return make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="log_loss",
                max_iter=max_iter,
                tol=1e-3,
                class_weight="balanced",
                random_state=seed,
            ),
        )
    if model_name == "ridge":
        return make_pipeline(StandardScaler(), RidgeClassifier(class_weight="balanced"))
    if model_name == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=max_iter,
                class_weight="balanced",
                random_state=seed,
                solver="liblinear",
            ),
        )
    if model_name == "logistic-lbfgs":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=max_iter,
                class_weight="balanced",
                random_state=seed,
            ),
        )
    raise ValueError(f"Unknown diagnostics model: {model_name!r}")


def _score_matrix(model: Any, X: np.ndarray) -> np.ndarray:
    estimator = model.steps[-1][1] if getattr(model, "steps", None) else model
    if isinstance(estimator, SGDClassifier) and hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=np.float32)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        return _softmax(scores)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X), dtype=np.float32)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=np.float32)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        return _softmax(scores)
    classes = np.asarray(model.classes_, dtype=object)
    predicted = np.asarray(model.predict(X), dtype=object)
    out = np.zeros((len(predicted), len(classes)), dtype=np.float32)
    class_to_index = {str(label): index for index, label in enumerate(classes.astype(str))}
    for row_index, label in enumerate(predicted.astype(str)):
        out[row_index, class_to_index.get(label, 0)] = 1.0
    return out


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(shifted)
    denom = np.sum(exp, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return (exp / denom).astype(np.float32)


def score_readouts_with_X(
    X: np.ndarray,
    readouts: Mapping[str, LayerReadout],
    *,
    target: str,
    split_column: str,
    top_k: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for layer, readout in readouts.items():
        rows = readout.rows.copy()
        X_layer = X[readout.row_index]
        scores = _score_matrix(readout.model, X_layer)
        predicted = readout.classes[np.argmax(scores, axis=1)].astype(str)
        actual = rows[target].astype(str).to_numpy()
        rows["actual"] = actual
        rows["predicted"] = predicted
        rows["correct"] = rows["actual"].astype(str) == rows["predicted"].astype(str)
        rows["confidence"] = np.max(scores, axis=1)
        rows["layer"] = layer
        rows["policy_call_key"] = _policy_call_key(rows)
        rows["contact_lead_policy_calls"] = [
            _lead_policy_calls(row, "first_contact_time_next_object")
            for row in rows.to_dict("records")
        ]
        rows["motion_lead_policy_calls"] = [
            _lead_policy_calls(row, "first_motion_time_next_object")
            for row in rows.to_dict("records")
        ]
        rows["contact_lead_bucket"] = rows["contact_lead_policy_calls"].map(_lead_bucket)
        rows["motion_lead_bucket"] = rows["motion_lead_policy_calls"].map(_lead_bucket)
        top = _topk_columns(scores, readout.classes, actual, top_k=top_k)
        frames.append(pd.concat([rows.reset_index(drop=True), top], axis=1))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _topk_columns(
    proba: np.ndarray,
    classes: Sequence[Any],
    actual: Sequence[Any],
    *,
    top_k: int,
) -> pd.DataFrame:
    class_array = np.asarray(classes, dtype=object).astype(str)
    order = np.argsort(-proba, axis=1)
    records: list[dict[str, Any]] = []
    for row_index, indices in enumerate(order):
        record: dict[str, Any] = {}
        top_indices = indices[: max(1, min(top_k, len(indices)))]
        labels = [str(class_array[index]) for index in top_indices]
        confidences = [float(proba[row_index, index]) for index in top_indices]
        actual_value = str(actual[row_index])
        for k in range(1, top_k + 1):
            label = labels[k - 1] if k <= len(labels) else ""
            confidence = confidences[k - 1] if k <= len(confidences) else np.nan
            record[f"top{k}_label"] = label
            record[f"top{k}_confidence"] = confidence
            record[f"top{k}_correct"] = actual_value in set(labels[:k])
        record["topk_labels"] = json.dumps(labels)
        record["topk_confidences"] = json.dumps(confidences)
        records.append(record)
    return pd.DataFrame.from_records(records)


def layer_split_metrics(
    predictions: pd.DataFrame,
    *,
    target: str,
    split_column: str,
) -> pd.DataFrame:
    del target
    records: list[dict[str, Any]] = []
    for (layer, split), group in predictions.groupby(["layer", split_column], sort=True):
        records.append(_metric_record(group, layer=str(layer), split=str(split)))
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    train_scores = frame.loc[frame["split"] == "train", ["layer", "balanced_accuracy"]].rename(
        columns={"balanced_accuracy": "train_balanced_accuracy"}
    )
    frame = frame.merge(train_scores, on="layer", how="left")
    frame["train_gap_balanced_accuracy"] = (
        frame["train_balanced_accuracy"] - frame["balanced_accuracy"]
    )
    return frame


def _metric_record(group: pd.DataFrame, *, layer: str, split: str) -> dict[str, Any]:
    actual = group["actual"].astype(str).to_numpy()
    predicted = group["predicted"].astype(str).to_numpy()
    return {
        "layer": layer,
        "split": split,
        "row_count": int(len(group)),
        "policy_call_count": _policy_call_count(group),
        "class_count": int(pd.Series(actual).nunique()),
        "balanced_accuracy": _balanced_accuracy(actual, predicted),
        "accuracy": float(accuracy_score(actual, predicted)) if len(group) else np.nan,
        "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0))
        if len(group)
        else np.nan,
        "top1_accuracy": _topk_accuracy(group, 1),
        "top2_accuracy": _topk_accuracy(group, 2),
        "top3_accuracy": _topk_accuracy(group, 3),
    }


def per_class_metrics(
    predictions: pd.DataFrame,
    *,
    target: str,
    split_column: str,
) -> pd.DataFrame:
    del target
    records: list[dict[str, Any]] = []
    for (layer, split), group in predictions.groupby(["layer", split_column], sort=True):
        actual = group["actual"].astype(str).to_numpy()
        predicted = group["predicted"].astype(str).to_numpy()
        labels = sorted(pd.Series(actual).dropna().astype(str).unique())
        if not labels:
            continue
        precision, recall, f1, support = precision_recall_fscore_support(
            actual,
            predicted,
            labels=labels,
            zero_division=0,
        )
        for label, p, r, f, s in zip(labels, precision, recall, f1, support, strict=False):
            binary_actual = actual == label
            binary_predicted = predicted == label
            records.append(
                {
                    "layer": str(layer),
                    "split": str(split),
                    "class": label,
                    "row_support": int(s),
                    "policy_call_support": _policy_call_count(
                        group.loc[group["actual"].astype(str) == label]
                    ),
                    "precision": float(p),
                    "recall": float(r),
                    "f1": float(f),
                    "one_vs_rest_balanced_accuracy": _balanced_accuracy(
                        binary_actual,
                        binary_predicted,
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def confusion_matrix_records(
    predictions: pd.DataFrame,
    *,
    target: str,
    split_column: str,
) -> pd.DataFrame:
    del target
    if predictions.empty:
        return pd.DataFrame()
    return (
        predictions.groupby(["layer", split_column, "actual", "predicted"], dropna=False)
        .agg(row_count=("actual", "size"), policy_call_count=("policy_call_key", "nunique"))
        .reset_index()
        .rename(columns={split_column: "split"})
    )


def support_tables(
    rows: pd.DataFrame,
    *,
    target: str,
    split_column: str,
) -> dict[str, pd.DataFrame]:
    policy_rows = _dedupe_policy_calls(rows)
    by_class = (
        policy_rows.groupby([split_column, target], dropna=False)
        .size()
        .reset_index(name="policy_call_count")
        .rename(columns={split_column: "split", target: "class"})
    )
    by_task = _group_count(policy_rows, [split_column, "task_id"])
    by_object = _group_count(policy_rows, [split_column, target]).rename(columns={target: "object"})
    by_phase = _group_count(policy_rows, [split_column, "task_phase"])
    by_flow_step = _group_count(policy_rows, [split_column, "next_object_flow_step_index"])
    return {
        "policy_call_support_by_class_split": by_class,
        "policy_call_support_by_task": by_task,
        "policy_call_support_by_object": by_object,
        "policy_call_support_by_phase": by_phase,
        "policy_call_support_by_flow_step": by_flow_step,
    }


def _group_count(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    existing = [column for column in columns if column in frame]
    if not existing:
        return pd.DataFrame()
    return (
        frame.groupby(existing, dropna=False)
        .size()
        .reset_index(name="policy_call_count")
        .rename(columns={existing[0]: "split"} if existing[0] != "split" else {})
    )


def lead_time_metrics(
    predictions: pd.DataFrame,
    *,
    target: str,
    split_column: str,
) -> pd.DataFrame:
    del target
    records: list[dict[str, Any]] = []
    for bucket_column in ["contact_lead_bucket", "motion_lead_bucket"]:
        for (layer, split, bucket), group in predictions.groupby(
            ["layer", split_column, bucket_column],
            dropna=False,
            sort=True,
        ):
            records.append(
                {
                    **_metric_record(group, layer=str(layer), split=str(split)),
                    "lead_kind": bucket_column.removesuffix("_bucket"),
                    "lead_bucket": str(bucket),
                }
            )
    return pd.DataFrame.from_records(records)


def bootstrap_intervals(
    predictions: pd.DataFrame,
    *,
    split_values: Sequence[str],
    group_columns: Sequence[str],
    runs: int,
    seed: int,
) -> pd.DataFrame:
    if runs <= 0 or predictions.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for split in split_values:
        split_rows = predictions.loc[predictions["split"].astype(str) == str(split)]
        if split_rows.empty:
            continue
        for group_column in group_columns:
            if group_column not in split_rows:
                continue
            groups = sorted(str(value) for value in split_rows[group_column].dropna().unique())
            if len(groups) < 2:
                continue
            samples = {"balanced_accuracy": [], "macro_f1": [], "top1_accuracy": []}
            grouped = {str(key): group for key, group in split_rows.groupby(group_column)}
            for _ in range(runs):
                chosen = rng.choice(groups, size=len(groups), replace=True)
                sample = pd.concat([grouped[str(key)] for key in chosen], ignore_index=True)
                samples["balanced_accuracy"].append(
                    _balanced_accuracy(sample["actual"], sample["predicted"])
                )
                samples["macro_f1"].append(
                    float(
                        f1_score(
                            sample["actual"].astype(str),
                            sample["predicted"].astype(str),
                            average="macro",
                            zero_division=0,
                        )
                    )
                )
                samples["top1_accuracy"].append(_topk_accuracy(sample, 1))
            for metric, values in samples.items():
                array = np.asarray(values, dtype=np.float32)
                records.append(
                    {
                        "split": str(split),
                        "group_column": group_column,
                        "group_count": len(groups),
                        "metric": metric,
                        "runs": int(runs),
                        "mean": float(np.mean(array)),
                        "ci_low": float(np.quantile(array, 0.025)),
                        "ci_high": float(np.quantile(array, 0.975)),
                    }
                )
    return pd.DataFrame.from_records(records)


def readout_battery_from_prepared(
    prepared: PreparedProbeData,
    *,
    target_names: Sequence[str],
    max_iter: int,
    seed: int,
    top_k: int,
    model_name: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for target in target_names:
        if target not in prepared.rows:
            frames.append(
                pd.DataFrame.from_records(
                    [
                        {
                            "target": target,
                            "status": "skipped",
                            "reason": f"target column {target!r} is not available",
                        }
                    ]
                )
            )
            continue
        X_target, rows_target = _drop_missing_target(prepared.X, prepared.rows, target)
        if rows_target.empty:
            frames.append(
                pd.DataFrame.from_records(
                    [{"target": target, "status": "skipped", "reason": "no non-missing rows"}]
                )
            )
            continue
        readouts = train_layer_readouts(
            X_target,
            rows_target,
            target=target,
            split_column=prepared.split_column,
            train_value=prepared.train_value,
            max_iter=max_iter,
            seed=seed,
            model_name=model_name,
        )
        if not readouts:
            frames.append(
                pd.DataFrame.from_records(
                    [
                        {
                            "target": target,
                            "status": "skipped",
                            "reason": "training split has fewer than two classes",
                        }
                    ]
                )
            )
            continue
        preds = score_readouts_with_X(
            X_target,
            readouts,
            target=target,
            split_column=prepared.split_column,
            top_k=top_k,
        )
        metrics = layer_split_metrics(
            preds,
            target=target,
            split_column=prepared.split_column,
        )
        metrics.insert(0, "target", target)
        metrics.insert(1, "status", "ok")
        frames.append(metrics)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _drop_missing_target(
    X: np.ndarray,
    rows: pd.DataFrame,
    target: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    missing = rows[target].isna() | (rows[target].astype(str) == "")
    kept = (~missing).to_numpy(dtype=bool)
    return X[kept], rows.loc[~missing].reset_index(drop=True)


def selection_aware_null(
    prepared: PreparedProbeData,
    *,
    shuffles: int,
    max_iter: int,
    seed: int,
    top_k: int,
    model_name: str,
) -> pd.DataFrame:
    del top_k
    rng = np.random.default_rng(seed)
    layer_groups = {
        _normalize_layer(layer): group.index.to_numpy()
        for layer, group in prepared.rows.groupby("layer", dropna=False, sort=True)
    }
    records: list[dict[str, Any]] = []
    start = time.monotonic()
    for run in range(shuffles):
        shuffled = _shuffle_labels_by_policy_call(
            prepared.rows,
            prepared.target,
            prepared.split_column,
            rng,
        )
        layer_records: list[dict[str, Any]] = []
        for layer, indices in layer_groups.items():
            layer_rows = prepared.rows.iloc[indices].reset_index(drop=True)
            y = shuffled[indices].astype(str)
            split_values = layer_rows[prepared.split_column].astype(str).to_numpy()
            train_mask = split_values == prepared.train_value
            if int(train_mask.sum()) == 0 or len(np.unique(y[train_mask])) < 2:
                continue
            model = _classifier(max_iter=max_iter, seed=seed + run, model_name=model_name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(prepared.X[indices][train_mask], y[train_mask])
            for split in [prepared.selection_value, prepared.test_value]:
                eval_mask = split_values == split
                if int(eval_mask.sum()) == 0:
                    continue
                pred = model.predict(prepared.X[indices][eval_mask])
                layer_records.append(
                    {
                        "run": run,
                        "layer": layer,
                        "split": split,
                        "score": _balanced_accuracy(y[eval_mask], pred),
                        "row_count": int(eval_mask.sum()),
                        "policy_call_count": _policy_call_count(layer_rows.loc[eval_mask]),
                    }
                )
        if layer_records:
            frame = pd.DataFrame.from_records(layer_records)
            selection_rows = frame.loc[frame["split"] == prepared.selection_value]
            if not selection_rows.empty:
                best = selection_rows.sort_values(
                    ["score", "layer"],
                    ascending=[False, True],
                ).iloc[0]
                selected_layer = str(best["layer"])
                selected = frame.loc[frame["layer"].astype(str) == selected_layer].copy()
                selected["selected_layer"] = selected_layer
                records.extend(selected.to_dict("records"))
        if (run + 1) % 25 == 0 or run + 1 == shuffles:
            elapsed = time.monotonic() - start
            print(
                f"selection_null_progress={run + 1}/{shuffles} "
                f"elapsed_sec={elapsed:.1f}",
                flush=True,
            )
    return pd.DataFrame.from_records(records)


def _shuffle_labels_by_policy_call(
    rows: pd.DataFrame,
    target: str,
    split_column: str,
    rng: np.random.Generator,
) -> np.ndarray:
    policy_rows = _dedupe_policy_calls(rows)
    shuffled = policy_rows[POLICY_CALL_KEYS + [split_column, target]].copy()
    shuffled[target] = shuffled[target].astype(str)
    for split, group in shuffled.groupby(split_column, dropna=False):
        del split
        values = group[target].to_numpy(copy=True)
        rng.shuffle(values)
        shuffled.loc[group.index, target] = values
    mapping = {
        (str(row["trace_id"]), int(row["policy_call_index"])): str(row[target])
        for row in shuffled.to_dict("records")
    }
    return np.asarray(
        [
            mapping[(str(row["trace_id"]), int(row["policy_call_index"]))]
            for row in rows.to_dict("records")
        ],
        dtype=object,
    )


def attach_error_browser_context(dataset: TraceDataset, predictions: pd.DataFrame) -> pd.DataFrame:
    events = _load_interaction_events(dataset)
    if events.empty:
        predictions["events_before"] = ""
        predictions["events_after"] = ""
        return predictions
    by_trace = {
        str(trace_id): group.sort_values("onset_timestep").reset_index(drop=True)
        for trace_id, group in events.groupby("trace_id", dropna=False)
    }
    records = predictions[["trace_id", "timestep"]].to_dict("records")
    predictions = predictions.copy()
    predictions["events_before"] = [
        _event_summary(by_trace, row["trace_id"], row["timestep"], before=True)
        for row in records
    ]
    predictions["events_after"] = [
        _event_summary(by_trace, row["trace_id"], row["timestep"], before=False)
        for row in records
    ]
    return predictions


def _load_interaction_events(dataset: TraceDataset) -> pd.DataFrame:
    artifact = _latest_artifact(dataset, OBJECT_FLOW_ARTIFACT_TYPE)
    if artifact is None:
        return pd.DataFrame()
    outputs = dict(artifact.method.get("outputs") or {})
    path = outputs.get("interaction_events")
    if not path:
        return pd.DataFrame()
    table_path = _artifact_output_path(dataset, str(path))
    if not table_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(table_path)


def _event_summary(
    by_trace: Mapping[str, pd.DataFrame],
    trace_id: Any,
    timestep: Any,
    *,
    before: bool,
    limit: int = 4,
) -> str:
    trace_events = by_trace.get(str(trace_id))
    if trace_events is None or trace_events.empty or pd.isna(timestep):
        return ""
    t = float(timestep)
    event_times = pd.to_numeric(trace_events["onset_timestep"], errors="coerce")
    if before:
        selected = trace_events.loc[event_times <= t].tail(limit)
    else:
        selected = trace_events.loc[event_times > t].head(limit)
    parts = []
    for row in selected.to_dict("records"):
        parts.append(
            f"{row.get('event_type')}:{row.get('object_name')}@{int(row.get('onset_timestep'))}"
        )
    return "; ".join(parts)


def build_summary(
    prepared: PreparedProbeData,
    layer_metrics: pd.DataFrame,
    supports: Mapping[str, pd.DataFrame],
    *,
    selected_layer: str,
    null_frame: pd.DataFrame,
) -> dict[str, Any]:
    policy_rows = _dedupe_policy_calls(prepared.rows)
    selected_metrics = layer_metrics.loc[layer_metrics["layer"] == selected_layer].copy()
    real_selection = selected_metrics.loc[
        selected_metrics["split"] == prepared.selection_value,
        "balanced_accuracy",
    ]
    real_test = selected_metrics.loc[
        selected_metrics["split"] == prepared.test_value,
        "balanced_accuracy",
    ]
    summary: dict[str, Any] = {
        "target": prepared.target,
        "cache_key": prepared.cache_key,
        "selected_layer": selected_layer,
        "feature_rows": int(len(prepared.rows)),
        "policy_call_count": _policy_call_count(prepared.rows),
        "episode_count": int(prepared.rows["trace_id"].nunique()),
        "class_count": int(policy_rows[prepared.target].astype(str).nunique()),
        "split_policy_call_counts": _split_counts(policy_rows, prepared.split_column),
        "layer_count": int(prepared.rows["layer"].nunique()),
        "filter_summary": prepared.filter_summary,
        "missing_summary": prepared.missing_summary,
        "selection_split": prepared.selection_value,
        "test_split": prepared.test_value,
        "selected_layer_selection_balanced_accuracy": _first_float(real_selection),
        "selected_layer_test_balanced_accuracy": _first_float(real_test),
    }
    support = supports.get("policy_call_support_by_class_split", pd.DataFrame())
    if not support.empty:
        summary["classes_by_split"] = {
            str(split): int(group["class"].astype(str).nunique())
            for split, group in support.groupby("split")
        }
    if not null_frame.empty and real_selection.size:
        selection_null = null_frame.loc[null_frame["split"] == prepared.selection_value]
        test_null = null_frame.loc[null_frame["split"] == prepared.test_value]
        summary["selection_aware_null"] = {
            "runs": int(selection_null["run"].nunique()),
            "selection_p_value": _p_value(
                selection_null["score"],
                float(real_selection.iloc[0]),
            ),
            "test_p_value": _p_value(test_null["score"], _first_float(real_test)),
            "selection_score_mean": _mean(selection_null["score"]),
            "selection_score_std": _std(selection_null["score"]),
            "test_score_mean": _mean(test_null["score"]),
            "test_score_std": _std(test_null["score"]),
        }
    return summary


def write_reports(
    output_dir: Path,
    *,
    summary: Mapping[str, Any],
    predictions: pd.DataFrame,
    layer_metrics: pd.DataFrame,
    per_class: pd.DataFrame,
    confusion: pd.DataFrame,
    supports: Mapping[str, pd.DataFrame],
    lead_time: pd.DataFrame,
    bootstrap: pd.DataFrame,
    battery: pd.DataFrame,
    null_frame: pd.DataFrame,
) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_table(_error_browser_frame(predictions), output_dir / "probe_error_browser.parquet")
    _write_table(layer_metrics, output_dir / "layer_split_metrics.parquet")
    _write_table(per_class, output_dir / "per_class_metrics.parquet")
    _write_table(confusion, output_dir / "confusion_matrix.parquet")
    _write_table(lead_time, output_dir / "lead_time_metrics.parquet")
    _write_table(bootstrap, output_dir / "bootstrap_ci.parquet", skip_empty=True)
    _write_table(battery, output_dir / "readout_battery_metrics.parquet", skip_empty=True)
    _write_table(null_frame, output_dir / "selection_aware_null.parquet", skip_empty=True)
    for name, table in supports.items():
        _write_table(table, output_dir / f"{name}.parquet")


def _write_table(frame: pd.DataFrame, path: Path, *, skip_empty: bool = False) -> None:
    if skip_empty and frame.empty and path.exists():
        return
    if frame.empty:
        pd.DataFrame().to_parquet(path, index=False)
    else:
        frame.to_parquet(path, index=False)


def _error_browser_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        "split",
        "trace_id",
        "episode_id",
        "task_id",
        "prompt",
        "timestep",
        "observation_timestep",
        "policy_call_index",
        "layer",
        "model_site_id",
        "token_space_id",
        "actual",
        "predicted",
        "correct",
        "confidence",
        "top1_label",
        "top1_confidence",
        "top1_correct",
        "top2_label",
        "top2_confidence",
        "top2_correct",
        "top3_label",
        "top3_confidence",
        "top3_correct",
        "topk_labels",
        "topk_confidences",
        "is_pre_contact",
        "is_pre_motion",
        "is_pre_lift",
        "task_phase",
        "next_manipulated_object",
        "active_manipulated_object",
        "active_receptacle_object",
        "current_contact_object",
        "current_moved_object",
        "current_lifted_object",
        "next_flow_step_index",
        "active_flow_step_index",
        "next_object_flow_step_index",
        "first_contact_time_next_object",
        "first_motion_time_next_object",
        "first_lift_time_next_object",
        "contact_lead_policy_calls",
        "motion_lead_policy_calls",
        "contact_lead_bucket",
        "motion_lead_bucket",
        "candidate_objects",
        "visible_candidate_objects",
        "visible_candidate_count",
        "events_before",
        "events_after",
    ]
    columns = [column for column in wanted if column in predictions]
    return predictions[columns].copy()


def _default_output_dir(dataset: TraceDataset, artifact_id: str | None) -> Path:
    selected = artifact_id or _latest_probe_artifact_id(dataset)
    if selected:
        return dataset._dataset_artifact_root() / "artifacts" / selected / "diagnostics"
    return dataset._dataset_artifact_root() / "reports" / "pi05_probe_diagnostics"


def _latest_probe_artifact_id(dataset: TraceDataset) -> str | None:
    table = dataset.artifact_index
    if table.empty or "artifact_type" not in table:
        return None
    probes = table.loc[table["artifact_type"].astype(str) == "probe_suite"].copy()
    if probes.empty:
        return None
    probes = probes.sort_values("created_utc", ascending=False, na_position="last")
    return str(probes.iloc[0]["artifact_id"])


def _dedupe_policy_calls(rows: pd.DataFrame) -> pd.DataFrame:
    keys = [key for key in POLICY_CALL_KEYS if key in rows]
    if len(keys) != len(POLICY_CALL_KEYS):
        return rows.drop_duplicates().copy()
    return rows.drop_duplicates(subset=keys, keep="first").copy()


def _policy_call_count(rows: pd.DataFrame) -> int:
    if not set(POLICY_CALL_KEYS).issubset(rows.columns):
        return int(len(rows))
    return int(len(rows[POLICY_CALL_KEYS].drop_duplicates()))


def _policy_call_key(rows: pd.DataFrame) -> pd.Series:
    return rows["trace_id"].astype(str) + "#" + rows["policy_call_index"].astype(str)


def _split_counts(rows: pd.DataFrame, split_column: str) -> dict[str, int]:
    return {
        str(split): int(len(group))
        for split, group in rows.groupby(split_column, dropna=False, sort=True)
    }


def _lead_policy_calls(row: Mapping[str, Any], event_column: str) -> float:
    event_time = row.get(event_column)
    observation = row.get("observation_timestep", row.get("timestep"))
    if event_time is None or observation is None or pd.isna(event_time) or pd.isna(observation):
        return np.nan
    start = row.get("env_timestep_start")
    end = row.get("env_timestep_end")
    stride = 50.0
    if start is not None and end is not None and not pd.isna(start) and not pd.isna(end):
        stride = max(1.0, float(end) - float(start) + 1.0)
    delta = float(event_time) - float(observation)
    if delta <= 0:
        return 0.0
    return float(math.ceil(delta / stride))


def _lead_bucket(value: Any) -> str:
    if value is None or pd.isna(value):
        return "missing"
    calls = int(value)
    if calls <= 0:
        return "0_same_call_or_past"
    if calls == 1:
        return "1_policy_call"
    if calls == 2:
        return "2_policy_calls"
    if 3 <= calls <= 5:
        return "3_5_policy_calls"
    return "gt_5_policy_calls"


def _topk_accuracy(group: pd.DataFrame, k: int) -> float:
    column = f"top{k}_correct"
    if column not in group or group.empty:
        return np.nan
    return float(group[column].fillna(False).astype(bool).mean())


def _balanced_accuracy(actual: Sequence[Any], predicted: Sequence[Any]) -> float:
    actual_series = pd.Series(actual).astype(str)
    predicted_series = pd.Series(predicted).astype(str)
    if actual_series.empty:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return float(balanced_accuracy_score(actual_series, predicted_series))


def _select_best_layer(
    readouts: Mapping[str, LayerReadout],
    target: str,
    selection_value: str,
    split_column: str,
    top_k: int,
) -> str:
    del readouts, target, selection_value, split_column, top_k
    raise ValueError("Requested selected layer is absent from trained readouts.")


def _normalize_layer(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return str(int(number))
    return str(number)


def _p_value(null_scores: pd.Series, real_score: float | None) -> float | None:
    if real_score is None or pd.isna(real_score) or null_scores.empty:
        return None
    scores = pd.to_numeric(null_scores, errors="coerce").dropna()
    if scores.empty:
        return None
    return float((1 + (scores >= float(real_score)).sum()) / (len(scores) + 1))


def _mean(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else None


def _std(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.std(ddof=0)) if not numeric.empty else None


def _first_float(values: pd.Series) -> float | None:
    if values.empty:
        return None
    value = values.iloc[0]
    if value is None or pd.isna(value):
        return None
    return float(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _print_report(output_dir: Path, summary: Mapping[str, Any]) -> None:
    print(f"diagnostics_dir={output_dir}")
    for key in [
        "target",
        "selected_layer",
        "feature_rows",
        "policy_call_count",
        "episode_count",
        "class_count",
        "selected_layer_selection_balanced_accuracy",
        "selected_layer_test_balanced_accuracy",
    ]:
        print(f"{key}={summary.get(key)}")
    null = summary.get("selection_aware_null")
    if isinstance(null, Mapping):
        print(f"selection_null_runs={null.get('runs')}")
        print(f"selection_null_p_value={null.get('selection_p_value')}")
        print(f"test_null_p_value={null.get('test_p_value')}")


if __name__ == "__main__":
    main()
