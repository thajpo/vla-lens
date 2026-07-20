"""Small probe-suite wrapper with metadata baselines."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    r2_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True, slots=True)
class ProbeResult:
    feature: str
    target: str
    probe_type: str
    score: float
    baseline_score: float
    n_train: int
    n_test: int
    metadata_baseline: str
    model: str = "linear"
    split_value: str = "test"
    primary_metric: str = "score"
    details: dict[str, Any] = field(default_factory=dict)
    prediction_records: list[dict[str, Any]] = field(default_factory=list)
    all_prediction_records: list[dict[str, Any]] = field(default_factory=list)
    model_state: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "target": self.target,
            "probe_type": self.probe_type,
            "score": self.score,
            "baseline_score": self.baseline_score,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "metadata_baseline": self.metadata_baseline,
            "model": self.model,
            "split_value": self.split_value,
            "primary_metric": self.primary_metric,
            "details": self.details,
            "prediction_records": self.prediction_records,
            "all_prediction_records": self.all_prediction_records,
            "model_state": self.model_state,
        }


def run_probe_suite(
    rows: pd.DataFrame,
    features: dict[str, np.ndarray],
    targets: Iterable[str],
    split_column: str = "split",
    train_value: str = "train",
    test_value: str = "test",
    eval_values: list[str] | None = None,
    metadata_baseline_columns: list[str] | None = None,
    target_kinds: dict[str, str] | None = None,
    probe_models: list[str] | None = None,
) -> pd.DataFrame:
    """Fit simple probes and compare them with categorical metadata baselines."""
    metadata_baseline_columns = metadata_baseline_columns or []
    eval_values = eval_values or [test_value]
    probe_models = probe_models or ["linear"]
    target_kinds = target_kinds or {}
    split = rows[split_column].astype(str).to_numpy()
    train_mask = split == train_value
    results: list[ProbeResult] = []

    for target in targets:
        y = rows[target].to_numpy()
        target_kind = target_kinds.get(target)
        for feature_name, X in features.items():
            for eval_value in eval_values:
                eval_mask = split == eval_value
                for model_name in probe_models:
                    if target_kind == "classification" or (
                        target_kind is None and _is_classification(y)
                    ):
                        result = _classification_result(
                            rows,
                            X,
                            y,
                            train_mask,
                            eval_mask,
                            feature_name,
                            target,
                            metadata_baseline_columns,
                            split_column=split_column,
                            eval_value=eval_value,
                            model_name=model_name,
                        )
                    elif target_kind == "regression" or target_kind is None:
                        result = _regression_result(
                            rows,
                            X,
                            y.astype(np.float32),
                            train_mask,
                            eval_mask,
                            feature_name,
                            target,
                            metadata_baseline_columns,
                            split_column=split_column,
                            eval_value=eval_value,
                            model_name=model_name,
                        )
                    else:
                        raise ValueError(f"Unknown target kind for {target!r}: {target_kind!r}")
                    if result is not None:
                        results.append(result)
    return pd.DataFrame.from_records([result.to_record() for result in results])


def _classification_result(
    rows: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
    feature_name: str,
    target: str,
    metadata_columns: list[str],
    split_column: str,
    eval_value: str,
    model_name: str,
) -> ProbeResult | None:
    y_train = y[train_mask]
    y_eval = y[eval_mask]
    if len(y_eval) == 0 or len(np.unique(y_train)) < 2:
        return None
    probe = _classification_probe(model_name)
    probe.fit(X[train_mask], y_train)
    y_pred = probe.predict(X[eval_mask])
    y_all_pred = probe.predict(X)
    score = float(balanced_accuracy_score(y_eval, y_pred))
    accuracy = float(accuracy_score(y_eval, y_pred))
    macro_f1 = float(f1_score(y_eval, y_pred, average="macro", zero_division=0))
    per_class_recall = _per_class_recall(y_eval, y_pred)
    confidence = _prediction_confidence(probe, X[eval_mask])
    all_confidence = _prediction_confidence(probe, X)
    proba = _prediction_proba(probe, X[eval_mask])
    label_log_loss = _safe_log_loss(y_eval, proba, _probe_classes(probe))

    baseline_score = _dummy_classifier_score(y_train, y_eval)
    baseline_name = "majority"
    baseline_details = [{"baseline": baseline_name, "score": baseline_score}]
    if metadata_columns:
        for columns in [[column] for column in metadata_columns] + [metadata_columns]:
            metadata_score = _metadata_classifier_score(rows, y, train_mask, eval_mask, columns)
            if metadata_score is None:
                continue
            name = "+".join(columns)
            baseline_details.append({"baseline": name, "score": metadata_score})
            if metadata_score > baseline_score:
                baseline_score = metadata_score
                baseline_name = name

    eval_rows = rows.loc[eval_mask].reset_index(drop=True)
    prediction_records = _prediction_records(
        eval_rows,
        y_eval,
        y_pred,
        confidence,
        target=target,
        split_value=eval_value,
        split_column=split_column,
    )
    all_prediction_records = _prediction_records(
        rows.reset_index(drop=True),
        y,
        y_all_pred,
        all_confidence,
        target=target,
        split_value=None,
        split_column=split_column,
    )
    return ProbeResult(
        feature=feature_name,
        target=target,
        probe_type="classification",
        score=score,
        baseline_score=baseline_score,
        n_train=int(train_mask.sum()),
        n_test=int(eval_mask.sum()),
        metadata_baseline=baseline_name,
        model=model_name,
        split_value=eval_value,
        primary_metric="balanced_accuracy",
        details={
            "model": model_name,
            "split": eval_value,
            "primary_metric": "balanced_accuracy",
            "balanced_accuracy": score,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "log_loss": label_log_loss,
            "per_class_recall": per_class_recall,
            "metadata_baselines": baseline_details,
            "confusion_matrix": _confusion_records(y_eval, y_pred),
            "test_predictions": prediction_records[:50],
            "test_episode_summary": _episode_prediction_summary(
                eval_rows,
                y_eval,
                y_pred,
                confidence,
            ),
        },
        prediction_records=prediction_records,
        all_prediction_records=all_prediction_records,
        model_state=_linear_model_state(
            probe,
            probe_type="classification",
            model_name=model_name,
        ),
    )


def _regression_result(
    rows: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
    feature_name: str,
    target: str,
    metadata_columns: list[str],
    split_column: str,
    eval_value: str,
    model_name: str,
) -> ProbeResult | None:
    y_train = y[train_mask]
    y_eval = y[eval_mask]
    if len(y_eval) == 0:
        return None
    probe = _regression_probe(model_name)
    probe.fit(X[train_mask], y_train)
    pred = np.asarray(probe.predict(X[eval_mask]))
    all_pred = np.asarray(probe.predict(X))
    r2 = float(r2_score(y_eval, pred))
    mae = float(mean_absolute_error(y_eval, pred))
    baseline = np.full_like(y_eval, float(np.mean(y_train)), dtype=np.float32)
    baseline_r2 = float(r2_score(y_eval, baseline))
    baseline_mae = float(mean_absolute_error(y_eval, baseline))
    score = -mae
    baseline_score = -baseline_mae
    baseline_name = "train_mean"
    baseline_details = [
        {
            "baseline": baseline_name,
            "score": baseline_score,
            "mae": baseline_mae,
            "r2": baseline_r2 if np.isfinite(baseline_r2) else None,
        }
    ]
    if metadata_columns:
        for columns in [[column] for column in metadata_columns] + [metadata_columns]:
            metadata_score = _metadata_regressor_score(rows, y, train_mask, eval_mask, columns)
            if metadata_score is None:
                continue
            baseline_details.append(metadata_score)
            if float(metadata_score["score"]) > baseline_score:
                baseline_score = float(metadata_score["score"])
                baseline_name = str(metadata_score["baseline"])
    prediction_records = _regression_prediction_records(
        rows.loc[eval_mask].reset_index(drop=True),
        y_eval,
        pred,
        target=target,
        split_value=eval_value,
        split_column=split_column,
    )
    all_prediction_records = _regression_prediction_records(
        rows.reset_index(drop=True),
        y,
        all_pred,
        target=target,
        split_value=None,
        split_column=split_column,
    )
    return ProbeResult(
        feature=feature_name,
        target=target,
        probe_type="regression",
        score=score,
        baseline_score=baseline_score,
        n_train=int(train_mask.sum()),
        n_test=int(eval_mask.sum()),
        metadata_baseline=baseline_name,
        model=model_name,
        split_value=eval_value,
        primary_metric="negative_mae",
        details={
            "model": model_name,
            "split": eval_value,
            "primary_metric": "negative_mae",
            "r2": r2 if np.isfinite(r2) else None,
            "mae": mae,
            "baseline_r2": baseline_r2 if np.isfinite(baseline_r2) else None,
            "baseline_mae": baseline_mae,
            "metadata_baselines": baseline_details,
            "test_predictions": prediction_records[:50],
        },
        prediction_records=prediction_records,
        all_prediction_records=all_prediction_records,
        model_state=_linear_model_state(probe, probe_type="regression", model_name=model_name),
    )


def _classification_probe(model_name: str) -> Any:
    if model_name == "linear":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
    if model_name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64,),
                alpha=1e-4,
                max_iter=300,
                random_state=0,
            ),
        )
    raise ValueError(f"Unknown classification probe model: {model_name!r}")


def _regression_probe(model_name: str) -> Any:
    if model_name == "linear":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if model_name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64,),
                alpha=1e-4,
                max_iter=300,
                random_state=0,
            ),
        )
    raise ValueError(f"Unknown regression probe model: {model_name!r}")


def _is_classification(y: np.ndarray) -> bool:
    if y.dtype == bool or y.dtype.kind in {"O", "U", "S", "b"}:
        return True
    return len(np.unique(y[~pd.isna(y)])) <= 20


def _dummy_classifier_score(y_train: np.ndarray, y_test: np.ndarray) -> float:
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    return float(balanced_accuracy_score(y_test, dummy.predict(np.zeros((len(y_test), 1)))))


def _metadata_classifier_score(
    rows: pd.DataFrame,
    y: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    columns: list[str],
) -> float | None:
    train_meta = rows.loc[train_mask, columns].astype(str)
    eval_meta = rows.loc[test_mask, columns].astype(str)
    y_train = y[train_mask]
    y_test = y[test_mask]
    if len(y_test) == 0 or len(np.unique(y_train)) < 2:
        return None
    model = make_pipeline(
        ColumnTransformer(
            [("categorical", OneHotEncoder(handle_unknown="ignore"), columns)],
            remainder="drop",
        ),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit(train_meta, y_train)
    return float(balanced_accuracy_score(y_test, model.predict(eval_meta)))


def _metadata_regressor_score(
    rows: pd.DataFrame,
    y: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    columns: list[str],
) -> dict[str, Any] | None:
    train_meta = rows.loc[train_mask, columns].astype(str)
    eval_meta = rows.loc[test_mask, columns].astype(str)
    y_train = y[train_mask]
    y_test = y[test_mask]
    if len(y_test) == 0 or len(y_train) == 0:
        return None
    model = make_pipeline(
        ColumnTransformer(
            [("categorical", OneHotEncoder(handle_unknown="ignore"), columns)],
            remainder="drop",
        ),
        Ridge(alpha=1.0),
    )
    model.fit(train_meta, y_train)
    pred = np.asarray(model.predict(eval_meta), dtype=np.float32)
    mae = float(mean_absolute_error(y_test, pred))
    r2 = float(r2_score(y_test, pred))
    return {
        "baseline": "+".join(columns),
        "score": -mae,
        "mae": mae,
        "r2": r2 if np.isfinite(r2) else None,
    }


def _prediction_confidence(probe: Any, X: np.ndarray) -> np.ndarray:
    if X.shape[0] == 0:
        return np.empty(0, dtype=np.float32)
    if hasattr(probe, "predict_proba"):
        proba = probe.predict_proba(X)
        return np.max(proba, axis=1).astype(np.float32)
    if hasattr(probe, "decision_function"):
        margin = np.asarray(probe.decision_function(X), dtype=np.float32)
        if margin.ndim == 1:
            return np.abs(margin)
        return np.max(margin, axis=1) - np.partition(margin, -2, axis=1)[:, -2]
    return np.full(X.shape[0], np.nan, dtype=np.float32)


def _prediction_proba(probe: Any, X: np.ndarray) -> np.ndarray | None:
    if X.shape[0] == 0 or not hasattr(probe, "predict_proba"):
        return None
    return np.asarray(probe.predict_proba(X), dtype=np.float32)


def _probe_classes(probe: Any) -> list[Any]:
    estimator = probe.steps[-1][1] if getattr(probe, "steps", None) else probe
    return list(getattr(estimator, "classes_", []))


def _safe_log_loss(
    y_true: np.ndarray,
    proba: np.ndarray | None,
    labels: list[Any],
) -> float | None:
    if proba is None or proba.size == 0 or not labels:
        return None
    try:
        return float(log_loss(y_true, proba, labels=labels))
    except ValueError:
        return None


def _per_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    labels = sorted(str(value) for value in pd.Series(y_true).dropna().unique())
    recalls = recall_score(
        pd.Series(y_true).astype(str),
        pd.Series(y_pred).astype(str),
        labels=labels,
        average=None,
        zero_division=0,
    )
    return {label: float(value) for label, value in zip(labels, recalls, strict=False)}


def _confusion_records(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, Any]]:
    frame = pd.DataFrame({"actual": y_true.astype(str), "predicted": y_pred.astype(str)})
    counts = frame.groupby(["actual", "predicted"], dropna=False).size().reset_index(name="count")
    return [
        {
            "actual": str(record["actual"]),
            "predicted": str(record["predicted"]),
            "count": int(record["count"]),
        }
        for record in counts.to_dict("records")
    ]


def _prediction_records(
    rows: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    *,
    target: str,
    split_value: str | None,
    split_column: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in rows.iterrows():
        records.append(
            {
                **_prediction_join_keys(
                    row,
                    target=target,
                    split_value=split_value,
                    split_column=split_column,
                ),
                "target_kind": "classification",
                "target_dim": 0,
                "target_value": str(y_true[index]),
                "prediction_value": str(y_pred[index]),
                "trace_id": str(row.get("trace_id", "")),
                "episode_id": str(row.get("episode_id", "")),
                "timestep": _optional_int(row.get("timestep")),
                "actual": str(y_true[index]),
                "predicted": str(y_pred[index]),
                "correct": bool(str(y_true[index]) == str(y_pred[index])),
                "confidence": _optional_float(confidence[index])
                if index < len(confidence)
                else None,
                "prediction_kind": "class_label",
            }
        )
    return records


def _episode_prediction_summary(
    rows: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
) -> list[dict[str, Any]]:
    if rows.empty or "trace_id" not in rows:
        return []
    frame = rows[["trace_id", "episode_id", "timestep"]].copy()
    frame["actual"] = y_true.astype(str)
    frame["predicted"] = y_pred.astype(str)
    frame["correct"] = frame["actual"] == frame["predicted"]
    frame["confidence"] = confidence[: len(frame)] if len(confidence) >= len(frame) else np.nan
    summaries: list[dict[str, Any]] = []
    for trace_id, group in frame.groupby("trace_id", dropna=False, sort=True):
        summaries.append(
            {
                "trace_id": str(trace_id),
                "episode_id": str(group["episode_id"].iloc[0]),
                "samples": int(len(group)),
                "accuracy": float(group["correct"].mean()),
                "mean_confidence": _finite_mean(group["confidence"].to_numpy()),
                "actual": str(group["actual"].mode().iloc[0]) if not group.empty else "",
                "predicted": str(group["predicted"].mode().iloc[0]) if not group.empty else "",
            }
        )
    return summaries


def _regression_prediction_records(
    rows: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    target: str,
    split_value: str | None,
    split_column: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in rows.iterrows():
        records.append(
            {
                **_prediction_join_keys(
                    row,
                    target=target,
                    split_value=split_value,
                    split_column=split_column,
                ),
                "target_kind": "regression",
                "target_dim": 0,
                "target_value": _optional_float(y_true[index]),
                "prediction_value": _optional_float(y_pred[index]),
                "trace_id": str(row.get("trace_id", "")),
                "episode_id": str(row.get("episode_id", "")),
                "timestep": _optional_int(row.get("timestep")),
                "actual": _optional_float(y_true[index]),
                "predicted": _optional_float(y_pred[index]),
                "error": _optional_float(float(y_pred[index]) - float(y_true[index])),
                "prediction_kind": "continuous_value",
            }
        )
    return records


def _prediction_join_keys(
    row: pd.Series,
    *,
    target: str,
    split_value: str | None,
    split_column: str,
) -> dict[str, Any]:
    trace_id = str(row.get("trace_id", ""))
    policy_call = _optional_int(row.get("policy_call_index", row.get("policy_call")))
    generation_step = _optional_axis_value(row.get("generation_step"))
    token_index = _optional_int(row.get("token_index"))
    token_space_id = _optional_str(row.get("token_space_id"))
    model_site_id = _optional_str(row.get("model_site_id", row.get("activation")))
    timestep = _optional_int(row.get("timestep"))
    parts = [
        trace_id,
        str(policy_call),
        str(generation_step),
        str(token_space_id),
        str(token_index),
        str(model_site_id),
        target,
    ]
    example_id = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]
    record_split = (
        split_value
        if split_value is not None
        else _optional_str(row.get(split_column))
        or _optional_str(row.get("split"))
        or _optional_str(row.get("eval_split"))
        or ""
    )
    return {
        "example_id": example_id,
        "split": record_split,
        "task_id": _optional_str(row.get("task_id")),
        "task_phase": _optional_str(row.get("task_phase")),
        "target_name": target,
        "target_timestep": timestep,
        "policy_call_index": policy_call,
        "generation_step": generation_step,
        "model_site_id": model_site_id,
        "token_space_id": token_space_id,
        "token_index": token_index,
        "input_row_index": _optional_int(row.get("input_row_index")),
        "prepared_row_index": _optional_int(row.get("prepared_row_index")),
        "source_feature_row_index": _optional_int(row.get("source_feature_row_index")),
        "active_manipulated_object": _optional_str(row.get("active_manipulated_object")),
        "probe_object_name": _optional_str(row.get("probe_object_name")),
        "probe_object_base_name": _optional_str(row.get("probe_object_base_name")),
        "probe_object_role_manipulated": _optional_bool(row.get("probe_object_role_manipulated")),
        "probe_object_role_receptacle": _optional_bool(row.get("probe_object_role_receptacle")),
        "probe_object_role_distractor": _optional_bool(row.get("probe_object_role_distractor")),
        "probe_object_prompt_mentioned": _optional_bool(row.get("probe_object_prompt_mentioned")),
    }


def _linear_model_state(probe: Any, *, probe_type: str, model_name: str) -> dict[str, Any]:
    scaler = probe.steps[0][1] if getattr(probe, "steps", None) else None
    estimator = probe.steps[-1][1] if getattr(probe, "steps", None) else probe
    state: dict[str, Any] = {
        "probe_type": probe_type,
        "model": model_name,
        "weights_space": "normalized_feature_space",
    }
    if scaler is not None:
        state["feature_mean"] = np.asarray(getattr(scaler, "mean_", [])).copy()
        state["feature_scale"] = np.asarray(getattr(scaler, "scale_", [])).copy()
    if hasattr(estimator, "coef_"):
        state["weights"] = np.asarray(estimator.coef_).copy()
    if hasattr(estimator, "intercept_"):
        state["bias"] = np.asarray(estimator.intercept_).reshape(-1).copy()
    if hasattr(estimator, "coefs_"):
        state["layer_weights"] = [np.asarray(value).copy() for value in estimator.coefs_]
    if hasattr(estimator, "intercepts_"):
        state["layer_biases"] = [np.asarray(value).copy() for value in estimator.intercepts_]
    if hasattr(estimator, "activation"):
        state["activation"] = str(estimator.activation)
    if hasattr(estimator, "out_activation_"):
        state["out_activation"] = str(estimator.out_activation_)
    if hasattr(estimator, "classes_"):
        state["classes"] = [_class_value(item) for item in estimator.classes_]
    return state


def _class_value(value: Any) -> str | int | float | bool | None:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _optional_str(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    return bool(value)


def _optional_axis_value(value: Any) -> int | str | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _finite_mean(values: np.ndarray) -> float | None:
    array = np.asarray(values, dtype=np.float32)
    if not np.isfinite(array).any():
        return None
    return float(np.nanmean(array))
