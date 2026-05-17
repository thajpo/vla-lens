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
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
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
    details: dict[str, Any] = field(default_factory=dict)
    prediction_records: list[dict[str, Any]] = field(default_factory=list)
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
            "details": self.details,
            "prediction_records": self.prediction_records,
            "model_state": self.model_state,
        }


def run_probe_suite(
    rows: pd.DataFrame,
    features: dict[str, np.ndarray],
    targets: Iterable[str],
    split_column: str = "split",
    train_value: str = "train",
    test_value: str = "test",
    metadata_baseline_columns: list[str] | None = None,
    target_kinds: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Fit simple probes and compare them with categorical metadata baselines."""
    metadata_baseline_columns = metadata_baseline_columns or []
    target_kinds = target_kinds or {}
    split = rows[split_column].astype(str).to_numpy()
    train_mask = split == train_value
    test_mask = split == test_value
    results: list[ProbeResult] = []

    for target in targets:
        y = rows[target].to_numpy()
        target_kind = target_kinds.get(target)
        for feature_name, X in features.items():
            if target_kind == "classification" or (target_kind is None and _is_classification(y)):
                result = _classification_result(
                    rows,
                    X,
                    y,
                    train_mask,
                    test_mask,
                    feature_name,
                    target,
                    metadata_baseline_columns,
                )
            elif target_kind == "regression" or target_kind is None:
                result = _regression_result(
                    rows, X, y.astype(np.float32), train_mask, test_mask, feature_name, target
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
    test_mask: np.ndarray,
    feature_name: str,
    target: str,
    metadata_columns: list[str],
) -> ProbeResult | None:
    y_train = y[train_mask]
    y_test = y[test_mask]
    if len(y_test) == 0 or len(np.unique(y_train)) < 2:
        return None
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    probe.fit(X[train_mask], y_train)
    y_pred = probe.predict(X[test_mask])
    score = float(accuracy_score(y_test, y_pred))
    confidence = _prediction_confidence(probe, X[test_mask])

    baseline_score = _dummy_classifier_score(y_train, y_test)
    baseline_name = "majority"
    if metadata_columns:
        metadata_score = _metadata_classifier_score(
            rows,
            y,
            train_mask,
            test_mask,
            metadata_columns,
        )
        if metadata_score > baseline_score:
            baseline_score = metadata_score
            baseline_name = "+".join(metadata_columns)

    test_rows = rows.loc[test_mask].reset_index(drop=True)
    prediction_records = _prediction_records(
        test_rows,
        y_test,
        y_pred,
        confidence,
        target=target,
        split_value="test",
    )
    return ProbeResult(
        feature=feature_name,
        target=target,
        probe_type="classification",
        score=score,
        baseline_score=baseline_score,
        n_train=int(train_mask.sum()),
        n_test=int(test_mask.sum()),
        metadata_baseline=baseline_name,
        details={
            "confusion_matrix": _confusion_records(y_test, y_pred),
            "test_predictions": prediction_records[:50],
            "test_episode_summary": _episode_prediction_summary(
                test_rows,
                y_test,
                y_pred,
                confidence,
            ),
        },
        prediction_records=prediction_records,
        model_state=_linear_model_state(probe, probe_type="classification"),
    )


def _regression_result(
    rows: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    feature_name: str,
    target: str,
) -> ProbeResult | None:
    y_train = y[train_mask]
    y_test = y[test_mask]
    if len(y_test) == 0:
        return None
    probe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    probe.fit(X[train_mask], y_train)
    pred = probe.predict(X[test_mask])
    score = float(r2_score(y_test, pred))
    baseline = np.full_like(y_test, float(np.mean(y_train)), dtype=np.float32)
    baseline_score = float(r2_score(y_test, baseline))
    if not np.isfinite(score):
        score = -float(mean_absolute_error(y_test, pred))
        baseline_score = -float(mean_absolute_error(y_test, baseline))
    prediction_records = _regression_prediction_records(
        rows.loc[test_mask].reset_index(drop=True),
        y_test,
        pred,
        target=target,
        split_value="test",
    )
    return ProbeResult(
        feature=feature_name,
        target=target,
        probe_type="regression",
        score=score,
        baseline_score=baseline_score,
        n_train=int(train_mask.sum()),
        n_test=int(test_mask.sum()),
        metadata_baseline="train_mean",
        details={
            "test_predictions": prediction_records[:50],
        },
        prediction_records=prediction_records,
        model_state=_linear_model_state(probe, probe_type="regression"),
    )


def _is_classification(y: np.ndarray) -> bool:
    if y.dtype == bool or y.dtype.kind in {"O", "U", "S", "b"}:
        return True
    return len(np.unique(y[~pd.isna(y)])) <= 20


def _dummy_classifier_score(y_train: np.ndarray, y_test: np.ndarray) -> float:
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    return float(accuracy_score(y_test, dummy.predict(np.zeros((len(y_test), 1)))))


def _metadata_classifier_score(
    rows: pd.DataFrame,
    y: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    columns: list[str],
) -> float:
    train_meta = rows.loc[train_mask, columns].astype(str)
    test_meta = rows.loc[test_mask, columns].astype(str)
    y_train = y[train_mask]
    y_test = y[test_mask]
    model = make_pipeline(
        ColumnTransformer(
            [("categorical", OneHotEncoder(handle_unknown="ignore"), columns)],
            remainder="drop",
        ),
        LogisticRegression(max_iter=1000),
    )
    model.fit(train_meta, y_train)
    return float(accuracy_score(y_test, model.predict(test_meta)))


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
    split_value: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in rows.iterrows():
        records.append(
            {
                **_prediction_join_keys(row, target=target, split_value=split_value),
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
    split_value: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in rows.iterrows():
        records.append(
            {
                **_prediction_join_keys(row, target=target, split_value=split_value),
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
    split_value: str,
) -> dict[str, Any]:
    trace_id = str(row.get("trace_id", ""))
    policy_call = _optional_int(row.get("policy_call_index", row.get("policy_call")))
    generation_step = _optional_int(row.get("generation_step"))
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
    return {
        "example_id": example_id,
        "split": split_value,
        "task_id": _optional_str(row.get("task_id")),
        "target_name": target,
        "target_timestep": timestep,
        "policy_call_index": policy_call,
        "generation_step": generation_step,
        "model_site_id": model_site_id,
        "token_space_id": token_space_id,
        "token_index": token_index,
        "input_row_index": _optional_int(row.get("input_row_index")),
    }


def _linear_model_state(probe: Any, *, probe_type: str) -> dict[str, Any]:
    scaler = probe.steps[0][1] if getattr(probe, "steps", None) else None
    estimator = probe.steps[-1][1] if getattr(probe, "steps", None) else probe
    state: dict[str, Any] = {
        "probe_type": probe_type,
        "weights_space": "normalized_feature_space",
    }
    if scaler is not None:
        state["feature_mean"] = np.asarray(getattr(scaler, "mean_", []), dtype=np.float32)
        state["feature_scale"] = np.asarray(getattr(scaler, "scale_", []), dtype=np.float32)
    if hasattr(estimator, "coef_"):
        state["weights"] = np.asarray(estimator.coef_, dtype=np.float32)
    if hasattr(estimator, "intercept_"):
        state["bias"] = np.asarray(estimator.intercept_, dtype=np.float32).reshape(-1)
    if hasattr(estimator, "classes_"):
        state["classes"] = [str(item) for item in estimator.classes_]
    return state


def _optional_str(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


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
