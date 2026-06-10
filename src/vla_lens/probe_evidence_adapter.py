"""Index-backed adapters from existing probe artifacts into ProbeEvidenceBundle."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from vla_lens.dataset.index import (
    ARTIFACT_INDEX,
    MODEL_SITE_INDEX,
    PROBE_EPISODE_INDEX,
    PROBE_PREDICTIONS,
)
from vla_lens.probe_evidence import (
    ArrayRef,
    FailureCaseEvidence,
    LensCapability,
    LensGeometry,
    LensProvenanceEvidence,
    LensRun,
    ModelLocusEvidence,
    ModelLocusRef,
    PredictionEvidence,
    ProbeEvidenceBundle,
    ProbeLensArtifact,
    RankedMoment,
    RankedMomentsEvidence,
    ScoreSeriesEvidence,
    UnavailableReason,
)
from vla_lens.server.indexed import _read_table

DEFAULT_SCORE_THRESHOLD = 0.5
DEFAULT_RANK_LIMIT = 50


def indexed_probe_evidence_bundle_payload(
    root: Path,
    probe_id: str,
    query: Mapping[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Return a canonical ProbeEvidenceBundle payload for one indexed probe."""
    query = query or {}
    limit = _positive_int(_query_value(query, "limit"), DEFAULT_RANK_LIMIT)
    bundle = probe_evidence_bundle_from_index(root, probe_id, rank_limit=limit)
    return bundle.to_dict()


def probe_evidence_bundle_from_index(
    root: Path,
    probe_id: str,
    *,
    rank_limit: int = DEFAULT_RANK_LIMIT,
) -> ProbeEvidenceBundle:
    """Build a validated ProbeEvidenceBundle from existing dashboard index tables.

    This is Phase 2's adapter seam. It consumes the current fast dashboard index
    rather than raw traces, which keeps the research-facing UI off ad hoc episode
    JSON traversal.
    """
    artifacts = _read_table(root / ARTIFACT_INDEX)
    artifact_row = _artifact_row(artifacts, probe_id)
    predictions = _probe_rows(_read_table(root / PROBE_PREDICTIONS), probe_id)
    episode_rows = _probe_rows(_read_table(root / PROBE_EPISODE_INDEX), probe_id)
    model_sites = _read_table(root / MODEL_SITE_INDEX)
    metrics = _json_mapping(artifact_row.get("metrics"))
    method = _json_mapping(artifact_row.get("method"))
    display = _json_mapping(artifact_row.get("display"))
    arrays = _json_mapping(artifact_row.get("arrays"))
    artifact = _probe_lens_artifact(artifact_row, metrics, method, display)
    run = _lens_run(root, artifact, artifact_row, predictions, episode_rows)
    geometry = _lens_geometry(metrics, method, predictions, episode_rows)
    primitives = [
        _provenance_evidence(artifact, run, metrics, method, display),
        *_score_series_evidence(artifact, run, predictions),
        *_ranked_moment_evidence(artifact, run, predictions, episode_rows, rank_limit=rank_limit),
        *_prediction_evidence(artifact, run, predictions, episode_rows),
        *_model_locus_evidence(artifact, run, predictions, episode_rows, model_sites),
        *_failure_case_evidence(artifact, run, predictions, episode_rows, rank_limit=rank_limit),
    ]
    capabilities = _capabilities(geometry, primitives)
    geometry = LensGeometry(
        temporal_scope=geometry.temporal_scope,
        output_kind=geometry.output_kind,
        input_basis=geometry.input_basis,
        locus_kind=geometry.locus_kind,
        capabilities=capabilities,
    )
    unavailable = _unavailable_reasons(
        geometry,
        primitives,
        predictions=predictions,
        episode_rows=episode_rows,
        arrays=arrays,
    )
    return ProbeEvidenceBundle(
        bundle_id=f"probe-evidence:{run.lens_run_id}",
        artifact=artifact,
        run=run,
        geometry=geometry,
        capabilities=capabilities,
        primitives=tuple(primitives),
        unavailable=unavailable,
    )


def _artifact_row(artifacts: pd.DataFrame, probe_id: str) -> Mapping[str, Any]:
    if artifacts.empty or "artifact_id" not in artifacts:
        raise KeyError(probe_id)
    rows = artifacts.loc[artifacts["artifact_id"].astype(str) == probe_id]
    if "artifact_type" in rows:
        rows = rows.loc[rows["artifact_type"].astype(str) == "probe_suite"]
    if rows.empty:
        raise KeyError(probe_id)
    return rows.iloc[0].to_dict()


def _probe_rows(frame: pd.DataFrame, probe_id: str) -> pd.DataFrame:
    if frame.empty or "probe_id" not in frame:
        return pd.DataFrame()
    return frame.loc[frame["probe_id"].astype(str) == probe_id].copy()


def _probe_lens_artifact(
    row: Mapping[str, Any],
    metrics: Mapping[str, Any],
    method: Mapping[str, Any],
    display: Mapping[str, Any],
) -> ProbeLensArtifact:
    artifact_id = str(row.get("artifact_id") or "")
    source = _source_from_artifact(metrics, method)
    return ProbeLensArtifact(
        lens_id=artifact_id,
        lens_version=str(
            method.get("probe_artifact_schema_version")
            or method.get("schema_version")
            or row.get("created_utc")
            or "indexed"
        ),
        name=str(row.get("name") or artifact_id),
        target=_optional_str(metrics.get("target") or display.get("target")),
        source_model={"model_id": _optional_str(source.get("model_id")) or "unknown"},
        source=source,
        training=_training_from_artifact(metrics, method),
        created_at=_optional_str(row.get("created_utc")),
    )


def _lens_run(
    root: Path,
    artifact: ProbeLensArtifact,
    row: Mapping[str, Any],
    predictions: pd.DataFrame,
    episode_rows: pd.DataFrame,
) -> LensRun:
    trace_ids = _unique_strings(
        pd.concat(
            [
                predictions.get("trace_id", pd.Series(dtype=object)),
                episode_rows.get("trace_id", pd.Series(dtype=object)),
            ],
            ignore_index=True,
        )
    )
    created = _optional_str(row.get("created_utc")) or datetime.now(UTC).isoformat()
    return LensRun(
        lens_run_id=f"indexed:{artifact.lens_id}:{_dataset_id(root)}",
        lens_id=artifact.lens_id,
        lens_version=artifact.lens_version,
        dataset_id=_dataset_id(root),
        episode_ids=tuple(trace_ids),
        computed_at=created,
        result_version="probe_evidence.indexed.v1",
        status="complete" if len(predictions) or len(episode_rows) else "partial",
        evidence_bundle_id=f"probe-evidence:indexed:{artifact.lens_id}:{_dataset_id(root)}",
    )


def _lens_geometry(
    metrics: Mapping[str, Any],
    method: Mapping[str, Any],
    predictions: pd.DataFrame,
    episode_rows: pd.DataFrame,
) -> LensGeometry:
    rows = predictions if not predictions.empty else episode_rows
    temporal_scope = "policy_call" if _has_non_null(rows, "policy_call_index") else "episode"
    input_basis = _input_basis(method, metrics)
    locus_kind = "model_locus" if _has_model_locus(rows, metrics, method) else "none"
    return LensGeometry(
        temporal_scope=temporal_scope,
        output_kind="scalar",
        input_basis=input_basis,
        locus_kind=locus_kind,
        capabilities=(),
    )


def _capabilities(
    geometry: LensGeometry,
    primitives: Sequence[Any],
) -> tuple[LensCapability, ...]:
    kinds = {primitive.kind for primitive in primitives}
    capabilities: list[LensCapability] = []
    if "score_series" in kinds:
        capabilities.extend(["score_series", "thresholding"])
    if "ranked_moments" in kinds:
        capabilities.extend(["ranked_moments", "uncertainty"])
    if "prediction" in kinds:
        capabilities.append("prediction")
    if "model_locus" in kinds and geometry.locus_kind == "model_locus":
        capabilities.append("model_locus_view")
    if "failure_case" in kinds:
        capabilities.append("failure_cases")
    return tuple(dict.fromkeys(capabilities))


def _provenance_evidence(
    artifact: ProbeLensArtifact,
    run: LensRun,
    metrics: Mapping[str, Any],
    method: Mapping[str, Any],
    display: Mapping[str, Any],
) -> LensProvenanceEvidence:
    return LensProvenanceEvidence(
        kind="provenance",
        lens_id=artifact.lens_id,
        lens_run_id=run.lens_run_id,
        fields={
            "Prediction": artifact.target or artifact.name,
            "Input": _input_label(method, metrics),
            "Output": _optional_str(display.get("output") or metrics.get("output")) or "scalar",
            "Objective": _objective_label(method, metrics),
            "Split": _split_label(method),
            "Model site": _optional_str(metrics.get("best_feature")) or "unknown",
        },
    )


def _score_series_evidence(
    artifact: ProbeLensArtifact,
    run: LensRun,
    predictions: pd.DataFrame,
) -> tuple[ScoreSeriesEvidence, ...]:
    if predictions.empty or "trace_id" not in predictions:
        return ()
    score_column = _score_column(predictions)
    if score_column is None:
        return ()
    time_axis = "policy_call" if _has_non_null(predictions, "policy_call_index") else "timestep"
    out: list[ScoreSeriesEvidence] = []
    for trace_id, group in predictions.groupby(predictions["trace_id"].astype(str), sort=False):
        values = pd.to_numeric(group[score_column], errors="coerce").dropna()
        if values.empty:
            continue
        out.append(
            ScoreSeriesEvidence(
                kind="score_series",
                lens_id=artifact.lens_id,
                lens_run_id=run.lens_run_id,
                episode_id=str(trace_id),
                time_axis=time_axis,
                values_ref=ArrayRef(
                    uri=f"indexed://probe_predictions/{artifact.lens_id}/{trace_id}/{score_column}",
                    format="indexed_table_column",
                    shape=(int(len(values)),),
                    dtype="float64",
                ),
                summary={
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "mean": float(values.mean()),
                },
                threshold=DEFAULT_SCORE_THRESHOLD,
            )
        )
    return tuple(out)


def _ranked_moment_evidence(
    artifact: ProbeLensArtifact,
    run: LensRun,
    predictions: pd.DataFrame,
    episode_rows: pd.DataFrame,
    *,
    rank_limit: int,
) -> tuple[RankedMomentsEvidence, ...]:
    rows = predictions if not predictions.empty else episode_rows
    score_column = _score_column(rows)
    if rows.empty or score_column is None:
        return ()
    scored = rows.copy()
    scored["_score"] = pd.to_numeric(scored[score_column], errors="coerce")
    scored = scored.loc[scored["_score"].notna()]
    if scored.empty:
        return ()
    out = [
        RankedMomentsEvidence(
            kind="ranked_moments",
            lens_id=artifact.lens_id,
            lens_run_id=run.lens_run_id,
            ranking="top",
            moments=_moments(scored.sort_values("_score", ascending=False).head(rank_limit)),
        ),
        RankedMomentsEvidence(
            kind="ranked_moments",
            lens_id=artifact.lens_id,
            lens_run_id=run.lens_run_id,
            ranking="bottom",
            moments=_moments(scored.sort_values("_score", ascending=True).head(rank_limit)),
        ),
    ]
    uncertain = scored.assign(_uncertain=(scored["_score"] - DEFAULT_SCORE_THRESHOLD).abs())
    out.append(
        RankedMomentsEvidence(
            kind="ranked_moments",
            lens_id=artifact.lens_id,
            lens_run_id=run.lens_run_id,
            ranking="uncertain",
            moments=_moments(uncertain.sort_values("_uncertain", ascending=True).head(rank_limit)),
        )
    )
    return tuple(out)


def _prediction_evidence(
    artifact: ProbeLensArtifact,
    run: LensRun,
    predictions: pd.DataFrame,
    episode_rows: pd.DataFrame,
) -> tuple[PredictionEvidence, ...]:
    rows = predictions if not predictions.empty else episode_rows
    if rows.empty or "trace_id" not in rows:
        return ()
    out: list[PredictionEvidence] = []
    for row in rows.head(200).to_dict("records"):
        prediction = _json_scalar(row.get("prediction_value", row.get("predicted")))
        if prediction is None:
            continue
        out.append(
            PredictionEvidence(
                kind="prediction",
                lens_id=artifact.lens_id,
                lens_run_id=run.lens_run_id,
                episode_id=str(row.get("trace_id") or row.get("episode_id") or ""),
                timestep=_optional_int(row.get("timestep")),
                policy_call=_optional_int(row.get("policy_call_index")),
                prediction=prediction,
                label=_json_scalar(row.get("target_value", row.get("actual"))),
                confidence=_optional_float(row.get("confidence")),
                correct=_optional_bool(row.get("correct")),
                split=_split_category(row.get("split_category") or row.get("split")),
            )
        )
    return tuple(out)


def _model_locus_evidence(
    artifact: ProbeLensArtifact,
    run: LensRun,
    predictions: pd.DataFrame,
    episode_rows: pd.DataFrame,
    model_sites: pd.DataFrame,
) -> tuple[ModelLocusEvidence, ...]:
    rows = predictions if not predictions.empty else episode_rows
    if rows.empty:
        return ()
    out: list[ModelLocusEvidence] = []
    seen: set[tuple[str, int | None, int | None, str | None]] = set()
    for row in rows.to_dict("records"):
        locus = _model_locus_from_row(row, model_sites) or _model_locus_from_artifact(artifact)
        if locus is None:
            continue
        trace_id = _optional_str(row.get("trace_id"))
        key = (
            trace_id or "",
            _optional_int(row.get("timestep")),
            _optional_int(row.get("policy_call_index")),
            locus.model_site_id,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(
            ModelLocusEvidence(
                kind="model_locus",
                lens_id=artifact.lens_id,
                lens_run_id=run.lens_run_id,
                episode_id=trace_id,
                timestep=_optional_int(row.get("timestep")),
                policy_call=_optional_int(row.get("policy_call_index")),
                locus=locus,
                source_label=locus.model_site_id or locus.module or _layer_label(locus.layer),
            )
        )
        if len(out) >= 200:
            break
    return tuple(out)


def _failure_case_evidence(
    artifact: ProbeLensArtifact,
    run: LensRun,
    predictions: pd.DataFrame,
    episode_rows: pd.DataFrame,
    *,
    rank_limit: int,
) -> tuple[FailureCaseEvidence, ...]:
    rows = predictions if not predictions.empty else episode_rows
    if rows.empty or not _has_labels(rows):
        return ()
    score_column = _score_column(rows)
    correct = _correct_series(rows)
    if correct is None:
        return ()
    wrong = rows.loc[correct == False].copy()  # noqa: E712
    if score_column is not None and not wrong.empty:
        wrong["_score"] = pd.to_numeric(wrong[score_column], errors="coerce")
        wrong = wrong.sort_values("_score", ascending=False)
    return (
        FailureCaseEvidence(
            kind="failure_case",
            lens_id=artifact.lens_id,
            lens_run_id=run.lens_run_id,
            ranking="high_confidence_wrong",
            moments=_moments(wrong.head(rank_limit)) if not wrong.empty else (),
        ),
    )


def _unavailable_reasons(
    geometry: LensGeometry,
    primitives: Sequence[Any],
    *,
    predictions: pd.DataFrame,
    episode_rows: pd.DataFrame,
    arrays: Mapping[str, Any],
) -> tuple[UnavailableReason, ...]:
    kinds = {primitive.kind for primitive in primitives}
    reasons: list[UnavailableReason] = []
    if "score_series" not in kinds:
        reasons.append(
            UnavailableReason(
                capability="score_series",
                panel_id="score_series",
                reason="missing_scores",
                message="Score series unavailable because no numeric probe scores were indexed.",
            )
        )
    if "contribution" not in kinds:
        reason = (
            "pooled_representation"
            if geometry.input_basis == "pooled_layer_activation"
            else "not_computed"
        )
        message = (
            "Contribution breakdown unavailable because the indexed probe path does not expose "
            "aligned activation values and probe weights."
        )
        if arrays.get("weights") or arrays.get("direction"):
            message = (
                "Contribution breakdown unavailable because weights exist but aligned "
                "activation values were not materialized in this evidence adapter."
            )
        reasons.append(
            UnavailableReason(
                capability="contribution_breakdown",
                panel_id="contribution",
                reason=reason,
                message=message,
            )
        )
    rows = predictions if not predictions.empty else episode_rows
    if "failure_case" not in kinds:
        reason = "missing_labels" if not _has_labels(rows) else "not_computed"
        message = (
            "Failure cases unavailable because no labels or proxy targets exist."
            if reason == "missing_labels"
            else "Failure cases unavailable because failure evidence was not computed."
        )
        reasons.append(
            UnavailableReason(
                capability="failure_cases",
                panel_id="failure_cases",
                reason=reason,
                message=message,
            )
        )
    if "model_locus" not in kinds:
        reasons.append(
            UnavailableReason(
                capability="model_locus_view",
                panel_id="model_locus",
                reason="missing_model_locus",
                message="Model locus unavailable because indexed probe rows have no model site.",
            )
        )
    return tuple(reasons)


def _moments(rows: pd.DataFrame) -> tuple[RankedMoment, ...]:
    return tuple(
        RankedMoment(
            episode_id=str(row.get("trace_id") or row.get("episode_id") or ""),
            timestep=_optional_int(row.get("timestep")),
            policy_call=_optional_int(row.get("policy_call_index")),
            score=_optional_float(row.get("_score", row.get("confidence"))),
            prediction=_json_scalar(row.get("prediction_value", row.get("predicted"))),
            label=_json_scalar(row.get("target_value", row.get("actual"))),
            confidence=_optional_float(row.get("confidence")),
        )
        for row in rows.to_dict("records")
    )


def _model_locus_from_row(
    row: Mapping[str, Any],
    model_sites: pd.DataFrame,
) -> ModelLocusRef | None:
    site = _optional_str(row.get("model_site_id") or row.get("feature"))
    if not site:
        return None
    site_rows = pd.DataFrame()
    if not model_sites.empty:
        for column in ("site_id", "name"):
            if column in model_sites:
                site_rows = model_sites.loc[model_sites[column].astype(str) == site]
                if not site_rows.empty:
                    break
    site_row = site_rows.iloc[0].to_dict() if not site_rows.empty else {}
    return ModelLocusRef(
        model_site_id=site,
        layer=_optional_int(row.get("layer", site_row.get("layer"))),
        module=_optional_str(site_row.get("module")),
        stream=_optional_str(site_row.get("tensor_type")),
    )


def _model_locus_from_artifact(artifact: ProbeLensArtifact) -> ModelLocusRef | None:
    source = artifact.source
    site = _optional_str(source.get("model_site_id") or source.get("module"))
    layer = _optional_int(source.get("layer"))
    stream = _optional_str(source.get("stream"))
    if not site and layer is None and stream is None:
        return None
    return ModelLocusRef(
        model_site_id=site,
        layer=layer,
        module=_optional_str(source.get("module")),
        stream=stream,
    )


def _layer_label(layer: int | None) -> str | None:
    return f"Layer {layer}" if layer is not None else None


def _source_from_artifact(metrics: Mapping[str, Any], method: Mapping[str, Any]) -> dict[str, Any]:
    selector = _selector_from_method(method)
    best_feature = _optional_str(metrics.get("best_feature"))
    return {
        "model_id": _optional_str(selector.get("model_id") or selector.get("model")),
        "module": _optional_str(selector.get("module") or best_feature),
        "layer": _first_int(selector.get("layers")),
        "stream": _optional_str(selector.get("tensor_type")),
        "token_scope": _token_scope(selector),
        "model_site_id": best_feature,
    }


def _training_from_artifact(
    metrics: Mapping[str, Any],
    method: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "dataset_id": _optional_str(method.get("dataset_id")),
        "split": _split_label(method),
        "objective": _objective_label(method, metrics),
        "metrics": {
            key: value
            for key, value in metrics.items()
            if key in {"best_score", "best_delta", "best_model", "best_feature", "target"}
        },
    }


def _selector_from_method(method: Mapping[str, Any]) -> Mapping[str, Any]:
    input_spec = method.get("input")
    if isinstance(input_spec, Mapping) and isinstance(input_spec.get("selector"), Mapping):
        return input_spec["selector"]
    selector = method.get("selector")
    return selector if isinstance(selector, Mapping) else {}


def _input_basis(method: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    text = " ".join(
        str(value).lower()
        for value in [
            method.get("input_basis"),
            method.get("feature_transform"),
            metrics.get("best_feature"),
            _selector_from_method(method).get("reduce_tokens"),
        ]
        if value is not None
    )
    if "sae" in text:
        return "sae_feature"
    if "attention_head" in text or "attention head" in text or "attention" in text:
        return "attention_head_output"
    if "pool" in text or "mean" in text:
        return "pooled_layer_activation"
    return "layer_activation"


def _input_label(method: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    selector = _selector_from_method(method)
    best_feature = _optional_str(metrics.get("best_feature"))
    layers = selector.get("layers")
    layer_label = f"layers {layers}" if layers else "selected model activations"
    return best_feature or layer_label


def _objective_label(method: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    model = _optional_str(metrics.get("best_model")) or _optional_str(method.get("model"))
    target = _optional_str(metrics.get("target")) or "target"
    return " ".join(part for part in [model, "probe for", target] if part)


def _split_label(method: Mapping[str, Any]) -> str | None:
    split = method.get("split")
    if isinstance(split, Mapping):
        return _optional_str(split.get("selection_value") or split.get("test_value"))
    return _optional_str(method.get("split_column"))


def _score_column(rows: pd.DataFrame) -> str | None:
    for column in ("confidence", "score", "prediction_value", "correct_rate"):
        if column in rows:
            numeric = pd.to_numeric(rows[column], errors="coerce")
            if numeric.notna().any():
                return column
    return None


def _has_model_locus(
    rows: pd.DataFrame,
    metrics: Mapping[str, Any],
    method: Mapping[str, Any],
) -> bool:
    selector = _selector_from_method(method)
    return _has_non_null(rows, "model_site_id") or _has_non_null(rows, "feature") or bool(
        _optional_str(metrics.get("best_feature"))
        or _optional_str(selector.get("module"))
        or _first_int(selector.get("layers")) is not None
    )


def _has_labels(rows: pd.DataFrame) -> bool:
    return _has_non_null(rows, "actual") or _has_non_null(rows, "target_value") or _has_non_null(
        rows, "correct"
    )


def _correct_series(rows: pd.DataFrame) -> pd.Series | None:
    if "correct" in rows:
        correct = rows["correct"]
        if correct.notna().any():
            return correct.map(_optional_bool)
    actual_column = "target_value" if _has_non_null(rows, "target_value") else "actual"
    predicted_column = (
        "prediction_value" if _has_non_null(rows, "prediction_value") else "predicted"
    )
    if actual_column not in rows or predicted_column not in rows:
        return None
    actual = rows[actual_column].map(_json_scalar)
    predicted = rows[predicted_column].map(_json_scalar)
    mask = actual.notna() & predicted.notna()
    if not mask.any():
        return None
    out = pd.Series([None] * len(rows), index=rows.index, dtype=object)
    out.loc[mask] = actual.loc[mask] == predicted.loc[mask]
    return out


def _has_non_null(frame: pd.DataFrame, column: str) -> bool:
    return not frame.empty and column in frame and frame[column].notna().any()


def _json_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if value is None or (isinstance(value, float) and value != value):
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _json_scalar(value: Any) -> str | bool | int | float | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _query_value(query: Mapping[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    return values[0] if values else None


def _positive_int(value: str | None, fallback: int) -> int:
    try:
        parsed = int(value) if value is not None else fallback
    except ValueError:
        parsed = fallback
    return max(1, min(parsed, 500))


def _dataset_id(root: Path) -> str:
    return root.name


def _unique_strings(values: pd.Series) -> list[str]:
    return sorted(str(value) for value in values.dropna().unique() if str(value))


def _optional_str(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    text = str(value)
    return text if text and text != "nan" else None


def _optional_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _split_category(value: Any) -> str | None:
    text = (_optional_str(value) or "").lower().replace("-", "_")
    if not text:
        return None
    if text in {"train", "training"}:
        return "train"
    if text.startswith("val") or "heldout" in text or "held_out" in text:
        return "validation"
    if text.startswith("test"):
        return "test"
    return "missing"


def _first_int(value: Any) -> int | None:
    if isinstance(value, Sequence) and not isinstance(value, str):
        for item in value:
            parsed = _optional_int(item)
            if parsed is not None:
                return parsed
    return _optional_int(value)


def _token_scope(selector: Mapping[str, Any]) -> str:
    if selector.get("reduce_tokens") or selector.get("reduction"):
        return "pooled"
    if selector.get("token_index") is not None:
        return "single_token"
    if selector.get("token_kind"):
        return "all_tokens"
    return "unknown"


__all__ = [
    "indexed_probe_evidence_bundle_payload",
    "probe_evidence_bundle_from_index",
]
