"""Pre-training review helpers for probe specs.

The preflight path intentionally stops before fitting a probe. It resolves the
same selected rows, target labels, filters, split values, and baseline columns
that training will see, then emits a small audit packet for human review.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from vla_lens.probes.experiment_cards import (
    experiment_card_from_preflight,
    format_experiment_card_markdown,
)
from vla_lens.probes.representation_options import probe_representation_options
from vla_lens.probes.workflow_artifacts import _probe_target, _value_counts
from vla_lens.probes.workflow_prepare import (
    _apply_missing_policy,
    _apply_row_expansion,
    _apply_row_filters,
    _attach_episode_metadata,
    _ensure_selection_split,
    _ensure_split,
)
from vla_lens.probes.workflow_spec import (
    baseline_columns,
    normalize_probe_spec,
    specialized_probe_family,
)
from vla_lens.probes.workflow_targets import (
    _normalize_target_spec,
    _resolve_probe_target,
    _target_name,
)
from vla_lens.selectors import ActivationQuery
from vla_lens.traces import TraceDataset

DEFAULT_MIN_CLASS_SUPPORT = 20
DEFAULT_LARGE_SWEEP_READOUTS = 100


def probe_preflight_report(
    dataset: TraceDataset,
    spec: Mapping[str, Any],
    *,
    min_class_support: int = DEFAULT_MIN_CLASS_SUPPORT,
    large_sweep_readouts: int = DEFAULT_LARGE_SWEEP_READOUTS,
) -> dict[str, Any]:
    """Return an auditable review packet for a probe spec before training."""
    family = specialized_probe_family(spec)
    if family is not None:
        return _specialized_probe_preflight_report(dataset, spec, family=family)
    normalized = normalize_probe_spec(spec)
    features = dict(normalized["features"])
    selector = _selector_from_features(features)
    feature_matrix = dataset.select_model_sites(selector).materialize(cache=True)
    X, rows = feature_matrix.X, feature_matrix.rows
    if rows.empty or X.shape[0] == 0:
        raise ValueError(f"Probe selector matched no activation rows: {selector.to_dict()}")

    selected_row_count = int(len(rows))
    representation = probe_representation_options(
        dataset,
        rows,
        selected=normalized.get("representation"),
        selector=selector,
    )
    rows = _attach_episode_metadata(rows, dataset)
    X, rows, expansion_summary = _apply_row_expansion(
        X,
        rows,
        dataset,
        normalized.get("row_expand"),
    )
    target_spec = _normalize_target_spec(normalized["target"])
    target_name = _target_name(target_spec)
    rows = _resolve_probe_target(dataset, rows, target_spec)
    X, rows, filter_summary = _apply_row_filters(X, rows, normalized.get("row_filter"))
    X, rows, missing_summary = _apply_missing_policy(
        X,
        rows,
        target_name,
        policy=str(target_spec.get("missing_policy") or "error"),
    )
    split = dict(normalized.get("split") or {})
    split_column = str(split.get("column", "split"))
    train_value = str(split.get("train_value", "train"))
    test_value = str(split.get("test_value", "test"))
    selection_value = str(split.get("selection_value", test_value))
    eval_values = [str(value) for value in split.get("eval_values", [test_value])]
    rows = _ensure_split(
        rows,
        split_column,
        train_value=train_value,
        test_value=test_value,
        split_kind=str(split.get("kind", "random_episode")),
    )
    rows, validation_summary = _ensure_selection_split(
        rows,
        split_column,
        train_value=train_value,
        selection_value=selection_value,
        test_value=test_value,
        split_kind=str(split.get("kind", "random_episode")),
    )

    target_info = _probe_target(target_name, rows, target_spec=target_spec)
    target_kind = str(target_info.get("kind") or "classification")
    split_summary = _split_summary(rows, split_column)
    baseline_info = _baseline_info(normalized, rows, target_name, target_spec)
    target_summary = _target_summary(rows, target_name, split_column, target_kind)
    sweep_info = _sweep_info(normalized, rows, eval_values)
    warnings = _preflight_warnings(
        rows=rows,
        target_name=target_name,
        target_kind=target_kind,
        split_column=split_column,
        train_value=train_value,
        test_value=test_value,
        selection_value=selection_value,
        eval_values=eval_values,
        baseline_info=baseline_info,
        target_summary=target_summary,
        sweep_info=sweep_info,
        min_class_support=min_class_support,
        large_sweep_readouts=large_sweep_readouts,
    )
    warnings.extend(_representation_warnings(representation))

    payload = _jsonable(
        {
            "name": str(normalized.get("name") or "Probe"),
            "question": _optional_str(normalized.get("question")),
            "hypothesis_family": _optional_str(normalized.get("hypothesis_family")),
            "intended_claim": _optional_str(normalized.get("intended_claim")),
            "dataset_root": str(dataset.root),
            "selector": selector.to_dict(),
            "representation": representation,
            "feature_matrix": {
                "selected_rows": selected_row_count,
                "rows_after_filters": int(len(rows)),
                "feature_dim": int(X.shape[1]) if X.ndim == 2 else None,
                "estimated_feature_bytes": int(X.nbytes),
                "cache_key": getattr(feature_matrix, "cache_key", None),
            },
            "target": target_info,
            "split": {
                "kind": str(split.get("kind", "random_episode")),
                "column": split_column,
                "train_value": train_value,
                "selection_value": selection_value,
                "test_value": test_value,
                "eval_values": eval_values,
                "automatic_validation": validation_summary,
                "summary": split_summary,
            },
            "filters": filter_summary,
            "row_expansion": expansion_summary,
            "missing_target": missing_summary,
            "baselines": baseline_info,
            "probe": {
                "models": list(dict.fromkeys(str(v) for v in _probe_models(normalized))),
                "primary_model": str(_probe_models(normalized)[0]),
            },
            "sweep": sweep_info,
            "target_summary": target_summary,
            "warnings": warnings,
        }
    )
    payload["experiment_card"] = experiment_card_from_preflight(payload)
    return payload


def probe_experiment_card(
    dataset: TraceDataset,
    spec: Mapping[str, Any],
    *,
    min_class_support: int = DEFAULT_MIN_CLASS_SUPPORT,
    large_sweep_readouts: int = DEFAULT_LARGE_SWEEP_READOUTS,
) -> dict[str, Any]:
    """Return the small human-facing card for a planned probe run."""

    report = probe_preflight_report(
        dataset,
        spec,
        min_class_support=min_class_support,
        large_sweep_readouts=large_sweep_readouts,
    )
    return dict(report["experiment_card"])


def format_probe_preflight_markdown(report: Mapping[str, Any], *, details: bool = False) -> str:
    """Render the short experiment card, with diagnostic tables on request."""
    if report.get("preflight_kind") == "specialized_review":
        return _format_specialized_probe_preflight_markdown(report)
    if not details:
        card = report.get("experiment_card") or experiment_card_from_preflight(report)
        return format_experiment_card_markdown(card)
    lines = [
        f"# Probe Preflight: {report.get('name', 'Probe')}",
        "",
    ]
    if report.get("question"):
        lines.extend(["## Question", str(report["question"]), ""])
    if report.get("hypothesis_family") or report.get("intended_claim"):
        lines.extend(
            [
                "## Research Framing",
                f"- Hypothesis family: {report.get('hypothesis_family') or '-'}",
                f"- Intended claim: {report.get('intended_claim') or '-'}",
                "",
            ]
        )

    target = dict(report.get("target") or {})
    feature_matrix = dict(report.get("feature_matrix") or {})
    split = dict(report.get("split") or {})
    baselines = dict(report.get("baselines") or {})
    sweep = dict(report.get("sweep") or {})
    probe = dict(report.get("probe") or {})
    row_expansion = dict(report.get("row_expansion") or {})
    representation = dict(report.get("representation") or {})
    lines.extend(
        [
            "## Training Surface",
            f"- Dataset: `{report.get('dataset_root')}`",
            f"- Target: `{target.get('name')}` ({target.get('kind')})",
            f"- Target source: `{target.get('source')}`",
            f"- Feature rows: {feature_matrix.get('rows_after_filters')} "
            f"after filters / {feature_matrix.get('selected_rows')} selected",
            f"- Feature dim: {feature_matrix.get('feature_dim')}",
            f"- Row expansion: {row_expansion.get('kind', 'none')} "
            f"({row_expansion.get('input_rows', feature_matrix.get('rows_after_filters'))} -> "
            f"{row_expansion.get('output_rows', feature_matrix.get('rows_after_filters'))} rows)",
            f"- Split: `{split.get('column')}` train=`{split.get('train_value')}` "
            f"select=`{split.get('selection_value')}` test=`{split.get('test_value')}`",
            f"- Probe models: {', '.join(probe.get('models') or [])}",
            f"- Planned readouts: {sweep.get('planned_readout_count')}",
            "",
        ]
    )

    lines.extend(["## Representation Options", ""])
    option_rows = [
        (
            str(option.get("label") or option.get("kind")),
            str(option.get("status") or "unknown"),
            str(option.get("question") or "-"),
            str(option.get("reason") or "-"),
        )
        for option in representation.get("options") or []
    ]
    lines.extend(
        _markdown_table(
            ["Input structure", "Status", "Research question", "What is needed"],
            option_rows,
        )
        or ["_No representation options available._"]
    )
    selected_representation = dict(representation.get("selected") or {})
    lines.extend(
        [
            "",
            f"Selected: `{selected_representation.get('kind', 'mean_pool')}`",
            "",
        ]
    )

    lines.extend(["## Baselines", ""])
    baseline_rows = [
        ("Metadata columns", ", ".join(baselines.get("available_columns") or []) or "-"),
        ("Missing columns", ", ".join(baselines.get("missing_columns") or []) or "-"),
        (
            "Suspicious columns",
            ", ".join(baselines.get("suspicious_columns") or []) or "-",
        ),
    ]
    lines.extend(_markdown_table(["Item", "Value"], baseline_rows))
    lines.append("")

    lines.extend(["## Split Summary", ""])
    split_values = dict((split.get("summary") or {}).get("values") or {})
    split_episodes = dict((split.get("summary") or {}).get("episodes") or {})
    split_rows = [
        (key, str(value), str(split_episodes.get(key, "-")))
        for key, value in sorted(split_values.items())
    ]
    lines.extend(_markdown_table(["Split", "Rows", "Episodes"], split_rows) or ["_No split rows._"])
    lines.append("")

    target_summary = dict(report.get("target_summary") or {})
    if target.get("kind") == "classification":
        lines.extend(["## Class Support", ""])
        support_rows = []
        for split_name, counts in sorted((target_summary.get("by_split") or {}).items()):
            for label, count in sorted(dict(counts).items()):
                support_rows.append((split_name, label, str(count)))
        lines.extend(
            _markdown_table(["Split", "Class", "Rows"], support_rows)
            or ["_No class support available._"]
        )
        lines.append("")
    else:
        lines.extend(["## Regression Target", ""])
        regression_rows = [
            (split_name, json.dumps(stats, sort_keys=True))
            for split_name, stats in sorted((target_summary.get("by_split") or {}).items())
        ]
        lines.extend(_markdown_table(["Split", "Stats"], regression_rows) or ["_No stats._"])
        lines.append("")

    warnings = list(report.get("warnings") or [])
    lines.extend(["## Warnings", ""])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- No automatic preflight warnings.")
    lines.append("")
    lines.extend(
        [
            "## Review Reminder",
            "This report does not prove scientific usefulness. It checks that the planned probe "
            "has visible labels, splits, baselines, and sweep scope before training.",
        ]
    )
    return "\n".join(lines)


def _specialized_probe_preflight_report(
    dataset: TraceDataset,
    spec: Mapping[str, Any],
    *,
    family: str,
) -> dict[str, Any]:
    """Describe a dedicated study without pretending to run generic selection."""
    payload = dict(spec)
    probe = dict(payload.get("probe") or {})
    split = dict(payload.get("split") or {})
    warnings: list[str] = []

    if family == "geometry_study":
        feature_specs = [dict(value) for value in payload.get("features") or []]
        selectors = [_selector_from_features(value).to_dict() for value in feature_specs]
        target = {
            "name": "object_pose_targets",
            "kind": "multi_output_regression",
            "values": [str(value) for value in payload.get("targets") or []],
            "source": "object-centered scene state and policy-call labels",
        }
        cohort = {
            "row_unit": "object-policy-call row",
            "object_column": str(payload.get("object_column") or "primary_target_object"),
            "filters": ["resolved object", "finite pose target", "aligned feature row"],
        }
        controls = [str(value) for value in payload.get("baseline") or []] + [
            "train_mean",
            "previous_or_initial_pose",
            "zero_position_update",
            "identity_relative_rotation",
        ]
        models = [str(value) for value in probe.get("models") or ["ridge"]]
        sweep_axes = ["feature_id", "layer", "pca_dim", "model"]
        if "ridge" in models:
            sweep_axes.append("ridge_alpha")
        options = _geometry_representation_options(dataset, feature_specs)
        selected = {
            "kind": "declared_multi_feature_pooling",
            "feature_ids": [
                str(value.get("id") or f"feature_{i}") for i, value in enumerate(feature_specs)
            ],
        }
        runner = "run_vla_lens_geometry_study.py"
    else:
        source_id = str(payload.get("source_probe_artifact_id") or "")
        source = _source_artifact(dataset, source_id)
        source_ready = source is not None
        if not source_ready:
            warnings.append(f"Required source probe artifact `{source_id}` is not available.")
        inherited_split = dict((source or {}).get("method", {}).get("split") or {})
        split = (
            split
            or inherited_split
            or {
                "kind": "inherited_from_source_artifact",
                "column": "split",
                "train_value": "train",
                "selection_value": "val_heldout_task",
                "test_value": "test_heldout_task",
            }
        )
        layers = [int(value) for value in probe.get("layers") or [0, 4, 8, 12, 17]]
        models = [str(value) for value in probe.get("models") or ["linear", "mlp"]]
        sweep_axes = ["layer", "model"]
        selectors = [
            {
                "source_probe_artifact_id": source_id,
                "camera_name": str(payload.get("camera_name") or "agentview"),
                "layers": layers,
            }
        ]
        if family == "object_roi_identity_study":
            target = {
                "name": "visible_object_identity",
                "kind": "classification",
                "source": "source artifact object vocabulary and known image-region boxes",
            }
            cohort = {
                "row_unit": "initial visible object instance",
                "camera_name": str(payload.get("camera_name") or "agentview"),
                "filters": ["supported object", "visible projected box", "available visual tokens"],
            }
            controls = [
                "whole_image",
                "task_scene_box",
                "wrong_object_roi",
                "background_roi",
                "shuffled_training_labels",
            ]
            selected_kind = "object_roi"
            runner = "run_vla_lens_object_roi_identity_study.py"
        else:
            target = {
                "name": "queried_object_patch_overlap",
                "kind": "classification_and_localization",
                "source": "object query joined to source visual-token patches and object boxes",
            }
            cohort = {
                "row_unit": "object-query patch example",
                "camera_name": str(payload.get("camera_name") or "agentview"),
                "negative_ratio": int(dict(payload.get("sampling") or {}).get("negative_ratio", 3)),
                "filters": ["supported visible object", "available visual tokens"],
            }
            controls = [
                "fixed_object_spatial_map",
                "query_xy",
                "prompt_scene_query_xy",
                "wrong_object_query",
                "within_task_shuffled_activation",
                "fixed_patch_position_permutation",
            ]
            selected_kind = "object_conditioned"
            runner = "run_vla_lens_object_query_localization_study.py"
            matched_id = str(payload.get("matched_scene_artifact_id") or "")
            if matched_id and _source_artifact(dataset, matched_id) is None:
                warnings.append(
                    f"Matched-scene artifact `{matched_id}` is unavailable; "
                    "displacement control is blocked."
                )
        options = _artifact_backed_representation_options(
            selected_kind=selected_kind,
            source_ready=source_ready,
            runner=runner,
        )
        selected = {"kind": selected_kind}

    report = _jsonable(
        {
            "preflight_kind": "specialized_review",
            "study_family": family,
            "name": str(payload.get("name") or family),
            "question": _optional_str(payload.get("question")),
            "hypothesis_family": _optional_str(payload.get("hypothesis_family")),
            "intended_claim": _optional_str(payload.get("intended_claim")),
            "dataset_root": str(dataset.root),
            "runner": {"name": runner, "status": _selected_option_status(options, selected)},
            "target": target,
            "cohort": cohort,
            "selectors": selectors,
            "representation": {"selected": selected, "options": options},
            "split": split,
            "baselines": {
                "configured": list(dict.fromkeys(controls)),
                "controls": list(dict.fromkeys(controls)),
            },
            "probe": {"models": models, "primary_model": models[0] if models else None, **probe},
            "sweep": {"columns": sweep_axes},
            "warnings": warnings,
        }
    )
    return report


def _source_artifact(dataset: TraceDataset, artifact_id: str) -> dict[str, Any] | None:
    if not artifact_id:
        return None
    path = dataset._dataset_artifact_root() / "artifacts" / artifact_id / "artifact.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _geometry_representation_options(
    dataset: TraceDataset, feature_specs: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    for index, feature in enumerate(feature_specs):
        selector = _selector_from_features(feature)
        matched = not dataset.select_model_sites(selector)._matching_model_sites().empty
        options.append(
            {
                "kind": str(feature.get("id") or f"feature_{index}"),
                "label": str(feature.get("id") or f"Feature family {index}"),
                "status": "ready" if matched else "blocked",
                "runner": "run_vla_lens_geometry_study.py",
                "reason": "Matching captured model sites are available."
                if matched
                else "No captured model sites match this declared selector.",
            }
        )
    token_data = any(
        str(feature.get("tensor_type")) == "hidden_tokens"
        and option.get("status") == "ready"
        for feature, option in zip(feature_specs, options, strict=True)
    )
    options.extend(
        [
            {
                "kind": "object_conditioned_pose",
                "label": "Object-conditioned token decoding",
                "status": "data_ready" if token_data else "blocked",
                "runner": "specialized runner required",
                "reason": (
                    "Token-preserving object-pose decoding is not implemented "
                    "by the pooled geometry runner."
                ),
            },
            {
                "kind": "set_decoder",
                "label": "Object-set decoding",
                "status": "blocked",
                "runner": "specialized runner required",
                "reason": "This spec does not declare complete unordered object-set labels.",
            },
        ]
    )
    return options


def _artifact_backed_representation_options(
    *, selected_kind: str, source_ready: bool, runner: str
) -> list[dict[str, Any]]:
    return [
        {
            "kind": "mean_pool",
            "label": "Whole-image mean baseline",
            "status": "ready" if source_ready else "blocked",
            "runner": runner,
            "reason": "A low-capacity baseline only; it is not the requested study representation.",
        },
        {
            "kind": selected_kind,
            "label": "Known object region"
            if selected_kind == "object_roi"
            else "Explicit object query",
            "status": "ready" if source_ready else "blocked",
            "runner": runner,
            "reason": "The dedicated runner and source token artifact are available."
            if source_ready
            else "The dedicated runner exists, but its source token artifact is missing.",
        },
        {
            "kind": "set_decoder",
            "label": "Unordered object-set decoder",
            "status": "data_ready" if source_ready else "blocked",
            "runner": "specialized runner required",
            "reason": (
                "The source contains token/object data, but this study does not "
                "implement set decoding."
                if source_ready
                else "Source token/object data is unavailable."
            ),
        },
    ]


def _selected_option_status(
    options: Sequence[Mapping[str, Any]], selected: Mapping[str, Any]
) -> str:
    selected_kinds = set(selected.get("feature_ids") or [selected.get("kind")])
    statuses = [
        str(option.get("status")) for option in options if option.get("kind") in selected_kinds
    ]
    return "ready" if statuses and all(value == "ready" for value in statuses) else "blocked"


def _format_specialized_probe_preflight_markdown(report: Mapping[str, Any]) -> str:
    target = dict(report.get("target") or {})
    cohort = dict(report.get("cohort") or {})
    split = dict(report.get("split") or {})
    representation = dict(report.get("representation") or {})
    probe = dict(report.get("probe") or {})
    runner = dict(report.get("runner") or {})
    split_values = (
        f"`{split.get('train_value', '-')}` / "
        f"`{split.get('selection_value', '-')}` / "
        f"`{split.get('test_value', '-')}`"
    )
    lines = [
        f"# Specialized Probe Preflight: {report.get('name', 'Probe')}",
        "",
        "## Question",
        str(report.get("question") or "-"),
        "",
        "## Target and cohort",
        f"- Target: `{target.get('name')}` ({target.get('kind')})",
        f"- Construction: {target.get('source')}",
        f"- Row unit: {cohort.get('row_unit')}",
        f"- Cohort: `{json.dumps(cohort, sort_keys=True)}`",
        "",
        "## Representation choices",
        "",
    ]
    rows = [
        (
            str(value.get("label")),
            str(value.get("status")),
            str(value.get("runner")),
            str(value.get("reason")),
        )
        for value in representation.get("options") or []
    ]
    lines.extend(_markdown_table(["Representation", "Status", "Runner", "Reason"], rows))
    lines.extend(
        [
            "",
            f"Selected: `{json.dumps(representation.get('selected') or {}, sort_keys=True)}`",
            "",
            "## Split",
            f"- Kind: `{split.get('kind', 'inherited')}`",
            f"- Train / select / test: {split_values}",
            "",
            "## Baselines and controls",
            "- " + ", ".join(dict(report.get("baselines") or {}).get("controls") or []),
            "",
            "## Model battery and sweep",
            "- Models: " + ", ".join(str(value) for value in probe.get("models") or []),
            "- Sweep: " + ", ".join(dict(report.get("sweep") or {}).get("columns") or []),
            f"- Specialized runner: `{runner.get('name')}` ({runner.get('status')})",
            "",
            "## Warnings",
        ]
    )
    lines.extend(
        [f"- {value}" for value in report.get("warnings") or []] or ["- No automatic warnings."]
    )
    lines.extend(
        ["", "This is a review-only preflight. It does not train or materialize probe features."]
    )
    return "\n".join(lines)


def _selector_from_features(features: Mapping[str, Any]) -> ActivationQuery:
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


def _split_summary(rows: pd.DataFrame, split_column: str) -> dict[str, Any]:
    if split_column not in rows:
        return {"values": {"all": int(len(rows))}, "episodes": {}}
    out: dict[str, Any] = {"values": _value_counts(rows[split_column])}
    if "trace_id" in rows:
        by_episode = rows[[split_column, "trace_id"]].drop_duplicates()
        out["episodes"] = _value_counts(by_episode[split_column])
    return out


def _baseline_info(
    spec: Mapping[str, Any],
    rows: pd.DataFrame,
    target_name: str,
    target_spec: Mapping[str, Any],
) -> dict[str, Any]:
    configured = [str(item) for item in spec.get("baseline") or []]
    columns = baseline_columns(configured)
    available = [column for column in columns if column in rows.columns]
    missing = [column for column in columns if column not in rows.columns]
    target_column = str(target_spec.get("column") or target_name)
    suspicious = [
        column
        for column in available
        if column == target_name or column == target_column or column.lower() == target_name.lower()
    ]
    return {
        "configured": configured,
        "available_columns": available,
        "missing_columns": missing,
        "suspicious_columns": suspicious,
        "majority_baseline_is_implicit": True,
    }


def _target_summary(
    rows: pd.DataFrame,
    target_name: str,
    split_column: str,
    target_kind: str,
) -> dict[str, Any]:
    if target_kind == "classification":
        by_split: dict[str, dict[str, int]] = {}
        for split_value, group in rows.groupby(split_column, dropna=False, sort=True):
            by_split[str(split_value)] = _value_counts(group[target_name])
        return {
            "kind": "classification",
            "overall": _value_counts(rows[target_name]),
            "by_split": by_split,
            "class_count": int(rows[target_name].astype(str).nunique(dropna=False)),
        }

    numeric = pd.to_numeric(rows[target_name], errors="coerce")
    by_split = {}
    for split_value, group in rows.assign(__target_numeric=numeric).groupby(
        split_column, dropna=False, sort=True
    ):
        values = group["__target_numeric"].dropna()
        by_split[str(split_value)] = _numeric_summary(values)
    return {
        "kind": "regression",
        "overall": _numeric_summary(numeric.dropna()),
        "by_split": by_split,
    }


def _sweep_info(
    spec: Mapping[str, Any],
    rows: pd.DataFrame,
    eval_values: Sequence[str],
) -> dict[str, Any]:
    columns = _normalize_sweep_columns(spec.get("sweep", "layer"))
    missing = [column for column in columns if column not in rows.columns]
    group_count = 1 if not columns else 0
    values: dict[str, list[str]] = {}
    if columns and not missing:
        group_key: str | list[str] = columns[0] if len(columns) == 1 else columns
        group_count = int(rows.groupby(group_key, dropna=False, sort=True).ngroups)
        for column in columns:
            values[column] = sorted(str(value) for value in rows[column].dropna().unique())
    models = _probe_models(spec)
    return {
        "columns": columns,
        "missing_columns": missing,
        "group_count": group_count,
        "values": values,
        "eval_split_count": int(len(eval_values)),
        "model_count": int(len(models)),
        "planned_readout_count": int(group_count * len(eval_values) * len(models)),
    }


def _preflight_warnings(
    *,
    rows: pd.DataFrame,
    target_name: str,
    target_kind: str,
    split_column: str,
    train_value: str,
    test_value: str,
    selection_value: str,
    eval_values: Sequence[str],
    baseline_info: Mapping[str, Any],
    target_summary: Mapping[str, Any],
    sweep_info: Mapping[str, Any],
    min_class_support: int,
    large_sweep_readouts: int,
) -> list[str]:
    warnings: list[str] = []
    split_values = set(rows[split_column].astype(str)) if split_column in rows else set()
    if train_value not in split_values:
        warnings.append(f"Train split `{train_value}` has no selected rows.")
    for value in eval_values:
        if value not in split_values:
            warnings.append(f"Eval split `{value}` has no selected rows.")
    if selection_value == test_value:
        warnings.append("Selection split equals test split; sweep selection can leak final claims.")
    if not baseline_info.get("available_columns"):
        warnings.append("No metadata baseline columns are available after row construction.")
    if baseline_info.get("missing_columns"):
        warnings.append(
            "Configured metadata baselines are missing: "
            + ", ".join(str(v) for v in baseline_info["missing_columns"])
        )
    if baseline_info.get("suspicious_columns"):
        warnings.append(
            "Metadata baseline includes the target column; this is likely leakage: "
            + ", ".join(str(v) for v in baseline_info["suspicious_columns"])
        )
    if "policy_call_index" in rows.columns and "policy_call_index" not in baseline_info.get(
        "available_columns", []
    ):
        warnings.append(
            "`policy_call_index` is available but not configured as a metadata baseline."
        )
    if sweep_info.get("missing_columns"):
        warnings.append(
            "Sweep columns are missing from selected rows: "
            + ", ".join(str(v) for v in sweep_info["missing_columns"])
        )
    planned = int(sweep_info.get("planned_readout_count") or 0)
    if planned > large_sweep_readouts:
        warnings.append(
            f"Planned sweep has {planned} readouts; use validation-only selection "
            "and null controls."
        )
    if target_kind == "classification":
        by_split = dict(target_summary.get("by_split") or {})
        low = [
            f"{split}/{label}={count}"
            for split, counts in by_split.items()
            for label, count in dict(counts).items()
            if split in set(eval_values) and int(count) < min_class_support
        ]
        if low:
            warnings.append(
                f"Low held-out class support below {min_class_support}: " + ", ".join(low[:12])
            )
        overall_classes = set(dict(target_summary.get("overall") or {}))
        missing_classes = [
            f"{split}/{label}"
            for split, counts in by_split.items()
            for label in sorted(overall_classes - set(dict(counts)))
            if split in set(eval_values)
        ]
        if missing_classes:
            warnings.append(
                "Eval split lacks some target classes seen overall: "
                + ", ".join(missing_classes[:12])
            )
        train_classes = set(dict(by_split.get(train_value) or {}))
        unseen_eval_classes = [
            f"{split}/{label}"
            for split, counts in by_split.items()
            for label in sorted(set(dict(counts)) - train_classes)
            if split in set(eval_values)
        ]
        if unseen_eval_classes:
            warnings.append(
                "Eval split contains target classes absent from train; "
                "classification readouts cannot predict unseen labels: "
                + ", ".join(unseen_eval_classes[:12])
            )
        warnings.extend(
            _sweep_group_support_warnings(
                rows=rows,
                target_name=target_name,
                split_column=split_column,
                train_value=train_value,
                eval_values=eval_values,
                sweep_info=sweep_info,
                min_class_support=min_class_support,
            )
        )
    else:
        by_split = dict(target_summary.get("by_split") or {})
        zero_var = [
            split
            for split, stats in by_split.items()
            if split in set(eval_values) and float(dict(stats).get("std", 0.0) or 0.0) == 0.0
        ]
        if zero_var:
            warnings.append(
                "Regression target has zero variance in eval split(s): " + ", ".join(zero_var)
            )
    if target_name == "task_phase":
        warnings.append("`task_phase` is behavior-derived; treat high scores as sanity checks.")
    return warnings


def _representation_warnings(representation: Mapping[str, Any]) -> list[str]:
    selected = str(dict(representation.get("selected") or {}).get("kind") or "mean_pool")
    option = next(
        (
            dict(value)
            for value in representation.get("options") or []
            if str(dict(value).get("kind")) == selected
        ),
        {},
    )
    status = str(option.get("status") or "unknown")
    if status == "ready":
        runner = str(option.get("runner") or "generic_probe")
        if runner == "generic_probe":
            return []
        return [
            f"Selected representation `{selected}` is runnable for "
            f"{option.get('runnable_scope')}; use the specialized `{runner}` runner."
        ]
    if status == "data_ready":
        return [
            f"Selected representation `{selected}` has the required captured data, "
            f"but needs the specialized `{option.get('runner')}` runner before training."
        ]
    return [
        f"Selected representation `{selected}` is not ready: "
        f"{option.get('reason') or 'requirements are unknown.'}"
    ]


def _sweep_group_support_warnings(
    *,
    rows: pd.DataFrame,
    target_name: str,
    split_column: str,
    train_value: str,
    eval_values: Sequence[str],
    sweep_info: Mapping[str, Any],
    min_class_support: int,
) -> list[str]:
    columns = [str(column) for column in sweep_info.get("columns") or []]
    if not columns or any(column not in rows for column in columns):
        return []
    needed = [split_column, target_name, *columns]
    if any(column not in rows for column in needed):
        return []

    group_key: str | list[str] = columns[0] if len(columns) == 1 else columns
    warnings: list[str] = []
    low_eval: list[str] = []
    low_class_eval: list[str] = []
    single_class_eval: list[str] = []
    untrainable: list[str] = []
    eval_set = {str(value) for value in eval_values}
    for group_value, group in rows.groupby(group_key, dropna=False, sort=True):
        label = _sweep_group_label(columns, group_value)
        train_group = group.loc[group[split_column].astype(str) == str(train_value)]
        if train_group[target_name].astype(str).nunique(dropna=False) < 2:
            untrainable.append(label)
        for split_value, split_group in group.groupby(split_column, dropna=False, sort=True):
            split_name = str(split_value)
            if split_name not in eval_set:
                continue
            row_count = int(len(split_group))
            if row_count < min_class_support:
                low_eval.append(f"{label}/{split_name}={row_count}")
            for target_value, count in split_group[target_name].value_counts(
                dropna=False,
                sort=True,
            ).items():
                if int(count) < min_class_support:
                    low_class_eval.append(
                        f"{label}/{split_name}/{target_value}={int(count)}"
                    )
            if split_group[target_name].astype(str).nunique(dropna=False) < 2:
                single_class_eval.append(f"{label}/{split_name}")
    if low_eval:
        warnings.append(
            "Low sweep-group eval support below "
            f"{min_class_support}: " + ", ".join(low_eval[:12])
        )
    if single_class_eval:
        warnings.append(
            "Sweep-group eval split has only one target class: "
            + ", ".join(single_class_eval[:12])
        )
    if low_class_eval:
        warnings.append(
            "Low sweep-group eval class support below "
            f"{min_class_support}: " + ", ".join(low_class_eval[:12])
        )
    if untrainable:
        warnings.append(
            "Sweep-group train split has fewer than two target classes; those readouts "
            "will be skipped: "
            + ", ".join(untrainable[:12])
        )
    return warnings


def _sweep_group_label(columns: Sequence[str], value: Any) -> str:
    if len(columns) == 1:
        return f"{columns[0]}={value}"
    values = value if isinstance(value, tuple) else (value,)
    return ",".join(f"{column}={item}" for column, item in zip(columns, values, strict=False))


def _normalize_sweep_columns(value: Any) -> list[str]:
    if isinstance(value, str):
        if value in {"", "none", "null"}:
            return []
        return [value]
    if value is None:
        return []
    return [str(item) for item in value if str(item) not in {"", "none", "null"}]


def _probe_models(spec: Mapping[str, Any]) -> list[str]:
    probe = spec.get("probe") if isinstance(spec.get("probe"), Mapping) else {}
    models = (
        probe.get("models", ["linear", "mlp"])
        if isinstance(probe, Mapping)
        else ["linear", "mlp"]
    )
    if isinstance(models, str):
        return [models]
    parsed = [str(model) for model in models]
    return parsed or ["linear", "mlp"]


def _numeric_summary(values: pd.Series) -> dict[str, float | int | None]:
    if values.empty:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(values.count()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    if not rows:
        return []
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_escape_markdown_cell(str(cell)) for cell in row) + " |")
    return lines


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|")


def _optional_str(value: Any) -> str | None:
    return None if value in {None, ""} else str(value)


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
