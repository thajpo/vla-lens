"""Human-readable probe experiment cards.

An experiment card separates choices that change the research claim from
method defaults and execution details.  The same small structure is emitted by
preflight and stored in replayable probe artifacts.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

EXPERIMENT_CARD_SCHEMA_VERSION = 1


def experiment_card_from_preflight(report: Mapping[str, Any]) -> dict[str, Any]:
    """Build the review card shown before probe training."""

    target = dict(report.get("target") or {})
    selector = dict(report.get("selector") or {})
    split = dict(report.get("split") or {})
    baselines = dict(report.get("baselines") or {})
    probe = dict(report.get("probe") or {})
    sweep = dict(report.get("sweep") or {})
    feature_matrix = dict(report.get("feature_matrix") or {})
    return _card(
        name=str(report.get("name") or "Probe"),
        question=_optional_str(report.get("question")),
        intended_claim=_optional_str(report.get("intended_claim")),
        target=target,
        selector=selector,
        split=split,
        baselines={
            "configured": list(baselines.get("configured") or []),
            "available": list(baselines.get("available_columns") or []),
            "missing": list(baselines.get("missing_columns") or []),
        },
        evaluation={
            "selection_split": split.get("selection_value"),
            "test_split": split.get("test_value"),
            "evaluation_splits": list(split.get("eval_values") or []),
            "independent_units": ["selected row", "episode"],
            "confidence_intervals": "not configured by the generic probe runner",
        },
        method_choices={
            "probe_models": list(probe.get("models") or []),
            "primary_model": probe.get("primary_model"),
            "sweep_columns": list(sweep.get("columns") or []),
            "planned_readouts": sweep.get("planned_readout_count"),
            "token_reduction": selector.get("reduce_tokens"),
            "feature_dtype": selector.get("dtype"),
        },
        execution={
            "selected_rows": feature_matrix.get("selected_rows"),
            "rows_after_filters": feature_matrix.get("rows_after_filters"),
            "feature_dim": feature_matrix.get("feature_dim"),
            "estimated_feature_bytes": feature_matrix.get("estimated_feature_bytes"),
            "runtime_estimate": report.get("runtime_estimate") or "not available",
            "raw_activation_tensors_copied": False,
            "feature_cache": "derived and removable",
        },
        warnings=list(report.get("warnings") or []),
        stage="planned",
    )


def experiment_card_from_artifact_fields(
    *,
    name: str,
    research: Mapping[str, Any],
    input_info: Mapping[str, Any],
    target: Mapping[str, Any],
    split: Mapping[str, Any],
    probe: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    metadata_baselines: Sequence[str],
    sweep: Any,
    metrics: Mapping[str, Any],
    uncertainty: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the card stored after the selected probe has been fitted."""

    selector = dict(input_info.get("selector") or {})
    feature_shape = list(input_info.get("feature_shape") or [])
    estimated_bytes = input_info.get("feature_matrix_bytes")
    return _card(
        name=name,
        question=_optional_str(research.get("question")),
        intended_claim=_optional_str(research.get("intended_claim")),
        target=dict(target),
        selector=selector,
        split=dict(split),
        baselines={"configured": list(metadata_baselines), "available": list(metadata_baselines)},
        evaluation={
            **dict(evaluation),
            "confidence_intervals": dict(uncertainty),
        },
        method_choices={
            "probe_models": list(probe.get("models") or []),
            "selected_model": dict(probe.get("best_model_state") or {}).get("model"),
            "selected_readout": dict(probe.get("best_model_state") or {}),
            "sweep": sweep,
            "token_reduction": selector.get("reduce_tokens"),
            "feature_dtype": input_info.get("dtype"),
        },
        execution={
            "source_episode_count": metrics.get("source_episode_count"),
            "row_count": metrics.get("sample_count"),
            "feature_shape": feature_shape,
            "feature_dim": feature_shape[1] if len(feature_shape) == 2 else None,
            "estimated_feature_bytes": estimated_bytes,
            "raw_activation_tensors_copied": False,
            "feature_cache": "derived and removable",
        },
        warnings=[],
        stage="completed",
    )


def format_experiment_card_markdown(card: Mapping[str, Any]) -> str:
    """Render an experiment card without exposing every configuration field."""

    claim = dict(card.get("claim_controls") or {})
    target = dict(claim.get("target") or {})
    model_input = dict(claim.get("model_input") or {})
    split = dict(claim.get("generalization") or {})
    evaluation = dict(claim.get("evaluation") or {})
    method = dict(card.get("method_choices") or {})
    execution = dict(card.get("execution_details") or {})
    lines = [f"# Probe Experiment Card: {card.get('name', 'Probe')}", ""]
    if card.get("question"):
        lines.extend(["## Question", str(card["question"]), ""])
    if card.get("intended_claim"):
        lines.extend(["## Intended claim", str(card["intended_claim"]), ""])
    lines.extend(
        [
            "## Choices that change the claim",
            f"- Prediction target: `{target.get('name')}` ({target.get('kind')})",
            f"- Target source: `{target.get('source')}`",
            f"- Model input: `{model_input.get('module') or model_input.get('name')}`",
            f"- Token handling: `{model_input.get('token_reduction')}`",
            f"- Held-out rule: `{split.get('kind')}` using `{split.get('column')}`",
            f"- Train / select / test: `{split.get('train_value')}` / "
            f"`{split.get('selection_value')}` / `{split.get('test_value')}`",
            "- Baselines: " + _joined(claim.get("baselines")),
            "- Evaluation units: " + _joined(evaluation.get("independent_units")),
            "",
            "## Method choices",
            "- Probe models: " + _joined(method.get("probe_models")),
            "- Primary or selected model: `"
            f"{method.get('primary_model') or method.get('selected_model')}`",
            "- Sweep: " + _joined(method.get("sweep_columns") or method.get("sweep")),
            f"- Planned readouts: {method.get('planned_readouts', '-')}",
            "",
            "## Execution details",
            f"- Rows: {execution.get('rows_after_filters', execution.get('row_count', '-'))}",
            "- Feature dimension: "
            f"{execution.get('feature_dim', execution.get('feature_shape', '-'))}",
            f"- Estimated feature memory: {_human_bytes(execution.get('estimated_feature_bytes'))}",
            f"- Runtime estimate: {execution.get('runtime_estimate', 'not available')}",
            "- Raw activation tensors copied into artifact: no",
            "- Reusable feature cache: derived and removable",
            "",
        ]
    )
    warnings = list(card.get("warnings") or [])
    lines.extend(["## Warnings", ""])
    lines.extend([f"- {warning}" for warning in warnings] or ["- No automatic warnings."])
    lines.extend(
        [
            "",
            "## Saved result",
            "A replayable result supports `explain`, `replay`, and `use`. "
            "Legacy results explain what was reported but may not support replay or use.",
        ]
    )
    return "\n".join(lines)


def _card(
    *,
    name: str,
    question: str | None,
    intended_claim: str | None,
    target: Mapping[str, Any],
    selector: Mapping[str, Any],
    split: Mapping[str, Any],
    baselines: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    method_choices: Mapping[str, Any],
    execution: Mapping[str, Any],
    warnings: Sequence[str],
    stage: str,
) -> dict[str, Any]:
    available_baselines = list(baselines.get("available") or baselines.get("configured") or [])
    return {
        "schema_version": EXPERIMENT_CARD_SCHEMA_VERSION,
        "stage": stage,
        "name": name,
        "question": question,
        "intended_claim": intended_claim,
        "claim_controls": {
            "target": dict(target),
            "model_input": {
                "name": selector.get("name"),
                "module": selector.get("module"),
                "layers": selector.get("layers"),
                "tensor_type": selector.get("tensor_type"),
                "token_kind": selector.get("token_kind"),
                "token_reduction": selector.get("reduce_tokens"),
                "generation_step": selector.get("generation_step"),
            },
            "generalization": dict(split),
            "baselines": available_baselines,
            "evaluation": dict(evaluation),
        },
        "method_choices": dict(method_choices),
        "execution_details": dict(execution),
        "warnings": list(warnings),
        "actions": {
            "explain": "show the question and experiment choices",
            "replay": "reproduce saved predictions from the capture without fitting",
            "use": "apply the fitted probe to compatible features",
        },
    }


def _joined(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, Mapping):
        return ", ".join(f"{key}={item}" for key, item in value.items()) or "-"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return ", ".join(str(item) for item in value) or "-"
    return str(value)


def _human_bytes(value: Any) -> str:
    if value is None or value == "":
        return "not available"
    amount = float(value)
    for unit in ["B", "KB", "MB", "GB"]:
        if amount < 1024.0 or unit == "GB":
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{amount:.1f} GB"


def _optional_str(value: Any) -> str | None:
    return None if value is None or value == "" else str(value)
