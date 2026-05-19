"""Validation for sealed ``.vlatrace`` bundles.

The validator enforces the data-integrity invariant: a trace may contain
captured episode data and trace-local derived summaries, but it must not depend
on raw capture paths or legacy payload locations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from vla_lens.traces import TraceBundle, TraceDataset

FORBIDDEN_LINEAGE_KEYS = frozenset(
    {
        "raw_path",
        "source_path",
        "legacy_path",
        "raw_legacy_path",
        "_pi05_legacy_root",
        "capture_root",
        "captures_root",
        "rollout_path",
        "vlm_path",
        "expert_path",
    }
)
FORBIDDEN_LINEAGE_FRAGMENTS = (
    "/vlm/call_",
    "/expert/call_",
    "\\vlm\\call_",
    "\\expert\\call_",
    "_pi05_legacy_root",
)

CAPTURE_PROFILE_REQUIREMENTS: dict[str, dict[str, tuple[str, ...]]] = {
    "rollout": {
        "episode_arrays": ("executed_actions", "action_chunks"),
        "model_tensors": (),
    },
    "representation": {
        "episode_arrays": ("executed_actions", "action_chunks", "generation_actions"),
        "model_tensors": ("hidden_tokens",),
    },
    "features": {
        "episode_arrays": ("executed_actions", "action_chunks", "generation_actions"),
        "model_tensors": ("hidden_tokens",),
    },
    "mechanistic_light": {
        "episode_arrays": ("executed_actions", "action_chunks", "generation_actions"),
        "model_tensors": ("hidden_tokens", "attention"),
    },
    "mechanistic_sampled": {
        "episode_arrays": ("executed_actions", "action_chunks", "generation_actions"),
        "model_tensors": ("hidden_tokens", "attention"),
    },
    "mechanistic_heavy": {
        "episode_arrays": ("executed_actions", "action_chunks", "generation_actions"),
        "model_tensors": ("hidden_tokens", "attention"),
    },
    "mechanistic_all": {
        "episode_arrays": ("executed_actions", "action_chunks", "generation_actions"),
        "model_tensors": ("hidden_tokens", "attention"),
    },
    "internals_sampled": {
        "episode_arrays": ("executed_actions", "action_chunks", "generation_actions"),
        "model_tensors": ("hidden_tokens", "attention"),
    },
    "audit_sampled": {
        "episode_arrays": ("executed_actions", "action_chunks", "generation_actions"),
        "model_tensors": ("hidden_tokens", "attention"),
    },
    "full": {
        "episode_arrays": ("executed_actions", "action_chunks", "generation_actions"),
        "model_tensors": (),
    },
    "audit_full": {
        "episode_arrays": ("executed_actions", "action_chunks", "generation_actions"),
        "model_tensors": (),
    },
    "custom": {
        "episode_arrays": ("executed_actions", "action_chunks"),
        "model_tensors": (),
    },
}

FULL_REQUIRED_MODEL_SITE_ROLES: tuple[str, ...] = (
    "input_embeddings",
    "residual_pre_attention",
    "attention_norm_output",
    "q",
    "k",
    "v",
    "pre_mask_scores",
    "post_mask_logits",
    "attention_probs",
    "attn_output_pre_o_proj",
    "o_proj",
    "residual_post_attention",
    "residual_pre_mlp",
    "mlp_norm_output",
    "mlp_gate",
    "mlp_up",
    "mlp_intermediate",
    "mlp_down",
    "mlp_output",
    "residual_post_mlp",
    "adarms_scale",
    "adarms_shift",
    "adarms_gate",
    "action_head_input",
    "action_head_output",
    "kv_cache_key",
    "kv_cache_value",
    "attention_mask",
    "causal_mask",
    "position_ids",
    "rope_cos",
    "rope_sin",
    "rope_metadata",
)


@dataclass(frozen=True, slots=True)
class TraceValidationResult:
    trace_id: str
    valid: bool
    errors: tuple[dict[str, Any], ...] = ()
    warnings: tuple[dict[str, Any], ...] = ()
    manifest_hash: str | None = None
    capture_profile: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["errors"] = list(self.errors)
        payload["warnings"] = list(self.warnings)
        return payload


@dataclass(frozen=True, slots=True)
class DatasetValidationResult:
    valid: bool
    traces: tuple[TraceValidationResult, ...] = ()
    errors: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "traces": [trace.to_dict() for trace in self.traces],
        }


def manifest_hash(bundle: TraceBundle) -> str:
    payload = (bundle.path / TraceBundle.MANIFEST).read_bytes()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def validate_trace_bundle(bundle: TraceBundle) -> TraceValidationResult:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    trace_id = _safe_trace_id(bundle)

    for required in [
        TraceBundle.MANIFEST,
        TraceBundle.TIMESTEPS,
        TraceBundle.POLICY_CALLS,
        TraceBundle.GENERATION_STEPS,
        TraceBundle.STREAMS,
        TraceBundle.TOKEN_SPACES,
        TraceBundle.TOKENS,
        TraceBundle.ROBOT_STATE,
        TraceBundle.SCENE_STATE,
        TraceBundle.CAMERA_STATE,
        TraceBundle.EVALUATION,
        TraceBundle.IMAGE_PREPROCESSING,
        TraceBundle.PROMPT_METADATA,
        TraceBundle.ACTION_NORMALIZATION,
        TraceBundle.ARRAY_INDEX,
        TraceBundle.MODEL_SITES,
        TraceBundle.ARTIFACT_INDEX,
        TraceBundle.CAPTURE_REQUEST,
        TraceBundle.CAPTURE_PLAN,
        TraceBundle.CAPTURE_REPORT,
        TraceBundle.FINGERPRINTS,
    ]:
        if not (bundle.path / required).exists():
            errors.append(
                _issue("missing_file", f"Missing required file {required}", file=required)
            )
    if errors:
        return TraceValidationResult(trace_id=trace_id, valid=False, errors=tuple(errors))

    manifest_payload = json.loads((bundle.path / TraceBundle.MANIFEST).read_text(encoding="utf-8"))
    _check_forbidden_lineage(manifest_payload, "manifest", errors)

    if bundle.manifest.length < 0:
        errors.append(_issue("invalid_manifest", "Trace length cannot be negative"))
    if bundle.manifest.schema_version == "":
        errors.append(_issue("invalid_manifest", "schema_version is required"))

    if len(bundle.timesteps) != bundle.manifest.length:
        errors.append(
            _issue(
                "timestep_length_mismatch",
                "timesteps row count must match manifest.length",
                expected=bundle.manifest.length,
                actual=len(bundle.timesteps),
            )
        )
    if "timestep" not in bundle.timesteps.columns:
        errors.append(_issue("missing_column", "timesteps.parquet requires a timestep column"))

    _validate_index_table(bundle, bundle.array_index, "array_index", errors)
    _validate_index_table(bundle, bundle.model_sites, "model_sites", errors)
    _validate_semantic_tables(bundle, errors)
    _validate_fingerprints(bundle, errors)

    profile = _capture_profile(bundle)
    _validate_profile(bundle, profile, errors, warnings)

    return TraceValidationResult(
        trace_id=trace_id,
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        manifest_hash=manifest_hash(bundle),
        capture_profile=profile,
    )


def validate_trace_dataset(dataset: TraceDataset) -> DatasetValidationResult:
    trace_results = tuple(validate_trace_bundle(bundle) for bundle in dataset.bundles)
    return DatasetValidationResult(
        valid=all(result.valid for result in trace_results),
        traces=trace_results,
    )


def _validate_index_table(
    bundle: TraceBundle,
    table: pd.DataFrame,
    table_name: str,
    errors: list[dict[str, Any]],
) -> None:
    required = {"name", "relative_path", "storage_format", "shape", "axes", "metadata"}
    if table.empty:
        return
    missing = sorted(required - set(table.columns))
    if missing:
        errors.append(_issue("missing_column", f"{table_name} is missing columns", columns=missing))
        return
    names = table["name"].astype(str).tolist()
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        errors.append(
            _issue("duplicate_array", f"{table_name} has duplicate names", names=duplicates)
        )
    for index, row in enumerate(table.to_dict("records")):
        context = f"{table_name}[{index}]"
        relative_path = Path(str(row["relative_path"]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            errors.append(
                _issue("invalid_path", "Storage paths must be bundle-relative", context=context)
            )
            continue
        storage_path = bundle.path / relative_path
        if not storage_path.exists():
            errors.append(
                _issue(
                    "missing_storage",
                    "Indexed storage path does not exist",
                    context=context,
                    relative_path=str(relative_path),
                )
            )
        _check_forbidden_lineage(row, context, errors)
        axes = _loads(row.get("axes"), default=[])
        shape = _loads(row.get("shape"), default=[])
        if len(axes) != len(shape):
            errors.append(
                _issue(
                    "axis_shape_mismatch",
                    "Axis count must match shape rank",
                    context=context,
                    axes=axes,
                    shape=shape,
                )
            )


def _validate_semantic_tables(bundle: TraceBundle, errors: list[dict[str, Any]]) -> None:
    _require_columns(bundle.policy_calls, "policy_calls", {"policy_call_index"}, errors)
    _require_columns(
        bundle.generation_steps,
        "generation_steps",
        {"policy_call_index", "generation_step"},
        errors,
    )
    _require_columns(bundle.streams, "streams", {"stream_id", "name", "modality"}, errors)
    _require_columns(
        bundle.token_spaces,
        "token_spaces",
        {"token_space_id", "stream_id", "token_count"},
        errors,
    )
    _require_columns(bundle.tokens, "tokens", {"token_space_id", "token_index"}, errors)

    policy_call_ids = _string_values(bundle.policy_calls, "policy_call_index")
    if policy_call_ids:
        _validate_refs(
            bundle.generation_steps,
            table_name="generation_steps",
            column="policy_call_index",
            allowed=policy_call_ids,
            errors=errors,
        )

    stream_ids = _string_values(bundle.streams, "stream_id")
    if stream_ids:
        _validate_refs(
            bundle.token_spaces,
            table_name="token_spaces",
            column="stream_id",
            allowed=stream_ids,
            errors=errors,
        )

    token_space_ids = _string_values(bundle.token_spaces, "token_space_id")
    if token_space_ids:
        _validate_refs(
            bundle.tokens,
            table_name="tokens",
            column="token_space_id",
            allowed=token_space_ids,
            errors=errors,
        )
        for column in ["token_space_id", "query_token_space_id", "key_token_space_id"]:
            _validate_refs(
                bundle.model_sites,
                table_name="model_sites",
                column=column,
                allowed=token_space_ids,
                errors=errors,
            )

    for table_name, frame in {
        "robot_state": bundle.robot_state,
        "scene_state": bundle.scene_state,
        "camera_state": bundle.camera_state,
        "evaluation": bundle.evaluation,
        "image_preprocessing": bundle.image_preprocessing,
        "prompt_metadata": bundle.prompt_metadata,
        "action_normalization": bundle.action_normalization,
    }.items():
        if not frame.empty:
            _check_forbidden_lineage(frame.to_dict("records"), table_name, errors)


def _require_columns(
    frame: pd.DataFrame,
    table_name: str,
    required: set[str],
    errors: list[dict[str, Any]],
) -> None:
    if frame.empty:
        return
    missing = sorted(required - set(frame.columns))
    if missing:
        errors.append(_issue("missing_column", f"{table_name} is missing columns", columns=missing))


def _validate_refs(
    frame: pd.DataFrame,
    *,
    table_name: str,
    column: str,
    allowed: set[str],
    errors: list[dict[str, Any]],
) -> None:
    if frame.empty or column not in frame.columns:
        return
    values = frame[column].dropna().astype(str)
    invalid = sorted({value for value in values if value and value not in allowed})
    if invalid:
        errors.append(
            _issue(
                "invalid_reference",
                f"{table_name}.{column} contains unknown references",
                table=table_name,
                column=column,
                values=invalid,
            )
        )


def _string_values(frame: pd.DataFrame, column: str) -> set[str]:
    if frame.empty or column not in frame.columns:
        return set()
    return {str(value) for value in frame[column].dropna().tolist()}


def _validate_profile(
    bundle: TraceBundle,
    profile: str,
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    requirements = CAPTURE_PROFILE_REQUIREMENTS.get(profile)
    if requirements is None:
        warnings.append(
            _issue("unknown_capture_profile", "Unknown capture profile", profile=profile)
        )
        return
    episode_names = set(bundle.array_index.get("name", pd.Series(dtype=str)).astype(str))
    tensor_types = set(bundle.model_sites.get("tensor_type", pd.Series(dtype=str)).astype(str))
    missing_arrays = sorted(set(requirements["episode_arrays"]) - episode_names)
    missing_tensors = sorted(set(requirements["model_tensors"]) - tensor_types)
    if missing_arrays:
        errors.append(
            _issue(
                "profile_missing_episode_arrays",
                "Capture profile requirements not met",
                missing=missing_arrays,
                profile=profile,
            )
        )
    if missing_tensors:
        errors.append(
            _issue(
                "profile_missing_model_tensors",
                "Capture profile requirements not met",
                missing=missing_tensors,
                profile=profile,
            )
        )
    if profile in {"full", "audit_full"}:
        _validate_full_profile_model_sites(bundle, errors)
    if profile in {"full", "audit_full"} and bundle.capture_report.get("missing_model_sites"):
        errors.append(
            _issue(
                "profile_full_incomplete",
                "Full profile cannot omit declared raw model sites",
                missing=bundle.capture_report.get("missing_model_sites"),
            )
        )


def _validate_fingerprints(bundle: TraceBundle, errors: list[dict[str, Any]]) -> None:
    fingerprints = bundle.fingerprints
    required = {
        "algorithm",
        "fingerprint_schema_version",
        "trajectory_fingerprint",
        "context_fingerprint",
        "trace_schema_fingerprint",
        "trace_fingerprint",
    }
    missing = sorted(required - set(fingerprints))
    if missing:
        errors.append(
            _issue(
                "missing_fingerprints",
                "Trace fingerprints are required for probe-grade provenance",
                missing=missing,
            )
        )
        return
    if fingerprints.get("algorithm") != "sha256":
        errors.append(
            _issue(
                "invalid_fingerprint_algorithm",
                "Trace fingerprints must use sha256",
                algorithm=fingerprints.get("algorithm"),
            )
        )
    manifest_fingerprints = bundle.manifest.metadata.get("fingerprints")
    if manifest_fingerprints != fingerprints:
        errors.append(
            _issue(
                "fingerprint_mismatch",
                "manifest.metadata.fingerprints must match tables/fingerprints.json",
                location="manifest.metadata.fingerprints",
            )
        )
    report_fingerprints = bundle.capture_report.get("fingerprints")
    if report_fingerprints != fingerprints:
        errors.append(
            _issue(
                "fingerprint_mismatch",
                "capture_report.fingerprints must match tables/fingerprints.json",
                location="capture_report.fingerprints",
            )
        )


def _validate_full_profile_model_sites(
    bundle: TraceBundle,
    errors: list[dict[str, Any]],
) -> None:
    sites = bundle.model_sites
    if sites.empty:
        errors.append(
            _issue(
                "profile_full_missing_raw_sites",
                "Full profile requires declared raw model-forward sites",
                missing=list(FULL_REQUIRED_MODEL_SITE_ROLES),
            )
        )
        return
    if "role" not in sites.columns:
        errors.append(_issue("missing_column", "model_sites is missing columns", columns=["role"]))
        return

    materialization = (
        sites["materialization"].astype(str)
        if "materialization" in sites.columns
        else pd.Series("", index=sites.index)
    )
    exactness = (
        sites["exactness"].astype(str)
        if "exactness" in sites.columns
        else pd.Series("", index=sites.index)
    )
    raw_exact = sites.loc[(materialization == "raw") & (exactness == "exact")]
    required_site_names = bundle.capture_report.get("required_model_sites")
    if isinstance(required_site_names, list) and required_site_names:
        raw_exact_names = set(raw_exact["name"].dropna().astype(str))
        missing_names = sorted(
            str(name) for name in required_site_names if str(name) not in raw_exact_names
        )
        if missing_names:
            errors.append(
                _issue(
                    "profile_full_missing_raw_sites",
                    "Full profile requires every declared exact raw model-forward site",
                    missing=missing_names,
                )
            )
    raw_exact_roles = set(raw_exact["role"].dropna().astype(str))
    missing_roles = sorted(set(FULL_REQUIRED_MODEL_SITE_ROLES) - raw_exact_roles)
    if missing_roles:
        errors.append(
            _issue(
                "profile_full_missing_raw_sites",
                "Full profile requires exact raw model-forward sites; summaries do not count",
                missing=missing_roles,
            )
        )


def _capture_profile(bundle: TraceBundle) -> str:
    metadata = dict(bundle.manifest.metadata or {})
    value = metadata.get("capture_profile") or metadata.get("vlatrace_profile")
    return str(value or "rollout")


def _check_forbidden_lineage(value: Any, context: str, errors: list[dict[str, Any]]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            child = f"{context}.{key_text}"
            if key_text in FORBIDDEN_LINEAGE_KEYS:
                errors.append(
                    _issue(
                        "forbidden_lineage_key",
                        "Raw/legacy lineage keys are not allowed",
                        context=child,
                    )
                )
            _check_forbidden_lineage(item, child, errors)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _check_forbidden_lineage(item, f"{context}[{index}]", errors)
        return
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        has_forbidden_fragment = any(
            fragment.replace("\\", "/") in normalized for fragment in FORBIDDEN_LINEAGE_FRAGMENTS
        )
        if has_forbidden_fragment:
            errors.append(
                _issue(
                    "forbidden_lineage_value",
                    "Raw capture path fragments are not allowed",
                    context=context,
                )
            )


def _safe_trace_id(bundle: TraceBundle) -> str:
    try:
        return bundle.manifest.trace_id
    except Exception:
        return bundle.path.name


def _loads(value: Any, *, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def _issue(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"code": code, "message": message, **extra}


__all__ = [
    "CAPTURE_PROFILE_REQUIREMENTS",
    "DatasetValidationResult",
    "FULL_REQUIRED_MODEL_SITE_ROLES",
    "TraceValidationResult",
    "manifest_hash",
    "validate_trace_bundle",
    "validate_trace_dataset",
]
