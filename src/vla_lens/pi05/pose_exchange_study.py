"""Build staged patch-study jobs from captured pose-exchange pair manifests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_LAYERS = (0, 4, 8, 12, 16)
DEFAULT_REGIONS = ("target", "distractor", "both", "complement")
SUPPORTED_REGIONS = (
    *DEFAULT_REGIONS,
    "main_camera",
    "wrist_camera",
    "active_images",
    "language_active",
    "full_prefix",
)
CONFIRMATION_CONTROLS = (
    "recipient_self_patch",
    "donor_self_patch",
    "shuffled_donor",
    "random_matched_norm",
    "wrong_region",
)
WRONG_REGION_BY_REGION = {
    "target": "distractor",
    "distractor": "target",
    "both": "complement",
    "complement": "both",
}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    collection = _read_json(args.pairs)
    job = build_pose_exchange_study_job(
        collection,
        study_id=args.study_id,
        phase=args.phase,
        pair_ids=_csv_strings(args.pair_ids) or None,
        layers=_csv_ints(args.layers),
        token_regions=_csv_strings(args.token_regions),
        control_seed=int(args.control_seed),
    )
    _write_json_atomic(args.output, job)
    print(json.dumps({"output": str(args.output), **study_job_summary(job)}, indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument(
        "--phase",
        choices=("localization", "confirmation"),
        default="localization",
    )
    parser.add_argument("--pair-ids", help="Comma-separated subset; default is all valid pairs")
    parser.add_argument(
        "--layers",
        default=",".join(str(layer) for layer in DEFAULT_LAYERS),
    )
    parser.add_argument("--token-regions", default=",".join(DEFAULT_REGIONS))
    parser.add_argument("--control-seed", type=int, default=20260725)
    return parser.parse_args(argv)


def build_pose_exchange_study_job(
    collection: Mapping[str, Any],
    *,
    study_id: str,
    phase: str,
    pair_ids: Sequence[str] | None = None,
    layers: Sequence[int] = DEFAULT_LAYERS,
    token_regions: Sequence[str] = DEFAULT_REGIONS,
    control_seed: int = 20260725,
) -> dict[str, Any]:
    """Turn saved pair evidence into a deterministic localization/confirmation job."""
    if phase not in {"localization", "confirmation"}:
        raise ValueError("phase must be localization or confirmation")
    raw_pairs = collection.get("pairs")
    if not isinstance(raw_pairs, Sequence) or isinstance(raw_pairs, str):
        raise ValueError("pair collection requires pairs")
    valid_pairs = [
        dict(_mapping(pair))
        for pair in raw_pairs
        if bool(_mapping(_mapping(pair).get("validation")).get("pair_valid"))
    ]
    available = {str(pair.get("pair_id")): pair for pair in valid_pairs}
    selected_ids = tuple(pair_ids or available)
    if not selected_ids:
        raise ValueError("study requires at least one valid pair")
    missing = [pair_id for pair_id in selected_ids if pair_id not in available]
    if missing:
        raise ValueError(f"requested pair IDs are missing or invalid: {missing}")
    selected_pairs = [available[pair_id] for pair_id in selected_ids]

    selected_layers = tuple(dict.fromkeys(int(layer) for layer in layers))
    if not selected_layers or any(layer < 0 for layer in selected_layers):
        raise ValueError("study layers must be unique non-negative integers")
    regions = tuple(dict.fromkeys(str(region) for region in token_regions))
    if not regions or any(region not in SUPPORTED_REGIONS for region in regions):
        raise ValueError(f"token regions must come from {SUPPORTED_REGIONS}")

    controls = CONFIRMATION_CONTROLS if phase == "confirmation" else ()
    shared_noise_refs = [
        f"{_recipient_trace_id(pair)}.flow_initial_noise[0]"
        for pair in selected_pairs
    ]
    axes: dict[str, Any] = {
        "phase": phase,
        "token_regions": list(regions),
    }
    if controls:
        unsupported = [region for region in regions if region not in WRONG_REGION_BY_REGION]
        if unsupported:
            raise ValueError(
                "confirmation wrong-region controls are not defined for broad regions: "
                f"{unsupported}"
            )
        axes["wrong_region_by_region"] = {
            region: WRONG_REGION_BY_REGION[region] for region in regions
        }

    return {
        "study": {
            "study_id": study_id,
            "question": (
                "Where do PI0.5 prefix states causally carry the action-relevant "
                "difference produced by exchanging the book and mug poses?"
            ),
            "hypothesis": (
                "Patching object-aligned donor tokens should move the recipient action "
                "toward the donor action more than background or matched controls."
            ),
            "pair_ids": list(selected_ids),
            "sites": [{"layer": layer} for layer in selected_layers],
            "controls": list(controls),
            "shared_noise_refs": shared_noise_refs,
            "thresholds": {
                "minimum_natural_delta_norm": 1e-6,
                "minimum_direction_agreement": 0.5,
                "minimum_transfer_fraction": 0.1,
                "maximum_donor_gap_remaining": 0.95,
                "minimum_control_margin": 0.05,
            },
            "axes": axes,
            "confounds": [
                "The intervention changes object identity at two poses at once.",
                "Image-patch regions approximate object pixels and include nearby context.",
                "This measures open-loop action chunks, not closed-loop task success.",
            ],
            "stopping_rule": (
                "Localization only ranks sites. Run confirmation controls on held-out "
                "pairs before making a specific causal claim."
                if phase == "localization"
                else "Require the same site/region effect across held-out pairs and controls."
            ),
            "provenance": {
                "pair_collection_schema": collection.get("schema_kind"),
                "pair_collection_version": collection.get("schema_version"),
                "phase": phase,
            },
        },
        "pairs": selected_pairs,
        "request_template": _request_template(
            phase=phase,
            control_seed=control_seed,
        ),
    }


def study_job_summary(job: Mapping[str, Any]) -> dict[str, Any]:
    study = _mapping(job.get("study"))
    pair_count = len(study.get("pair_ids") or ())
    site_count = len(study.get("sites") or ())
    region_count = len(_mapping(study.get("axes")).get("token_regions") or ())
    controls_per_trial = len(study.get("controls") or ())
    return {
        "study_id": study.get("study_id"),
        "phase": _mapping(study.get("axes")).get("phase"),
        "pair_count": pair_count,
        "site_count": site_count,
        "region_count": region_count,
        "planned_trial_count": pair_count * site_count * region_count,
        "model_calls_per_trial": 1 + controls_per_trial,
    }


def _request_template(*, phase: str, control_seed: int) -> dict[str, Any]:
    controls = []
    if phase == "confirmation":
        controls = [
            {"kind": "recipient_self_patch"},
            {"kind": "donor_self_patch"},
            {"kind": "shuffled_donor", "parameters": {"seed": control_seed}},
            {"kind": "random_matched_norm", "parameters": {"seed": control_seed}},
            {"kind": "wrong_region"},
        ]
    return {
        "runtime_adapter": "pi05",
        "target": {
            "kind": "activation_slice",
            "model_id": "lerobot/pi05_libero_finetuned",
            "model_family": "pi05",
            "model_site": "pi05.vlm.layers.{layer}.prefix.hidden_tokens",
            "tensor_type": "hidden_tokens",
            "token_space": "pi05.prefix",
            "metadata": {"research_question_id": "RQ-020", "phase": phase},
        },
        "intervention": {
            "request": {
                "operator": {
                    "operator": "source_patch",
                    "strength": 1.0,
                    "parameters": {"mode": "donor_source_patch"},
                },
                "schedule": {
                    "policy_calls": [0],
                    "generation_steps": "all",
                    "tokens": "target_tokens",
                },
                "outcome": {"kind": "action", "basis": ["raw"], "horizon": "full_chunk"},
                "controls": controls,
            }
        },
    }


def _recipient_trace_id(pair: Mapping[str, Any]) -> str:
    recipient = _mapping(pair.get("recipient"))
    policy_call = _mapping(recipient.get("policy_call"))
    trace = _mapping(recipient.get("trace"))
    trace_id = str(policy_call.get("trace_id") or trace.get("trace_id") or "").strip()
    if not trace_id:
        raise ValueError("pair recipient requires trace_id")
    return trace_id


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _csv_strings(value: str | None) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value or "").split(",") if item.strip())


def _csv_ints(value: str | None) -> tuple[int, ...]:
    return tuple(int(item) for item in _csv_strings(value))


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
