"""Build staged patch-study jobs from captured pose-exchange pair manifests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_LAYERS = (0, 4, 8, 12, 16)
DEFAULT_EXPERT_LAYERS = (*DEFAULT_LAYERS, 17)
DEFAULT_REGIONS = ("target", "distractor", "both", "complement")
VLM_REGIONS = (
    *DEFAULT_REGIONS,
    "main_camera",
    "wrist_camera",
    "active_images",
    "language_active",
    "full_prefix",
)
EXPERT_REGIONS = (
    "action_all",
    "action_first_10",
    "action_middle_10",
    "action_last_10",
)
VLM_CONFIRMATION_CONTROLS = (
    "recipient_self_patch",
    "donor_self_patch",
    "shuffled_donor",
    "random_matched_norm",
    "wrong_region",
)
EXPERT_CONFIRMATION_CONTROLS = (
    "recipient_self_patch",
    "donor_self_patch",
    "alpha_zero",
    "shuffled_donor",
    "random_matched_norm",
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
        stream=args.stream,
        pair_ids=_csv_strings(args.pair_ids) or None,
        layers=_csv_ints(args.layers) or None,
        token_regions=_csv_strings(args.token_regions) or None,
        generation_steps=_generation_step_selector(args.generation_steps),
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
    parser.add_argument(
        "--stream",
        choices=("vlm_prefix", "expert_action"),
        default="vlm_prefix",
    )
    parser.add_argument("--pair-ids", help="Comma-separated subset; default is all valid pairs")
    parser.add_argument("--layers", help="Comma-separated layers; defaults depend on stream")
    parser.add_argument(
        "--token-regions",
        help="Comma-separated scopes; defaults to object regions or action_all by stream",
    )
    parser.add_argument(
        "--generation-steps",
        default="all",
        help="all, comma-separated denoising steps, or a half-open range such as 0:5",
    )
    parser.add_argument("--control-seed", type=int, default=20260725)
    return parser.parse_args(argv)


def build_pose_exchange_study_job(
    collection: Mapping[str, Any],
    *,
    study_id: str,
    phase: str,
    stream: str = "vlm_prefix",
    pair_ids: Sequence[str] | None = None,
    layers: Sequence[int] | None = None,
    token_regions: Sequence[str] | None = None,
    generation_steps: str | Mapping[str, Any] = "all",
    control_seed: int = 20260725,
) -> dict[str, Any]:
    """Turn saved pair evidence into a deterministic localization/confirmation job."""
    if phase not in {"localization", "confirmation"}:
        raise ValueError("phase must be localization or confirmation")
    if stream not in {"vlm_prefix", "expert_action"}:
        raise ValueError("stream must be vlm_prefix or expert_action")
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

    default_layers = DEFAULT_LAYERS if stream == "vlm_prefix" else DEFAULT_EXPERT_LAYERS
    selected_layers = tuple(
        dict.fromkeys(int(layer) for layer in (layers or default_layers))
    )
    if not selected_layers or any(layer < 0 for layer in selected_layers):
        raise ValueError("study layers must be unique non-negative integers")
    default_regions = DEFAULT_REGIONS if stream == "vlm_prefix" else ("action_all",)
    regions = tuple(
        dict.fromkeys(str(region) for region in (token_regions or default_regions))
    )
    supported_regions = VLM_REGIONS if stream == "vlm_prefix" else EXPERT_REGIONS
    if not regions or any(region not in supported_regions for region in regions):
        raise ValueError(f"{stream} token regions must come from {supported_regions}")

    if phase == "confirmation":
        controls = (
            VLM_CONFIRMATION_CONTROLS
            if stream == "vlm_prefix"
            else EXPERT_CONFIRMATION_CONTROLS
        )
    else:
        controls = ()
    shared_noise_refs = [
        f"{_recipient_trace_id(pair)}.flow_initial_noise[0]"
        for pair in selected_pairs
    ]
    axes: dict[str, Any] = {
        "phase": phase,
        "stream": stream,
        "token_regions": list(regions),
        "generation_steps": _normalize_generation_steps(generation_steps),
    }
    if controls and stream == "vlm_prefix":
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
            "question": _question(stream),
            "hypothesis": _hypothesis(stream),
            "pair_ids": list(selected_ids),
            "sites": [_site_record(stream, layer) for layer in selected_layers],
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
                (
                    "Image-patch regions approximate object pixels and include nearby context."
                    if stream == "vlm_prefix"
                    else "Action positions are horizon slots, not independent semantic tokens."
                ),
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
                "stream": stream,
            },
        },
        "pairs": selected_pairs,
        "request_template": _request_template(
            phase=phase,
            stream=stream,
            generation_steps=_normalize_generation_steps(generation_steps),
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
        "stream": _mapping(study.get("axes")).get("stream"),
        "pair_count": pair_count,
        "site_count": site_count,
        "region_count": region_count,
        "planned_trial_count": pair_count * site_count * region_count,
        "model_calls_per_trial": 1 + controls_per_trial,
    }


def _request_template(
    *,
    phase: str,
    stream: str,
    generation_steps: str | Mapping[str, Any],
    control_seed: int,
) -> dict[str, Any]:
    controls = []
    if phase == "confirmation":
        controls = [
            {"kind": "recipient_self_patch"},
            {"kind": "donor_self_patch"},
        ]
        if stream == "expert_action":
            controls.append({"kind": "alpha_zero"})
        controls.extend(
            [
                {"kind": "shuffled_donor", "parameters": {"seed": control_seed}},
                {"kind": "random_matched_norm", "parameters": {"seed": control_seed}},
            ]
        )
        if stream == "vlm_prefix":
            controls.append({"kind": "wrong_region"})
    token_space = "pi05.prefix" if stream == "vlm_prefix" else "pi05.action_suffix"
    return {
        "runtime_adapter": "pi05",
        "target": {
            "kind": "activation_slice",
            "model_id": "lerobot/pi05_libero_finetuned",
            "model_family": "pi05",
            "model_site": "placeholder",
            "tensor_type": "hidden_tokens",
            "token_space": token_space,
            "metadata": {
                "research_question_id": "RQ-020" if stream == "vlm_prefix" else "RQ-022",
                "phase": phase,
                "stream": stream,
            },
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
                    "generation_steps": generation_steps,
                    "tokens": "target_tokens",
                },
                "outcome": {"kind": "action", "basis": ["raw"], "horizon": "full_chunk"},
                "controls": controls,
            }
        },
    }


def _site_record(stream: str, layer: int) -> dict[str, Any]:
    if stream == "vlm_prefix":
        model_site = f"pi05.vlm.layers.{layer}.prefix.hidden_tokens"
    else:
        model_site = f"pi05.expert.layers.{layer}.by_step.hidden_tokens"
    return {"layer": layer, "model_site": model_site}


def _question(stream: str) -> str:
    if stream == "expert_action":
        return (
            "Where does PI0.5's action expert carry the scene-driven action difference "
            "after it leaves the VLM prefix?"
        )
    return (
        "Where do PI0.5 prefix states causally carry the action-relevant difference "
        "produced by exchanging the book and mug poses?"
    )


def _hypothesis(stream: str) -> str:
    if stream == "expert_action":
        return (
            "Later expert layers should increasingly transfer the donor action when all "
            "50 action positions are patched at matching denoising steps."
        )
    return (
        "Patching object-aligned donor tokens should move the recipient action toward "
        "the donor action more than background or matched controls."
    )


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


def _generation_step_selector(value: str) -> str | dict[str, Any]:
    text = str(value).strip().lower()
    if text == "all":
        return "all"
    if ":" in text:
        parts = text.split(":")
        if len(parts) != 2 or not all(part.strip() for part in parts):
            raise ValueError("generation-step range must be start:end")
        return _normalize_generation_steps(
            {"start": int(parts[0]), "end": int(parts[1])}
        )
    return _normalize_generation_steps({"indices": list(_csv_ints(text))})


def _normalize_generation_steps(
    value: str | Mapping[str, Any],
) -> str | dict[str, Any]:
    if value == "all":
        return "all"
    selector = dict(_mapping(value))
    if "indices" in selector:
        indices = tuple(dict.fromkeys(int(item) for item in selector["indices"]))
        if not indices or any(item < 0 for item in indices):
            raise ValueError("generation-step indices must be unique non-negative integers")
        return {"indices": list(indices)}
    if "start" in selector or "end" in selector:
        start = int(selector.get("start", 0))
        end = int(selector.get("end", start))
        if start < 0 or end <= start:
            raise ValueError("generation-step range must satisfy 0 <= start < end")
        return {"start": start, "end": end}
    raise ValueError("generation steps must be all, indices, or a start:end range")


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
