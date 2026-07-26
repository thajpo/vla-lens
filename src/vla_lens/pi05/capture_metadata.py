"""PI0.5 capture metadata helpers."""

from __future__ import annotations

import argparse
import json
import re
from typing import Any


def _capture_design_metadata(args: argparse.Namespace) -> dict[str, Any]:
    capture_design = str(args.capture_design or "single_trace")
    trace_variant = _trace_variant_suffix(
        str(args.trace_variant or "") or str(args.counterfactual_role or "")
    )
    counterfactual = _counterfactual_metadata(args)
    metadata: dict[str, Any] = {"capture_design": capture_design}
    if trace_variant:
        metadata["trace_variant"] = trace_variant
    if counterfactual:
        metadata["counterfactual"] = counterfactual
        metadata.update(
            {
                "counterfactual_group_id": counterfactual.get("group_id", ""),
                "counterfactual_role": counterfactual.get("role", ""),
                "counterfactual_type": counterfactual.get("type", ""),
                "pair_index": counterfactual.get("pair_index"),
                "paired_trace_id": counterfactual.get("paired_trace_id", ""),
                "changed_fields": counterfactual.get("changed_fields", []),
                "matched_fields": counterfactual.get("matched_fields", []),
                "target_object_id": counterfactual.get("target_object_id", ""),
                "counterfactual_target_object_id": counterfactual.get(
                    "counterfactual_target_object_id",
                    "",
                ),
            }
        )
    return metadata


def trial_runtime_metadata(args: argparse.Namespace, *, legacy_seed: int) -> dict[str, Any]:
    return {
        "trial_id": str(args.trial_id or ""),
        "child_plan_id": str(args.child_plan_id or ""),
        "canonical_family_id": str(args.canonical_family_id or ""),
        "pool": str(args.pool or ""),
        "replicate_id": str(args.replicate_id or ""),
        "seed_identities": {
            "layout": int(args.layout_seed if args.layout_seed is not None else legacy_seed),
            "reset": int(args.reset_seed if args.reset_seed is not None else legacy_seed),
            "environment": int(
                args.environment_seed if args.environment_seed is not None else legacy_seed
            ),
            "policy": int(args.policy_seed if args.policy_seed is not None else legacy_seed),
            "flow_noise": int(
                args.flow_noise_seed if args.flow_noise_seed is not None else legacy_seed
            ),
        },
        "layout_id": args.layout_id,
    }

def _capture_design_request_metadata(args: argparse.Namespace) -> dict[str, Any]:
    metadata = _capture_design_metadata(args)
    metadata.pop("capture_design", None)
    return metadata

def _counterfactual_metadata(args: argparse.Namespace) -> dict[str, Any]:
    group_id = str(args.counterfactual_group_id or "").strip()
    role = _trace_variant_suffix(str(args.counterfactual_role or ""))
    counterfactual_type = str(args.counterfactual_type or "").strip()
    paired_trace_id = str(args.paired_trace_id or "").strip()
    changed_fields = _parse_list_argument(args.changed_fields)
    matched_fields = _parse_list_argument(args.matched_fields)
    target_object_id = str(args.target_object_id or "").strip()
    counterfactual_target_object_id = str(args.counterfactual_target_object_id or "").strip()
    if not any(
        (
            group_id,
            role,
            counterfactual_type,
            paired_trace_id,
            changed_fields,
            matched_fields,
            target_object_id,
            counterfactual_target_object_id,
        )
    ):
        return {}
    return {
        "group_id": group_id,
        "role": role,
        "type": counterfactual_type,
        "pair_index": args.pair_index,
        "paired_trace_id": paired_trace_id,
        "changed_fields": list(changed_fields),
        "matched_fields": list(matched_fields),
        "target_object_id": target_object_id,
        "counterfactual_target_object_id": counterfactual_target_object_id,
    }

def _parse_list_argument(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    text = str(value).strip()
    if not text:
        return ()
    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected list argument, got {value!r}")
        return tuple(str(item).strip() for item in parsed if str(item).strip())
    return tuple(item.strip() for item in text.split(",") if item.strip())


def _trace_variant_suffix(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")
