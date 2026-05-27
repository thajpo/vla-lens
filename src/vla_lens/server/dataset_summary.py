"""Top-level dataset summary response helpers."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from vla_lens.server.artifacts import _artifact_summary
from vla_lens.server.common import _json_scalar, _string_list
from vla_lens.server.metrics import _manifest_payload
from vla_lens.traces import TraceDataset
from vla_lens.workbench import workbench_manifest


def _dataset_payload(dataset: TraceDataset, *, include_workbench: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "root": str(dataset.root),
        "episodes": [_manifest_payload(bundle) for bundle in dataset.bundles],
        "counterfactual_pairs": _counterfactual_pairs_payload(dataset),
        "capabilities": _dataset_capabilities(dataset),
    }
    if include_workbench:
        workbench = workbench_manifest(dataset)
        payload.update(
            {
                "activation_sites": int(len(dataset.model_site_index)),
                "artifacts": _artifact_summary(dataset),
                "workbench": workbench,
            }
        )
    else:
        payload.update(
            {
                "activation_sites": int(len(dataset.model_site_index)),
                "artifacts": {"total": 0, "counts": {}},
            }
        )
    return payload


def _dataset_capabilities(dataset: TraceDataset) -> dict[str, Any]:
    model_sites = dataset.model_site_index
    array_names = {
        str(name)
        for bundle in dataset.bundles
        for name in bundle.array_index.get("name", [])
        if str(name).strip()
    }
    token_space_rows = sum(len(bundle.token_spaces) for bundle in dataset.bundles)
    artifact_counts = _artifact_summary(dataset)["counts"]
    camera_names = sorted({camera for bundle in dataset.bundles for camera in bundle.cameras()})
    capabilities = {
        "robot_episodes": bool(dataset.bundles),
        "cameras": bool(camera_names),
        "policy_calls": any(not bundle.policy_calls.empty for bundle in dataset.bundles),
        "model_sites": not model_sites.empty,
        "token_spaces": token_space_rows > 0,
        "image_token_maps": _has_token_kind(model_sites, "image_patch"),
        "attention_maps": _has_tensor_type(model_sites, "attention"),
        "action_chunks": "action_chunks" in array_names,
        "action_generation": "generation_actions" in array_names,
        "architecture_graph": not model_sites.empty,
        "probe_artifacts": artifact_counts.get("probe_suite", 0) > 0,
        "intervention_artifacts": artifact_counts.get("intervention_run", 0) > 0,
    }
    model_families = sorted(
        {
            str(row.get("model_family"))
            for bundle in dataset.bundles
            for row in bundle.policy_calls.to_dict("records")
            if row.get("model_family")
        }
    )
    model_site_prefixes = sorted(
        {
            str(name).split(".", 1)[0]
            for name in model_sites.get("name", [])
            if str(name).strip()
        }
    )
    return {
        "available": sorted(name for name, available in capabilities.items() if available),
        "flags": capabilities,
        "camera_names": camera_names,
        "model_families": model_families,
        "model_site_prefixes": model_site_prefixes,
    }


def _has_tensor_type(model_sites: pd.DataFrame, tensor_type: str) -> bool:
    return "tensor_type" in model_sites and bool(
        (model_sites["tensor_type"].astype(str) == tensor_type).any()
    )


def _has_token_kind(model_sites: pd.DataFrame, token_kind: str) -> bool:
    return "token_kind" in model_sites and bool(
        (model_sites["token_kind"].astype(str) == token_kind).any()
    )


def _counterfactual_pairs_response(dataset: TraceDataset) -> dict[str, Any]:
    pairs = _counterfactual_pairs_payload(dataset)
    return {"pairs": pairs, "count": len(pairs)}


def _counterfactual_pairs_payload(dataset: TraceDataset) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    group_metadata: dict[str, dict[str, Any]] = {}
    for bundle in dataset.bundles:
        metadata = dict(bundle.manifest.metadata or {})
        counterfactual = _counterfactual_metadata_from_manifest(metadata)
        group_id = str(counterfactual.get("group_id") or "").strip()
        if not group_id:
            continue
        member = {
            "trace_id": bundle.manifest.trace_id,
            "episode_id": bundle.manifest.episode_id,
            "role": str(counterfactual.get("role") or ""),
            "pair_index": _json_scalar(counterfactual.get("pair_index")),
            "paired_trace_id": str(counterfactual.get("paired_trace_id") or ""),
            "target_object_id": str(counterfactual.get("target_object_id") or ""),
            "counterfactual_target_object_id": str(
                counterfactual.get("counterfactual_target_object_id") or ""
            ),
            "outcome": bundle.manifest.outcome,
            "prompt": bundle.manifest.prompt,
        }
        grouped.setdefault(group_id, []).append(member)
        group_metadata.setdefault(
            group_id,
            {
                "group_id": group_id,
                "type": str(counterfactual.get("type") or ""),
                "changed_fields": _string_list(counterfactual.get("changed_fields")),
                "matched_fields": _string_list(counterfactual.get("matched_fields")),
            },
        )
    pairs: list[dict[str, Any]] = []
    for group_id, members in grouped.items():
        members.sort(key=_counterfactual_member_sort_key)
        pairs.append({**group_metadata[group_id], "members": members})
    pairs.sort(key=lambda pair: str(pair.get("group_id") or ""))
    return pairs


def _counterfactual_metadata_from_manifest(metadata: Mapping[str, Any]) -> dict[str, Any]:
    nested = metadata.get("counterfactual")
    counterfactual = dict(nested) if isinstance(nested, Mapping) else {}
    aliases = {
        "counterfactual_group_id": "group_id",
        "counterfactual_role": "role",
        "counterfactual_type": "type",
        "paired_trace_id": "paired_trace_id",
        "pair_index": "pair_index",
        "changed_fields": "changed_fields",
        "matched_fields": "matched_fields",
        "target_object_id": "target_object_id",
        "counterfactual_target_object_id": "counterfactual_target_object_id",
    }
    for source, target in aliases.items():
        if target not in counterfactual and source in metadata:
            counterfactual[target] = metadata[source]
    return counterfactual


def _counterfactual_member_sort_key(member: Mapping[str, Any]) -> tuple[int, int, str]:
    role_order = {"clean": 0, "control": 1, "corrupt": 2, "intervention": 3}
    role = str(member.get("role") or "")
    try:
        pair_index = int(member.get("pair_index"))
    except (TypeError, ValueError):
        pair_index = 10_000
    return (pair_index, role_order.get(role, 100), str(member.get("trace_id") or ""))
