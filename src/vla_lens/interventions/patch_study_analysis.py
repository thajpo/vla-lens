"""Cohort summaries and confidence intervals for saved patch studies."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from vla_lens.interventions.counterfactuals import counterfactual_action_metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_root", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args(argv)
    payload = save_patch_study_analysis(
        args.study_root,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(_compact_report(payload), indent=2, sort_keys=True))


def save_patch_study_analysis(
    study_root: Path,
    *,
    bootstrap_samples: int = 20_000,
    seed: int = 20260725,
) -> dict[str, Any]:
    """Recompute a compact cohort view from reconstructable run artifacts."""
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    artifact = _read_json(study_root / "artifact.json")
    records, controls = _run_records(study_root)
    summaries = summarize_patch_records(
        records,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    control_summaries = summarize_patch_records(
        controls,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        group_columns=("layer", "token_region", "control_kind"),
    )
    specificity = _specificity_summary(summaries, control_summaries)
    study = _mapping(artifact.get("study"))
    pairs = [
        {
            "pair_id": pair.get("pair_id"),
            "recipient_trace_id": _pair_trace_id(pair, "recipient"),
            "donor_trace_id": _pair_trace_id(pair, "donor"),
            "target_object": _mapping(pair.get("recipe")).get("target_object"),
            "distractor_object": _mapping(pair.get("recipe")).get("distractor_object"),
            "media": dict(_mapping(pair.get("media"))),
        }
        for pair in artifact.get("pairs") or ()
        if isinstance(pair, Mapping)
    ]
    payload = {
        "schema_kind": "vla_lens.patch_study_analysis",
        "schema_version": 1,
        "study_id": study.get("study_id") or study_root.name,
        "question": study.get("question"),
        "hypothesis": study.get("hypothesis"),
        "phase": _mapping(study.get("axes")).get("phase"),
        "status": "completed",
        "pair_count": len(pairs),
        "planned_trial_count": len(records),
        "controls": list(study.get("controls") or ()),
        "layers": sorted({int(record["layer"]) for record in records}),
        "token_regions": list(
            dict.fromkeys(str(record["token_region"]) for record in records)
        ),
        "summary": summaries,
        "control_summary": control_summaries,
        "specificity": specificity,
        "pairs": pairs,
        "bootstrap": {
            "method": "percentile bootstrap over counterfactual pairs",
            "samples": bootstrap_samples,
            "confidence": 0.95,
            "seed": seed,
            "note": "Five-pair intervals are exploratory, not population guarantees.",
        },
        "headline": _headline(summaries),
        "files": {
            "records": "analysis_records.parquet",
            "summary": "cohort_summary.parquet",
            "controls": "control_summary.parquet",
        },
    }
    _write_json_atomic(study_root / "analysis.json", payload)
    pd.DataFrame(records).to_parquet(study_root / "analysis_records.parquet", index=False)
    pd.DataFrame(summaries).to_parquet(study_root / "cohort_summary.parquet", index=False)
    pd.DataFrame(control_summaries).to_parquet(
        study_root / "control_summary.parquet", index=False
    )
    return payload


def summarize_patch_records(
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
    group_columns: tuple[str, ...] = ("layer", "token_region"),
) -> list[dict[str, Any]]:
    if not records:
        return []
    frame = pd.DataFrame(records)
    rng = np.random.default_rng(seed)
    summaries: list[dict[str, Any]] = []
    grouper: str | list[str] = (
        group_columns[0] if len(group_columns) == 1 else list(group_columns)
    )
    for key, group in frame.groupby(grouper, sort=True, dropna=False):
        keys = (key,) if len(group_columns) == 1 else tuple(key)
        values = group["transfer_fraction"].astype(float).to_numpy()
        direction = group["direction_agreement"].astype(float).to_numpy()
        recovery = group["donor_recovery"].astype(float).to_numpy()
        low, high = _bootstrap_mean_interval(values, bootstrap_samples, rng)
        summary = {
            column: _json_scalar(value)
            for column, value in zip(group_columns, keys, strict=True)
        }
        summary.update(
            {
                "pair_count": int(len(group)),
                "transfer_mean": float(np.mean(values)),
                "transfer_ci95_low": low,
                "transfer_ci95_high": high,
                "transfer_min": float(np.min(values)),
                "transfer_max": float(np.max(values)),
                "positive_pair_count": int(np.sum(values > 0.0)),
                "direction_agreement_mean": float(np.mean(direction)),
                "donor_recovery_mean": float(np.mean(recovery)),
                "patch_delta_l2_mean": float(
                    group["patch_delta_norm"].astype(float).mean()
                ),
                "natural_delta_l2_mean": float(
                    group["natural_delta_norm"].astype(float).mean()
                ),
                "localized_pair_count": int(
                    (group.get("verdict", pd.Series(dtype=str)) == "localized_transfer").sum()
                ),
            }
        )
        summaries.append(summary)
    return summaries


def _run_records(study_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    action_group = None
    for path in sorted((study_root / "runs").glob("*.json")):
        payload = _read_json(path)
        planned = _mapping(payload.get("planned_trial"))
        run = _mapping(payload.get("run"))
        transfer = _mapping(_mapping(run.get("display")).get("counterfactual_transfer"))
        metrics = _mapping(transfer.get("metrics"))
        decision = _mapping(transfer.get("decision"))
        record = {
            "run_id": run.get("run_id"),
            "pair_id": planned.get("pair_id"),
            "layer": int(planned.get("layer") or 0),
            "token_region": planned.get("token_region"),
            "verdict": decision.get("verdict"),
            **_metric_record(metrics),
        }
        records.append(record)
        patch_trials = payload.get("patch_trials") or ()
        if not any(_mapping(trial).get("control_kind") for trial in patch_trials):
            continue
        if action_group is None:
            import zarr

            action_group = zarr.open_group(str(study_root / "actions.zarr"), mode="r")
        actions: dict[str, np.ndarray] = {}
        for raw_trial in patch_trials:
            trial = _mapping(raw_trial)
            action = _mapping(trial.get("action"))
            array_ref = str(action.get("array_ref") or "")
            if "actions.zarr/" not in array_ref:
                continue
            key = str(trial.get("control_kind") or trial.get("trial_kind"))
            actions[key] = np.asarray(
                action_group[array_ref.split("actions.zarr/", 1)[1]], dtype=np.float32
            )
        if not {"recipient", "donor"} <= set(actions):
            continue
        for control_kind, action in actions.items():
            if control_kind in {"recipient", "donor", "patched"}:
                continue
            control_metrics = counterfactual_action_metrics(
                actions["recipient"], actions["donor"], action
            )
            controls.append(
                {
                    "run_id": run.get("run_id"),
                    "pair_id": planned.get("pair_id"),
                    "layer": int(planned.get("layer") or 0),
                    "token_region": planned.get("token_region"),
                    "control_kind": control_kind,
                    "verdict": "control",
                    **_metric_record(control_metrics.to_dict()),
                }
            )
    return records, controls


def _metric_record(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        name: float(metrics.get(name) or 0.0)
        for name in (
            "natural_delta_norm",
            "patch_delta_norm",
            "direction_agreement",
            "transfer_fraction",
            "donor_gap_remaining",
            "donor_recovery",
            "off_direction_norm",
            "off_direction_fraction",
        )
    }


def _bootstrap_mean_interval(
    values: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    indexes = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indexes].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def _headline(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not summaries:
        return {}
    best = max(summaries, key=lambda item: float(item.get("transfer_mean") or 0.0))
    return {
        "best_layer": best.get("layer"),
        "best_token_region": best.get("token_region"),
        "best_transfer_mean": best.get("transfer_mean"),
        "best_transfer_ci95": [
            best.get("transfer_ci95_low"),
            best.get("transfer_ci95_high"),
        ],
    }


def _specificity_summary(
    summaries: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    negative_controls = [
        item
        for item in controls
        if item.get("control_kind")
        not in {"recipient_self_patch", "donor_self_patch"}
    ]
    out: list[dict[str, Any]] = []
    for summary in summaries:
        matching = [
            item
            for item in negative_controls
            if item.get("layer") == summary.get("layer")
            and item.get("token_region") == summary.get("token_region")
        ]
        if not matching:
            continue
        strongest = max(matching, key=lambda item: float(item["transfer_mean"]))
        main = float(summary["transfer_mean"])
        control = float(strongest["transfer_mean"])
        out.append(
            {
                "layer": summary.get("layer"),
                "token_region": summary.get("token_region"),
                "main_transfer_mean": main,
                "strongest_control_kind": strongest.get("control_kind"),
                "strongest_control_transfer_mean": control,
                "specificity_margin": main - control,
                "main_beats_control": main > control,
            }
        )
    return out


def _compact_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: payload.get(key)
        for key in (
            "study_id",
            "phase",
            "pair_count",
            "planned_trial_count",
            "headline",
        )
    }


def _pair_trace_id(pair: Mapping[str, Any], role: str) -> str | None:
    spec = _mapping(pair.get(role))
    policy_call = _mapping(spec.get("policy_call"))
    trace = _mapping(spec.get("trace"))
    value = policy_call.get("trace_id") or trace.get("trace_id")
    return str(value) if value else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _json_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


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


__all__ = ["save_patch_study_analysis", "summarize_patch_records"]


if __name__ == "__main__":
    main()
