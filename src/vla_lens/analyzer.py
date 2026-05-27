"""Dataset diagnostics for choosing defensible interpretability analyses."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from vla_lens.artifacts import LensArtifact
from vla_lens.traces import TraceDataset

DIAGNOSTIC_TYPE = "dataset_diagnostics"


def dataset_fingerprint(dataset: TraceDataset) -> str:
    """Fingerprint trace contents that should invalidate dataset guidance."""
    episodes = _records(
        dataset.episode_index,
        [
            "trace_id",
            "episode_id",
            "task_id",
            "prompt",
            "model_id",
            "env_id",
            "robot_id",
            "outcome",
            "length",
            "seed",
            "scene_id",
            "layout_id",
            "benchmark",
            "target_object",
            "object",
        ],
    )
    model_sites = _records(
        dataset.model_site_index,
        [
            "trace_id",
            "name",
            "module",
            "layer",
            "tensor_type",
            "token_kind",
            "generation_step",
            "family",
            "role",
            "segment",
            "materialization",
            "exactness",
            "token_space_id",
            "query_token_space_id",
            "key_token_space_id",
            "parent_site_id",
            "summary_type",
            "shape",
            "axes",
            "metadata",
        ],
    )
    timesteps = _timestep_summary(dataset)
    payload = {"episodes": episodes, "model_sites": model_sites, "timesteps": timesteps}
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def diagnostics_status(dataset: TraceDataset) -> dict[str, Any]:
    fingerprint = dataset_fingerprint(dataset)
    latest = latest_diagnostics_artifact(dataset)
    latest_fingerprint = (latest.metrics or {}).get("dataset_fingerprint") if latest else None
    return {
        "fingerprint": fingerprint,
        "stale": latest_fingerprint != fingerprint,
        "latest": latest.to_dict() if latest else None,
    }


def run_dataset_diagnostics(dataset: TraceDataset) -> LensArtifact:
    """Analyze dataset affordances and save a reusable diagnostics artifact."""
    fingerprint = dataset_fingerprint(dataset)
    episode_index = dataset.episode_index.copy()
    model_sites = dataset.model_site_index.copy()
    factors = _factor_coverage(episode_index)
    splits = _split_feasibility(episode_index)
    modalities = _available_modalities(dataset, model_sites)
    recommendations = _recommendations(episode_index, modalities, splits)
    avoid = _avoid_or_delay(episode_index, modalities, splits)
    warnings = _warnings(episode_index, factors, splits)
    key_stats = _key_stats(episode_index, modalities)
    summary = _summary(episode_index, splits, warnings, key_stats)

    artifact = LensArtifact(
        artifact_id=f"{DIAGNOSTIC_TYPE}-{fingerprint}",
        artifact_type=DIAGNOSTIC_TYPE,
        name="Dataset diagnostics",
        group_id="dataset_diagnostics",
        scope="dataset",
        selector={"episodes": "all", "fingerprint_source": "episodes+model_sites+timesteps"},
        method={"workflow": "run_dataset_diagnostics", "version": 1},
        metrics={
            "dataset_fingerprint": fingerprint,
            "episode_count": int(len(episode_index)),
            "activation_site_count": int(len(model_sites)),
            "warning_count": int(len(warnings)),
            "recommended_count": int(len(recommendations)),
        },
        display={
            "summary": summary,
            "key_stats": key_stats,
            "factor_coverage": factors,
            "split_feasibility": splits,
            "available_modalities": modalities,
            "recommended_artifacts": recommendations,
            "avoid_or_delay": avoid,
            "warnings": warnings,
        },
        tags=("diagnostics", "dataset"),
        source_trace_ids=tuple(sorted(str(value) for value in episode_index["trace_id"])),
    )
    return dataset.save_artifact(artifact)


def latest_diagnostics_artifact(dataset: TraceDataset) -> LensArtifact | None:
    table = dataset.artifact_index
    if table.empty or "artifact_type" not in table:
        return None
    matches = table.loc[table["artifact_type"].astype(str) == DIAGNOSTIC_TYPE].copy()
    if matches.empty:
        return None
    matches = matches.sort_values("created_utc", ascending=False, na_position="last")
    for artifact_id in matches["artifact_id"].astype(str):
        try:
            return dataset.load_artifact(artifact_id)
        except (FileNotFoundError, KeyError):
            continue
    return None


def _summary(
    episode_index: pd.DataFrame,
    splits: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    key_stats: list[dict[str, Any]],
) -> str:
    episode_count = len(episode_index)
    usable = [item["name"] for item in splits if item["status"] in {"recommended", "usable"}]
    outcome = next((item for item in key_stats if item["key"] == "outcome_balance"), None)
    outcome_text = f" Outcome balance: {outcome['value']}." if outcome else ""
    if usable:
        return (
            f"{episode_count} episodes. Best-supported split ideas: {', '.join(usable[:2])}."
            f"{outcome_text} {len(warnings)} design issue(s) need attention."
        )
    return (
        f"{episode_count} episodes. Most analyses need caution."
        f"{outcome_text} {len(warnings)} design issue(s) need attention."
    )


def _key_stats(frame: pd.DataFrame, modalities: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _outcome_stat(frame),
        _task_repetition_stat(frame),
        _seed_stat(frame),
        _call_alignment_stat(frame, modalities),
    ]


def _outcome_stat(frame: pd.DataFrame) -> dict[str, Any]:
    if "outcome" not in frame or frame.empty:
        return {
            "key": "outcome_balance",
            "label": "Outcome balance",
            "value": "missing",
            "detail": "No outcome column found.",
            "status": "caution",
        }
    counts = frame["outcome"].astype(str).value_counts()
    total = int(counts.sum())
    parts = [
        f"{key} {int(value)} ({int(round(100 * int(value) / total))}%)"
        for key, value in counts.items()
    ]
    status = "caution" if _is_imbalanced(frame["outcome"]) else "good"
    return {
        "key": "outcome_balance",
        "label": "Outcome balance",
        "value": " / ".join(parts),
        "detail": "Any outcome predictor should beat always guessing the largest group.",
        "status": status,
    }


def _task_repetition_stat(frame: pd.DataFrame) -> dict[str, Any]:
    if "task_id" not in frame or frame.empty:
        return {
            "key": "task_repetition",
            "label": "Repeated tasks",
            "value": "missing",
            "detail": "No task_id column found.",
            "status": "caution",
        }
    counts = frame["task_id"].astype(str).value_counts()
    repeated = int((counts > 1).sum())
    total = int(counts.shape[0])
    status = "good" if repeated else "bad"
    return {
        "key": "task_repetition",
        "label": "Repeated tasks",
        "value": f"{repeated}/{total}",
        "detail": "No repeated tasks means task identity can be confused with behavior.",
        "status": status,
    }


def _seed_stat(frame: pd.DataFrame) -> dict[str, Any]:
    if "seed" not in frame or frame.empty:
        return {
            "key": "seed_variation",
            "label": "Seed variation",
            "value": "missing",
            "detail": "No seed is recorded, so this dataset cannot test seed generalization.",
            "status": "bad",
        }
    seeds = frame["seed"].dropna().astype(str)
    unique = int(seeds.nunique())
    if "task_id" in frame:
        repeated_seed_tasks = int((frame.groupby("task_id")["seed"].nunique() > 1).sum())
        detail = f"{repeated_seed_tasks} task(s) have multiple seeds."
    else:
        repeated_seed_tasks = 0
        detail = "Cannot check whether seeds repeat within task."
    status = "good" if repeated_seed_tasks else "bad"
    return {
        "key": "seed_variation",
        "label": "Seed variation",
        "value": f"{unique} unique",
        "detail": detail,
        "status": status,
    }


def _call_alignment_stat(frame: pd.DataFrame, modalities: dict[str, Any]) -> dict[str, Any]:
    available = bool(modalities.get("model_calls", {}).get("available"))
    has_frames = bool(modalities.get("frames", {}).get("available"))
    status = "good" if available and has_frames else "caution"
    value = "yes" if available else "missing"
    return {
        "key": "call_alignment",
        "label": "Decision review",
        "value": value,
        "detail": "Frames can be matched to policy decisions, so videos and plots stay in sync."
        if has_frames
        else "Frames are missing, so visual review artifacts are unavailable.",
        "status": status,
    }


def _factor_coverage(frame: pd.DataFrame) -> list[dict[str, Any]]:
    candidates = [
        "task_id",
        "outcome",
        "seed",
        "scene_id",
        "layout_id",
        "benchmark",
        "target_object",
        "object",
        "model_id",
        "env_id",
    ]
    rows: list[dict[str, Any]] = []
    for column in candidates:
        if column not in frame:
            continue
        values = frame[column].dropna().astype(str)
        counts = values.value_counts(dropna=False)
        rows.append(
            {
                "factor": column,
                "observed": int(values.shape[0]),
                "unique": int(counts.shape[0]),
                "repeated": bool((counts > 1).any()),
                "top_values": [
                    {"value": str(key), "count": int(value)}
                    for key, value in counts.head(6).to_dict().items()
                ],
            }
        )
    return rows


def _split_feasibility(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = [
        _random_episode_split(frame),
        _seed_split(frame),
        _within_task_outcome_split(frame),
        _heldout_task_split(frame),
        _factor_holdout(frame, "scene_id", "Held-out scene/layout"),
        _factor_holdout(frame, "target_object", "Held-out object"),
        _temporal_split(frame),
    ]
    return rows


def _random_episode_split(frame: pd.DataFrame) -> dict[str, Any]:
    if len(frame) >= 8:
        status = "usable"
        reason = "Enough episodes for a smoke-test split, but the claim is weak."
    elif len(frame) >= 4:
        status = "caution"
        reason = "Very small sample; use only as a smoke test."
    else:
        status = "avoid"
        reason = "Too few episodes for a meaningful holdout."
    return {
        "key": "random_episode",
        "name": "Random episode",
        "status": status,
        "claim": "Checks whether a probe has any obvious signal.",
        "reason": reason,
    }


def _seed_split(frame: pd.DataFrame) -> dict[str, Any]:
    if "seed" not in frame:
        return _split("seed_holdout", "Held-out seed", "unavailable", "No seed column exists.")
    seed_counts = frame["seed"].dropna().astype(str).value_counts()
    if seed_counts.empty or seed_counts.shape[0] <= 1:
        return _split(
            "seed_holdout",
            "Held-out seed",
            "avoid",
            "Only one seed value is present, so a seed holdout is uninformative.",
        )
    tasks_per_seed = frame.groupby("seed")["task_id"].nunique() if "task_id" in frame else None
    seeds_per_task = frame.groupby("task_id")["seed"].nunique() if "task_id" in frame else None
    if seeds_per_task is not None and int((seeds_per_task > 1).sum()) == 0:
        return _split(
            "seed_holdout",
            "Held-out seed",
            "avoid",
            "No task appears under multiple seeds; seed split would mostly waste time.",
        )
    if tasks_per_seed is not None and int((tasks_per_seed > 1).sum()) > 0:
        return _split(
            "seed_holdout",
            "Held-out seed",
            "usable",
            "Seeds cross multiple tasks, so a seed holdout can test initial-condition robustness.",
        )
    return _split(
        "seed_holdout",
        "Held-out seed",
        "caution",
        "Multiple seeds exist, but crossing is limited. Inspect confounds first.",
    )


def _within_task_outcome_split(frame: pd.DataFrame) -> dict[str, Any]:
    if not {"task_id", "outcome"}.issubset(frame.columns):
        return _split(
            "within_task_outcome",
            "Within-task outcome",
            "unavailable",
            "Requires task_id and outcome columns.",
        )
    outcome_counts = frame.groupby("task_id")["outcome"].nunique()
    mixed_tasks = int((outcome_counts > 1).sum())
    if mixed_tasks > 0:
        return _split(
            "within_task_outcome",
            "Within-task success/failure",
            "recommended",
            f"{mixed_tasks} task(s) contain both outcomes; use this to separate task identity "
            "from behavior.",
        )
    return _split(
        "within_task_outcome",
        "Within-task success/failure",
        "avoid",
        "No task has both success and failure, so outcome is confounded with task.",
    )


def _heldout_task_split(frame: pd.DataFrame) -> dict[str, Any]:
    if "task_id" not in frame:
        return _split("task_holdout", "Held-out task", "unavailable", "No task_id column exists.")
    unique_tasks = int(frame["task_id"].dropna().nunique())
    if unique_tasks >= 5:
        return _split(
            "task_holdout",
            "Held-out task",
            "usable",
            "Enough task diversity to test transfer across instructions.",
        )
    if unique_tasks >= 2:
        return _split(
            "task_holdout",
            "Held-out task",
            "caution",
            "Possible, but too few tasks for stable estimates.",
        )
    return _split("task_holdout", "Held-out task", "avoid", "Only one task is present.")


def _factor_holdout(frame: pd.DataFrame, column: str, name: str) -> dict[str, Any]:
    if column not in frame:
        return _split(column, name, "unavailable", f"No {column} column exists.")
    counts = frame[column].dropna().astype(str).value_counts()
    if counts.shape[0] >= 3 and bool((counts > 1).any()):
        return _split(column, name, "usable", f"{column} has repeated values across episodes.")
    if counts.shape[0] >= 2:
        return _split(column, name, "caution", f"{column} varies but repetitions are sparse.")
    return _split(column, name, "avoid", f"{column} does not vary enough.")


def _temporal_split(frame: pd.DataFrame) -> dict[str, Any]:
    max_length = int(frame["length"].max()) if "length" in frame and len(frame) else 0
    if max_length >= 8:
        return _split(
            "temporal_holdout",
            "Temporal holdout",
            "usable",
            "Episodes are long enough to compare early vs late calls/timesteps.",
        )
    return _split(
        "temporal_holdout",
        "Temporal holdout",
        "caution",
        "Episodes are short; temporal probes may be noisy.",
    )


def _split(key: str, name: str, status: str, reason: str) -> dict[str, Any]:
    claims = {
        "recommended": "Strong fit for the current dataset.",
        "usable": "Supported, but interpret with ordinary probe caveats.",
        "caution": "Possible, but likely noisy or confounded.",
        "avoid": "Do not prioritize this for this dataset.",
        "unavailable": "Required metadata or arrays are missing.",
    }
    return {"key": key, "name": name, "status": status, "claim": claims[status], "reason": reason}


def _available_modalities(dataset: TraceDataset, model_sites: pd.DataFrame) -> dict[str, Any]:
    cameras = sorted({camera for bundle in dataset.bundles for camera in bundle.cameras()})
    arrays = dataset.bundles[0].array_index if dataset.bundles else pd.DataFrame()
    array_names = set(arrays["name"].astype(str)) if not arrays.empty else set()
    activation_names = " ".join(model_sites.get("name", pd.Series(dtype=str)).astype(str))
    modules = " ".join(model_sites.get("module", pd.Series(dtype=str)).astype(str))
    return {
        "frames": {"available": bool(cameras), "cameras": cameras},
        "executed_actions": {"available": bool({"action", "executed_actions"} & array_names)},
        "action": {"available": "action" in array_names},
        "action_chunks": {"available": "action_chunks" in array_names},
        "generation_actions": {"available": "generation_actions" in array_names},
        "vlm_hidden": {"available": "vlm" in modules or "vlm" in activation_names},
        "expert_hidden": {"available": "expert" in modules or "expert" in activation_names},
        "attention": {"available": "attn" in modules or "attention" in activation_names},
        "model_calls": {"available": _has_model_calls(dataset)},
    }


def _recommendations(
    frame: pd.DataFrame,
    modalities: dict[str, Any],
    splits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if frame.get("outcome") is not None and frame["outcome"].nunique() > 1:
        outcome_counts = _counts_text(frame["outcome"])
        task_counts = frame["task_id"].nunique() if "task_id" in frame else "unknown"
        rows.append(
            {
                "title": "Outcome-by-layer probe with task baseline",
                "why": (
                    f"Outcome varies ({outcome_counts}) across {task_counts} task(s); "
                    "compare against task metadata before trusting model features."
                ),
                "artifact_type": "probe_suite",
            }
        )
    if modalities["expert_hidden"]["available"]:
        rows.append(
            {
                "title": "Expert/action representation probes",
                "why": (
                    "Expert hidden states are present for action tokens, so you can inspect "
                    "which internal features move with planned actions."
                ),
                "artifact_type": "probe_suite",
            }
        )
    if modalities["generation_actions"]["available"]:
        rows.append(
            {
                "title": "Denoising commitment analysis",
                "why": (
                    "Denoising action arrays are present; action formation can be "
                    "tracked over steps."
                ),
                "artifact_type": "action_generation",
            }
        )
    if modalities["frames"]["available"] and modalities["model_calls"]["available"]:
        cameras = modalities["frames"].get("cameras", [])
        rows.append(
            {
                "title": "Policy-decision episode videos",
                "why": (
                    f"{len(cameras)} camera stream(s) can be matched to policy decisions; "
                    "compressed videos are useful review artifacts."
                ),
                "artifact_type": "episode_video",
            }
        )
    if any(item["status"] == "recommended" for item in splits):
        rows.append(
            {
                "title": "Probe with recommended split first",
                "why": (
                    "The dataset supports at least one stronger split than random episode holdout."
                ),
                "artifact_type": "probe_suite",
            }
        )
    return rows


def _avoid_or_delay(
    frame: pd.DataFrame,
    modalities: dict[str, Any],
    splits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = [
        {"title": item["name"], "why": item["reason"]}
        for item in splits
        if item["status"] == "avoid"
    ]
    if frame.get("outcome") is not None and _is_imbalanced(frame["outcome"]):
        counts = _counts_text(frame["outcome"])
        rows.append(
            {
                "title": "Raw outcome probes without baselines",
                "why": f"Outcome counts are {counts}; score alone will be misleading.",
            }
        )
    if not modalities["attention"]["available"]:
        rows.append(
            {
                "title": "Strong attention-mechanism claims",
                "why": "No explicit attention tensors are indexed in the opened dataset.",
            }
        )
    return rows


def _warnings(
    frame: pd.DataFrame,
    factors: list[dict[str, Any]],
    splits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if frame.get("outcome") is not None and _is_imbalanced(frame["outcome"]):
        rows.append(
            {
                "title": "Outcome imbalance",
                "detail": (
                    f"Outcome counts are {_counts_text(frame['outcome'])}; "
                    "compare against always guessing the largest group."
                ),
            }
        )
    task_factor = next((item for item in factors if item["factor"] == "task_id"), None)
    if task_factor and int(task_factor["unique"]) == len(frame):
        rows.append(
            {
                "title": "Mostly one episode per task",
                "detail": "Task holdout is possible, but within-task outcome claims are weak.",
            }
        )
    return rows


def _is_imbalanced(values: pd.Series) -> bool:
    counts = values.dropna().astype(str).value_counts()
    if counts.shape[0] < 2:
        return False
    return float(counts.min() / counts.sum()) < 0.35


def _counts(values: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in values.astype(str).value_counts().items()}


def _counts_text(values: pd.Series) -> str:
    counts = values.astype(str).value_counts()
    total = int(counts.sum())
    return ", ".join(
        f"{key} {int(value)} ({int(round(100 * int(value) / total))}%)"
        for key, value in counts.items()
    )


def _has_model_calls(dataset: TraceDataset) -> bool:
    return any(not bundle.policy_calls.empty for bundle in dataset.bundles)


def _timestep_summary(dataset: TraceDataset) -> list[dict[str, Any]]:
    table = dataset.timestep_index
    if table.empty:
        return []
    rows: list[dict[str, Any]] = []
    for trace_id, group in table.groupby("trace_id", dropna=False):
        rows.append(
            {
                "trace_id": str(trace_id),
                "rows": int(len(group)),
                "model_calls": int(group["policy_call_index"].nunique())
                if "policy_call_index" in group
                else 0,
            }
        )
    return rows


def _records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    present = [column for column in columns if column in frame]
    records = frame[present].sort_values(present[:1]).to_dict("records")
    return [{key: _json_scalar(value) for key, value in record.items()} for record in records]


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return _json_scalar(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
