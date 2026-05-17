"""Action-generation artifacts for flow-action VLA traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from vla_lens.artifacts import LensArtifact
from vla_lens.traces import TraceBundle, TraceDataset
from vla_lens.workbench import AnalysisRunSpec, save_analysis_run


@dataclass(frozen=True, slots=True)
class SavedActionGeneration:
    artifact: LensArtifact
    commitment: np.ndarray
    executed_error: np.ndarray
    delta_to_final: np.ndarray | None = None
    step_delta: np.ndarray | None = None
    final_vs_executed: np.ndarray | None = None


def save_action_generation_artifact(
    dataset: TraceDataset,
    *,
    name: str = "Action generation summary",
) -> SavedActionGeneration:
    """Summarize generation/action formation across every compatible episode."""
    summaries: list[dict[str, Any]] = []
    commitments: list[np.ndarray] = []
    executed_errors: list[np.ndarray] = []
    delta_to_final_rows: list[np.ndarray] = []
    step_delta_rows: list[np.ndarray] = []
    final_vs_executed_rows: list[np.ndarray] = []

    for bundle in dataset.bundles:
        summary = _bundle_summary(bundle)
        if summary is None:
            continue
        summaries.append(summary["display"])
        commitments.append(summary["commitment"])
        executed_errors.append(summary["executed_error"])
        delta_to_final_rows.append(summary["delta_to_final"])
        step_delta_rows.append(summary["step_delta"])
        final_vs_executed_rows.append(summary["final_vs_executed"])

    if not summaries:
        raise ValueError(
            "No bundles contain generation actions, action chunks, and executed actions"
        )

    commitment_array = _pad_3d(commitments)
    executed_error_array = _pad_2d(executed_errors)
    delta_to_final_array = _pad_4d(delta_to_final_rows)
    step_delta_array = _pad_4d(step_delta_rows)
    final_vs_executed_array = _pad_4d(final_vs_executed_rows)
    aggregate = _aggregate_summary(summaries)
    artifact = LensArtifact.create(
        artifact_type="action_generation",
        name=name,
        group_id="action_generation",
        scope="dataset",
        selector={"episodes": "all", "arrays": ["generation_actions", "action_chunks"]},
        method={
            "workflow": "save_action_generation_artifact",
            "metrics": [
                "delta_to_final",
                "step_delta",
                "final_vs_executed",
            ],
        },
        metrics=aggregate["metrics"],
        display={
            "kind": "action_generation",
            "summary": aggregate["summary"],
            "episodes": summaries,
            "interpretation_notes": [
                (
                    "delta_to_final measures how far each generation step is from "
                    "the final action chunk."
                ),
                "step_delta measures step-to-step action change during generation.",
                (
                    "final_vs_executed compares each final planned horizon action with the "
                    "corresponding executed action when it exists."
                ),
            ],
        },
        tags=("action", "generation", "flow"),
        source_trace_ids=tuple(item["trace_id"] for item in summaries),
    )
    saved = dataset.save_artifact(
        artifact,
        arrays={
            "commitment": commitment_array,
            "executed_vs_predicted": executed_error_array,
            "delta_to_final": delta_to_final_array,
            "step_delta": step_delta_array,
            "final_vs_executed": final_vs_executed_array,
        },
    )
    save_analysis_run(
        dataset,
        AnalysisRunSpec(
            run_id=saved.artifact_id,
            workflow="action_stabilization",
            inputs=saved.selector,
            outputs=tuple(saved.arrays),
            provenance={"artifact_id": saved.artifact_id},
        ),
    )
    return SavedActionGeneration(
        artifact=saved,
        commitment=commitment_array,
        executed_error=executed_error_array,
        delta_to_final=delta_to_final_array,
        step_delta=step_delta_array,
        final_vs_executed=final_vs_executed_array,
    )


def _bundle_summary(bundle: TraceBundle) -> dict[str, Any] | None:
    try:
        generation = np.asarray(bundle.generation_actions(mmap=True), dtype=np.float32)
        chunks = np.asarray(bundle.action_chunks(mmap=True), dtype=np.float32)
        actions = np.asarray(bundle.actions(mmap=True), dtype=np.float32)
    except KeyError:
        return None
    if generation.ndim != 4 or chunks.ndim != 3 or actions.ndim != 2:
        return None

    calls = _policy_call_timesteps(bundle, fallback_count=generation.shape[0])
    call_count = min(len(calls), generation.shape[0], chunks.shape[0])
    if call_count == 0:
        return None
    generation = generation[:call_count]
    chunks = chunks[:call_count]
    calls = calls[:call_count]

    final = generation[:, -1:, :, :]
    commitment = np.linalg.norm(generation - final, axis=(-1, -2))
    delta_to_final = np.linalg.norm(generation - final, axis=-1)
    start_distance = np.maximum(commitment[:, :1], 1e-8)
    relative = commitment / start_distance
    step_delta = np.zeros_like(commitment)
    horizon_step_delta = np.zeros_like(delta_to_final)
    if commitment.shape[1] > 1:
        step_delta[:, 1:] = np.linalg.norm(np.diff(generation, axis=1), axis=(-1, -2))
        horizon_step_delta[:, 1:, :] = np.linalg.norm(np.diff(generation, axis=1), axis=-1)

    executed_error = np.full(call_count, np.nan, dtype=np.float32)
    final_vs_executed = np.full(
        (call_count, generation.shape[2], generation.shape[3]),
        np.nan,
        dtype=np.float32,
    )
    for index, timestep in enumerate(calls):
        if 0 <= timestep < actions.shape[0]:
            dim = min(actions.shape[-1], chunks.shape[-1])
            executed_error[index] = float(
                np.linalg.norm(actions[timestep, :dim] - chunks[index, 0, :dim])
            )
            for horizon in range(generation.shape[2]):
                executed_index = timestep + horizon
                if executed_index < actions.shape[0]:
                    final_vs_executed[index, horizon, :dim] = (
                        generation[index, -1, horizon, :dim] - actions[executed_index, :dim]
                    )

    action_norm = np.linalg.norm(chunks[:, 0], axis=-1)
    display = {
        "trace_id": bundle.manifest.trace_id,
        "episode_id": bundle.manifest.episode_id,
        "task": bundle.manifest.task_id,
        "outcome": bundle.manifest.outcome,
        "policy_decisions": int(call_count),
        "generation_steps": int(generation.shape[1]),
        "horizon": int(generation.shape[2]),
        "action_dim": int(generation.shape[3]),
        "final_commitment_mean": _finite_mean(commitment[:, -1]),
        "initial_commitment_mean": _finite_mean(commitment[:, 0]),
        "executed_vs_predicted_mean": _finite_mean(executed_error),
        "action_norm_mean": _finite_mean(action_norm),
        "commitment_curve": _round_list(np.nanmean(commitment, axis=0)),
        "relative_commitment_curve": _round_list(np.nanmean(relative, axis=0)),
        "step_delta_curve": _round_list(np.nanmean(step_delta, axis=0)),
        "delta_to_final_shape": [int(item) for item in delta_to_final.shape],
        "final_vs_executed_shape": [int(item) for item in final_vs_executed.shape],
        "unstable_calls": _unstable_calls(
            calls=calls,
            commitment=commitment,
            relative=relative,
            executed_error=executed_error,
        ),
    }
    return {
        "display": display,
        "commitment": commitment.astype(np.float32),
        "executed_error": executed_error.astype(np.float32),
        "delta_to_final": delta_to_final.astype(np.float32),
        "step_delta": horizon_step_delta.astype(np.float32),
        "final_vs_executed": final_vs_executed.astype(np.float32),
    }


def _policy_call_timesteps(bundle: TraceBundle, *, fallback_count: int) -> list[int]:
    calls = bundle.policy_calls
    if not calls.empty:
        column = "observation_timestep" if "observation_timestep" in calls else "env_timestep_start"
        if column in calls:
            return [int(value) for value in calls[column].tolist()]
    return list(range(fallback_count))


def _aggregate_summary(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    initial = np.array([item["initial_commitment_mean"] for item in summaries], dtype=np.float32)
    executed = np.array(
        [item["executed_vs_predicted_mean"] for item in summaries],
        dtype=np.float32,
    )
    decisions = np.array([item["policy_decisions"] for item in summaries], dtype=np.float32)
    outcomes = _counts([item.get("outcome") for item in summaries])
    outcome_summary = _outcome_summary(summaries)
    summary = (
        f"{len(summaries)} episode(s). Mean initial commitment "
        f"{_finite_mean(initial):.3g}; mean executed-vs-predicted error "
        f"{_finite_mean(executed):.3g}."
    )
    return {
        "summary": summary,
        "metrics": {
            "episode_count": int(len(summaries)),
            "policy_decision_count": int(np.nansum(decisions)),
            "mean_initial_commitment": _finite_mean(initial),
            "mean_executed_vs_predicted": _finite_mean(executed),
            "outcome_counts": outcomes,
            "outcome_summary": outcome_summary,
        },
    }


def _outcome_summary(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    outcomes = sorted({str(item.get("outcome")) for item in summaries})
    for outcome in outcomes:
        group = [item for item in summaries if str(item.get("outcome")) == outcome]
        if not group:
            continue
        initial = np.array([item["initial_commitment_mean"] for item in group], dtype=np.float32)
        executed = np.array(
            [item["executed_vs_predicted_mean"] for item in group],
            dtype=np.float32,
        )
        action_norm = np.array([item["action_norm_mean"] for item in group], dtype=np.float32)
        rows.append(
            {
                "outcome": outcome,
                "episodes": int(len(group)),
                "mean_initial_commitment": _finite_mean(initial),
                "mean_executed_vs_predicted": _finite_mean(executed),
                "mean_action_norm": _finite_mean(action_norm),
            }
        )
    return rows


def _unstable_calls(
    *,
    calls: list[int],
    commitment: np.ndarray,
    relative: np.ndarray,
    executed_error: np.ndarray,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    initial = commitment[:, 0] if commitment.shape[1] else np.zeros(commitment.shape[0])
    final_relative = relative[:, -1] if relative.shape[1] else np.zeros(relative.shape[0])
    for call_index, timestep in enumerate(calls):
        records.append(
            {
                "call_index": int(call_index),
                "timestep": int(timestep),
                "initial_commitment": _finite_value(initial[call_index]),
                "final_relative_commitment": _finite_value(final_relative[call_index]),
                "executed_vs_predicted": _finite_value(executed_error[call_index]),
                "instability_score": _finite_value(
                    initial[call_index] + np.nan_to_num(executed_error[call_index], nan=0.0)
                ),
            }
        )
    return sorted(
        records,
        key=lambda item: float(item.get("instability_score") or 0.0),
        reverse=True,
    )[:8]


def _pad_3d(arrays: list[np.ndarray]) -> np.ndarray:
    max_a = max(array.shape[0] for array in arrays)
    max_b = max(array.shape[1] for array in arrays)
    out = np.full((len(arrays), max_a, max_b), np.nan, dtype=np.float32)
    for index, array in enumerate(arrays):
        out[index, : array.shape[0], : array.shape[1]] = array
    return out


def _pad_4d(arrays: list[np.ndarray]) -> np.ndarray:
    max_a = max(array.shape[0] for array in arrays)
    max_b = max(array.shape[1] for array in arrays)
    max_c = max(array.shape[2] for array in arrays)
    out = np.full((len(arrays), max_a, max_b, max_c), np.nan, dtype=np.float32)
    for index, array in enumerate(arrays):
        out[index, : array.shape[0], : array.shape[1], : array.shape[2]] = array
    return out


def _pad_2d(arrays: list[np.ndarray]) -> np.ndarray:
    max_a = max(array.shape[0] for array in arrays)
    out = np.full((len(arrays), max_a), np.nan, dtype=np.float32)
    for index, array in enumerate(arrays):
        out[index, : array.shape[0]] = array
    return out


def _finite_mean(values: Any) -> float:
    array = np.asarray(values, dtype=np.float32)
    if not np.isfinite(array).any():
        return 0.0
    return float(np.nanmean(array))


def _finite_value(value: Any) -> float | None:
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _round_list(values: Any) -> list[float | None]:
    out: list[float | None] = []
    for value in np.asarray(values, dtype=np.float32).tolist():
        if value is None or not np.isfinite(value):
            out.append(None)
        else:
            out.append(round(float(value), 6))
    return out


def _counts(values: list[Any]) -> Mapping[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return out
