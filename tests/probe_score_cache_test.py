from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from vla_lens import (
    ActivationSpec,
    ArraySpec,
    TraceManifest,
    TraceRecord,
    create_synthetic_trace_dataset,
    write_lerobot_trace_record,
)
from vla_lens.dataset import build_dataset_index
from vla_lens.probes import dump_probe_spec, train_probe_artifact_from_spec
from vla_lens.probes.score_cache import (
    _normalized_value,
    _sweep_value_from_best_state,
    read_probe_score_cache,
    refresh_probe_score_cache,
)
from vla_lens.server import _probe_index_payload
from vla_lens.traces import TraceDataset


def test_probe_score_cache_parses_legacy_multi_sweep_feature_name():
    assert _sweep_value_from_best_state(
        ["layer", "policy_call_index"],
        {"feature": "layer=4.0, policy_call_index=6", "sweep_value": None},
    ) == {"layer": "4.0", "policy_call_index": "6"}
    assert _normalized_value("4.0") == _normalized_value(4.0) == "4"


def test_probe_score_cache_refresh_scores_new_episodes_without_retraining(tmp_path):
    root = tmp_path / "demo"
    dataset = create_synthetic_trace_dataset(root, num_episodes=5, timesteps=8)
    initial_trace_ids = [bundle.manifest.trace_id for bundle in dataset.bundles]
    pd.DataFrame(
        {
            "trace_id": initial_trace_ids,
            "split": ["train", "train", "train", "train", "test"],
        }
    ).to_csv(dataset.root / "probe_splits.csv", index=False)

    saved = train_probe_artifact_from_spec(dataset, yaml.safe_load(_refreshable_probe_spec()))
    artifact_dir = dataset.root / "vla_lens" / "artifacts" / saved.artifact.artifact_id
    predictions_path = artifact_dir / "predictions.parquet"
    original_metrics = dict(saved.artifact.metrics)
    original_eval_prediction_count = len(pd.read_parquet(predictions_path))

    _append_probe_refresh_trace(root, trace_id="synthetic_005", outcome="failure")
    refreshed = TraceDataset.open(root)
    trace_ids = [bundle.manifest.trace_id for bundle in refreshed.bundles]
    pd.DataFrame(
        {
            "trace_id": trace_ids,
            "split": ["train", "train", "train", "train", "test", "new"],
        }
    ).to_csv(refreshed.root / "probe_splits.csv", index=False)

    result = refresh_probe_score_cache(refreshed, saved.artifact.artifact_id)
    score_cache = read_probe_score_cache(refreshed, saved.artifact.artifact_id)

    assert result.trace_count == 6
    assert trace_ids[-1] in set(score_cache["trace_id"].astype(str))
    assert set(score_cache["split"].dropna().astype(str)).issuperset({"train", "test", "new"})
    assert refreshed.load_artifact(saved.artifact.artifact_id).metrics == original_metrics
    assert len(pd.read_parquet(predictions_path)) == original_eval_prediction_count

    build_dataset_index(refreshed.root, overwrite=True)
    indexed = TraceDataset.open(refreshed.root)
    index_payload = _probe_index_payload(indexed)
    probe_index = next(
        probe for probe in index_payload["probes"] if probe["name"] == "Refreshable outcome probe"
    )
    assert probe_index["by_trace"][trace_ids[-1]]["available"] is True
    assert probe_index["prediction_summary"]["unscored"] == 0


def _refreshable_probe_spec() -> str:
    return dump_probe_spec(
        {
            "name": "Refreshable outcome probe",
            "target": {"kind": "outcome"},
            "features": {
                "module": "action_head.layers.*.resid",
                "tensor_type": "resid",
                "token_kind": "action",
                "reduction": "mean",
            },
            "split": {
                "kind": "random_episode",
                "column": "split_sidecar_split",
                "test_value": "test",
            },
            "baseline": ["majority_class"],
            "probe": {"models": ["linear"]},
            "sweep": "layer",
        }
    )


def _append_probe_refresh_trace(root, *, trace_id: str, outcome: str) -> None:
    timesteps = 8
    horizon = 8
    action_dim = 7
    hidden = 12
    rng = np.random.default_rng(99)
    policy_call_timesteps = np.arange(0, timesteps, 4, dtype=np.int32)
    policy_calls = pd.DataFrame(
        {
            "policy_call_index": np.arange(len(policy_call_timesteps), dtype=np.int32),
            "episode_id": trace_id,
            "observation_timestep": policy_call_timesteps,
            "env_timestep_start": policy_call_timesteps,
            "env_timestep_end": np.minimum(policy_call_timesteps + 3, timesteps - 1),
        }
    )
    write_lerobot_trace_record(
        TraceRecord(
            manifest=TraceManifest(
                trace_id=trace_id,
                episode_id=trace_id,
                task_id="pick_red_cube",
                prompt="pick up the red cube",
                model_id="synthetic-flow-policy",
                env_id="synthetic-tabletop",
                robot_id="synthetic-7dof",
                outcome=outcome,
                length=timesteps,
                metadata={"split": "new", "target_object": "red_cube"},
            ),
            timesteps=pd.DataFrame(
                {
                    "timestep": np.arange(timesteps, dtype=np.int32),
                    "policy_call_index": np.arange(timesteps, dtype=np.int32) // 4,
                    "horizon_index": np.arange(timesteps, dtype=np.int32) % 4,
                    "reward": np.linspace(
                        0.0,
                        0.25 if outcome == "failure" else 1.0,
                        timesteps,
                        dtype=np.float32,
                    ),
                }
            ),
            policy_calls=policy_calls,
            generation_steps=pd.DataFrame(
                {
                    "policy_call_index": np.repeat(policy_calls["policy_call_index"], 5),
                    "generation_step": list(range(5)) * len(policy_calls),
                }
            ),
            streams=pd.DataFrame(
                {
                    "stream_id": ["action"],
                    "name": ["action"],
                    "modality": ["action"],
                }
            ),
            token_spaces=pd.DataFrame(
                {
                    "token_space_id": ["synthetic.action_suffix"],
                    "stream_id": ["action"],
                    "token_count": [horizon],
                }
            ),
            tokens=pd.DataFrame(
                {
                    "token_space_id": ["synthetic.action_suffix"] * horizon,
                    "token_index": list(range(horizon)),
                    "token_kind": ["action"] * horizon,
                }
            ),
            episode_arrays={
                "frames.main": ArraySpec(
                    np.zeros((timesteps, 96, 128, 3), dtype=np.uint8),
                    ["timestep", "height", "width", "rgb"],
                ),
                "frames.wrist": ArraySpec(
                    np.zeros((timesteps, 96, 128, 3), dtype=np.uint8),
                    ["timestep", "height", "width", "rgb"],
                ),
                "executed_actions": ArraySpec(
                    np.zeros((timesteps, action_dim), dtype=np.float32),
                    ["timestep", "action_dim"],
                ),
                "action_chunks": ArraySpec(
                    np.zeros((len(policy_calls), horizon, action_dim), dtype=np.float32),
                    ["policy_call", "horizon", "action_dim"],
                ),
                "generation_actions": ArraySpec(
                    np.zeros((len(policy_calls), 5, horizon, action_dim), dtype=np.float32),
                    ["policy_call", "generation_step", "horizon", "action_dim"],
                ),
            },
            model_arrays=[
                ActivationSpec(
                    name=f"action_head.layers.{layer}.resid",
                    array=rng.normal(0.0, 0.05, size=(timesteps, horizon, hidden)).astype(
                        np.float32
                    ),
                    axes=["timestep", "token", "channel"],
                    module=f"action_head.layers.{layer}.resid",
                    layer=layer,
                    tensor_type="resid",
                    token_kind="action",
                    token_space_id="synthetic.action_suffix",
                )
                for layer in range(6)
            ],
        ),
        root,
    )
