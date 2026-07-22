from __future__ import annotations

from pathlib import Path

import pandas as pd

from vla_lens import create_synthetic_trace_dataset
from vla_lens.probes import (
    format_probe_preflight_markdown,
    probe_preflight_report,
    train_probe_artifact_from_spec,
)


def test_probe_preflight_reports_sweep_baselines_and_leakage(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=6, timesteps=8)
    _write_probe_splits(
        dataset.root,
        ["train", "train", "train", "train", "val_heldout_task", "test_heldout_task"],
    )

    report = probe_preflight_report(dataset, _outcome_probe_spec(), min_class_support=5)

    assert report["name"] == "Synthetic outcome preflight"
    assert report["question"] == "Can outcome be decoded from synthetic action hidden states?"
    assert report["target"]["name"] == "outcome"
    assert report["target"]["kind"] == "classification"
    assert report["split"]["summary"]["episodes"] == {
        "test_heldout_task": 1,
        "train": 4,
        "val_heldout_task": 1,
    }
    assert report["baselines"]["available_columns"] == [
        "outcome",
        "task_id",
        "policy_call_index",
    ]
    assert report["baselines"]["suspicious_columns"] == ["outcome"]
    assert report["sweep"]["columns"] == ["layer", "policy_call_index"]
    assert report["sweep"]["group_count"] == 4
    assert report["sweep"]["planned_readout_count"] == 8
    assert any("target column" in warning for warning in report["warnings"])
    assert any("lacks some target classes" in warning for warning in report["warnings"])


def test_probe_preflight_markdown_is_reviewable(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=6, timesteps=8)
    _write_probe_splits(
        dataset.root,
        ["train", "train", "train", "train", "val_heldout_task", "test_heldout_task"],
    )

    markdown = format_probe_preflight_markdown(
        probe_preflight_report(dataset, _outcome_probe_spec(), min_class_support=5)
    )

    assert "# Probe Experiment Card: Synthetic outcome preflight" in markdown
    assert "## Choices that change the claim" in markdown
    assert "## Method choices" in markdown
    assert "## Execution details" in markdown
    assert "Baselines:" in markdown
    assert "outcome" in markdown
    assert "Planned readouts: 8" in markdown
    assert "target column" in markdown


def test_probe_training_persists_research_framing_from_spec(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=6, timesteps=8)
    _write_probe_splits(
        dataset.root,
        ["train", "train", "train", "train", "val_heldout_task", "test_heldout_task"],
    )

    saved = train_probe_artifact_from_spec(dataset, _outcome_probe_spec())

    assert saved.artifact.method["research"] == {
        "hypothesis_family": "test fixture",
        "intended_claim": "synthetic outcome information is decodable",
        "question": "Can outcome be decoded from synthetic action hidden states?",
    }
    assert saved.artifact.display["research"] == saved.artifact.method["research"]
    artifact_path = Path(str(saved.artifact.path))
    if not artifact_path.is_absolute():
        candidates = [
            dataset.root / artifact_path,
            dataset._dataset_artifact_root() / artifact_path,
        ]
        artifact_path = next((path for path in candidates if path.exists()), candidates[0])
    predictions_path = artifact_path.parent / "predictions.parquet"
    predictions = pd.read_parquet(predictions_path)
    assert set(predictions["feature"].astype(str)) == {saved.artifact.metrics["best_feature"]}
    assert set(predictions["split"].astype(str)) == {"test_heldout_task", "val_heldout_task"}


def _write_probe_splits(root, splits: list[str]) -> None:
    pd.DataFrame(
        {
            "trace_id": [f"synthetic_{idx:03d}" for idx in range(len(splits))],
            "split": splits,
        }
    ).to_csv(root / "probe_splits.csv", index=False)


def _outcome_probe_spec() -> dict:
    return {
        "name": "Synthetic outcome preflight",
        "question": "Can outcome be decoded from synthetic action hidden states?",
        "hypothesis_family": "test fixture",
        "intended_claim": "synthetic outcome information is decodable",
        "target": {"kind": "outcome"},
        "features": {
            "module": "action_head.layers.*.resid",
            "tensor_type": "resid",
            "token_kind": "action",
            "layers": [0, 1],
            "timesteps": "all",
            "policy_calls": "all",
            "reduction": "mean",
            "dtype": "float32",
        },
        "split": {
            "kind": "heldout_task",
            "column": "split_sidecar_split",
            "train_value": "train",
            "selection_value": "val_heldout_task",
            "test_value": "test_heldout_task",
            "eval_values": ["val_heldout_task", "test_heldout_task"],
        },
        "baseline": ["majority_class", "outcome", "task_id", "policy_call_index"],
        "probe": {"models": ["linear"]},
        "sweep": ["layer", "policy_call_index"],
    }
