from __future__ import annotations

import shutil
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vla_lens import create_synthetic_trace_dataset
from vla_lens.artifacts import LensArtifact
from vla_lens.probes import (
    NonReplayableProbeError,
    ProbeArtifactError,
    load_probe_artifact,
    train_probe_artifact_from_spec,
)


@pytest.mark.parametrize("target", ["classification", "regression"])
def test_saved_linear_probe_explains_replays_and_runs_without_refitting(tmp_path, target):
    dataset = _split_dataset(tmp_path)
    saved = train_probe_artifact_from_spec(dataset, _probe_spec(target))

    probe = load_probe_artifact(dataset, saved.artifact.artifact_id)
    explanation = probe.explain()
    shutil.rmtree(dataset.cache_dir())
    replay = probe.replay()
    contract = probe.contract

    assert explanation["replayable"] is True
    assert explanation["experiment_card"]["claim_controls"]["target"]["kind"] == target
    assert replay.matched is True
    assert replay.mismatch_count == 0
    assert contract is not None
    assert contract["schema_version"] == 1
    assert contract["uncertainty"]["confidence_intervals"]["status"] == "not_computed"
    assert contract["source"]["source_rows_fingerprint"].startswith("sha256:")
    assert contract["source"]["source_sites_fingerprint"].startswith("sha256:")
    assert contract["source"]["feature_matrix_fingerprint"].startswith("sha256:")
    assert contract["run_spec"]["name"] == f"Replayable {target} probe"
    assert set(saved.artifact.arrays) >= {
        "weights",
        "bias",
        "feature_mean",
        "feature_scale",
    }
    assert not any("feature_matrix" in name for name in saved.artifact.arrays)

    feature_dim = int(contract["model"]["feature_dim"])
    predictions = probe.predict(np.zeros((3, feature_dim), dtype=np.float32))
    assert predictions.shape == (3,)
    with pytest.raises(ProbeArtifactError, match="expects"):
        probe.predict(np.zeros((1, feature_dim + 1), dtype=np.float32))


def test_legacy_probe_remains_explainable_but_is_not_claimed_as_replayable(tmp_path):
    dataset = _split_dataset(tmp_path)
    saved = dataset.save_artifact(
        LensArtifact.create(
            artifact_type="probe_suite",
            name="Old probe",
            scope="dataset",
            method={"research": {"question": "What did this old probe test?"}},
        )
    )

    probe = load_probe_artifact(dataset, saved.artifact_id)

    assert probe.explain()["experiment_card"]["question"] == ("What did this old probe test?")
    assert probe.capabilities["legacy"] is True
    assert probe.capabilities["replay"] is False
    with pytest.raises(NonReplayableProbeError, match="predates"):
        probe.replay()


def test_loader_rejects_non_probe_artifact(tmp_path):
    dataset = _split_dataset(tmp_path)
    saved = dataset.save_artifact(
        LensArtifact.create(artifact_type="report", name="Not a probe", scope="dataset")
    )

    with pytest.raises(ProbeArtifactError, match="not 'probe_suite'"):
        load_probe_artifact(dataset, saved.artifact_id)


def test_replay_rejects_an_unknown_contract_version(tmp_path):
    dataset = _split_dataset(tmp_path)
    saved = train_probe_artifact_from_spec(dataset, _probe_spec("classification"))
    probe = load_probe_artifact(dataset, saved.artifact.artifact_id)
    method = deepcopy(dict(probe.artifact.method))
    method["probe_run_contract"]["schema_version"] = 999
    probe.artifact = replace(probe.artifact, method=method)

    with pytest.raises(ProbeArtifactError, match="Unsupported probe-run schema version"):
        probe.replay()


def _split_dataset(tmp_path: Path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=6, timesteps=8)
    pd.DataFrame(
        {
            "trace_id": [f"synthetic_{index:03d}" for index in range(6)],
            "split": ["train", "train", "train", "train", "test", "test"],
        }
    ).to_csv(dataset.root / "probe_splits.csv", index=False)
    return dataset


def _probe_spec(target: str) -> dict:
    target_spec = (
        {"kind": "outcome"}
        if target == "classification"
        else {
            "kind": "regression",
            "name": "timestep",
            "source": "row",
            "column": "timestep",
        }
    )
    return {
        "name": f"Replayable {target} probe",
        "question": f"Can this {target} target be decoded?",
        "intended_claim": "The held-out target is linearly decodable.",
        "target": target_spec,
        "features": {
            "module": "action_head.layers.*.resid",
            "tensor_type": "resid",
            "token_kind": "action",
            "layers": [0],
            "reduction": "mean",
            "dtype": "float32",
        },
        "split": {
            "kind": "heldout_task",
            "column": "split_sidecar_split",
            "train_value": "train",
            "test_value": "test",
        },
        "baseline": ["majority_class"],
        "probe": {"models": ["linear"]},
        "sweep": "layer",
    }
