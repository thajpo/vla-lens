from __future__ import annotations

import shutil
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr

from vla_lens import ActivationQuery, create_synthetic_trace_dataset
from vla_lens.artifacts import LensArtifact
from vla_lens.probes import (
    NonReplayableProbeError,
    ProbeArtifactError,
    load_probe_artifact,
    train_probe_artifact,
    train_probe_artifact_from_spec,
    workflow_training,
)
from vla_lens.probes.run_artifacts import (
    _compare_predictions,
    dataframe_fingerprint,
    probe_label_sources,
)
from vla_lens.probes.workflow_models import _floating_replay_tolerance
from vla_lens.probes.workflow_prepare import latest_loadable_artifact
from vla_lens.probes.workflow_types import INTERACTION_METRICS_ARTIFACT_TYPE


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
    assert contract["schema_version"] == 2
    intervals = contract["uncertainty"]["confidence_intervals"]
    assert intervals["status"] == "not_computed"
    assert "Fewer than two" in intervals["reason"]
    assert contract["run_spec"]["split"]["selection_value"] == "validation"
    assert contract["run_spec"]["split"]["automatic_validation"]["created"] is True
    assert contract["source"]["source_rows_fingerprint"].startswith("sha256:")
    assert contract["source"]["source_sites_fingerprint"].startswith("sha256:")
    assert contract["source"]["feature_matrix_fingerprint"].startswith("sha256:")
    assert set(contract["model"]["array_fingerprints"]) <= set(saved.artifact.arrays)
    assert contract["readouts"]
    assert contract["run_spec"]["name"] == f"Replayable {target} probe"
    assert set(saved.artifact.arrays) >= {
        "weights",
        "bias",
        "feature_mean",
        "feature_scale",
    }
    assert not any("feature_matrix" in name for name in saved.artifact.arrays)
    artifact_dir = dataset.root / "vla_lens" / "artifacts" / saved.artifact.artifact_id
    candidates = pd.read_parquet(artifact_dir / "candidate_results.parquet")
    assert candidates["readout_id"].nunique() >= 1
    assert candidates["selected"].any()
    assert candidates.loc[candidates["selected"], "fitted_state_retained"].all()

    feature_dim = int(contract["model"]["feature_dim"])
    predictions = probe.predict(np.zeros((3, feature_dim), dtype=np.float32))
    integer_predictions = probe.predict(np.zeros((3, feature_dim), dtype=np.int64))
    assert predictions.shape == (3,)
    for readout in contract["readouts"]:
        readout_dim = int(readout["model"]["feature_dim"])
        readout_predictions = probe.predict(
            np.zeros((2, readout_dim), dtype=np.float32),
            readout_id=str(readout["readout_id"]),
        )
        assert readout_predictions.shape == (2,)
    np.testing.assert_array_equal(
        integer_predictions,
        probe.predict(np.zeros((3, feature_dim), dtype=np.float64)),
    )
    if target == "classification":
        assert not predictions.dtype.hasobject
        prediction_path = tmp_path / "predictions.npy"
        np.save(prediction_path, predictions, allow_pickle=False)
        np.testing.assert_array_equal(
            np.load(prediction_path, allow_pickle=False),
            predictions,
        )
    else:
        tolerance = contract["model"]["prediction_tolerance"]
        assert tolerance["absolute"] > 0
        assert tolerance["relative"] > 0
        assert tolerance["operation_count"] >= feature_dim
        assert replay.absolute_tolerance == tolerance["absolute"]
        assert replay.relative_tolerance == tolerance["relative"]
    with pytest.raises(ProbeArtifactError, match="expects"):
        probe.predict(np.zeros((1, feature_dim + 1), dtype=np.float32))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_probe_rejects_nonfinite_features_before_inference(tmp_path, value):
    dataset = _split_dataset(tmp_path)
    saved = train_probe_artifact_from_spec(dataset, _probe_spec("classification"))
    probe = load_probe_artifact(dataset, saved.artifact.artifact_id)
    contract = probe.contract
    assert contract is not None
    features = np.zeros((1, int(contract["model"]["feature_dim"])), dtype=np.float32)
    features[0, 0] = value

    with pytest.raises(ProbeArtifactError, match="only finite values"):
        probe.predict(features)


def test_probe_validates_every_fitted_array_before_use_and_replay(tmp_path):
    dataset = _split_dataset(tmp_path)
    saved = train_probe_artifact_from_spec(dataset, _probe_spec("classification"))
    probe = load_probe_artifact(dataset, saved.artifact.artifact_id)
    contract = probe.contract
    assert contract is not None

    for name, relative_path in saved.artifact.arrays.items():
        stored = zarr.open_array(
            str(dataset._dataset_artifact_root() / relative_path),
            mode="a",
        )
        original = np.asarray(stored[...]).copy()
        changed = original.copy()
        changed.reshape(-1)[0] += 1
        stored[...] = changed
        readout_id = name.split("__", 1)[0] if "__" in name else None
        model = (
            next(
                readout["model"]
                for readout in contract["readouts"]
                if readout["readout_id"] == readout_id
            )
            if readout_id is not None
            else contract["model"]
        )
        candidate_features = np.zeros((1, int(model["feature_dim"])), dtype=np.float32)
        with pytest.raises(ProbeArtifactError, match=rf"array '{name}' changed"):
            probe.predict(candidate_features, readout_id=readout_id)
        stored[...] = original

    weights_path = dataset._dataset_artifact_root() / saved.artifact.arrays["weights"]
    stored = zarr.open_array(str(weights_path), mode="a")
    changed = np.asarray(stored[...]).copy()
    changed.reshape(-1)[0] += 1
    stored[...] = changed

    with pytest.raises(ProbeArtifactError, match="array 'weights' changed"):
        probe.replay()


def test_activation_query_round_trips_temporal_slices():
    query = ActivationQuery(
        timesteps=slice(1, 7, 2),
        policy_calls=slice(None, 4, 1),
    )

    restored = ActivationQuery.from_dict(query.to_dict())

    assert restored.timesteps == slice(1, 7, 2)
    assert restored.policy_calls == slice(None, 4, 1)


def test_replay_round_trips_a_slice_selector(tmp_path):
    dataset = _split_dataset(tmp_path)
    selector = ActivationQuery(
        module="action_head.layers.*.resid",
        layers=[0],
        tensor_type="resid",
        token_kind="action",
        timesteps=slice(0, None, 2),
        policy_calls=slice(None, None, 2),
        reduce_tokens="mean",
        dtype="float32",
    )
    saved = train_probe_artifact(
        dataset,
        name="Slice selector probe",
        selector=selector,
        target="outcome",
    )

    assert load_probe_artifact(dataset, saved.artifact.artifact_id).replay().matched is True


def test_replay_rebuilds_features_without_trusting_the_training_cache(tmp_path):
    dataset = _split_dataset(tmp_path)
    saved = train_probe_artifact_from_spec(dataset, _probe_spec("classification"))
    probe = load_probe_artifact(dataset, saved.artifact.artifact_id)
    bundle = dataset.bundle("synthetic_004")
    model_site = bundle.model_sites.loc[
        bundle.model_sites["name"].astype(str) == "action_head.layers.0.resid"
    ].iloc[0]
    activation = zarr.open_array(
        str(bundle.path / str(model_site["relative_path"])),
        mode="a",
    )
    activation[0] = np.asarray(activation[0]) + 10.0

    with pytest.raises(ProbeArtifactError, match="Prepared probe features changed"):
        probe.replay()


def test_replay_validates_saved_source_sites(tmp_path):
    dataset = _split_dataset(tmp_path)
    saved = train_probe_artifact_from_spec(dataset, _probe_spec("classification"))
    probe = load_probe_artifact(dataset, saved.artifact.artifact_id)
    contract = probe.contract
    assert contract is not None
    path = dataset.root / contract["source"]["source_sites_path"]
    rows = pd.read_parquet(path)
    rows.loc[0, "trace_id"] = "changed-trace"
    rows.to_parquet(path, index=False)

    with pytest.raises(ProbeArtifactError, match="Saved source-site rows changed"):
        probe.replay()


def test_replay_validates_saved_prediction_row_order(tmp_path):
    dataset = _split_dataset(tmp_path)
    saved = train_probe_artifact_from_spec(dataset, _probe_spec("classification"))
    probe = load_probe_artifact(dataset, saved.artifact.artifact_id)
    contract = probe.contract
    assert contract is not None
    path = dataset.root / contract["source"]["scored_predictions_path"]
    rows = pd.read_parquet(path).iloc[::-1].reset_index(drop=True)
    rows.to_parquet(path, index=False)

    with pytest.raises(ProbeArtifactError, match="Saved scored predictions changed"):
        probe.replay()


def test_probe_is_not_registered_when_a_sidecar_write_fails(tmp_path, monkeypatch):
    dataset = _split_dataset(tmp_path)
    artifact_id = "probe-suite-staging-failure"
    monkeypatch.setattr(workflow_training, "make_artifact_id", lambda *_: artifact_id)
    original_to_parquet = pd.DataFrame.to_parquet

    def fail_source_sites(frame, path, *args, **kwargs):
        if Path(path).name == "source_sites.parquet":
            raise OSError("simulated sidecar failure")
        return original_to_parquet(frame, path, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_source_sites)

    with pytest.raises(OSError, match="simulated sidecar failure"):
        train_probe_artifact_from_spec(dataset, _probe_spec("classification"))

    artifact_dir = dataset._dataset_artifact_root() / "artifacts" / artifact_id
    assert not artifact_dir.exists()
    assert artifact_id not in set(dataset.artifact_index.get("artifact_id", []))


def test_dataframe_fingerprints_survive_parquet_round_trips(tmp_path):
    frame = pd.DataFrame(
        {
            "trace_id": ["trace-1", "trace-2"],
            "sample": [1, 2],
            "optional": [None, "value"],
            "coordinates": [[1, 2], [3, 4]],
        }
    )
    path = tmp_path / "rows.parquet"
    frame.to_parquet(path, index=False)

    fingerprint = dataframe_fingerprint(frame)

    assert dataframe_fingerprint(pd.read_parquet(path)) == fingerprint
    assert dataframe_fingerprint(frame.iloc[::-1].reset_index(drop=True)) != fingerprint


def test_dataframe_fingerprints_distinguish_infinities_from_missing_values(tmp_path):
    frame = pd.DataFrame({"value": [np.nan, np.inf, -np.inf, 1.0]})
    path = tmp_path / "special-values.parquet"
    frame.to_parquet(path, index=False)

    fingerprint = dataframe_fingerprint(frame)

    assert dataframe_fingerprint(pd.read_parquet(path)) == fingerprint
    for row_index in [1, 2]:
        changed = frame.copy()
        changed.loc[row_index, "value"] = np.nan
        assert dataframe_fingerprint(changed) != fingerprint
    changed = frame.copy()
    changed.loc[1, "value"] = -np.inf
    assert dataframe_fingerprint(changed) != fingerprint


def test_regression_prediction_comparison_uses_absolute_and_relative_tolerance():
    tolerance = {"absolute": 1e-8, "relative": 1e-6}
    within = _compare_predictions(
        artifact_id="probe",
        probe_type="regression",
        replayed=np.array([1e8 + 50.0, 1e-10 + 5e-9]),
        saved=np.array([1e8, 1e-10]),
        feature_fingerprint="sha256:features",
        prediction_tolerance=tolerance,
    )
    outside = _compare_predictions(
        artifact_id="probe",
        probe_type="regression",
        replayed=np.array([1e8 + 200.0, 1e-10 + 2e-8]),
        saved=np.array([1e8, 1e-10]),
        feature_fingerprint="sha256:features",
        prediction_tolerance=tolerance,
    )

    assert within.matched is True
    assert within.absolute_tolerance == tolerance["absolute"]
    assert within.relative_tolerance == tolerance["relative"]
    assert outside.matched is False
    assert outside.mismatch_count == 2


def test_regression_replay_tolerance_never_accepts_large_prediction_drift():
    feature_dim = 128
    tolerance = _floating_replay_tolerance(
        {
            "weights": np.ones(feature_dim, dtype=np.float16),
            "bias": np.zeros(1, dtype=np.float16),
            "feature_mean": np.zeros(feature_dim, dtype=np.float16),
            "feature_scale": np.ones(feature_dim, dtype=np.float16),
        },
        feature_matrix=np.ones((2, feature_dim), dtype=np.float16),
        predictions=np.array([1.0, 2.0], dtype=np.float16),
    )

    assert tolerance["estimated_relative_error"] > tolerance["maximum_relative_error"]
    scale = max(1.0, tolerance["prediction_scale"])
    assert tolerance["absolute"] / scale + tolerance["relative"] <= 1e-4
    replay = _compare_predictions(
        artifact_id="probe",
        probe_type="regression",
        replayed=np.array([2.0]),
        saved=np.array([1.0]),
        feature_fingerprint="sha256:features",
        prediction_tolerance=tolerance,
    )
    assert replay.matched is False


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


def test_v1_probe_contract_remains_explainable_but_is_not_replayable(tmp_path):
    dataset = _split_dataset(tmp_path)
    saved = train_probe_artifact_from_spec(dataset, _probe_spec("classification"))
    probe = load_probe_artifact(dataset, saved.artifact.artifact_id)
    method = deepcopy(dict(probe.artifact.method))
    method["probe_run_contract"]["schema_version"] = 1
    method["probe_run_contract"]["model"].pop("array_fingerprints")
    probe.artifact = replace(probe.artifact, method=method)

    explanation = probe.explain()

    assert explanation["experiment_card"]["name"] == "Replayable classification probe"
    assert explanation["replayable"] is False
    assert explanation["capabilities"]["explain"] is True
    assert explanation["capabilities"]["replay"] is False
    assert explanation["capabilities"]["use"] is False
    with pytest.raises(NonReplayableProbeError, match="version 1 predates"):
        probe.predict(np.zeros((1, 4), dtype=np.float32))
    with pytest.raises(NonReplayableProbeError, match="version 1 predates"):
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


def test_label_provenance_uses_newest_loadable_artifact(tmp_path):
    dataset = _split_dataset(tmp_path)
    loadable = dataset.save_artifact(
        replace(
            LensArtifact.create(
                artifact_type=INTERACTION_METRICS_ARTIFACT_TYPE,
                name="Loadable labels",
                scope="dataset",
            ),
            created_utc="2026-01-01T00:00:00+00:00",
        )
    )
    stale = replace(
        LensArtifact.create(
            artifact_type=INTERACTION_METRICS_ARTIFACT_TYPE,
            name="Missing labels",
            scope="dataset",
        ),
        created_utc="2026-02-01T00:00:00+00:00",
    )
    dataset.__dict__["artifact_index"] = pd.DataFrame.from_records(
        [loadable.to_record(), stale.to_record()]
    )

    selected = latest_loadable_artifact(dataset, INTERACTION_METRICS_ARTIFACT_TYPE)
    sources = probe_label_sources(dataset)

    assert selected is not None
    assert selected.artifact_id == loadable.artifact_id
    assert [source["artifact_id"] for source in sources] == [loadable.artifact_id]


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
