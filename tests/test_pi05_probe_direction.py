from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from vla_lens.artifacts import LensArtifact
from vla_lens.interventions import TargetSpec
from vla_lens.pi05.intervention_executor import PI05ActionInterventionExecutor
from vla_lens.pi05.probe_direction import resolve_object_roi_probe_direction
from vla_lens.pi05.replay import PolicyCallReplayInputs, ReplayConfig
from vla_lens.synthetic import create_synthetic_trace_dataset


def _saved_probe(dataset):
    artifact_id = "rq015-linear-object-roi-test"
    relative = f"artifacts/{artifact_id}"
    artifact_dir = dataset._dataset_artifact_root() / relative
    artifact_dir.mkdir(parents=True)
    trace_id = dataset.bundles[0].manifest.trace_id
    model_id = dataset.bundles[0].manifest.model_id
    tables = {
        "instances": pd.DataFrame.from_records(
            [
                {
                    "trace_id": trace_id,
                    "source_row_index": 0,
                    "object_index": 1,
                    "object_name": "target_object",
                    "roi_patch_indices": [1, 2],
                    "wrong_roi_patch_indices": [0],
                }
            ]
        ),
        "predictions": pd.DataFrame.from_records(
            [
                {
                    "method": "object_roi",
                    "model": "linear",
                    "layer": 8,
                    "instance_index": 0,
                    "object_index": 1,
                    "correct": True,
                }
            ]
        ),
        "vocabulary": pd.DataFrame.from_records(
            [
                {"object_index": 0, "object_name": "wrong_object"},
                {"object_index": 1, "object_name": "target_object"},
                {"object_index": 2, "object_name": "other_object"},
            ]
        ),
        "token_metadata": pd.DataFrame.from_records(
            [
                {
                    "token_index": index,
                    "patch_index": index,
                    "token_space_id": "pi05.prefix",
                    "camera_id": "main",
                    "camera_slot_index": 0,
                    "token_kind": "image",
                    "prefix_mask": True,
                }
                for index in range(4)
            ]
        ),
        "source_sites": pd.DataFrame.from_records(
            [
                {
                    "trace_id": trace_id,
                    "name": "pi05.vlm.layers.8.prefix.hidden_tokens",
                    "axes": '["policy_call", "token", "channel"]',
                    "shape": "[1, 4, 2048]",
                }
            ]
        ),
        "source_rows": pd.DataFrame.from_records(
            [
                {
                    "trace_id": trace_id,
                    "policy_call_index": 0,
                    "sample_index": 0,
                    "model_id": model_id,
                }
            ]
        ),
    }
    for name, table in tables.items():
        table.to_parquet(artifact_dir / f"{name}.parquet", index=False)

    components = np.zeros((2, 2048), dtype=np.float64)
    components[0, 0] = 1.0
    components[1, 1] = 1.0
    arrays = {
        "selected_0_classes": np.array([0, 1, 2]),
        "selected_0_feature_mean": np.array([0.25, -0.5]),
        "selected_0_feature_scale": np.array([2.0, 3.0]),
        "selected_0_weights_0": np.array(
            [[-1.0, 2.0, 0.5], [0.25, -0.5, 1.5]], dtype=np.float64
        ),
        "selected_0_biases_0": np.array([-0.2, 0.4, 0.1]),
        "channel_input_center": np.zeros(2048),
        "channel_input_scale": np.ones(2048),
        "channel_pca_center": np.zeros(2048),
        "channel_components": components,
    }
    outputs = {name: f"{relative}/{name}.parquet" for name in tables}
    artifact = LensArtifact(
        artifact_id=artifact_id,
        artifact_type="object_roi_identity_study",
        name="RQ-015 linear object ROI test",
        scope="dataset",
        method={
            "schema_version": 1,
            "models": {
                "object_roi": {
                    "prefix": "selected_0",
                    "layer": 8,
                    "model": "linear",
                    "feature_dim": 2,
                }
            },
            "outputs": outputs,
        },
    )
    dataset.save_artifact(artifact, arrays=arrays)
    return artifact_id


def _target(dataset, artifact_id, *, tokens=(1, 2)) -> TargetSpec:
    return TargetSpec(
        kind="contrast_direction",
        source_artifact_id=artifact_id,
        source_artifact_type="object_roi_identity_study",
        model_id=dataset.bundles[0].manifest.model_id,
        model_family="pi05",
        model_site="pi05.vlm.layers.8.prefix.hidden_tokens",
        layer=8,
        token_space="pi05.prefix",
        token_selector={"indices": list(tokens)},
        representation={
            "method": "object_roi",
            "instance_index": 0,
            "target_class": 1,
            "target_name": "target_object",
            "contrast_class": "class_mean",
            "contrast_name": "class_mean",
        },
    )


def _resolved(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=8)
    artifact_id = _saved_probe(dataset)
    direction = resolve_object_roi_probe_direction(
        dataset,
        _target(dataset, artifact_id),
        trace_id=dataset.bundles[0].manifest.trace_id,
        policy_call_index=0,
    )
    return dataset, direction


def test_probe_direction_exactly_replays_add_and_project_out_math(tmp_path):
    _dataset, direction = _resolved(tmp_path)
    raw = np.zeros(2048, dtype=np.float64)
    strength = 0.25

    standardized_before = direction.standardized_features(raw)
    standardized_after = direction.standardized_features(raw + direction.add_delta(strength))
    np.testing.assert_allclose(
        standardized_after - standardized_before,
        strength * direction.compact_direction,
        atol=1e-12,
    )
    centered_weights = direction.classifier_weights[:, 1] - np.mean(
        direction.classifier_weights, axis=1
    )
    assert direction.score_difference(raw + direction.add_delta(strength)) - (
        direction.score_difference(raw)
    ) == pytest.approx(strength * np.linalg.norm(centered_weights), abs=1e-12)

    projected = raw + direction.project_out_delta(raw, 1.0)
    assert direction.standardized_features(projected) @ direction.compact_direction == (
        pytest.approx(0.0, abs=1e-12)
    )
    assert direction.provenance["projection_reconstruction_max_abs"] < 1e-12
    assert len(direction.provenance["array_sha256"]) == 9


def test_probe_direction_controls_are_orthogonal_or_raw_norm_matched(tmp_path):
    _dataset, direction = _resolved(tmp_path)
    random = direction.random_control(17)
    wrong = direction.with_class_pair(0, "class_mean", purpose="wrong_identity_control")

    assert random.compact_direction @ direction.compact_direction == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.norm(random.add_delta(1.0)) == pytest.approx(
        np.linalg.norm(direction.add_delta(1.0)), rel=1e-12
    )
    assert np.linalg.norm(wrong.add_delta(1.0)) == pytest.approx(
        np.linalg.norm(direction.add_delta(1.0)), rel=1e-12
    )


def test_probe_direction_rejects_request_that_disagrees_with_saved_roi(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=8)
    artifact_id = _saved_probe(dataset)

    with pytest.raises(ValueError, match="token indices disagree"):
        resolve_object_roi_probe_direction(
            dataset,
            _target(dataset, artifact_id, tokens=(0,)),
            trace_id=dataset.bundles[0].manifest.trace_id,
            policy_call_index=0,
        )


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)
        self.device = "fake-device"
        self.dtype = "bfloat16"

    @property
    def shape(self):
        return self.value.shape

    def clone(self):
        return FakeTensor(self.value.copy())

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value.copy()

    def mean(self, *, dim):
        return FakeTensor(self.value.mean(axis=dim))

    def __getitem__(self, key):
        return FakeTensor(self.value[key])

    def __setitem__(self, key, value):
        self.value[key] = value.value if isinstance(value, FakeTensor) else value

    def __add__(self, other):
        return FakeTensor(self.value + (other.value if isinstance(other, FakeTensor) else other))


class FakeLayer:
    def forward(self, hidden):
        return hidden, "preserved-tail"


class FakePolicy:
    def __init__(self):
        self.layer = FakeLayer()
        language_model = SimpleNamespace(layers=[FakeLayer() for _ in range(9)])
        language_model.layers[8] = self.layer
        paligemma = SimpleNamespace(language_model=language_model)
        self.model = SimpleNamespace(
            paligemma_with_expert=SimpleNamespace(paligemma=paligemma)
        )

    def reset(self):
        pass

    def predict_action_chunk(self, observation, *, noise):
        del observation, noise
        hidden = np.zeros((1, 4, 2048), dtype=np.float32)
        hidden[:, 1:3, 0] = 1.0
        hidden[:, 1:3, 1] = -0.25
        output, tail = self.layer.forward(FakeTensor(hidden))
        assert tail == "preserved-tail"
        roi = output.value[:, 1:3, :2].mean(axis=1)
        return FakeTensor(np.repeat(roi[:, None, :], 2, axis=1))


class FakeTorch:
    @staticmethod
    def no_grad():
        return nullcontext()

    @staticmethod
    def as_tensor(value, *, device, dtype):
        del device, dtype
        return FakeTensor(value)


def _replay_inputs(trace_id):
    return PolicyCallReplayInputs(
        config=ReplayConfig(
            benchmark="libero_object",
            task_id=0,
            layout_id=0,
            seed=0,
            horizon=2,
            obs_size=224,
        ),
        trace_id=trace_id,
        policy_call_index=0,
        observation_timestep=0,
        policy_call={"policy_call_index": 0},
        stored_action_chunk=np.zeros((2, 2), dtype=np.float32),
        initial_noise=np.zeros((2, 2), dtype=np.float32),
        initial_noise_ref="flow_initial_noise[0]",
        initial_noise_exactness="exact",
    )


def _request(direction):
    return {
        "target": {
            "kind": "contrast_direction",
            "source_artifact_id": direction.artifact_id,
            "source_artifact_type": direction.artifact_type,
            "model_family": "pi05",
            "model_site": direction.model_site,
            "layer": direction.layer,
            "token_space": "pi05.prefix",
            "token_selector": {"indices": list(direction.token_indices)},
            "representation": {
                "method": "object_roi",
                "instance_index": direction.instance_index,
                "target_class": direction.target_class,
                "contrast_class": direction.contrast_class,
            },
        },
        "intervention": {
            "request": {
                "operator": {
                    "operator": "project_out_direction",
                    "strength": 1.0,
                    "parameters": {"mode": "artifact_probe_direction"},
                },
                "schedule": {
                    "policy_calls": [0],
                    "generation_steps": "all",
                    "tokens": "target_tokens",
                },
                "controls": [
                    {"kind": "random_direction", "parameters": {"seed": 9}},
                    {
                        "kind": "wrong_feature",
                        "parameters": {
                            "role": "wrong_identity",
                            "target_class": 0,
                            "contrast_class": "class_mean",
                        },
                    },
                    {"kind": "wrong_token", "parameters": {"role": "wrong_roi"}},
                ],
            }
        },
    }


def test_executor_applies_probe_direction_and_three_runtime_free_controls(tmp_path):
    dataset, direction = _resolved(tmp_path)
    policy = FakePolicy()
    executor = PI05ActionInterventionExecutor(
        runtime=SimpleNamespace(torch=FakeTorch(), policy=policy),
        replay_inputs=_replay_inputs(dataset.bundles[0].manifest.trace_id),
        observation={},
        initial_noise=FakeTensor(np.zeros((1, 2, 2))),
        probe_direction=direction,
    )
    request = _request(direction)
    original_forward = policy.layer.forward

    intervention = executor.run_intervention(request)
    random = executor.run_control(request, control_kind="random_direction_control")
    wrong_identity = executor.run_control(request, control_kind="wrong_feature")
    wrong_roi = executor.run_control(request, control_kind="wrong_token")

    assert intervention.runtime["claim_eligible"] is True
    assert intervention.runtime["hook_calls"] == 1
    assert intervention.runtime["operator_semantics"].endswith("leaving classifier intercept")
    assert set(
        [random.control_kind, wrong_identity.control_kind, wrong_roi.control_kind]
    ) == {"matched_random", "wrong_identity", "wrong_roi"}
    assert all(
        np.isfinite(trial.metrics["probe_margin_after"])
        for trial in (intervention, random, wrong_identity, wrong_roi)
    )
    assert policy.layer.forward == original_forward
