from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd

from vla_lens.interventions import (
    ActionBasisRequest,
    ActionBasisResult,
    action_delta_metrics,
    resolve_action_basis,
)
from vla_lens.synthetic import create_synthetic_trace_dataset
from vla_lens.traces import ArraySpec, TraceBundle, TraceManifest


def _minimal_action_bundle(path) -> TraceBundle:
    return TraceBundle.create(
        path,
        manifest=TraceManifest(
            trace_id="minimal-action",
            episode_id="minimal-action",
            task_id="minimal",
            prompt="minimal",
            model_id="minimal-model",
            env_id="minimal-env",
            robot_id="minimal-robot",
            outcome="unknown",
            length=2,
        ),
        episode_arrays={
            "action_chunks": ArraySpec(
                np.zeros((1, 2, 3), dtype=np.float32),
                ["policy_call", "horizon", "action_dim"],
            )
        },
        action_normalization=pd.DataFrame(
            {
                "normalization_id": ["identity"],
                "mode": ["identity"],
                "stats_ref": [""],
                "normalized_action_array_ref": ["action_chunks"],
            }
        ),
    )


def test_raw_action_basis_resolves_on_synthetic_action_chunks(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=8)
    bundle = dataset.bundles[0]

    result = resolve_action_basis(bundle, ActionBasisRequest(basis=("raw",)))
    raw = result.resolution("raw")

    assert result.status == "ok"
    assert raw is not None
    assert raw.available is True
    assert raw.source_dimensions["indices"] == list(range(7))
    assert raw.units["kind"] == "native_saved_action_units"
    assert raw.sign_convention["kind"] == "native_saved_action_sign"
    assert raw.action_schema_ref.action_array_ref == "action_chunks"


def test_named_action_bases_resolve_from_metadata(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=8)

    result = resolve_action_basis(
        dataset.bundles[0],
        ActionBasisRequest(basis=("gripper", "eef_delta_xyz", "rotation", "speed")),
    )
    status = result.to_dict()["basis_status"]

    assert result.status == "ok"
    assert status["gripper"]["source_dimensions"]["indices"] == [6]
    assert status["eef_delta_xyz"]["source_dimensions"]["indices"] == [0, 1, 2]
    assert status["rotation"]["source_dimensions"]["indices"] == [3, 4, 5]
    assert status["speed"]["basis_resolution"] == {
        "kind": "l2_norm",
        "source_basis": "eef_delta_xyz",
    }
    assert status["gripper"]["units"]["values"]["gripper"] == "normalized gripper command"
    assert status["gripper"]["sign_convention"]["description"]
    assert status["eef_delta_xyz"]["coordinate_frame"] == "end_effector"


def test_missing_named_action_basis_metadata_is_partial(tmp_path):
    bundle = _minimal_action_bundle(tmp_path / "minimal")

    result = resolve_action_basis(
        bundle,
        ActionBasisRequest(basis=("raw", "gripper", "eef_delta_xyz")),
    )
    status = result.to_dict()["basis_status"]

    assert result.status == "partial"
    assert result.available == ("raw",)
    assert result.missing == ("gripper", "eef_delta_xyz")
    assert status["raw"]["available"] is True
    assert status["gripper"]["available"] is False
    assert status["eef_delta_xyz"]["available"] is False
    assert result.warnings


def test_action_delta_metrics_include_normalized_delta_and_side_effect_score(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=8)
    bundle = dataset.bundles[0]
    basis = resolve_action_basis(bundle, ActionBasisRequest(basis=("raw", "gripper")))
    stored = np.asarray(bundle.action_chunks(mmap=True)[0], dtype=np.float32)
    intervened = stored.copy()
    noop = stored.copy()
    intervened[:, 6] += 0.5
    intervened[:, 0] += 0.25

    metrics = action_delta_metrics(
        stored_original=stored,
        noop=noop,
        intervened=intervened,
        basis_result=basis,
        intended_basis="gripper",
    )

    assert metrics["gripper"]["intervened_minus_stored_original"] > 0.0
    assert metrics["gripper"]["side_effect_score"] == 0.0
    assert metrics["raw"]["intervened_minus_noop"] > metrics["gripper"]["intervened_minus_noop"]
    assert metrics["raw"]["side_effect_score"] > 0.0
    assert metrics["raw"]["normalized_delta"] == metrics["raw"]["intervened_minus_stored_original"]


def test_action_basis_result_roundtrips_json(tmp_path):
    bundle = _minimal_action_bundle(tmp_path / "minimal")
    result = resolve_action_basis(bundle, ActionBasisRequest(basis=("raw", "gripper")))

    loaded = ActionBasisResult.from_dict(json.loads(json.dumps(result.to_dict())))

    assert loaded.status == "partial"
    assert loaded.available == ("raw",)
    assert loaded.missing == ("gripper",)
    assert loaded.normalization.normalization_id == "identity"


def test_action_basis_import_does_not_load_heavy_runtime_dependencies():
    code = """
import sys
import vla_lens.interventions.action_basis
banned = {"torch", "lerobot", "libero", "robosuite"}
loaded = sorted(name for name in banned if name in sys.modules)
if loaded:
    raise SystemExit("loaded heavy modules: " + ", ".join(loaded))
"""

    subprocess.run([sys.executable, "-c", code], check=True)
