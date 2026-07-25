from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from vla_lens import create_synthetic_trace_dataset
from vla_lens.probes import load_probe_spec, probe_preflight_report
from vla_lens.probes.workflow_spec import specialized_probe_family

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs" / "probes"


@pytest.mark.parametrize(
    ("config_name", "family", "selected_kind", "target_name"),
    [
        (
            "pi05_broad_1000_object_roi_identity_study.yaml",
            "object_roi_identity_study",
            "object_roi",
            "visible_object_identity",
        ),
        (
            "pi05_broad_1000_object_query_localization_study.yaml",
            "object_query_localization_study",
            "object_conditioned",
            "queried_object_patch_overlap",
        ),
        (
            "pi05_broad_1000_nonlinear_pose_capacity_study.yaml",
            "geometry_study",
            "declared_multi_feature_pooling",
            "object_pose_targets",
        ),
    ],
)
def test_specialized_probe_preflight_cli_and_report(
    tmp_path: Path,
    config_name: str,
    family: str,
    selected_kind: str,
    target_name: str,
) -> None:
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=2, timesteps=2)
    spec_path = CONFIG_ROOT / config_name

    spec = load_probe_spec(spec_path)
    report = probe_preflight_report(dataset, spec)

    assert report["preflight_kind"] == "specialized_review"
    assert report["study_family"] == family
    assert report["question"] == spec["question"]
    assert report["target"]["name"] == target_name
    assert report["cohort"]["row_unit"]
    assert report["representation"]["selected"]["kind"] == selected_kind
    assert report["representation"]["options"]
    assert report["split"]
    assert report["baselines"]["controls"]
    assert report["probe"]["models"]
    assert report["sweep"]["columns"]
    assert all(selector.get("tensor_type") != "hidden_mean" for selector in report["selectors"])

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "preflight_vla_lens_probe.py"),
            str(dataset.root),
            "--spec",
            str(spec_path),
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cli_report = json.loads(completed.stdout)
    assert cli_report["study_family"] == family
    assert cli_report["representation"]["selected"]["kind"] == selected_kind
    assert "hidden_mean" not in completed.stdout


@pytest.mark.parametrize(
    "config_name",
    [
        "pi05_broad_1000_identity_patch_localization_study.yaml",
        "pi05_broad_1000_image_location_study.yaml",
        "pi05_broad_1000_object_motion_followup_study.yaml",
    ],
)
def test_lookalike_study_specs_do_not_dispatch_to_specialized_preflight(
    config_name: str,
) -> None:
    payload = yaml.safe_load((CONFIG_ROOT / config_name).read_text(encoding="utf-8"))

    assert specialized_probe_family(payload) is None
