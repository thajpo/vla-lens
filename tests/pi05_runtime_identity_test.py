from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from vla_lens.pi05.batch_capture import (
    _capture_commands,
    _read_episode_plan,
    _validate_batch_config,
    _validate_episode_rows,
)
from vla_lens.pi05.capture import parse_args
from vla_lens.pi05.capture_runner import _flow_noise_call_seed, _seed_bundle
from vla_lens.pi05.runtime_identity import (
    canonical_component_identities,
    canonical_sha256,
    require_immutable_revision,
    resolve_immutable_checkpoint,
)

REVISION = "a" * 40


def _exact_config(tmp_path: Path) -> dict[str, object]:
    return {
        "dataset_id": "dataset-a",
        "output_root": str(tmp_path),
        "model_id": "lerobot/pi05_libero_finetuned",
        "model_revision": REVISION,
        "capture_profile": "mechanistic_sampled",
        "storage_dtype": "float16",
        "obs_size": 256,
        "device": "cuda",
        "dtype": "bfloat16",
        "estimated_mb_per_episode": 1,
        "minimum_free_gb_after_capture": 0,
    }


def _write_exact_plan(path: Path) -> None:
    fields = [
        "dataset_id",
        "benchmark",
        "task_id",
        "seed",
        "split",
        "capture_profile",
        "trial_id",
        "child_plan_id",
        "canonical_family_id",
        "pool",
        "replicate_id",
        "reset_seed",
        "environment_seed",
        "policy_seed",
        "flow_noise_seed",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "dataset_id": "dataset-a",
                "benchmark": "libero_object",
                "task_id": 1,
                "seed": 101,
                "split": "foundation",
                "capture_profile": "mechanistic_sampled",
                "trial_id": "trial-001",
                "child_plan_id": "rq024-foundation-r1",
                "canonical_family_id": "family-01",
                "pool": "discovery",
                "replicate_id": "replicate-1",
                "reset_seed": 201,
                "environment_seed": 301,
                "policy_seed": 401,
                "flow_noise_seed": 501,
            }
        )


def test_canonical_component_hashes_are_order_independent() -> None:
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})
    identities = canonical_component_identities(
        obs_size=256,
        model_id="model",
        model_revision=REVISION,
        device="cuda",
        dtype="bfloat16",
    )
    assert identities["camera_config_sha256"].startswith("sha256:")
    assert identities["controller_config"] == {"control_mode": "relative"}


def test_immutable_checkpoint_receipt_records_exact_snapshot_manifest(tmp_path: Path) -> None:
    snapshot = tmp_path / REVISION
    snapshot.mkdir()
    (snapshot / "config.json").write_text('{"model":"pi05"}\n', encoding="utf-8")

    resolved, receipt = resolve_immutable_checkpoint(
        "org/model",
        REVISION,
        snapshot_download=lambda **_: str(snapshot),
    )

    assert resolved == snapshot.resolve()
    assert receipt["requested_revision"] == REVISION
    assert receipt["resolved_revision"] == REVISION
    assert receipt["snapshot_manifest_sha256"].startswith("sha256:")
    assert receipt["files"][0]["path"] == "config.json"


def test_mutable_checkpoint_revision_is_rejected() -> None:
    with pytest.raises(ValueError, match="40-character"):
        require_immutable_revision("main")


def test_exact_plan_retains_trial_and_independent_seed_identities(tmp_path: Path) -> None:
    plan = tmp_path / "episodes.csv"
    _write_exact_plan(plan)
    config = _exact_config(tmp_path)

    _validate_batch_config(config, exact=True, has_episode_plan=True)
    rows = _read_episode_plan(plan, exact=True)
    _validate_episode_rows(rows, exact=True)
    command = _capture_commands(config, tmp_path, rows, exact=True)[0].command

    assert rows[0].child_plan_id == "rq024-foundation-r1"
    assert rows[0].canonical_family_id == "family-01"
    assert rows[0].flow_noise_seed == 501
    for flag, value in {
        "--model-revision": REVISION,
        "--trial-id": "trial-001",
        "--child-plan-id": "rq024-foundation-r1",
        "--canonical-family-id": "family-01",
        "--pool": "discovery",
        "--reset-seed": "201",
        "--environment-seed": "301",
        "--policy-seed": "401",
        "--flow-noise-seed": "501",
    }.items():
        assert command[command.index(flag) + 1] == value
    assert "--exact-runtime" in command


def test_exact_capture_parser_requires_complete_single_trial() -> None:
    with pytest.raises(ValueError, match="trial/child/family/pool/replicate"):
        parse_args(["--model-revision", REVISION, "--exact-runtime", "--episodes", "1"])

    args = parse_args(
        [
            "--model-revision",
            REVISION,
            "--exact-runtime",
            "--episodes",
            "1",
            "--trial-id",
            "trial-1",
            "--child-plan-id",
            "rq024-foundation-r1",
            "--canonical-family-id",
            "family-1",
            "--pool",
            "discovery",
            "--replicate-id",
            "replicate-1",
            "--reset-seed",
            "1",
            "--environment-seed",
            "2",
            "--policy-seed",
            "3",
            "--flow-noise-seed",
            "4",
        ]
    )
    assert _seed_bundle(args, 99) == (1, 2, 3, 4)
    assert _flow_noise_call_seed(4, 0) == _flow_noise_call_seed(4, 0)
    assert _flow_noise_call_seed(4, 0) != _flow_noise_call_seed(4, 1)


def test_persisted_plan_runtime_identity_is_machine_readable(tmp_path: Path) -> None:
    plan = tmp_path / "episodes.csv"
    _write_exact_plan(plan)
    rows = _read_episode_plan(plan, exact=True)
    config = _exact_config(tmp_path)

    from vla_lens.pi05.batch_capture import _write_plan_files

    _write_plan_files(tmp_path, config=config, rows=rows)
    receipt = json.loads((tmp_path / "runtime_identity.json").read_text(encoding="utf-8"))
    assert receipt["kind"] == "vla_lens.pi05_runtime_identity"
    assert receipt["model"]["revision"] == REVISION
    assert receipt["components"]["preprocessor_config_sha256"].startswith("sha256:")
