from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from vla_lens.artifacts import LensArtifact
from vla_lens.research_guardrails import (
    check_dataset_trust,
    lint_episode_plan,
    lint_research_configs,
    validate_audit_capture_contract,
    validate_probe_claim_artifact,
)
from vla_lens.synthetic import create_synthetic_trace_dataset


def test_research_config_linter_accepts_current_configs_with_runtime_warnings():
    report = lint_research_configs(Path(__file__).resolve().parents[1])

    assert report.valid, report.to_dict()
    assert any(issue.code == "runtime_field_ignored_by_wrapper" for issue in report.warnings)


def test_research_config_linter_blocks_broad_1000_without_episode_plan(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "pi05_broad_1000_bad.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "pi05-broad-1000-bad",
                "dataset_id": "pi05-broad-1000-bad",
                "output_root": "runs/pi05-broad-1000-bad",
                "capture_profile": "mechanistic_sampled",
                "start_seed": 1,
                "seeds_per_task": 10,
                "benchmarks": ["libero_object"],
                "task_ids": {"train_seen_task": [0]},
                "seed_splits_for_train_seen_tasks": {"train": [1]},
                "device": "cuda",
                "dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )

    report = lint_research_configs(tmp_path)

    assert not report.valid
    assert "broad_1000_requires_episode_plan" in _codes(report.errors)


def test_research_config_linter_blocks_broad_audit_profile(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "pi05_diverse_500_audit.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "pi05-diverse-500-audit",
                "dataset_id": "pi05-diverse-500-audit",
                "output_root": "runs/pi05-diverse-500-audit",
                "capture_profile": "audit_windowed",
                "start_seed": 1,
                "seeds_per_task": 10,
                "benchmarks": ["libero_object", "libero_goal", "libero_spatial"],
                "task_ids": {"train_seen_task": [0, 1]},
                "seed_splits_for_train_seen_tasks": {"train": [1]},
            }
        ),
        encoding="utf-8",
    )

    report = lint_research_configs(tmp_path)

    assert not report.valid
    assert "broad_config_audit_profile" in _codes(report.errors)


def test_episode_plan_linter_blocks_broad_audit_rows(tmp_path):
    plan = tmp_path / "episode_plan.csv"
    pd.DataFrame.from_records(
        [
            {
                "dataset_id": "pi05-broad-1000-mech-light",
                "benchmark": "libero_goal",
                "task_id": 0,
                "seed": seed,
                "split": "train" if seed < 25 else "test_heldout_task",
                "capture_profile": "audit_full",
            }
            for seed in range(30)
        ]
    ).to_csv(plan, index=False)

    report = lint_episode_plan(plan)

    assert not report.valid
    assert "broad_episode_plan_audit_profile" in _codes(report.errors)


def test_dataset_trust_gate_checks_schema_splits_activation_outcomes_and_artifacts(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=4, timesteps=4)
    pd.DataFrame.from_records(
        [
            {
                "dataset_id": "synthetic",
                "trace_id": trace_id,
                "split": split,
                "benchmark": "synthetic",
                "task_id": index,
                "seed": index,
                "capture_profile": "mechanistic_sampled",
            }
            for index, (trace_id, split) in enumerate(
                [
                    ("synthetic_000", "train"),
                    ("synthetic_001", "train"),
                    ("synthetic_002", "val_heldout_task"),
                    ("synthetic_003", "test_heldout_task"),
                ]
            )
        ]
    ).to_csv(dataset.root / "probe_splits.csv", index=False)

    report = check_dataset_trust(dataset.root)

    assert report.valid, report.to_dict()
    assert report.summary["episodes"] == 4
    assert report.summary["activation_coverage_ratio"] == 1.0
    assert report.summary["artifacts"]["count"] > 0


def test_probe_claim_gate_classifies_and_rejects_overclaims():
    artifact = _probe_artifact_payload()

    smoke = validate_probe_claim_artifact(artifact)
    overclaim = validate_probe_claim_artifact(artifact, claimed_level="candidate_mechanism")

    assert smoke.classified_level == "decodable"
    assert smoke.valid
    assert not overclaim.valid
    assert overclaim.issues[0].code == "claim_level_missing_evidence"


def test_audit_capture_contract_requires_narrow_evidence_plan():
    report = validate_audit_capture_contract(
        {
            "kind": "audit_capture_contract",
            "question": "Does expert layer 8 write the target-contact feature?",
            "hypothesis": "Layer 8 MLP output shifts a contact-relevant action direction.",
            "source_dataset": {"root": "/data/pi05-broad-1000-mech-light"},
            "source_traces": ["trace-a", "trace-b"],
            "capture_profile": "audit_windowed",
            "sites": {"modules": ["pi05.expert.layers.8", "pi05.expert.layers.9"]},
            "episode_selection": {"basis": "probe_false_positive_review"},
            "budget": {"max_episodes": 2, "max_estimated_gb": 12},
            "evidence_plan": {"required_outputs": ["model sites", "fingerprints"]},
            "stop_rules": ["dataset trust gate fails"],
        }
    )

    assert report.valid, report.to_dict()


def test_audit_capture_contract_rejects_broad_matrix_fields():
    report = validate_audit_capture_contract(
        {
            "kind": "audit_capture_contract",
            "question": "too broad",
            "hypothesis": "too broad",
            "source_dataset": {"root": "/data"},
            "source_traces": "all",
            "capture_profile": "audit_full",
            "sites": {"modules": ["*"]},
            "episode_selection": {"basis": "all"},
            "budget": {"max_episodes": 100, "max_estimated_gb": 500},
            "evidence_plan": {"required_outputs": ["everything"]},
            "stop_rules": ["none"],
            "benchmarks": ["libero_object"],
            "task_ids": {"train_seen_task": [0]},
            "start_seed": 0,
            "seeds_per_task": 100,
        }
    )

    assert not report.valid
    assert {"audit_contract_too_broad", "audit_contract_contains_broad_matrix"} <= _codes(
        report.errors
    )


def _probe_artifact_payload() -> LensArtifact:
    return LensArtifact.from_dict(
        {
            "artifact_id": "probe-suite-demo",
            "artifact_type": "probe_suite",
            "name": "demo",
            "scope": "dataset",
            "selector": {"module": "pi05.expert.layers.*"},
            "method": {
                "source": {"source_traces": ["trace-a"]},
                "input": {"selector": {"module": "pi05.expert.layers.*"}},
                "target": {"name": "target_moved"},
                "split": {"kind": "heldout_task", "group_key": "trace_id"},
                "evaluation": {
                    "primary_metric": "balanced_accuracy",
                    "eval_splits": ["val_heldout_task", "test_heldout_task"],
                },
                "outputs": {
                    "predictions": "artifacts/probe-suite-demo/predictions.parquet",
                    "per_split_metrics": "artifacts/probe-suite-demo/per_split_metrics.parquet",
                    "null_metrics": "artifacts/probe-suite-demo/null_metrics.parquet",
                },
                "metadata_baseline_columns": ["benchmark", "task_id"],
            },
            "metrics": {
                "sample_count": 100,
                "best_delta": 0.08,
                "best_baseline": 0.66,
            },
            "arrays": {},
            "display": {
                "data_quality": [{"name": "split_episodes", "status": "ok"}],
                "target_distribution": {"true": 40, "false": 60},
            },
            "tags": ["probe"],
            "source_trace_ids": ["trace-a"],
            "created_utc": "2026-05-26T00:00:00+00:00",
        }
    )


def _codes(issues) -> set[str]:
    return {issue.code for issue in issues}
