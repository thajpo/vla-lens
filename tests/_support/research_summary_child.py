from pathlib import Path

from vla_lens.research_child import study_fingerprint
from vla_lens.research_plan import load_research_plan, research_plan_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[2]
PROGRAM = load_research_plan(
    REPO_ROOT / "configs/campaigns/rq024_controlled_scene_to_behavior.yaml"
)
SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64
SHA_D = "sha256:" + "d" * 64


def locked_ref(name: str, sha256: str) -> dict:
    return {"id": name, "type": f"{name}_manifest", "path": f"locked/{name}.json", "sha256": sha256}


def gate(
    gate_id: str,
    role: str,
    value_id: str,
    operator: str,
    threshold: float,
    unit: str,
) -> dict:
    return {
        "id": gate_id,
        "role": role,
        "value_id": value_id,
        "operator": operator,
        "threshold": threshold,
        "unit": unit,
        "evidence_artifact_type": "analysis_evidence",
    }


def child(study_id: str, *, result_kind: str = "effect_estimate") -> dict:
    study = next(item for item in PROGRAM["studies"] if item["id"] == study_id)
    predecessors = [
        {
            "sequence": index + 1,
            "event_id": f"{predecessor.lower()}-result-recorded",
            "event_sha256": SHA_D,
        }
        for index, predecessor in enumerate(study["entry_conditions"]["requires_all_completed"])
    ]
    expected_count = 72 if study_id == "FOUNDATION" else 10
    expected_units = (
        {"task_families": 24, "scene_clusters": 24, "noise_repeats": 0, "rollouts": 72}
        if study_id == "FOUNDATION"
        else {"task_families": 6, "scene_clusters": 24, "noise_repeats": 4, "rollouts": 36}
    )
    if study_id == "FOUNDATION":
        gates = [
            gate(
                "discovery_eligible_count",
                "positive",
                "discovery_eligible_count",
                "greater_than_or_equal",
                6,
                "task_object_families",
            ),
            gate(
                "confirmation_eligible_count",
                "positive",
                "confirmation_eligible_count",
                "greater_than_or_equal",
                6,
                "task_object_families",
            ),
            gate(
                "discovery_total_count",
                "integrity",
                "discovery_total_count",
                "equal",
                12,
                "task_object_families",
            ),
            gate(
                "confirmation_total_count",
                "integrity",
                "confirmation_total_count",
                "equal",
                12,
                "task_object_families",
            ),
            gate(
                "trial_matrix_complete",
                "integrity",
                "completed_trial_count",
                "equal",
                72,
                "rollout_trials",
            ),
            gate(
                "either_pool_below_six",
                "negative",
                "minimum_pool_eligible_count",
                "less",
                6,
                "task_object_families",
            ),
        ]
    else:
        gates = [
            gate(
                "trial_matrix_complete", "integrity", "accounted_trial_count", "equal", 10, "trials"
            ),
            gate(
                "primary_interval_above_zero",
                "positive",
                "primary_interval_low",
                "greater",
                0,
                "dimensionless_gain",
            ),
            gate(
                "effect_below_useful_margin",
                "negative",
                "primary_interval_high",
                "less",
                0.2,
                "dimensionless_gain",
            ),
        ]
    defaults = list(PROGRAM["child_contract_defaults"]["required_lock_fields"])
    additions = list(study["child_contract"]["additional_lock_fields"])
    locked_choices = {name: f"locked:{name}" for name in additions}
    return {
        "schema_version": 1,
        "kind": "vla_lens.research_child",
        "child_plan_id": f"rq024-{study_id.lower()}-r1",
        "revision": 1,
        "prepared_by": "planner-agent",
        "program": {
            "path": "configs/campaigns/rq024_controlled_scene_to_behavior.yaml",
            "program_id": PROGRAM["program_id"],
            "fingerprint": research_plan_fingerprint(PROGRAM),
        },
        "study": {
            "id": study_id,
            "fingerprint": study_fingerprint(study),
            "phase": study["phase"],
        },
        "predecessor_result_events": predecessors,
        "claim": {
            "result_kind": result_kind,
            "question": study["question"],
            "allowed_conclusions": list(study["allowed_conclusions"]),
            "forbidden_conclusions": list(study["forbidden_conclusions"]),
        },
        "cohort": {
            "family_pool": study["data_scope"]["family_pool"],
            "pool_phase": study["data_scope"]["pool_phase"],
            "requires_gate": study["data_scope"]["requires_gate"],
            "read_namespaces": list(study["data_scope"]["read_namespaces"]),
            "write_namespace": study["data_scope"]["write_namespace"],
            "selection_allowed": study["data_scope"]["selection_allowed"],
            "manifest": locked_ref("cohort", SHA_A),
            "exposure_log": locked_ref("exposure", SHA_B),
        },
        "trials": {
            "manifest": locked_ref("trials", SHA_C),
            "expected_count": expected_count,
            "stable_id_fields": ["child_plan_id", "trial_id"],
            "seed_domains": ["environment", "policy", "flow_noise"],
            "expected_independent_units": expected_units,
        },
        "measurement": {
            "primary": {
                "metric_id": study["primary_claim"]["metric_id"],
                "formula": study["primary_claim"]["definition"],
                "implementation_id": f"{study_id.lower()}-metric-v1",
                "unit": study["primary_claim"]["unit"],
                "direction": study["primary_claim"]["direction"],
                "minimum_useful_effect": (
                    "six eligible families in each fixed pool" if study_id == "FOUNDATION" else 0.2
                ),
            },
            "controls": list(study["controls"]),
            "strongest_control_metric_id": "irrelevant_object_gain",
            "inference": {
                "method": "hierarchical_bootstrap",
                "level": 0.95,
                "grouping_unit": "task_object_family",
                "replicates": 10000,
                "seed": 24001,
            },
        },
        "decision": {
            "gate_components": gates,
            "positive_combiner": "all",
            "negative_combiner": "all",
            "inconclusive_rule": "neither positive nor negative gate passes",
            "invalid_conditions": ["trial matrix invalid"],
        },
        "runtime": {
            "model": {"repo_id": "pi05", "revision": "commit", "snapshot_manifest_sha256": SHA_A},
            "environment": {
                "backend": "rocm",
                "package_receipt": locked_ref("environment", SHA_A),
                "camera_config_sha256": SHA_A,
                "controller_config_sha256": SHA_A,
                "preprocessor_config_sha256": SHA_A,
                "postprocessor_config_sha256": SHA_A,
            },
            "code": {"implementation_commit": "a" * 40, "source_tree_sha256": SHA_A},
            "runner": {
                "entrypoint": "scripts/pi05_batch_capture.sh",
                "argv": ["--backend", "rocm"],
                "config": locked_ref("runner", SHA_A),
            },
        },
        "budget": {
            "max_model_calls": min(10, study["budget"]["max_model_calls"]),
            "max_action_generations": min(10, study["budget"]["max_action_generations"]),
            "max_full_rollouts": (
                72 if study_id == "FOUNDATION" else min(10, study["budget"]["max_full_rollouts"])
            ),
            "max_simulator_steps": min(100, study["budget"]["max_simulator_steps"]),
            "max_probe_fits": min(10, study["budget"].get("max_probe_fits", 0)),
            "max_persistent_gb": min(1, study["budget"]["max_additional_persistent_gb"]),
            "max_ephemeral_gb": min(1, study["budget"]["max_ephemeral_gb"]),
            "min_free_space_gb": PROGRAM["program_budget"]["min_free_space_gb"],
        },
        "output": {
            "root": "/mnt/new-volume/vla-lens/rq024",
            "namespace": f"rq024/{study_id.lower()}",
            "attempt_ledger": "events/trials",
            "required_artifact_types": list(study["required_outputs"]),
        },
        "completion": {
            "valid_trial_statuses": ["completed"],
            "technical_retry_rule": "append only",
            "resume_identity_fields": ["child_fingerprint", "trial_id"],
        },
        "protocol_lock": {
            "required_lock_fields": list(dict.fromkeys([*defaults, *additions])),
            "locked_choices": locked_choices,
        },
    }
