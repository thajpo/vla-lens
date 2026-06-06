from __future__ import annotations

import json
import subprocess
import sys

import pytest

from vla_lens.interventions import (
    ActionBasisRequest,
    ContextSpec,
    ControlSpec,
    DonorSpec,
    InterventionOperatorSpec,
    InterventionScheduleSpec,
    OutcomeSpec,
    PolicyCallRef,
    PreflightCheck,
    RecipientSpec,
    RuntimePreflightResult,
    RuntimeResolution,
    TargetSpec,
    TraceRef,
)


def _roundtrip(value, factory):
    encoded = json.loads(json.dumps(value.to_dict()))
    return factory(encoded)


def test_target_spec_roundtrip():
    target = TargetSpec(
        kind="probe_direction",
        source_artifact_id="probe-gripper-close",
        source_artifact_type="probe_suite",
        model_id="pi0.5",
        model_family="pi05",
        model_site="pi05.expert.layers.12.hidden_tokens",
        site_id="expert_l12_hidden",
        layer=12,
        tensor_type="hidden_state",
        token_space="action",
        token_selector={"indices": "all"},
        generation_step_selector={"steps": "all"},
        reduction="mean",
        representation={"kind": "vector", "array_ref": "artifact://probe/coef"},
        metadata={"source": "pytest"},
    )

    loaded = _roundtrip(target, TargetSpec.from_dict)

    assert loaded == target
    assert loaded.source_artifact_id == "probe-gripper-close"


def test_artifact_derived_target_requires_source_artifact():
    with pytest.raises(ValueError, match="source_artifact_id"):
        TargetSpec(
            kind="probe_direction",
            model_site="pi05.expert.layers.12.hidden_tokens",
            representation={"kind": "vector"},
        )


def test_manual_target_allows_no_source_artifact():
    target = TargetSpec(
        kind="manual",
        model_site="pi05.expert.layers.12.hidden_tokens",
        token_selector={"token_space": "action"},
        metadata={"reason": "debug"},
    )

    assert target.source_artifact_id is None


def test_operator_schedule_outcome_specs_roundtrip():
    operator = InterventionOperatorSpec(
        operator="add_direction",
        strength=1.5,
        strengths=(-1.0, 0.0, 1.0),
        parameters={"normalization": "unit_norm"},
    )
    schedule = InterventionScheduleSpec(
        policy_calls=(7,),
        generation_steps={"start": 0, "stop": 8},
        tokens={"token_space": "action", "indices": "all"},
        action_horizon="full_chunk",
    )
    outcome = OutcomeSpec(
        kind="action",
        basis=("raw", "gripper"),
        metrics=("raw_delta", "normalized_delta", "side_effect_score"),
    )
    control = ControlSpec(
        kind="random_direction",
        parameters={"matched_norm": True},
        expected_effect="near_zero",
    )

    assert _roundtrip(operator, InterventionOperatorSpec.from_dict) == operator
    assert _roundtrip(schedule, InterventionScheduleSpec.from_dict) == schedule
    assert _roundtrip(outcome, OutcomeSpec.from_dict) == outcome
    assert _roundtrip(control, ControlSpec.from_dict) == control


def test_context_recipient_donor_refs_roundtrip():
    trace = TraceRef(trace_id="trace-1", dataset_id="demo", episode_id="episode-1")
    call = PolicyCallRef(trace_id="trace-1", policy_call_index=7, timestep=14)
    context = ContextSpec(
        dataset_id="demo",
        dataset_fingerprint="fingerprint-1",
        trace_id="trace-1",
        episode_id="episode-1",
        policy_call_index=7,
        instruction="close the gripper",
    )
    recipient = RecipientSpec(trace=trace, policy_call=call)
    donor = DonorSpec(trace=trace, policy_call=call, activation_ref="artifact://donor")

    assert _roundtrip(trace, TraceRef.from_dict) == trace
    assert _roundtrip(call, PolicyCallRef.from_dict) == call
    assert _roundtrip(context, ContextSpec.from_dict) == context
    assert _roundtrip(recipient, RecipientSpec.from_dict) == recipient
    assert _roundtrip(donor, DonorSpec.from_dict) == donor


def test_action_basis_request_records_provenance_fields():
    request = ActionBasisRequest(
        basis=("raw", "gripper"),
        action_schema_ref="trace://action_normalization",
        basis_resolution={"raw": "available", "gripper": "metadata"},
        units={"raw": "normalized", "gripper": "open-positive"},
        sign_convention={"gripper": "positive_closes"},
        source_dimensions={"gripper": [6]},
        normalization={"mean": 0.0, "std": 1.0},
    )

    loaded = _roundtrip(request, ActionBasisRequest.from_dict)

    assert loaded == request
    assert set(loaded.to_dict()) >= {
        "action_schema_ref",
        "basis_resolution",
        "units",
        "sign_convention",
        "source_dimensions",
        "normalization",
    }


def test_preflight_and_runtime_resolution_roundtrip():
    check = PreflightCheck(
        name="model_runtime_available",
        status="unavailable",
        ok=False,
        message="Normal dashboard environment cannot load PI0.5 runtime.",
    )
    preflight = RuntimePreflightResult(
        status="inspected_only",
        checks=(check,),
        missing_capabilities=("model_runtime",),
        capability_status={"policy_call_exists": True, "model_runtime_available": False},
    )
    runtime = RuntimeResolution(
        adapter="pi05",
        model_family="pi05",
        model_id="openpi/pi05",
        requested_target={"kind": "probe_direction"},
        resolved_hook={"module_path": "model.expert.layers.12"},
        resolved_tensor_shape=(1, 32, 2048),
        resolved_dtype="float16",
        resolved_device="cuda:0",
    )

    assert _roundtrip(check, PreflightCheck.from_dict) == check
    assert _roundtrip(preflight, RuntimePreflightResult.from_dict) == preflight
    assert _roundtrip(runtime, RuntimeResolution.from_dict) == runtime
    assert preflight.ok is False


def test_intervention_contract_import_does_not_load_heavy_runtime_dependencies():
    code = """
import sys
import vla_lens.interventions.specs
import vla_lens.interventions.results
banned = {"torch", "lerobot", "libero", "robosuite"}
loaded = sorted(name for name in banned if name in sys.modules)
if loaded:
    raise SystemExit("loaded heavy modules: " + ", ".join(loaded))
"""

    subprocess.run([sys.executable, "-c", code], check=True)
