from __future__ import annotations

import json

import numpy as np
import pandas as pd
import zarr

from vla_lens.interventions import (
    ActionOutcomeResult,
    ContextSpec,
    CounterfactualPairManifest,
    CounterfactualRecipe,
    DonorSpec,
    InterventionRun,
    InterventionTrial,
    PatchStudySpec,
    PatchStudyStore,
    PolicyCallRef,
    RecipientSpec,
    RuntimePreflightResult,
    RuntimeResolution,
    TargetSpec,
    TraceRef,
    expand_patch_study,
)


def _pair() -> CounterfactualPairManifest:
    return CounterfactualPairManifest(
        pair_id="pair-1",
        recipe=CounterfactualRecipe(
            kind="pose_exchange",
            target_object="caddy",
            distractor_object="mug",
            changed_variables=("caddy.pose", "mug.pose"),
            held_fixed={"prompt": True},
        ),
        recipient=RecipientSpec(
            trace=TraceRef(trace_id="recipient"),
            policy_call=PolicyCallRef(trace_id="recipient", policy_call_index=0),
        ),
        donor=DonorSpec(
            trace=TraceRef(trace_id="donor"),
            policy_call=PolicyCallRef(trace_id="donor", policy_call_index=0),
        ),
        validation={
            "token_regions": {
                "target": [1, 2],
                "wrong": [3, 4],
            }
        },
    )


def _study() -> PatchStudySpec:
    return PatchStudySpec(
        study_id="study-1",
        question="Does the patch transfer the natural action change?",
        hypothesis="A localized layer transfers some of the donor action.",
        pair_ids=("pair-1",),
        sites=({"layer": 8},),
        controls=("wrong_region",),
        shared_noise_refs=("noise-0",),
        axes={"token_regions": ["target"], "wrong_region": "wrong"},
    )


def _run(trial_id: str) -> InterventionRun:
    return InterventionRun(
        run_id=trial_id,
        title="test patch",
        status="ok",
        context=ContextSpec(
            dataset_id="demo",
            dataset_root_id="demo",
            dataset_fingerprint="abc",
            trace_id="recipient",
            policy_call_index=0,
        ),
        target=TargetSpec(
            kind="activation_slice",
            model_family="pi05",
            model_site="pi05.vlm.layers.8.prefix.hidden_tokens",
            layer=8,
            token_space="pi05.prefix",
            token_selector={"indices": [1, 2]},
        ),
        request={"operator": {"operator": "source_patch", "strength": 1.0}},
        preflight=RuntimePreflightResult(status="ok"),
        runtime_resolution=RuntimeResolution(
            adapter="pi05",
            model_family="pi05",
            requested_target={},
            resolved_hook={},
        ),
        trials=(
            InterventionTrial(
                trial_id="trial_noop",
                trial_kind="noop_rerun",
                runtime={"shared_noise_ref": "noise-0"},
            ),
            InterventionTrial(
                trial_id="trial_intervention",
                trial_kind="intervention",
                metrics={"realized_perturbation_l2": 2.0},
                runtime={
                    "shared_noise_ref": "noise-0",
                    "hook_calls": 1,
                    "recipient_token_indices": [1, 2],
                    "token_mapping_sha256": "mapping",
                },
            ),
            InterventionTrial(
                trial_id="trial_wrong_region",
                trial_kind="source_patch_control",
                control_kind="wrong_region",
                runtime={
                    "shared_noise_ref": "noise-0",
                    "hook_calls": 1,
                    "recipient_token_indices": [3, 4],
                },
            ),
        ),
        outcomes=(
            ActionOutcomeResult(
                basis="raw",
                horizon="full_chunk",
                baseline_trial_id="trial_noop",
                intervention_trial_id="trial_intervention",
                metrics={"raw_delta_norm": 0.5},
            ).to_dict(),
        ),
        controls=({"control_kind": "wrong_region", "status": "ok"},),
        outputs=("noop", "donor_shared_noise", "intervened", "control_wrong_region"),
        display={
            "counterfactual_transfer": {
                "decision": {
                    "verdict": "localized_transfer",
                    "summary": "Transfer gates passed; controls did not establish specificity.",
                }
            }
        },
    )


def test_patch_study_store_checkpoints_resumes_and_materializes_tables(tmp_path):
    study = _study()
    pair = _pair()
    plan = expand_patch_study(study, (pair,))
    store = PatchStudyStore(
        tmp_path / "study-1",
        study=study,
        pairs=(pair,),
        plan=plan,
        request_sha256="request-hash",
    )

    progress = store.prepare()
    assert progress.status == "planned"
    assert store.prepare() == progress

    arrays = {
        "noop": np.zeros((2, 7), dtype=np.float32),
        "donor_shared_noise": np.ones((2, 7), dtype=np.float32),
        "intervened": np.full((2, 7), 0.5, dtype=np.float32),
        "control_wrong_region": np.full((2, 7), 0.1, dtype=np.float32),
    }
    store.record_run(plan[0], _run(plan[0].trial_id), arrays)
    assert store.is_completed(plan[0].trial_id)

    artifact = store.finalize()

    assert artifact.provenance["status"] == "completed"
    assert len(artifact.trials) == 4
    assert len(artifact.action_arrays) == 4
    assert artifact.decisions[0].verdict == "localized_transfer"
    assert pd.read_parquet(store.root / "pairs.parquet").shape[0] == 1
    assert pd.read_parquet(store.root / "trials.parquet").shape[0] == 4
    assert pd.read_parquet(store.root / "failures.parquet").empty
    assert np.array_equal(
        np.asarray(zarr.open_group(str(store.actions_path), mode="r")[plan[0].trial_id]["noop"]),
        arrays["noop"],
    )
    saved = json.loads((store.root / "artifact.json").read_text(encoding="utf-8"))
    assert saved["study"]["study_id"] == "study-1"


def test_patch_study_store_keeps_partial_failure_for_resume(tmp_path):
    study = _study()
    pair = _pair()
    plan = expand_patch_study(study, (pair,))
    store = PatchStudyStore(
        tmp_path / "study-1",
        study=study,
        pairs=(pair,),
        plan=plan,
        request_sha256="request-hash",
    )
    store.prepare()

    store.record_failure(plan[0], RuntimeError("GPU out of memory"))
    artifact = store.finalize()

    assert store.progress().status == "partial"
    assert store.is_failed(plan[0].trial_id)
    assert artifact.provenance["failed_trial_count"] == 1
    failures = pd.read_parquet(store.root / "failures.parquet")
    assert failures.loc[0, "error_type"] == "RuntimeError"
