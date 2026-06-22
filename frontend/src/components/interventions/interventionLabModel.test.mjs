import assert from "node:assert/strict";
import test from "node:test";

import {
  buildBackendTargetInterventionSeed,
  buildInspectedInterventionRecord,
  buildInterventionRequest,
  liveRunAvailable,
} from "./interventionLabModel.ts";

const draft = {
  artifactId: "probe-a",
  artifactType: "probe_suite",
  basis: ["raw", "gripper"],
  controls: ["random_direction"],
  datasetFingerprint: "fingerprint-a",
  datasetId: "demo",
  modelFamily: "pi05",
  modelSite: "pi05.expert.layers.12.by_step.hidden_tokens",
  operator: "add_direction",
  policyCallIndex: 2,
  runId: "run-a",
  strength: 1.25,
  title: "Probe intervention",
  tokenSpace: "pi05.expert_context",
  traceId: "trace-a",
};

const preflight = {
  status: "inspected_only",
  ok: false,
  checks: [{ name: "model_runtime_available", status: "unavailable" }],
  warnings: ["runtime unavailable"],
  errors: [],
  runtime_resolution: { adapter: "pi05" },
  missing_capabilities: ["model_runtime_available"],
  capability_status: { model_runtime_available: false },
  target_resolution: {},
  action_basis_status: {},
  runtime_environment: {},
};

test("intervention lab builds a typed request from a probe artifact", () => {
  const request = buildInterventionRequest(draft);

  assert.equal(request.runtime_adapter, "pi05");
  assert.equal(request.target.kind, "probe_direction");
  assert.equal(request.target.source_artifact_id, "probe-a");
  assert.equal(request.target.metadata.target_source, "local_fallback");
  assert.equal(request.baseline.context.trace_id, "trace-a");
  assert.deepEqual(request.intervention.request.schedule.policy_calls, [2]);
  assert.deepEqual(request.intervention.request.outcome.basis, ["raw", "gripper"]);
  assert.deepEqual(request.intervention.request.controls, [{ kind: "random_direction" }]);
});

test("intervention lab preserves a backend-normalized target seed", () => {
  const request = buildInterventionRequest({
    ...draft,
    target: {
      kind: "probe_direction",
      source_artifact_id: "probe-a",
      source_artifact_type: "probe_suite",
      model_site: "pi05.action_head.input",
      token_space: "pi05.action_horizon",
      metadata: { artifact_family: "probe_suite", policy_call_index: "2" },
    },
  });

  assert.equal(request.target.kind, "probe_direction");
  assert.equal(request.target.model_site, "pi05.action_head.input");
  assert.equal(request.target.token_space, "pi05.action_horizon");
  assert.equal(request.target.metadata.artifact_family, "probe_suite");
  assert.equal(request.target.metadata.intended_basis, "gripper");
  assert.equal(request.target.metadata.target_source, undefined);
});

test("backend target seed carries a normalized TargetSpec into the lab draft", () => {
  const seed = buildBackendTargetInterventionSeed({
    artifactId: "probe-a",
    artifactType: "probe_suite",
    basis: ["episode_lens_view", "probe_contributors"],
    modelSite: "pi05.expert.layers.12.hidden_tokens",
    operator: "ablate",
    policyCallIndex: 3,
    target: {
      kind: "probe_direction",
      source_artifact_id: "probe-a",
      source_artifact_type: "probe_suite",
      model_site: "pi05.expert.layers.12.hidden_tokens",
      token_space: "pi05.expert_context",
      metadata: {
        artifact_family: "probe_suite",
        policy_call_index: "3",
        trace_id: "trace-a",
      },
    },
    title: "Intervene with Contact probe",
    tokenSpace: "pi05.expert_context",
    traceId: "trace-a",
  });

  assert.equal(seed.artifactId, "probe-a");
  assert.equal(seed.modelSite, "pi05.expert.layers.12.hidden_tokens");
  assert.equal(seed.policyCallIndex, 3);
  assert.equal(seed.title, "Intervene with Contact probe");
  assert.equal(seed.target.kind, "probe_direction");
  assert.equal(seed.target.source_artifact_id, "probe-a");
  assert.equal(seed.target.model_site, "pi05.expert.layers.12.hidden_tokens");
  assert.equal(seed.target.token_space, "pi05.expert_context");
  assert.equal(seed.target.metadata.trace_id, "trace-a");
});

test("target seed without backend TargetSpec leaves local fallback explicit", () => {
  const seed = buildBackendTargetInterventionSeed({
    artifactId: "probe-a",
    artifactType: "probe_suite",
    modelSite: "pi05.expert.layers.12.hidden_tokens",
    policyCallIndex: 3,
    traceId: "trace-a",
  });
  const request = buildInterventionRequest({
    ...draft,
    artifactId: seed.artifactId,
    artifactType: seed.artifactType,
    modelSite: seed.modelSite,
    policyCallIndex: seed.policyCallIndex,
    target: seed.target,
    traceId: seed.traceId,
  });

  assert.equal(seed.target, undefined);
  assert.equal(request.target.kind, "probe_direction");
  assert.equal(request.target.metadata.target_source, "local_fallback");
});

test("intervention lab saves unavailable runtime as inspected evidence", () => {
  const record = buildInspectedInterventionRecord(draft, preflight, "2026-06-06T00:00:00Z");

  assert.equal(record.run_id, "run-a");
  assert.equal(record.intervention_type, "intervention_record");
  assert.equal(record.readouts.status, "inspected_only");
  assert.deepEqual(record.readouts.claim.claim_strength, ["observation"]);
  assert.equal(record.provenance.source_artifact_id, "probe-a");
});

test("live run gate requires preflight ok and runtime capability", () => {
  assert.equal(liveRunAvailable(undefined), false);
  assert.equal(liveRunAvailable(preflight), false);
  assert.equal(
    liveRunAvailable({
      ...preflight,
      status: "ok",
      capability_status: { model_runtime_available: true },
    }),
    true,
  );
});
