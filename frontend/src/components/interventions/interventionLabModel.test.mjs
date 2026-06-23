import assert from "node:assert/strict";
import test from "node:test";

import {
  buildInspectedInterventionRecord,
  buildInterventionRequest,
  liveRunAvailable,
} from "./interventionLabModel.ts";
import { sourceArtifactIdForRun, summarizeInterventionRun } from "./interventionDisplay.ts";

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
  rankingMode: "probe_contribution",
  runId: "run-a",
  selectionSource: "probe_contributor",
  sourceObjectRef: {
    artifactId: "probe-a",
    artifactType: "probe_suite",
    feature: 42,
    kind: "probe_suite",
    label: "Target lifted probe",
    layer: 12,
    modelSite: "pi05.expert.layers.12.by_step.hidden_tokens",
    policyCallIndex: 2,
    probeId: "probe-a",
    rankingMode: "probe_contribution",
    timestep: 7,
    traceId: "trace-a",
  },
  strength: 1.25,
  title: "Probe intervention",
  tokenSpace: "pi05.expert_context",
  traceId: "trace-a",
  feature: 42,
  layer: 12,
  timestep: 7,
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
  assert.equal(request.target.selection_source, "probe_contributor");
  assert.equal(request.target.layer, 12);
  assert.equal(request.target.feature, 42);
  assert.equal(request.target.source_object_ref.artifact_id, "probe-a");
  assert.equal(request.target.source_object_ref.model_site, "pi05.expert.layers.12.by_step.hidden_tokens");
  assert.equal(request.target.source_object_ref.ranking_mode, "probe_contribution");
  assert.equal(request.baseline.context.trace_id, "trace-a");
  assert.equal(request.baseline.context.timestep, 7);
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
});

test("intervention lab saves unavailable runtime as inspected evidence", () => {
  const record = buildInspectedInterventionRecord(draft, preflight, "2026-06-06T00:00:00Z");

  assert.equal(record.run_id, "run-a");
  assert.equal(record.intervention_type, "intervention_record");
  assert.equal(record.readouts.status, "inspected_only");
  assert.deepEqual(record.readouts.claim.claim_strength, ["observation"]);
  assert.equal(record.provenance.source_artifact_id, "probe-a");
  assert.equal(record.provenance.source_object_ref.artifact_id, "probe-a");
  assert.equal(record.provenance.model_site, "pi05.expert.layers.12.by_step.hidden_tokens");
  assert.equal(record.provenance.feature, 42);
  assert.equal(sourceArtifactIdForRun(record), "probe-a");

  const summary = summarizeInterventionRun(record);
  assert.match(summary.sourceLabel, /Target lifted probe/);
  assert.match(summary.sourceLabel, /feature 42/);
});

test("intervention lab supports manual model-site targets without a probe artifact", () => {
  const request = buildInterventionRequest({
    ...draft,
    artifactId: "",
    artifactType: "",
    modelSite: "pi05.action_head.input",
    selectionSource: "manual_model_site",
    sourceObjectRef: {
      kind: "manual_model_site",
      modelSite: "pi05.action_head.input",
      traceId: "trace-a",
    },
    target: undefined,
  });

  assert.equal(request.target.kind, "manual");
  assert.equal(request.target.selection_source, "manual_model_site");
  assert.equal(request.target.model_site, "pi05.action_head.input");
  assert.equal(request.target.source_object_ref.kind, "manual_model_site");
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
