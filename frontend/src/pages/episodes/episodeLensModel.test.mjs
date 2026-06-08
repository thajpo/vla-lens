import assert from "node:assert/strict";
import test from "node:test";

import {
  lensDefaultApplicationKey,
  lensFeatureRows,
  lensReadoutLine,
  lensTimelineMarks,
  probeLayerReferencesFromLensView,
  probeTrainingLine,
  shouldApplyLensDefault,
} from "./episodeLensModel.ts";

const baseView = {
  schema_version: "episode_lens_view.v1",
  family: "probe_suite",
  available: true,
  unavailable_reason: null,
  lens: {
    artifact_id: "probe-target-contacted",
    artifact_type: "probe_suite",
    family: "probe_suite",
    display_name: "Target contacted",
    spec: { prediction: "Target contacted" },
  },
  episode: { trace_id: "trace-1", dataset_id: "demo", episode_index: 3 },
  current_selection: { trace_id: "trace-1" },
  resolved_selection: {
    trace_id: "trace-1",
    timestep: 42,
    policy_call_index: 6,
    model_site_id: "action_head.layers.2.resid",
    layer: 2,
    feature: 4,
    mode: "features",
  },
  recommended_selection: {
    trace_id: "trace-1",
    timestep: 42,
    policy_call_index: 6,
    model_site_id: "action_head.layers.2.resid",
    layer: 2,
    mode: "features",
  },
  readout: {
    predicted: true,
    actual: false,
    confidence: 0.97,
    correct: false,
    split: "test",
    verdict: "high_conf_wrong",
  },
  annotations: {
    pipeline: [
      {
        model_site_id: "action_head.layers.0.resid",
        layer: 0,
        role: "source",
        trained: true,
        selected: false,
        default: false,
        severity: "neutral",
        label: "Layer 0",
      },
      {
        model_site_id: "action_head.layers.2.resid",
        layer: 2,
        role: "source",
        trained: true,
        selected: true,
        default: true,
        severity: "high_conf_wrong",
        label: "Layer 2",
      },
    ],
    timeline: [
      {
        timestep: 42,
        policy_call_index: 6,
        kind: "prediction",
        value: 0.97,
        verdict: "high_conf_wrong",
        label: "Probe readout",
        selected: true,
      },
    ],
    overlays: [],
    callouts: [],
  },
  inspector: {
    default_mode: "features",
    default_ranking_id: "probe_contributors",
    pipeline_marks: [
      {
        model_site_id: "action_head.layers.0.resid",
        layer: 0,
        role: "source",
        trained: true,
        selected: false,
        default: false,
        severity: "neutral",
        label: "Layer 0",
      },
      {
        model_site_id: "action_head.layers.2.resid",
        layer: 2,
        role: "source",
        trained: true,
        selected: true,
        default: true,
        severity: "high_conf_wrong",
        label: "Layer 2",
      },
    ],
    timeline_marks: [
      {
        timestep: 42,
        policy_call_index: 6,
        kind: "prediction",
        value: 0.97,
        verdict: "high_conf_wrong",
        label: "Probe readout",
        selected: true,
      },
    ],
    overlay_marks: [],
    rankings: [
      { id: "probe_contributors", label: "Probe contributors", kind: "feature_ranking", available: true },
      { id: "raw_activations", label: "Raw activations", kind: "feature_ranking", available: true },
    ],
    callouts: [],
  },
  view: {
    probe: {
      training_spec: {
        model: "linear",
        probe_type: "classification",
        layers: [0, 2],
        policy_calls: [4, 5, 6],
      },
    },
    site_readout: {
      available: true,
      model_site_id: "action_head.layers.2.resid",
      layer: 2,
      policy_call_index: 6,
      ranking_basis: "linear_logit_contribution",
      ranking_order: "absolute_value",
      top_k: 2,
      total_features: 12,
      feature_contributors_available: true,
      probe_contribution_ranking_available: true,
      raw_activation_ranking_available: true,
      feature_contributors: [
        {
          feature: 4,
          weight: 0.5,
          normalized_activation: 0.8,
          activation: 1.4,
          contribution: 0.4,
          abs_contribution: 0.4,
          rank: 1,
          direction: "positive",
          sign_label: "supports True",
        },
      ],
      raw_activation_ranking: [
        { feature: 9, activation: -2.5, abs_activation: 2.5, rank: 1 },
      ],
    },
  },
  actions: [],
};

test("recommended LensView selection applies once per active artifact and trace", () => {
  assert.equal(lensDefaultApplicationKey(baseView), "probe-target-contacted:trace-1");
  assert.equal(shouldApplyLensDefault(baseView, ""), true);
  assert.equal(shouldApplyLensDefault(baseView, "probe-target-contacted:trace-1"), false);
});

test("pipeline annotations become probe layer refs for existing inspector", () => {
  const refs = probeLayerReferencesFromLensView(baseView);

  assert.equal(refs.length, 2);
  assert.equal(refs[1].artifactId, "probe-target-contacted");
  assert.equal(refs[1].modelSiteId, "action_head.layers.2.resid");
  assert.equal(refs[1].policyCall, 6);
  assert.equal(refs[1].confidence, 0.97);
  assert.equal(refs[0].trained, true);
  assert.equal(refs[0].selected, false);
  assert.equal(refs[1].selected, true);
});

test("LensView ranking helper switches between contributors and raw activations", () => {
  const contributorRows = lensFeatureRows(baseView, "probe_contribution");
  const rawRows = lensFeatureRows(baseView, "raw_activation");

  assert.equal(contributorRows[0].index, 4);
  assert.equal(contributorRows[0].value, 0.4);
  assert.equal(contributorRows[0].label, "supports True");
  assert.equal(rawRows[0].index, 9);
  assert.equal(rawRows[0].value, -2.5);
});

test("compact lens readout stays short enough for subordinate inspector context", () => {
  assert.equal(lensReadoutLine(baseView.readout), "High-conf wrong · confidence 0.970 · Test split");
});

test("LensView exposes temporal marks and compact probe training line", () => {
  assert.equal(lensTimelineMarks(baseView).length, 1);
  assert.equal(lensTimelineMarks(baseView)[0].policy_call_index, 6);
  assert.equal(probeTrainingLine(baseView), "Trained on L0, L2 · policy calls 4-6");
});
