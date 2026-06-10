import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  lensDefaultApplicationKey,
  lensFeatureRows,
  lensReadoutLine,
  lensTimelineMarks,
  probeEvidenceCallouts,
  probeEvidenceFeatureRows,
  probeEvidencePinPayload,
  probeEvidenceReadout,
  probeEvidenceSelection,
  probeEvidenceSiteReadoutFromBundle,
  probeSourceSitesFromEvidenceBundle,
  probeEvidenceTimelineMarks,
  probeLayerReferencesFromEvidenceBundle,
  probeLayerReferencesFromLensView,
  probeTrainingLine,
  shouldApplyLensDefault,
} from "./episodeLensModel.ts";
import { probeEvidenceDefaultSiteName } from "./useProbeEvidenceInspectorActions.ts";
import { episodeHashFromSelection } from "./useEpisodeHashSync.ts";

function readProbeEvidenceFixture(name) {
  const url = new URL(`../../../../tests/fixtures/probe_evidence/${name}.json`, import.meta.url);
  return JSON.parse(readFileSync(url, "utf8"));
}

const rawContributionBundle = readProbeEvidenceFixture("raw_layer_contribution");
const scalarBundle = readProbeEvidenceFixture("scalar_timestep");

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

test("ProbeEvidenceBundle selection preserves active run context and defaults model locus", () => {
  const selection = probeEvidenceSelection(rawContributionBundle, {
    activeTraceId: "episode-1",
    currentTimestep: 0,
    initialLensRunId: "run-probe-grasp-intent",
    initialResearchSelection: {
      dataset_id: "demo",
      episode_id: "episode-1",
      lens_id: "probe-grasp-intent",
      lens_run_id: "run-probe-grasp-intent",
      policy_call: 3,
      ranking: "top",
    },
  });

  assert.equal(selection.lens_id, "probe-grasp-intent");
  assert.equal(selection.lens_run_id, "run-probe-grasp-intent");
  assert.equal(selection.dataset_id, "demo");
  assert.equal(selection.episode_id, "episode-1");
  assert.equal(selection.policy_call, 3);
  assert.equal(selection.model_locus.model_site_id, "action_head.layers.8.resid");

  const refs = probeLayerReferencesFromEvidenceBundle(rawContributionBundle, selection);
  assert.equal(refs.length, 2);
  assert.equal(refs[0].modelSiteId, "action_head.layers.8.resid");
  assert.equal(refs[0].selected, true);
  assert.equal(refs[0].trained, true);
});

test("ProbeEvidenceBundle readout and site rows stay conservative about numeric contributors", () => {
  const selection = probeEvidenceSelection(rawContributionBundle, {
    activeTraceId: "episode-1",
    currentTimestep: 0,
    initialLensRunId: "run-probe-grasp-intent",
    policyCallIndex: 3,
  });
  const readout = probeEvidenceReadout(rawContributionBundle, selection);
  const siteReadout = probeEvidenceSiteReadoutFromBundle(
    rawContributionBundle,
    selection,
    "action_head.layers.8.resid",
  );
  const contributionRows = probeEvidenceFeatureRows(rawContributionBundle, selection, "probe_contribution");
  const rawRows = probeEvidenceFeatureRows(rawContributionBundle, selection, "raw_activation");
  const marks = probeEvidenceTimelineMarks(rawContributionBundle, selection);

  assert.equal(readout.predicted, true);
  assert.equal(readout.confidence, 0.88);
  assert.equal(readout.correct, true);
  assert.equal(readout.policy_call_index, 3);
  assert.equal(siteReadout.model_site_id, "action_head.layers.8.resid");
  assert.equal(siteReadout.ranking_basis, "numeric_only");
  assert.equal(siteReadout.feature_contributors_available, true);
  assert.equal(siteReadout.feature_contributors[0].feature, 42);
  assert.equal(siteReadout.feature_contributors[0].label, null);
  assert.match(contributionRows[0].title, /not a semantic feature claim/);
  assert.equal(rawRows.length, 0);
  assert.equal(marks.find((mark) => mark.policy_call_index === 3)?.selected, true);
});

test("ProbeEvidenceBundle pin payload saves research state, not page layout", () => {
  const selection = probeEvidenceSelection(rawContributionBundle, {
    activeTraceId: "episode-1",
    currentTimestep: 0,
    policyCallIndex: 3,
  });
  const pin = probeEvidencePinPayload(rawContributionBundle, selection, "inspect later", {
    feature: 42,
    modelSiteId: "action_head.layers.8.resid",
  });

  assert.equal(pin.label, "grasp intent");
  assert.equal(pin.note, "inspect later");
  assert.equal(pin.selection.episode_id, "episode-1");
  assert.equal(pin.selection.lens_run_id, "run-probe-grasp-intent");
  assert.equal(pin.selection.policy_call, 3);
  assert.equal(pin.selection.feature_id, "dim_42");
  assert.equal(pin.selection.model_locus.model_site_id, "action_head.layers.8.resid");
  assert.equal(pin.evidence.primitive_kind, "contribution");
  assert.equal(pin.evidence.model_site_id, "action_head.layers.8.resid");
  assert.equal(pin.evidence.selected_contributor, "dim_42");
  assert.equal(pin.evidence.claim_level, "numeric_only");
  assert.equal(pin.evidence.score, 0.88);
});

test("ProbeEvidenceBundle pin payload treats selected false positives as failure evidence", () => {
  const bundle = structuredClone(rawContributionBundle);
  bundle.primitives.push({
    kind: "failure_case",
    lens_id: "probe-grasp-intent",
    lens_run_id: "run-probe-grasp-intent",
    ranking: "false_positive",
    moments: [
      {
        episode_id: "episode-1",
        label: "False positive",
        policy_call: 3,
        score: 0.88,
        timestep: 0,
      },
    ],
  });
  const selection = {
    ...probeEvidenceSelection(rawContributionBundle, {
      activeTraceId: "episode-1",
      currentTimestep: 0,
      policyCallIndex: 3,
    }),
    ranking: "false_positive",
  };
  const pin = probeEvidencePinPayload(bundle, selection);

  assert.equal(pin.evidence.primitive_kind, "failure_case");
});

test("ProbeEvidenceBundle source sites choose one default from selected model locus", () => {
  const selection = probeEvidenceSelection(rawContributionBundle, {
    activeTraceId: "episode-1",
    currentTimestep: 0,
    policyCallIndex: 3,
  });
  const bundle = {
    ...rawContributionBundle,
    primitives: [
      ...rawContributionBundle.primitives,
      {
        episode_id: "episode-1",
        kind: "model_locus",
        lens_id: "probe-grasp-intent",
        lens_run_id: "run-probe-grasp-intent",
        locus: {
          layer: 12,
          model_site_id: "action_head.layers.12.resid",
          stream: "residual",
        },
        policy_call: 3,
        source_label: "Layer 12 residual stream",
        timestep: null,
      },
    ],
  };
  const sites = probeSourceSitesFromEvidenceBundle(bundle, selection);

  assert.equal(probeEvidenceDefaultSiteName(selection), "action_head.layers.8.resid");
  assert.deepEqual(
    sites.map((site) => [site.model_site_id, site.default]),
    [
      ["action_head.layers.8.resid", true],
      ["action_head.layers.12.resid", false],
    ],
  );
});

test("ProbeEvidenceBundle timeline and callouts expose available and unavailable evidence precisely", () => {
  const selection = probeEvidenceSelection(scalarBundle, {
    activeTraceId: "episode-1",
    currentTimestep: 7,
    initialLensRunId: "run-probe-target-contacted",
  });
  const marks = probeEvidenceTimelineMarks(scalarBundle, selection);
  const callouts = probeEvidenceCallouts(scalarBundle);

  assert.deepEqual(marks.map((mark) => [mark.kind, mark.timestep, mark.selected]), [
    ["ranked_moment", 7, true],
    ["prediction", 7, true],
  ]);
  assert.match(callouts.join("\n"), /Contribution breakdown unavailable/);
  assert.match(callouts.join("\n"), /Model locus unavailable/);
});

test("episode hash preserves canonical evidence selection over stale initial route fields", () => {
  const hash = episodeHashFromSelection({
    activeSelectedProbeArtifactId: "probe-grasp-intent",
    activeSelectedSiteName: "action_head.layers.8.resid",
    activeTraceId: "episode-1",
    clampedFeature: 42,
    inspectionMode: "features",
    lensRankingMode: "raw_activation",
    lensRunId: "stale-run",
    policyCallIndex: 3,
    researchSelection: {
      dataset_id: "demo",
      episode_id: "episode-1",
      lens_id: "probe-grasp-intent",
      lens_run_id: "run-probe-grasp-intent",
      ranking: "top",
      timestep: 7,
    },
  });

  assert.equal(
    hash,
    "#episode/episode-1?probe_id=probe-grasp-intent&lens_run_id=run-probe-grasp-intent&dataset_id=demo&rank=top&call=3&timestep=7&site=action_head.layers.8.resid&feature=42&ranking=raw_activation",
  );
});
