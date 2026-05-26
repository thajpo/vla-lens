import assert from "node:assert/strict";
import test from "node:test";

import {
  datasetBrowserCapabilityGates,
  episodeCapabilityGates,
  episodeQueryGates,
} from "./capabilityGating.ts";

const fullInputs = {
  activeTraceId: "fake_000",
  activeSelectedProbeArtifactId: "probe_001",
  activeSelectedSiteName: "fake.backbone.layers.0.hidden",
  activeCall: { index: 0 },
  attentionSiteName: "fake.attention.layers.0",
  effectiveSelectedExpertToken: 0,
  expertTokenSiteName: "fake.action_head.output",
  inspectorContext: "vlm",
  selectedPatch: { camera: "main", row: 1, col: 2 },
  selectedSiteHasFeatures: true,
};

test("missing capability manifest keeps legacy datasets fully enabled", () => {
  assert.deepEqual(episodeCapabilityGates(undefined), {
    hasPolicyCalls: true,
    hasModelSites: true,
    hasTokenSpaces: true,
    hasImageTokenMaps: true,
    hasAttentionMaps: true,
    hasActionGeneration: true,
    hasProbeArtifacts: true,
  });
  assert.deepEqual(datasetBrowserCapabilityGates(undefined), {
    hasProbeArtifacts: true,
  });
});

test("false model and probe capabilities disable dependent episode queries", () => {
  const capabilities = episodeCapabilityGates({
    policy_calls: false,
    model_sites: false,
    token_spaces: false,
    image_token_maps: false,
    attention_maps: false,
    action_generation: false,
    probe_artifacts: false,
  });
  const gates = episodeQueryGates(capabilities, fullInputs);

  assert.deepEqual(gates, {
    policyCalls: false,
    episodeProbes: false,
    activationSites: false,
    observationalComparisons: false,
    generation: false,
    activationSlice: false,
    imageTokenMap: false,
    patchFeatures: false,
    expertTokenActivations: false,
    expertTokenDetails: false,
    attentionMap: false,
    promptAttention: false,
    promptFeatureMap: false,
  });
});

test("query gates encode the narrow dependency for each inspector affordance", () => {
  const base = episodeCapabilityGates({});

  assert.equal(episodeQueryGates(base, fullInputs).imageTokenMap, true);
  assert.equal(episodeQueryGates(base, fullInputs).patchFeatures, true);
  assert.equal(episodeQueryGates(base, fullInputs).promptFeatureMap, true);
  assert.equal(
    episodeQueryGates(base, { ...fullInputs, inspectorContext: "attention" }).attentionMap,
    true,
  );
  assert.equal(
    episodeQueryGates(base, { ...fullInputs, inspectorContext: "expert" }).generation,
    true,
  );
  assert.equal(
    episodeQueryGates(base, { ...fullInputs, inspectorContext: "expert" }).expertTokenDetails,
    true,
  );
  assert.equal(
    episodeQueryGates(base, {
      ...fullInputs,
      expertTokenSiteName: "",
      inspectorContext: "expert",
    }).expertTokenDetails,
    false,
  );

  const withoutImageMaps = episodeCapabilityGates({ image_token_maps: false });
  assert.equal(episodeQueryGates(withoutImageMaps, fullInputs).imageTokenMap, false);
  assert.equal(episodeQueryGates(withoutImageMaps, fullInputs).patchFeatures, false);
  assert.equal(episodeQueryGates(withoutImageMaps, fullInputs).promptFeatureMap, true);

  const withoutTokenSpaces = episodeCapabilityGates({ token_spaces: false });
  assert.equal(
    episodeQueryGates(withoutTokenSpaces, {
      ...fullInputs,
      inspectorContext: "attention",
    }).attentionMap,
    false,
  );
  assert.equal(episodeQueryGates(withoutTokenSpaces, fullInputs).promptFeatureMap, false);

  const withoutProbeArtifacts = episodeCapabilityGates({ probe_artifacts: false });
  assert.equal(episodeQueryGates(withoutProbeArtifacts, fullInputs).episodeProbes, false);
  assert.equal(episodeQueryGates(withoutProbeArtifacts, fullInputs).observationalComparisons, false);
});

test("dataset browser disables probe index fetches when probes are absent", () => {
  assert.deepEqual(datasetBrowserCapabilityGates({ probe_artifacts: false }), {
    hasProbeArtifacts: false,
  });
});
