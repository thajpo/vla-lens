import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  defaultProbePanelSpecs,
  primitiveKinds,
  rankedMoments,
  selectAvailablePanels,
} from "./probeEvidence.ts";

function readFixture(name) {
  const url = new URL(`../../../tests/fixtures/probe_evidence/${name}.json`, import.meta.url);
  return JSON.parse(readFileSync(url, "utf8"));
}

const scalarBundle = readFixture("scalar_timestep");
const pooledBundle = readFixture("pooled_no_contribution");
const contributionBundle = readFixture("raw_layer_contribution");
const saeBundle = readFixture("sae_feature_contribution");
const headBundle = readFixture("attention_head_grouped");

function panelsById(bundle) {
  return Object.fromEntries(
    selectAvailablePanels(bundle, defaultProbePanelSpecs()).map((panel) => [panel.panel_id, panel]),
  );
}

test("probe evidence helpers consume shared Python-generated fixture payloads", () => {
  assert.deepEqual([...primitiveKinds(scalarBundle)].sort(), [
    "prediction",
    "provenance",
    "ranked_moments",
    "score_series",
  ]);
  assert.equal(rankedMoments(scalarBundle, "top")[0].timestep, 7);
  assert.deepEqual(rankedMoments(scalarBundle, "bottom")[0].episode_id, "episode-2");
  assert.deepEqual(pooledBundle, scalarBundle);
});

test("panel availability uses unavailable reasons instead of fake contribution/model panels", () => {
  const panels = panelsById(scalarBundle);

  assert.equal(panels.probe_provenance.available, true);
  assert.equal(panels.score_series.available, true);
  assert.equal(panels.ranked_moments.available, true);
  assert.equal(panels.contribution.available, false);
  assert.equal(panels.contribution.reason, "missing_contribution_basis");
  assert.equal(panels.model_locus.available, false);
  assert.equal(panels.model_locus.reason, "pooled_representation");
});

test("contribution-capable fixtures render model-locus and contribution panels", () => {
  for (const bundle of [contributionBundle, saeBundle, headBundle]) {
    const panels = panelsById(bundle);

    assert.equal(panels.contribution.available, true);
    assert.equal(panels.model_locus.available, true);
  }

  const contribution = contributionBundle.primitives.find((item) => item.kind === "contribution");
  const saeContribution = saeBundle.primitives.find((item) => item.kind === "contribution");
  const headContribution = headBundle.primitives.find((item) => item.kind === "contribution");

  assert.equal(contribution.claim_level, "numeric_only");
  assert.equal(saeContribution.claim_level, "human_labeled_feature");
  assert.equal(headContribution.claim_level, "grouped_model_locus");
});

test("geometry mismatch blocks panels even when capability and primitive exist", () => {
  const [availability] = selectAvailablePanels(contributionBundle, [
    {
      panel_id: "sae_only_model_locus",
      consumes: ["model_locus"],
      requires_capabilities: ["model_locus_view"],
      requires_geometry: { input_basis: "sae_feature" },
      unavailable_copy: "SAE-only model locus panel unavailable.",
    },
  ]);

  assert.equal(availability.available, false);
  assert.match(availability.reason, /geometry input_basis/);
});
