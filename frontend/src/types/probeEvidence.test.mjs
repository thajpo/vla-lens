import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  defaultProbePanelSpecs,
  primitiveKinds,
  probeEpisodeLensAdapter,
  rankedMoments,
  selectAvailablePanels,
  selectContributionClaimLevel,
  selectContributionRows,
  selectCurrentMomentEvidence,
  selectTopMoments,
  selectUnavailableReasons,
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
  assert.equal(panels.unavailable_reasons.available, true);
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

test("selector helpers expose current moment evidence without fake contribution rows", () => {
  assert.equal(selectTopMoments(scalarBundle)[0].score, 0.91);
  assert.deepEqual(selectTopMoments(scalarBundle, "uncertain"), []);
  assert.deepEqual(selectTopMoments(scalarBundle, "top", 0), []);
  assert.throws(() => selectTopMoments(scalarBundle, "top", -1), /limit/);

  const current = selectCurrentMomentEvidence(scalarBundle, {
    dataset_id: scalarBundle.run.dataset_id,
    lens_id: scalarBundle.artifact.lens_id,
    lens_run_id: scalarBundle.run.lens_run_id,
    episode_id: "episode-1",
    timestep: 7,
    ranking: "top",
  });

  assert.deepEqual(current.ranked_moments.map((moment) => moment.score), [0.91]);
  assert.deepEqual(current.predictions.map((prediction) => prediction.confidence), [0.91]);
  assert.deepEqual(current.score_series.map((series) => series.episode_id), ["episode-1"]);
  assert.deepEqual(current.contributions, []);
  assert.deepEqual(current.model_loci, []);
});

test("contribution selectors are selection-aware and preserve conservative claim level", () => {
  const rows = selectContributionRows(contributionBundle, { episode_id: "episode-1", policy_call: 3 });

  assert.deepEqual(rows.map((row) => row.key), ["dim_42"]);
  assert.deepEqual(selectContributionRows(contributionBundle, { episode_id: "episode-2" }), []);
  assert.equal(selectContributionClaimLevel(contributionBundle), "numeric_only");
  assert.equal(selectContributionClaimLevel(saeBundle), "human_labeled_feature");
  assert.equal(selectContributionClaimLevel(scalarBundle), null);
});

test("unavailable reason selector filters by panel and capability", () => {
  assert.deepEqual(
    selectUnavailableReasons(scalarBundle, { panel_id: "contribution" }).map((reason) => reason.reason),
    ["missing_contribution_basis"],
  );
  assert.deepEqual(
    selectUnavailableReasons(scalarBundle, { capability: "model_locus_view" }).map((reason) => reason.panel_id),
    ["model_locus"],
  );
});

test("probe episode lens adapter provides microscope seams from evidence only", () => {
  const selection = probeEpisodeLensAdapter.defaultSelection(contributionBundle);

  assert.equal(selection.episode_id, "episode-1");
  assert.equal(selection.policy_call, 3);
  assert.equal(selection.ranking, "top");

  const annotations = probeEpisodeLensAdapter.pipelineAnnotations(contributionBundle, selection);
  const channels = probeEpisodeLensAdapter.channelRanking(contributionBundle, selection);
  const timeline = probeEpisodeLensAdapter.timelineRows(contributionBundle);

  assert.deepEqual(new Set(annotations.map((annotation) => annotation.source)), new Set(["model_locus", "contribution"]));
  assert.deepEqual(channels.map((row) => row.key), ["dim_42"]);
  assert.equal(channels[0].claim_level, "numeric_only");
  assert.deepEqual(new Set(timeline.map((row) => row.source)), new Set(["ranked_moment", "prediction"]));
  assert.equal(probeEpisodeLensAdapter.interventionSeed(contributionBundle, selection), null);

  const scalarSelection = probeEpisodeLensAdapter.defaultSelection(scalarBundle);
  assert.deepEqual(probeEpisodeLensAdapter.channelRanking(scalarBundle, scalarSelection), []);
  assert.deepEqual(probeEpisodeLensAdapter.pipelineAnnotations(scalarBundle, scalarSelection), []);
});

test("probe episode lens adapter preserves origin claims for multiple contribution primitives", () => {
  const original = contributionBundle.primitives.find((primitive) => primitive.kind === "contribution");
  const bundle = {
    ...contributionBundle,
    primitives: [
      ...contributionBundle.primitives,
      {
        ...original,
        basis: "sae_feature",
        claim_level: "human_labeled_feature",
        items: [
          {
            key: "sae_99",
            value: 0.99,
            rank: 2,
            sign: "positive",
            label: "synthetic feature",
          },
        ],
      },
    ],
  };

  const rows = probeEpisodeLensAdapter.channelRanking(bundle, {
    episode_id: "episode-1",
    policy_call: 3,
  });

  assert.deepEqual(
    rows.map((row) => [row.key, row.basis, row.claim_level]),
    [
      ["dim_42", "raw_activation_dimension", "numeric_only"],
      ["sae_99", "sae_feature", "human_labeled_feature"],
    ],
  );
});

test("probe episode lens adapter default selection preserves fallback ranking", () => {
  const bundle = {
    ...scalarBundle,
    primitives: scalarBundle.primitives.filter(
      (primitive) => !(primitive.kind === "ranked_moments" && primitive.ranking === "top"),
    ),
  };

  const selection = probeEpisodeLensAdapter.defaultSelection(bundle);
  const current = selectCurrentMomentEvidence(bundle, selection);

  assert.equal(selection.ranking, "bottom");
  assert.equal(selection.episode_id, "episode-2");
  assert.equal(selection.timestep, 1);
  assert.deepEqual(current.ranked_moments.map((moment) => moment.episode_id), ["episode-2"]);
});
