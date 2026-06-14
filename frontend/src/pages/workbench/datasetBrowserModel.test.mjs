import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  probeEvidenceCueForEpisode,
  probeEvidenceContextForEpisode,
  probeEvidenceRankedRows,
  probeEvidenceSpec,
  probeCalibrationRows,
  probeConfusionRows,
  probeLensSpec,
  probeTrainingDetails,
  probeResultChartRows,
  probeScoredCohortDetail,
  probeSplitBarSegments,
  probeSplitChartRows,
} from "./datasetBrowserModel.ts";

function readProbeEvidenceFixture(name) {
  const url = new URL(`../../../../tests/fixtures/probe_evidence/${name}.json`, import.meta.url);
  return JSON.parse(readFileSync(url, "utf8"));
}

test("probe split rows use indexed split-level review stats without by_trace", () => {
  const probe = {
    artifact_id: "probe-a",
    name: "Probe A",
    split_summary: { test: 200, validation: 200 },
    prediction_summary: { correct: 45, incorrect: 355, scored: 400, unscored: 0 },
    review_stats: {
      confidentWrong: 330,
      highConfidence: 370,
      scored: 400,
      test: 200,
      train: 0,
      unscored: 0,
      validation: 200,
      wrong: 355,
    },
    review_stats_by_split: {
      test: { highConfWrong: 180, scored: 200, total: 200, wrong: 190 },
      validation: { highConfWrong: 150, scored: 200, total: 200, wrong: 165 },
    },
  };

  const splitRows = probeSplitChartRows(probe);
  assert.deepEqual(
    splitRows.map((row) => [row.id, row.scored, row.total, row.wrong, row.highConfWrong]),
    [
      ["train", 0, 0, 0, 0],
      ["validation", 200, 200, 165, 150],
      ["test", 200, 200, 190, 180],
    ],
  );

  const highConfidence = probeResultChartRows(probe).find((row) => row.id === "high_confidence");
  assert.equal(highConfidence?.value, 370);
});

test("probe split rows do not invent error counts from split totals", () => {
  const probe = {
    artifact_id: "probe-a",
    name: "Probe A",
    split_summary: { test: 200, validation: 200 },
    prediction_summary: { correct: 45, incorrect: 355, scored: 400, unscored: 0 },
  };

  const splitRows = probeSplitChartRows(probe);
  assert.equal(splitRows.find((row) => row.id === "validation")?.wrong, null);
  assert.equal(splitRows.find((row) => row.id === "validation")?.highConfWrong, null);
  assert.equal(splitRows.find((row) => row.id === "test")?.wrong, null);
  assert.equal(splitRows.find((row) => row.id === "test")?.highConfWrong, null);
});

test("probe split bar segments keep unknown correctness out of correct bars", () => {
  const segments = probeSplitBarSegments({
    highConfWrong: null,
    id: "validation",
    label: "Validation",
    scored: 306,
    total: 306,
    wrong: null,
  });

  assert.equal(segments.hasErrorCounts, false);
  assert.equal(segments.correctCount, 0);
  assert.equal(segments.correctWidth, 0);
  assert.equal(segments.unknownCount, 306);
  assert.equal(segments.unknownWidth, 100);
});

test("probe calibration rows ignore records without known correctness", () => {
  const rows = probeCalibrationRows([
    { available: true, confidence: 0.91, correct: true, trace_id: "correct-high" },
    { available: true, confidence: 0.93, correct: null, trace_id: "unknown-high" },
    { available: true, confidence: 0.11, correct: false, trace_id: "wrong-low" },
  ]);

  const high = rows.find((row) => row.label === ".80-1");
  const low = rows.find((row) => row.label === "0-.20");
  assert.equal(high?.total, 1);
  assert.equal(high?.accuracy, 1);
  assert.equal(low?.total, 1);
  assert.equal(low?.accuracy, 0);
});

test("probe confusion rows keep unknown correctness separate", () => {
  const rows = probeConfusionRows([
    { actual: "cube", available: true, correct: true, predicted: "cube", trace_id: "correct" },
    { actual: "cube", available: true, correct: false, predicted: "bowl", trace_id: "wrong" },
    { actual: "cube", available: true, correct: null, predicted: "cube", trace_id: "unknown" },
  ]);

  const cube = rows.find((row) => row.label === "cube -> cube");
  const bowl = rows.find((row) => row.label === "bowl -> cube");
  assert.deepEqual(
    [cube?.correct, cube?.wrong, cube?.unknown, cube?.total],
    [1, 0, 1, 2],
  );
  assert.deepEqual(
    [bowl?.correct, bowl?.wrong, bowl?.unknown, bowl?.total],
    [0, 1, 0, 1],
  );
});

test("probe scored cohort detail distinguishes compatible rows from ranked dataset total", () => {
  assert.equal(
    probeScoredCohortDetail(
      {
        confidentWrong: 33,
        heldoutScored: 231,
        heldoutWrong: 51,
        highConfidence: 394,
        scored: 577,
        test: 140,
        train: 346,
        unscored: 0,
        validation: 91,
        wrong: 90,
      },
      1000,
    ),
    "577 scored / 1000 ranked episodes",
  );
});

test("probe training details summarize artifact method contract", () => {
  const probe = {
    artifact_id: "probe-a",
    best_feature: "layer=0.0, policy_call_index=6",
    best_model: "linear",
    name: "Probe A",
    prediction_summary: {},
    split_summary: {},
    target: "target_contacted",
  };
  const details = probeTrainingDetails(probe, {
    artifact_id: "probe-a",
    artifact_type: "probe_suite",
    display: { feature_dim: 1024, row_count: 27880 },
    method: {
      evaluation: {
        primary_metric: "balanced_accuracy",
        selection_split: "val_heldout_task",
      },
      input: {
        feature_dim: 1024,
        selector: {
          generation_step: "final",
          layers: [0, 4, 8],
          module: "pi05.expert.layers.*",
          policy_calls: [0, 1, 2, 3, 4, 5, 6],
          reduce_tokens: "mean",
          tensor_type: "hidden_tokens",
          token_kind: "action",
        },
      },
      probe: {
        library: "sklearn",
        models: ["linear", "mlp"],
        primary_model: "linear",
        trained_on_split: "train",
        type: "classification",
      },
      split: {
        eval_values: ["val_heldout_task", "test_heldout_task"],
        kind: "heldout_task",
        selection_value: "val_heldout_task",
        train_value: "train",
      },
      target: {
        kind: "classification",
        name: "target_contacted",
        resolved_column: "target_contacted",
        source: "row",
      },
    },
    name: "Probe A",
  });

  const rows = new Map(details.rows.map((row) => [row.label, row]));
  assert.equal(details.unavailable, false);
  assert.match(rows.get("X")?.value ?? "", /pi05\.expert\.layers\.\*/);
  assert.match(rows.get("Y")?.value ?? "", /target_contacted/);
  assert.match(rows.get("Objective")?.value ?? "", /logistic regression/);
  assert.match(rows.get("Split")?.value ?? "", /heldout_task/);
  assert.match(rows.get("Metric")?.value ?? "", /balanced accuracy/);
});

test("probe training details name linear regression objective as ridge regression", () => {
  const probe = {
    artifact_id: "probe-a",
    best_model: "linear",
    name: "Probe A",
    prediction_summary: {},
    split_summary: {},
    target: "outcome_score",
  };
  const details = probeTrainingDetails(probe, {
    artifact_id: "probe-a",
    artifact_type: "probe_suite",
    method: {
      probe: { primary_model: "linear", type: "regression" },
      target: { kind: "regression", name: "outcome_score" },
    },
    name: "Probe A",
  });

  const objective = details.rows.find((row) => row.label === "Objective");
  assert.equal(objective?.value, "ridge regression");
});

test("probe lens spec stays short and human-readable", () => {
  const probe = {
    artifact_id: "probe-a",
    best_feature: "layer=0.0, policy_call_index=6",
    best_model: "linear",
    name: "Probe A",
    prediction_summary: {},
    split_summary: {},
    target: "target_contacted",
  };
  const spec = probeLensSpec(probe, {
    artifact_id: "probe-a",
    artifact_type: "probe_suite",
    method: {
      evaluation: { primary_metric: "balanced_accuracy" },
      input: {
        selector: {
          generation_step: "final",
          layers: [0, 4, 8, 12, 17],
          module: "pi05.expert.layers.*",
          tensor_type: "hidden_tokens",
          token_kind: "action",
        },
      },
      probe: {
        best_model_state: { classes: ["False", "True"], probe_type: "classification" },
        primary_model: "linear",
        type: "classification",
      },
      target: {
        kind: "classification",
        name: "target_contacted",
        source: "row",
      },
    },
    name: "Probe A",
  });

  assert.equal(spec.prediction.value, "Target contacted");
  assert.equal(spec.input.value, "Expert hidden states");
  assert.equal(spec.input.detail, "action tokens · layers 0, 4, 8, 12, 17 · final step");
  assert.equal(spec.output.value, "False / True");
  assert.equal(spec.objective.value, "Logistic regression");
});

test("probe evidence spec uses canonical bundle provenance for dataset lens summary", () => {
  const bundle = readProbeEvidenceFixture("scalar_timestep");
  const spec = probeEvidenceSpec(bundle);

  assert.equal(spec.prediction.value, "Target contacted");
  assert.equal(spec.input.value, "Pooled hidden states");
  assert.equal(spec.output.value, "Scalar");
  assert.equal(spec.objective.value, "logistic regression");
});

test("probe evidence ranked rows preserve lens run and moment selection context", () => {
  const bundle = readProbeEvidenceFixture("scalar_timestep");
  const probe = {
    artifact_id: "probe-target-contacted",
    name: "Target contacted",
    prediction_summary: {},
    split_summary: {},
    by_trace: {
      "episode-1": {
        available: true,
        confidence: 0.91,
        correct: true,
        predicted: true,
        row_count: 1,
        split_category: "test",
        trace_id: "episode-1",
      },
      "episode-2": {
        available: true,
        confidence: 0.1,
        correct: false,
        predicted: false,
        row_count: 1,
        split_category: "validation",
        trace_id: "episode-2",
      },
    },
  };
  const episodes = [
    { episode_id: "episode-1", trace_id: "episode-1", prompt: "pick cube" },
    { episode_id: "episode-2", trace_id: "episode-2", prompt: "push cube" },
  ];

  const rows = probeEvidenceRankedRows({
    bundle,
    episodes,
    probe,
    selectedDatasetId: "libero_spatial",
  });

  assert.deepEqual(rows.map((row) => [row.ranking, row.moment.episode_id, row.timeLabel]), [
    ["top", "episode-1", "timestep 7"],
    ["bottom", "episode-2", "timestep 1"],
  ]);
  assert.equal(rows[0].context.probeId, "probe-target-contacted");
  assert.equal(rows[0].context.lensRunId, bundle.run.lens_run_id);
  assert.equal(rows[0].context.researchSelection.dataset_id, "libero_spatial");
  assert.equal(rows[0].context.researchSelection.lens_run_id, bundle.run.lens_run_id);
  assert.equal(rows[0].context.researchSelection.ranking, "top");
  assert.equal(rows[0].context.policyCall, null);
});

test("probe evidence cue marks table rows with score and timeline location", () => {
  const bundle = readProbeEvidenceFixture("scalar_timestep");
  const cue = probeEvidenceCueForEpisode(bundle, { episode_id: "episode-1", trace_id: "episode-1" });

  assert.equal(cue.markerLabel, "Top timestep 7");
  assert.equal(cue.scoreLabel, "score 0.910");
  assert.equal(Math.round(cue.timelinePercent), 64);
});

test("probe evidence ranked rows keep top low and uncertain visible for top-heavy bundles", () => {
  const bundle = readProbeEvidenceFixture("scalar_timestep");
  const topPrimitive = bundle.primitives.find(
    (primitive) => primitive.kind === "ranked_moments" && primitive.ranking === "top",
  );
  const topMoments = Array.from({ length: 12 }, (_, index) => ({
    episode_id: `top-${index}`,
    score: 1 - index * 0.01,
    timestep: index,
  }));
  const topHeavyBundle = {
    ...bundle,
    primitives: [
      ...bundle.primitives.filter(
        (primitive) => !(primitive.kind === "ranked_moments" && primitive.ranking === "top"),
      ),
      { ...topPrimitive, moments: topMoments },
      {
        ...topPrimitive,
        ranking: "uncertain",
        moments: [{ episode_id: "uncertain-1", score: 0.5, timestep: 5 }],
      },
    ],
  };

  const rows = probeEvidenceRankedRows({
    bundle: topHeavyBundle,
    episodes: [],
    limit: 6,
    probe: {
      artifact_id: "probe-target-contacted",
      name: "Target contacted",
      prediction_summary: {},
      split_summary: {},
    },
  });

  assert.deepEqual([...new Set(rows.map((row) => row.ranking))], ["top", "bottom", "uncertain"]);
});

test("probe evidence context for compact rows uses full bundle beyond displayed row limit", () => {
  const bundle = readProbeEvidenceFixture("scalar_timestep");
  const probe = {
    artifact_id: "probe-target-contacted",
    name: "Target contacted",
    prediction_summary: {},
    split_summary: {},
  };
  const rows = probeEvidenceRankedRows({
    bundle,
    episodes: [{ episode_id: "episode-1", trace_id: "episode-1" }],
    limit: 1,
    probe,
  });
  assert.deepEqual(rows.map((row) => row.moment.episode_id), ["episode-1"]);

  const context = probeEvidenceContextForEpisode(
    probe,
    bundle,
    { episode_id: "episode-2", trace_id: "episode-2" },
    "heldout_dataset",
  );

  assert.equal(context.probeId, "probe-target-contacted");
  assert.equal(context.lensRunId, bundle.run.lens_run_id);
  assert.equal(context.researchSelection.dataset_id, "heldout_dataset");
  assert.equal(context.researchSelection.episode_id, "episode-2");
  assert.equal(context.researchSelection.ranking, "bottom");
});
