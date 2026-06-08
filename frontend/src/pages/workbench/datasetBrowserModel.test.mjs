import assert from "node:assert/strict";
import test from "node:test";

import {
  probeLensSpec,
  probeTrainingDetails,
  probeResultChartRows,
  probeScoredCohortDetail,
  probeSplitChartRows,
} from "./datasetBrowserModel.ts";

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
    "577 compatible scored / 1000 ranked episodes",
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
