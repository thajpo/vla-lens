import assert from "node:assert/strict";
import test from "node:test";

import {
  formatResearchProgress,
  formatResearchResult,
  groupResearchRuns,
  researchArtifactDestination,
} from "./researchRunsModel.ts";

function run(overrides = {}) {
  return {
    run_id: "campaign",
    parent_run_id: null,
    kind: "campaign",
    name: "Semantic object wave",
    question: "Where are objects represented?",
    status: "running",
    stage: "training",
    progress: { completed: 1, total: 3, unit: "experiments", fraction: 1 / 3 },
    artifact_ids: [],
    result: { metric: "", score: null, baseline: null, delta: null, verdict: "" },
    error: null,
    created_utc: "2026-07-22T10:00:00Z",
    updated_utc: "2026-07-22T11:00:00Z",
    provenance: {},
    ...overrides,
  };
}

test("research runs group children under their campaign and keep orphans visible", () => {
  const groups = groupResearchRuns([
    run(),
    run({ run_id: "child", parent_run_id: "campaign", kind: "probe" }),
    run({ run_id: "orphan", parent_run_id: "missing" }),
  ]);

  assert.deepEqual(groups.map((group) => group.run.run_id), ["campaign", "orphan"]);
  assert.deepEqual(groups[0].children.map((child) => child.run_id), ["child"]);
});

test("research progress and result copy retains the underlying counts", () => {
  assert.equal(
    formatResearchProgress({ completed: 1, total: 3, unit: "experiments" }),
    "1 / 3 experiments · 33%",
  );
  assert.equal(
    formatResearchResult({ metric: "AP", score: 0.76, baseline: 0.51, delta: 0.25, verdict: "positive" }),
    "0.760 vs 0.510 (+0.250)",
  );
});

test("artifact links use the run kind to choose the existing evidence screen", () => {
  assert.equal(researchArtifactDestination(run({ kind: "probe" })), "probes");
  assert.equal(researchArtifactDestination(run({ kind: "intervention_sweep" })), "interventions");
});
