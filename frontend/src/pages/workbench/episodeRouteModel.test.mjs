import assert from "node:assert/strict";
import test from "node:test";

import { evidencePinHash } from "../evidencePinsModel.ts";
import { parseEpisodeRoute } from "./episodeRouteModel.ts";

test("episode route parser restores evidence pin research state", () => {
  const hash = evidencePinHash({
    pin_id: "pin-1",
    created_utc: "2026-06-10T00:00:00+00:00",
    label: "Probe evidence",
    selection: {
      dataset_id: "demo",
      episode_id: "episode-1",
      lens_id: "probe-grasp-intent",
      lens_run_id: "run-probe-grasp-intent",
      policy_call: 3,
      ranking: "false_positive",
      timestep: 7,
      feature_id: "dim_42",
      model_locus: { model_site_id: "action_head.layers.8.resid" },
    },
    evidence: {
      primitive_kind: "failure_case",
      model_site_id: "action_head.layers.8.resid",
      selected_contributor: "dim_42",
    },
  });
  const route = parseEpisodeRoute(hash.replace("#episode/", ""));

  assert.equal(route.traceId, "episode-1");
  assert.equal(route.siteName, "action_head.layers.8.resid");
  assert.equal(route.feature, 42);
  assert.equal(route.researchSelection?.dataset_id, "demo");
  assert.equal(route.researchSelection?.lens_id, "probe-grasp-intent");
  assert.equal(route.researchSelection?.lens_run_id, "run-probe-grasp-intent");
  assert.equal(route.researchSelection?.ranking, "false_positive");
  assert.equal(route.researchSelection?.policy_call, 3);
  assert.equal(route.researchSelection?.timestep, 7);
  assert.equal(route.researchSelection?.feature_id, "dim_42");
  assert.equal(route.researchSelection?.model_locus?.model_site_id, "action_head.layers.8.resid");
});

test("episode route parser preserves arbitrary contributor ids", () => {
  const route = parseEpisodeRoute(
    "episode-1?probe_id=probe-a&lens_run_id=run-a&dataset_id=demo&call=2&site=attention.blocks.3&contributor=head_3_token_7",
  );

  assert.equal(route.feature, undefined);
  assert.equal(route.researchSelection?.feature_id, "head_3_token_7");
  assert.equal(route.researchSelection?.model_locus?.model_site_id, "attention.blocks.3");
});
