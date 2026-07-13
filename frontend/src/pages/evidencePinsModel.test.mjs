import assert from "node:assert/strict";
import test from "node:test";

import { evidencePinHash, evidencePinSummary } from "./evidencePinsModel.ts";

test("evidence pin hash reopens saved research state", () => {
  const pin = {
    pin_id: "pin-1",
    created_utc: "2026-06-10T00:00:00+00:00",
    label: "Probe evidence",
    selection: {
      dataset_id: "demo",
      episode_id: "episode-1",
      lens_id: "probe-grasp-intent",
      lens_run_id: "run-probe-grasp-intent",
      policy_call: 3,
      ranking: "top",
      timestep: 7,
      feature_id: "dim_42",
    },
    evidence: {
      model_site_id: "action_head.layers.8.resid",
      primitive_kind: "contribution",
      score: 0.88,
    },
  };

  assert.equal(
    evidencePinHash(pin),
    "#episode/episode-1?probe_id=probe-grasp-intent&lens_run_id=run-probe-grasp-intent&dataset_id=demo&rank=top&call=3&timestep=7&site=action_head.layers.8.resid&feature=42",
  );
  assert.equal(evidencePinSummary(pin), "top · call 3 · timestep 7 · score 0.880");
});

test("evidence pin hash falls back to interventions page when episode state is missing", () => {
  assert.equal(
    evidencePinHash({
      pin_id: "pin-2",
      created_utc: "2026-06-10T00:00:00+00:00",
      label: "Broken pin",
      selection: {},
      evidence: {},
    }),
    "#interventions",
  );
});

test("evidence pin hash preserves non-numeric contributors", () => {
  assert.equal(
    evidencePinHash({
      pin_id: "pin-3",
      created_utc: "2026-06-10T00:00:00+00:00",
      label: "String contributor",
      selection: {
        episode_id: "episode-1",
        feature_id: "head_3_token_7",
      },
      evidence: {
        model_site_id: "attention.blocks.3",
        primitive_kind: "contribution",
      },
    }),
    "#episode/episode-1?site=attention.blocks.3&contributor=head_3_token_7",
  );
});
