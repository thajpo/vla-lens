import assert from "node:assert/strict";
import test from "node:test";

import {
  discoveryArtifactEpisodeSearchParams,
  episodeLensViewSearchParams,
} from "./discoveryArtifactParams.ts";

test("discovery artifact episode params omit empty all-values", () => {
  const search = discoveryArtifactEpisodeSearchParams({
    cohort_preset: "all",
    limit: 12,
    offset: 0,
    prediction: "incorrect",
    q: "",
    rank_by: "interest",
    split: "test",
  });

  assert.equal(search.get("limit"), "12");
  assert.equal(search.get("offset"), "0");
  assert.equal(search.get("rank_by"), "interest");
  assert.equal(search.get("split"), "test");
  assert.equal(search.get("prediction"), "incorrect");
  assert.equal(search.has("cohort_preset"), false);
  assert.equal(search.has("q"), false);
});

test("episode LensView params include trace, selection, ranking mode, and top k", () => {
  const search = episodeLensViewSearchParams({
    feature: 12,
    model_site_id: "action_head.layers.2.resid",
    policy_call_index: 6,
    ranking_mode: "probe_contribution",
    timestep: 42,
    top_k: 25,
    trace_id: "trace-1",
  });

  assert.equal(search.get("trace_id"), "trace-1");
  assert.equal(search.get("timestep"), "42");
  assert.equal(search.get("policy_call_index"), "6");
  assert.equal(search.get("model_site_id"), "action_head.layers.2.resid");
  assert.equal(search.get("feature"), "12");
  assert.equal(search.get("ranking_mode"), "probe_contribution");
  assert.equal(search.get("top_k"), "25");
});
