import assert from "node:assert/strict";
import test from "node:test";

import { discoveryArtifactEpisodeSearchParams } from "./discoveryArtifactParams.ts";

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
