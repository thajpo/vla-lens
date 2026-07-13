import assert from "node:assert/strict";
import test from "node:test";

import {
  buildInterventionsHash,
  parseWorkbenchHash,
} from "./workbenchRouteModel.ts";

test("workbench route parser opens canonical interventions hashes", () => {
  const route = parseWorkbenchHash("#interventions/run-a");

  assert.equal(route.page, "interventions");
  assert.equal(route.interventionRunId, "run-a");
});

test("workbench route parser keeps old evidence hashes as interventions aliases", () => {
  const route = parseWorkbenchHash("#evidence/run-a");

  assert.equal(route.page, "interventions");
  assert.equal(route.interventionRunId, "run-a");
});

test("interventions hashes are written canonically", () => {
  assert.equal(buildInterventionsHash(), "#interventions");
  assert.equal(buildInterventionsHash("run a"), "#interventions/run%20a");
});
