import assert from "node:assert/strict";
import test from "node:test";

import {
  buildInterventionsHash,
  buildProbeHash,
  buildResearchHash,
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

test("research and probe detail hashes preserve the selected result", () => {
  const research = parseWorkbenchHash("#research/rq%20015");
  const probe = parseWorkbenchHash("#probes/probe%20artifact");

  assert.equal(research.page, "research");
  assert.equal(research.researchRunId, "rq 015");
  assert.equal(probe.page, "probes");
  assert.equal(probe.probeRunId, "probe artifact");
  assert.equal(buildResearchHash("rq 015"), "#research/rq%20015");
  assert.equal(buildProbeHash("probe artifact"), "#probes/probe%20artifact");
});
