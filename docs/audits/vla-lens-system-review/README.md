# VLA Lens System Review

Status: tracked static audit and consolidation backlog.

The audit inspected commit `882eeb83be8c8a69a80fc8f6ec8829c311ca4630`
(`master` on June 18, 2026). It describes the implemented system, identifies
research and software gaps, and separates verified code facts from proposed
architecture.

The audit predates two June 23 intervention branches. Use
[`08-repository-consolidation-plan.md`](08-repository-consolidation-plan.md)
for the explicit preservation and merge sequence before starting new feature
work.

## Reading Order

| Document | Job |
| --- | --- |
| [`00-executive-system-map.md`](00-executive-system-map.md) | Current system identity, maturity map, and highest-leverage opportunities. |
| [`01-domain-object-model.md`](01-domain-object-model.md) | Canonical identities, joins, axes, and vocabulary. |
| [`02-data-storage-and-indexing.md`](02-data-storage-and-indexing.md) | LeRobot/overlay storage, indexes, policy-call relation, and example manifests. |
| [`03-researcher-workflow-and-experiment-api.md`](03-researcher-workflow-and-experiment-api.md) | Experiment construction friction and proposed public method API. |
| [`04-capture-and-model-execution.md`](04-capture-and-model-execution.md) | PI0.5 runtime, temporal alignment, site ontology, reproducibility, and environment boundaries. |
| [`05-backend-frontend-state-and-ui.md`](05-backend-frontend-state-and-ui.md) | Selection state, backend/frontend contracts, drilldowns, and information architecture. |
| [`06-evidence-interventions-and-method-extensions.md`](06-evidence-interventions-and-method-extensions.md) | Evidence semantics, intervention readiness, and future analysis-method seams. |
| [`07-tests-observability-and-software-quality.md`](07-tests-observability-and-software-quality.md) | Test map, missing scientific invariants, observability, and a proposed normal-lane fixture. |
| [`08-repository-consolidation-plan.md`](08-repository-consolidation-plan.md) | Git preservation, baseline repair, branch reconciliation, documentation, and cleanup plan. |
| [`09-questions-for-owner.md`](09-questions-for-owner.md) | Product and architecture decisions that materially change implementation direction. |

## How To Use This Audit

1. Complete the repository consolidation plan and return `master` to a green,
   documented state.
2. Use the executive map to select the next architecture or product vertical.
3. Recheck any file/line references against the then-current commit before
   implementation; the audit is evidence from a fixed revision, not live API
   documentation.
4. Treat recommendations as candidate work. Product or architecture choices in
   `09-questions-for-owner.md` still require owner decisions.

The audit was static: it did not run PI0.5, LeRobot, LIBERO, model downloads,
simulators, capture, or GPU workloads.
