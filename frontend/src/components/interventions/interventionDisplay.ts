import type { InterventionRunRecord, InterventionSummary } from "../../types/interventions";

export function summarizeInterventionRun(run: InterventionRunRecord): InterventionSummary {
  const readouts = record(run.readouts);
  const provenance = record(run.provenance);
  const baseline = record(run.baseline);
  const baselineContext = record(baseline.context);
  const context = Object.keys(baselineContext).length ? baselineContext : provenance;
  const intervention = record(run.intervention);
  const request = record(intervention.request);
  const operator = record(request.operator);
  const outcome = record(request.outcome);
  const target = record(run.target);
  const claim = record(readouts.claim);
  const status = stringValue(readouts.status) || "inspected_only";
  return {
    claimLabels: claimLabels(claim, readouts),
    context,
    createdUtc: stringValue(readouts.created_utc) || stringValue(provenance.created_utc),
    operator: stringValue(operator.operator),
    outcomeKind: stringValue(outcome.kind),
    policyCall: stringValue(context.policy_call_index) || stringValue(provenance.policy_call_index) || "-",
    status,
    target: targetSummary(target),
    title: stringValue(readouts.title) || run.run_id,
    traceId: stringValue(context.trace_id) || stringValue(provenance.trace_id),
  };
}

export function statusClass(status: string): string {
  return `intervention-status ${status.replace(/[^a-z0-9_-]+/gi, "_")}`;
}

function targetSummary(target: Record<string, unknown>): string {
  const kind = stringValue(target.kind) || "target";
  const site = stringValue(target.model_site) || stringValue(target.site_id);
  const layer = stringValue(target.layer);
  return [kind, site, layer ? `L${layer}` : ""].filter(Boolean).join(" · ");
}

function claimLabels(
  claim: Record<string, unknown>,
  readouts: Record<string, unknown>,
): string[] {
  const labels = claim.claim_strength ?? claim.claim_strengths ?? readouts.claim_strengths ?? [];
  if (typeof labels === "string") {
    return [labels];
  }
  if (Array.isArray(labels)) {
    return labels.map(String).filter(Boolean);
  }
  return [];
}

function record(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function stringValue(value: unknown): string {
  if (value === null || typeof value === "undefined") {
    return "";
  }
  return String(value);
}
