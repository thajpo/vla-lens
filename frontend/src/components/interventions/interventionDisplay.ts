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
    sourceLabel: sourceSummary(run),
    status,
    target: targetSummary(target),
    title: stringValue(readouts.title) || run.run_id,
    traceId: stringValue(context.trace_id) || stringValue(provenance.trace_id),
  };
}

export function statusClass(status: string): string {
  return `intervention-status ${status.replace(/[^a-z0-9_-]+/gi, "_")}`;
}

export function sourceArtifactIdForRun(run: InterventionRunRecord): string {
  const provenance = record(run.provenance);
  const target = record(run.target);
  const provenanceSource = record(provenance.source_object_ref);
  const targetSource = record(target.source_object_ref);
  return (
    stringValue(provenanceSource.artifact_id) ||
    stringValue(targetSource.artifact_id) ||
    stringValue(provenance.source_artifact_id) ||
    stringValue(target.source_artifact_id)
  );
}

export function interventionsForSource(
  runs: InterventionRunRecord[],
  artifactId: string,
): InterventionRunRecord[] {
  if (!artifactId) {
    return [];
  }
  return runs.filter((run) => sourceArtifactIdForRun(run) === artifactId);
}

function targetSummary(target: Record<string, unknown>): string {
  const kind = stringValue(target.kind) || "target";
  const site = stringValue(target.model_site) || stringValue(target.site_id);
  const layer = stringValue(target.layer);
  const feature = stringValue(target.feature);
  return [kind, site, layer ? `L${layer}` : "", feature ? `feature ${feature}` : ""]
    .filter(Boolean)
    .join(" · ");
}

function sourceSummary(run: InterventionRunRecord): string {
  const provenance = record(run.provenance);
  const target = record(run.target);
  const source = record(provenance.source_object_ref);
  const fallbackSource = record(target.source_object_ref);
  const kind = stringValue(source.kind) || stringValue(fallbackSource.kind);
  const label = stringValue(source.label) || stringValue(fallbackSource.label);
  const artifactId =
    stringValue(source.artifact_id) ||
    stringValue(fallbackSource.artifact_id) ||
    stringValue(provenance.source_artifact_id) ||
    stringValue(target.source_artifact_id);
  const modelSite =
    stringValue(source.model_site) ||
    stringValue(fallbackSource.model_site) ||
    stringValue(provenance.model_site) ||
    stringValue(target.model_site);
  const layer = stringValue(source.layer) || stringValue(fallbackSource.layer) || stringValue(target.layer);
  const feature = stringValue(source.feature) || stringValue(fallbackSource.feature) || stringValue(target.feature);
  const sourceLabel = label || artifactId || kind;
  return [
    sourceLabel,
    modelSite,
    layer ? `L${layer}` : "",
    feature ? `feature ${feature}` : "",
  ].filter(Boolean).join(" · ");
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
