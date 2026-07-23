import type { InterventionRunRecord, InterventionSummary } from "../../types/interventions";

export type InterventionActionCondition = {
  actionRef: string;
  kind: "original" | "noop" | "intervention" | "control";
  label: string;
  metrics: Record<string, unknown>;
  status: string;
};

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

export function interventionActionComparison(
  run: InterventionRunRecord,
): InterventionActionCondition[] {
  const readouts = record(run.readouts);
  const trials = arrayRecords(readouts.trials);
  const outcomes = arrayRecords(readouts.outcomes);
  const controls = arrayRecords(readouts.controls);
  const controlMetrics = new Map<string, Record<string, unknown>>();
  for (const control of controls) {
    const metrics = record(control.metrics);
    for (const trialId of arrayStrings(control.trial_ids)) {
      controlMetrics.set(trialId, metrics);
    }
  }
  return trials.flatMap((trial) => {
    const outputs = record(trial.outputs);
    const actionRef = stringValue(outputs.action_ref);
    if (!actionRef) {
      return [];
    }
    const trialId = stringValue(trial.trial_id);
    const trialKind = stringValue(trial.trial_kind);
    const controlKind = stringValue(trial.control_kind);
    const kind = actionConditionKind(trialKind, controlKind);
    if (!kind) {
      return [];
    }
    const outcomeMetrics = outcomes
      .filter((outcome) => stringValue(outcome.intervention_trial_id) === trialId)
      .reduce((combined, outcome) => ({ ...combined, ...record(outcome.metrics) }), {});
    return [{
      actionRef,
      kind,
      label: actionConditionLabel(kind, controlKind || trialKind),
      metrics: {
        ...record(trial.metrics),
        ...controlMetrics.get(trialId),
        ...outcomeMetrics,
      },
      status: stringValue(trial.status) || "unknown",
    }];
  });
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

function actionConditionKind(
  trialKind: string,
  controlKind: string,
): InterventionActionCondition["kind"] | null {
  if (trialKind === "stored_original") {
    return "original";
  }
  if (trialKind === "noop" || trialKind === "noop_rerun") {
    return "noop";
  }
  if (trialKind === "intervention") {
    return "intervention";
  }
  if (controlKind || trialKind.includes("control")) {
    return "control";
  }
  return null;
}

function actionConditionLabel(
  kind: InterventionActionCondition["kind"],
  detail: string,
): string {
  if (kind === "original") {
    return "Original";
  }
  if (kind === "noop") {
    return "No-op";
  }
  if (kind === "intervention") {
    return "Intervention";
  }
  const label = detail.replace(/[_-]+/g, " ").replace(/^./, (value) => value.toUpperCase());
  return label.includes("control") ? label : `${label} control`;
}

function record(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function arrayRecords(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.filter(isRecord) : [];
}

function arrayStrings(value: unknown): string[] {
  return Array.isArray(value) ? value.map(String) : [];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringValue(value: unknown): string {
  if (value === null || typeof value === "undefined") {
    return "";
  }
  return String(value);
}
