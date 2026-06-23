import type {
  InterventionLabDraft,
  InterventionPreflightResult,
  InterventionRunRecord,
} from "../../types/interventions";

export function buildInterventionRequest(draft: InterventionLabDraft): Record<string, unknown> {
  const sourceObjectRef = sourceObjectRefRecord(draft);
  const metadata = {
    ...record(draft.target?.metadata),
    intended_basis: draft.basis.includes("gripper") ? "gripper" : "raw",
    ranking_mode: draft.rankingMode || undefined,
  };
  const target = draft.target
    ? {
        ...draft.target,
        feature: draft.feature ?? draft.target.feature,
        layer: draft.layer ?? draft.target.layer,
        metadata,
        model_site: draft.target.model_site ?? draft.modelSite,
        selection_source: draft.selectionSource ?? draft.target.selection_source,
        source_object_ref: Object.keys(sourceObjectRef).length
          ? sourceObjectRef
          : draft.target.source_object_ref,
      }
    : {
        kind: draft.artifactId ? "probe_direction" : "manual",
        source_artifact_id: draft.artifactId || undefined,
        source_artifact_type: draft.artifactType || undefined,
        feature: draft.feature ?? undefined,
        layer: draft.layer ?? undefined,
        model_family: draft.modelFamily || "pi05",
        model_site: draft.modelSite,
        selection_source: draft.selectionSource ?? (draft.artifactId ? "probe_model_locus" : "manual_model_site"),
        source_object_ref: Object.keys(sourceObjectRef).length ? sourceObjectRef : undefined,
        token_space: draft.tokenSpace,
        metadata,
      };
  return {
    runtime_adapter: "pi05",
    target,
    baseline: {
      context: {
        dataset_id: draft.datasetId,
        dataset_fingerprint: draft.datasetFingerprint,
        trace_id: draft.traceId,
        policy_call_index: draft.policyCallIndex,
        timestep: draft.timestep ?? undefined,
      },
    },
    intervention: {
      request: {
        operator: {
          operator: draft.operator,
          strength: draft.strength,
        },
        schedule: {
          policy_calls: [draft.policyCallIndex],
          generation_steps: "all",
          tokens: "target_tokens",
          action_horizon: "full_chunk",
        },
        outcome: {
          kind: "action",
          basis: draft.basis.length ? draft.basis : ["raw"],
          intended_basis: draft.basis.includes("gripper") ? "gripper" : "raw",
        },
        controls: draft.controls.map((kind) => ({ kind })),
      },
    },
  };
}

export function buildInspectedInterventionRecord(
  draft: InterventionLabDraft,
  preflight: InterventionPreflightResult,
  createdUtc: string,
): InterventionRunRecord {
  const request = buildInterventionRequest(draft);
  const target = record(request.target);
  const baseline = record(request.baseline);
  const intervention = record(request.intervention);
  const status = preflight.status === "ok" ? "inspected_only" : preflight.status;
  const runId = draft.runId || `intervention-lab-${createdUtc.replace(/[^0-9A-Za-z]+/g, "-")}`;
  const sourceObjectRef = sourceObjectRefRecord(draft);
  return {
    run_id: runId,
    intervention_type: "intervention_record",
    target,
    baseline,
    intervention,
    readouts: {
      title: draft.title || `Intervention at ${draft.modelSite || "selected site"}`,
      status,
      created_utc: createdUtc,
      preflight,
      trials: [],
      outcomes: [],
      controls: [],
      display: {
        source: "intervention_lab",
      },
      claim: {
        claim_strength: status === "failed" ? [] : ["observation"],
      },
    },
    outputs: [],
    provenance: {
      schema_kind: "vla_lens.intervention_run",
      schema_version: "0.1.0",
      dataset_id: draft.datasetId,
      dataset_fingerprint: draft.datasetFingerprint,
      trace_id: draft.traceId,
      policy_call_index: draft.policyCallIndex,
      timestep: draft.timestep ?? undefined,
      model_site: draft.modelSite || undefined,
      layer: draft.layer ?? undefined,
      feature: draft.feature ?? undefined,
      ranking_mode: draft.rankingMode || undefined,
      selection_source: draft.selectionSource || undefined,
      source_object_ref: Object.keys(sourceObjectRef).length ? sourceObjectRef : undefined,
      source_artifact_id: draft.artifactId || undefined,
      created_utc: createdUtc,
      ui_surface: "intervention_lab",
    },
  };
}

export function liveRunAvailable(preflight?: InterventionPreflightResult): boolean {
  if (!preflight) {
    return false;
  }
  return preflight.status === "ok" && preflight.capability_status.model_runtime_available === true;
}

function record(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function sourceObjectRefRecord(draft: InterventionLabDraft): Record<string, unknown> {
  const explicit = record(draft.sourceObjectRef);
  const value = {
    kind: stringOrUndefined(explicit.kind) ?? (draft.artifactId ? draft.artifactType || "probe_suite" : "manual_model_site"),
    artifact_id: stringOrUndefined(explicit.artifactId) ?? (draft.artifactId || undefined),
    artifact_type: stringOrUndefined(explicit.artifactType) ?? (draft.artifactType || undefined),
    feature: numberOrNullish(explicit.feature) ?? draft.feature ?? undefined,
    label: stringOrUndefined(explicit.label) ?? (draft.title || undefined),
    layer: numberOrNullish(explicit.layer) ?? draft.layer ?? undefined,
    lens_id: stringOrUndefined(explicit.lensId),
    model_site: stringOrUndefined(explicit.modelSite) ?? (draft.modelSite || undefined),
    policy_call_index: numberOrNullish(explicit.policyCallIndex) ?? draft.policyCallIndex,
    probe_id: stringOrUndefined(explicit.probeId) ?? (draft.artifactId || undefined),
    ranking_mode: stringOrUndefined(explicit.rankingMode) ?? (draft.rankingMode || undefined),
    timestep: numberOrNullish(explicit.timestep) ?? draft.timestep ?? undefined,
    trace_id: stringOrUndefined(explicit.traceId) ?? (draft.traceId || undefined),
  };
  return Object.fromEntries(
    Object.entries(value).filter(([, item]) => item !== undefined && item !== ""),
  );
}

function stringOrUndefined(value: unknown): string | undefined {
  if (value === null || value === undefined || value === "") {
    return undefined;
  }
  return String(value);
}

function numberOrNullish(value: unknown): number | null | undefined {
  if (value === null) {
    return null;
  }
  if (value === undefined || value === "") {
    return undefined;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}
