import type {
  InterventionLabDraft,
  InterventionLabSeed,
  InterventionPreflightResult,
  InterventionRunRecord,
} from "../../types/interventions";

export function buildBackendTargetInterventionSeed({
  artifactId,
  artifactType,
  basis,
  modelFamily,
  modelSite,
  operator,
  policyCallIndex,
  target,
  title,
  tokenSpace,
  traceId,
}: {
  artifactId: string;
  artifactType: string;
  basis?: string[];
  modelFamily?: string;
  modelSite: string;
  operator?: string;
  policyCallIndex: number;
  target?: Record<string, unknown>;
  title?: string;
  tokenSpace?: string;
  traceId: string;
}): InterventionLabSeed {
  return {
    artifactId,
    artifactType,
    basis,
    modelFamily,
    modelSite,
    operator,
    policyCallIndex,
    target,
    title,
    tokenSpace,
    traceId,
  };
}

export function buildInterventionRequest(draft: InterventionLabDraft): Record<string, unknown> {
  const target = draft.target
    ? {
        ...draft.target,
        metadata: {
          ...record(draft.target.metadata),
          intended_basis: draft.basis.includes("gripper") ? "gripper" : "raw",
        },
      }
    : {
        kind: draft.artifactId ? "probe_direction" : "manual",
        source_artifact_id: draft.artifactId || undefined,
        source_artifact_type: draft.artifactType || undefined,
        model_family: draft.modelFamily || "pi05",
        model_site: draft.modelSite,
        token_space: draft.tokenSpace,
        metadata: {
          intended_basis: draft.basis.includes("gripper") ? "gripper" : "raw",
          target_source: "local_fallback",
        },
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
