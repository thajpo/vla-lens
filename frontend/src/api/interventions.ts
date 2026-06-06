import { getJson, noStore, postJson } from "./client";
import type {
  InterventionPreflightResponse,
  InterventionRunResponse,
  InterventionRunSaveResponse,
  InterventionRunsResponse,
} from "../types/interventions";

export function fetchInterventionRuns(): Promise<InterventionRunsResponse> {
  return getJson<InterventionRunsResponse>("/api/intervention-runs", noStore());
}

export function fetchInterventionRun(runId: string): Promise<InterventionRunResponse> {
  return getJson<InterventionRunResponse>(
    `/api/intervention-runs/${encodeURIComponent(runId)}`,
    noStore(),
  );
}

export function preflightIntervention(payload: unknown): Promise<InterventionPreflightResponse> {
  return postJson<InterventionPreflightResponse>("/api/interventions/preflight", payload);
}

export function saveInterventionRun(payload: unknown): Promise<InterventionRunSaveResponse> {
  return postJson<InterventionRunSaveResponse>("/api/intervention-runs", payload);
}
