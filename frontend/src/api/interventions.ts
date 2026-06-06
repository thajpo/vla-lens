import { getJson, noStore } from "./client";
import type {
  InterventionRunResponse,
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
