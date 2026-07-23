import { getJson, noStore } from "./client";
import type { ResearchRunResponse, ResearchRunsResponse } from "../types/researchRuns";

export function fetchResearchRuns(): Promise<ResearchRunsResponse> {
  return getJson<ResearchRunsResponse>("/api/research-runs", noStore());
}

export function fetchResearchRun(runId: string): Promise<ResearchRunResponse> {
  return getJson<ResearchRunResponse>(
    `/api/research-runs/${encodeURIComponent(runId)}`,
    noStore(),
  );
}
