export type ResearchRunStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

export type ResearchProgress = {
  completed: number;
  total: number;
  unit: string;
  fraction?: number | null;
};

export type ResearchResultSummary = {
  metric: string;
  score?: number | null;
  baseline?: number | null;
  delta?: number | null;
  verdict: string;
};

export type ResearchRunRecord = {
  run_id: string;
  parent_run_id?: string | null;
  kind: string;
  name: string;
  question: string;
  status: ResearchRunStatus;
  stage: string;
  progress: ResearchProgress;
  artifact_ids: string[];
  result: ResearchResultSummary;
  error?: string | null;
  created_utc: string;
  updated_utc: string;
  started_utc?: string | null;
  completed_utc?: string | null;
  provenance: Record<string, unknown>;
};

export type ResearchRunsResponse = {
  research_runs: ResearchRunRecord[];
  total: number;
};

export type ResearchRunResponse = {
  research_run: ResearchRunRecord;
};
