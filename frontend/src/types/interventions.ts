export type InterventionStatus = "inspected_only" | "ok" | "partial" | "failed" | string;

export type InterventionRunRecord = {
  run_id: string;
  intervention_type: "intervention_record" | string;
  target: Record<string, unknown>;
  baseline: Record<string, unknown>;
  intervention: Record<string, unknown>;
  readouts: Record<string, unknown>;
  outputs: string[];
  provenance: Record<string, unknown>;
};

export type InterventionRunsResponse = {
  intervention_runs: InterventionRunRecord[];
  total: number;
};

export type InterventionRunResponse = {
  intervention_run: InterventionRunRecord;
};

export type InterventionSummary = {
  claimLabels: string[];
  context: Record<string, unknown>;
  createdUtc: string;
  operator: string;
  outcomeKind: string;
  policyCall: string;
  status: InterventionStatus;
  target: string;
  title: string;
  traceId: string;
};
