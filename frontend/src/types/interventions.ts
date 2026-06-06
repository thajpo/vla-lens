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

export type InterventionRunSaveResponse = InterventionRunsResponse & {
  intervention_run: InterventionRunRecord;
};

export type PreflightCheck = {
  name: string;
  status: string;
  message: string;
  ok?: boolean | null;
  warnings: string[];
  errors: string[];
  metadata: Record<string, unknown>;
};

export type InterventionPreflightResult = {
  status: InterventionStatus;
  ok: boolean;
  checks: PreflightCheck[];
  warnings: string[];
  errors: string[];
  runtime_resolution: Record<string, unknown>;
  missing_capabilities: string[];
  capability_status: Record<string, boolean>;
  target_resolution: Record<string, unknown>;
  action_basis_status: Record<string, unknown>;
  runtime_environment: Record<string, unknown>;
};

export type InterventionPreflightResponse = {
  preflight: InterventionPreflightResult;
};

export type InterventionLabDraft = {
  artifactId: string;
  artifactType: string;
  basis: string[];
  controls: string[];
  datasetFingerprint: string;
  datasetId: string;
  modelFamily: string;
  modelSite: string;
  operator: string;
  policyCallIndex: number;
  runId?: string;
  strength: number;
  target?: Record<string, unknown>;
  title?: string;
  tokenSpace: string;
  traceId: string;
};

export type InterventionLabSeed = Partial<InterventionLabDraft>;

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
