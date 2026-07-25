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
  feature?: number | null;
  layer?: number | null;
  modelFamily: string;
  modelSite: string;
  operator: string;
  policyCallIndex: number;
  rankingMode?: string;
  runId?: string;
  selectionSource?: string;
  sourceObjectRef?: InterventionSourceObjectRef;
  strength: number;
  target?: Record<string, unknown>;
  title?: string;
  tokenSpace: string;
  traceId: string;
  timestep?: number | null;
};

export type InterventionLabSeed = Partial<InterventionLabDraft>;

export type InterventionSourceObjectRef = {
  kind: string;
  artifactId?: string;
  artifactType?: string;
  feature?: number | null;
  label?: string;
  layer?: number | null;
  lensId?: string;
  modelSite?: string;
  policyCallIndex?: number | null;
  probeId?: string;
  rankingMode?: string;
  timestep?: number | null;
  traceId?: string;
};

export type InterventionSummary = {
  claimLabels: string[];
  context: Record<string, unknown>;
  createdUtc: string;
  operator: string;
  outcomeKind: string;
  policyCall: string;
  sourceLabel: string;
  status: InterventionStatus;
  target: string;
  title: string;
  traceId: string;
};

export type PatchStudyCell = {
  layer: number;
  token_region: string;
  pair_count: number;
  transfer_mean: number;
  transfer_ci95_low: number;
  transfer_ci95_high: number;
  direction_agreement_mean: number;
  donor_recovery_mean: number;
  localized_pair_count: number;
};

export type PatchStudyPair = {
  pair_id: string;
  recipient_trace_id: string;
  donor_trace_id: string;
  target_object?: string | null;
  distractor_object?: string | null;
};

export type PatchStudySpecificity = {
  layer: number;
  token_region: string;
  main_transfer_mean: number;
  strongest_control_kind: string;
  strongest_control_transfer_mean: number;
  specificity_margin: number;
  main_beats_control: boolean;
};

export type PatchStudyAnalysis = {
  study_id: string;
  question?: string | null;
  hypothesis?: string | null;
  phase?: string | null;
  stream?: "vlm_prefix" | "expert_action" | string | null;
  generation_steps?: "all" | { indices?: number[]; start?: number; end?: number } | null;
  status: string;
  pair_count: number;
  planned_trial_count: number;
  controls: string[];
  layers: number[];
  token_regions: string[];
  summary: PatchStudyCell[];
  specificity: PatchStudySpecificity[];
  pairs: PatchStudyPair[];
  headline: {
    best_layer?: number | null;
    best_token_region?: string | null;
    best_transfer_mean?: number | null;
    best_transfer_ci95?: [number, number] | null;
  };
};

export type PatchStudiesResponse = {
  patch_studies: PatchStudyAnalysis[];
  total: number;
};
