export type DatasetEpisode = {
  trace_id: string;
  episode_id: string;
  episode_index?: number | null;
  task_id?: string | null;
  prompt?: string | null;
  model_id?: string | null;
  env_id?: string | null;
  robot_id?: string | null;
  outcome?: string | null;
  length?: number | null;
  schema_version?: string | null;
  metadata?: Record<string, unknown>;
  probe_record?: ProbeEpisodeIndex;
};

export type DatasetPayload = {
  root: string;
  episode_count: number;
  activation_sites: number;
  capabilities?: {
    available: string[];
    flags: Record<string, boolean>;
    camera_names: string[];
    model_families: string[];
    model_site_prefixes: string[];
  };
  artifacts: {
    total: number;
    counts: Record<string, number>;
  };
  probes?: {
    total_predictions: number;
    total_episode_records?: number;
    probe_count: number;
  };
  index?: {
    schema_version?: string;
    dataset_fingerprint?: string;
    indexed_episode_count?: number;
    updated_utc?: string;
    tables?: Record<string, unknown>;
  };
  counterfactual_pairs?: CounterfactualPair[];
  workbench?: unknown;
};

export type ProbeEpisodeIndex = {
  trace_id: string;
  split?: string | null;
  split_category?: string | null;
  sidecar?: Record<string, unknown>;
  available: boolean;
  row_count: number;
  best_row_count?: number;
  actual?: string | boolean | number | null;
  predicted?: string | boolean | number | null;
  confidence?: number | null;
  correct?: boolean | null;
  correct_rate?: number | null;
  eval_split?: string | null;
  model?: string | null;
  feature?: string | null;
  policy_call_index?: number | null;
};

export type ProbeDatasetIndex = {
  artifact_id: string;
  name: string;
  target?: string | null;
  best_model?: string | null;
  best_feature?: string | null;
  best_score?: number | null;
  best_delta?: number | null;
  split_summary: Record<string, number>;
  prediction_summary: Record<string, number>;
  review_stats?: {
    confidentWrong?: number;
    heldoutScored?: number;
    heldoutWrong?: number;
    scored?: number;
    test?: number;
    train?: number;
    unscored?: number;
    validation?: number;
    wrong?: number;
  };
  by_trace?: Record<string, ProbeEpisodeIndex>;
};

export type ProbeIndexResponse = {
  probes: ProbeDatasetIndex[];
  total: number;
  trace_count: number;
  split_source?: string | null;
};

export type EpisodeFacetValue = {
  value: string;
  count: number;
};

export type EpisodePageResponse = {
  episodes: DatasetEpisode[];
  total: number;
  limit: number;
  offset: number;
  next_offset?: number | null;
  facets: Record<string, EpisodeFacetValue[]>;
  sort: string;
};

export type EpisodeNeighborsResponse = {
  trace_id: string;
  previous_trace_id?: string | null;
  next_trace_id?: string | null;
};

export type ProbeEvidenceResponse = {
  probe: ProbeDatasetIndex;
  episodes: DatasetEpisode[];
  total: number;
  limit: number;
};

export type DiscoveryArtifactFamily = {
  available: true;
  artifact_type: string;
  target_kind: string;
  operators: string[];
  outcomes: string[];
  required_controls: Record<string, string[]>;
  representation_kind: string;
  description: string;
  reason: string;
};

export type DiscoveryArtifactUnavailableFamily = {
  available: false;
  artifact_type: string;
  reason: string;
};

export type DiscoveryArtifactFamiliesResponse = {
  families: DiscoveryArtifactFamily[];
  total: number;
};

export type DiscoveryArtifactEpisodesResponse = EpisodePageResponse & {
  artifact: ArtifactRecord;
  available: boolean;
  family: DiscoveryArtifactFamily | DiscoveryArtifactUnavailableFamily;
  rank_by: string;
  reason: string;
};

export type DiscoveryArtifactReadoutResponse = {
  artifact: ArtifactRecord;
  available: boolean;
  family: DiscoveryArtifactFamily | DiscoveryArtifactUnavailableFamily;
  readout_type: string;
  reason: string;
  row_count: number;
  rows: Record<string, unknown>[];
  summary: Record<string, unknown>;
  target_hint: Record<string, unknown>;
  trace_id: string;
};

export type DiscoveryArtifactTargetResponse = {
  artifact: ArtifactRecord;
  available: boolean;
  family: DiscoveryArtifactFamily | DiscoveryArtifactUnavailableFamily;
  reason: string;
  target: Record<string, unknown> | null;
};

export type CounterfactualPairMember = {
  trace_id: string;
  episode_id?: string;
  role?: string;
  pair_index?: number | null;
  paired_trace_id?: string;
  target_object_id?: string;
  counterfactual_target_object_id?: string;
  outcome?: string | null;
  prompt?: string | null;
};

export type CounterfactualPair = {
  group_id: string;
  type?: string;
  changed_fields?: string[];
  matched_fields?: string[];
  members: CounterfactualPairMember[];
};

export type ObservationalComparisonEpisode = Pick<
  DatasetEpisode,
  "trace_id" | "episode_id" | "task_id" | "prompt" | "model_id" | "env_id" | "outcome" | "length"
> & {
  metadata?: Record<string, unknown>;
};

export type ObservationalComparisonCandidate = {
  trace_id: string;
  score: number;
  reasons: string[];
  episode: ObservationalComparisonEpisode;
  probe?: ProbeEpisodeIndex | null;
  metrics: {
    same_task?: boolean;
    same_prompt?: boolean;
    same_target_object?: boolean;
    different_outcome?: boolean;
    length_delta?: number;
    source_outcome?: string | null;
    candidate_outcome?: string | null;
    source_probe_correct?: boolean | null;
    candidate_probe_correct?: boolean | null;
    source_split_category?: string | null;
    candidate_split_category?: string | null;
    source_confidence?: number | null;
    candidate_confidence?: number | null;
    confidence_delta?: number | null;
  };
  contract: {
    source_trace_id: string;
    comparison_trace_id: string;
    method: string;
    causal: boolean;
    requires_live_intervention?: boolean;
  };
};

export type ObservationalComparisonsResponse = {
  artifact_type: "observational_counterfactual_comparison";
  artifact_id: string;
  name: string;
  causal: boolean;
  comparison_kind: string;
  source_trace_id: string;
  probe_id?: string | null;
  probe_name?: string | null;
  source: {
    episode: ObservationalComparisonEpisode;
    probe?: ProbeEpisodeIndex | null;
  };
  candidates: ObservationalComparisonCandidate[];
  total_candidates: number;
  limit: number;
  notes?: string;
};

export type EpisodeArtifactRecord = Record<string, unknown> & {
  artifact_id?: string;
  artifact_type?: string;
  name?: string;
  group_id?: string;
};

export type ArtifactRecord = EpisodeArtifactRecord & {
  scope?: string;
  artifact_scope?: string;
  metrics?: Record<string, unknown>;
  display?: Record<string, unknown>;
  method?: Record<string, unknown>;
  selector?: Record<string, unknown>;
  arrays?: Record<string, string>;
  created_utc?: string;
  source_trace_ids?: string[];
};

export type ArtifactListResponse = {
  artifacts: ArtifactRecord[];
  counts: Record<string, number>;
  total: number;
};

export type ArtifactArrayPreview = {
  name: string;
  path: string;
  shape: number[];
  dtype: string;
  summary: Record<string, unknown>;
  preview?: unknown;
};

export type ArtifactDetailResponse = {
  artifact: ArtifactRecord;
  arrays: ArtifactArrayPreview[];
};

export type EpisodeArrayRecord = Record<string, unknown> & {
  name?: string;
  relative_path?: string;
  storage_format?: string;
  shape?: string;
  dtype?: string;
  axes?: string;
};

export type EpisodeDetail = DatasetEpisode & {
  cameras: string[];
  artifacts: EpisodeArtifactRecord[];
  arrays: EpisodeArrayRecord[];
};

export type EpisodeAnnotation = {
  trace_id: string;
  starred: boolean;
  notes: string;
  updated_utc?: string | null;
};

export type EpisodeAnnotationResponse = {
  annotation: EpisodeAnnotation;
};

export type PolicyCall = {
  index: number;
  model_call_index?: number | null;
  env_timestep: number;
  segment_start: number;
  segment_end: number;
  segment_length: number;
};

export type PolicyCallsResponse = {
  calls: PolicyCall[];
  count: number;
  env_length: number;
};

export type NumericSeriesResponse = {
  values: number[];
};

export type MatrixSeriesResponse = {
  values: number[][];
};

export type EpisodeMetric = {
  key: string;
  label: string;
  domain: string;
  kind: string;
  description?: string;
  values: number[];
  x_values?: number[];
  x_label?: string;
  y_label?: string;
  y_unit?: string | null;
};

export type EpisodeMetricsResponse = {
  domains: { key: string; label: string }[];
  metrics: EpisodeMetric[];
};

export type EpisodeInteractionLabel = {
  primary_target_object: string;
  target_objects: string[];
  target_parse_status: string;
  first_moved_object: string;
  first_moved_timestep: number | null;
  first_moved_is_target: boolean;
  first_lifted_object: string;
  first_lifted_timestep: number | null;
  first_lifted_is_target: boolean;
  first_contacted_object: string;
  first_contact_timestep: number | null;
  scene_family?: string;
  task_verb?: string;
};

export type EpisodeObjectMetric = {
  object_name: string;
  object_base_name: string;
  object_kind: string;
  is_target_object: boolean;
  moved: boolean;
  lifted: boolean;
  contacted: boolean;
  movement_onset_timestep: number | null;
  lift_onset_timestep: number | null;
  contact_onset_timestep: number | null;
  max_displacement: number | null;
  max_z_delta: number | null;
};

export type EpisodeInteractionsResponse = {
  available: boolean;
  reason?: string;
  trace_id: string;
  artifact_id?: string;
  episode?: EpisodeInteractionLabel;
  quality?: Record<string, boolean>;
  objects: EpisodeObjectMetric[];
};

export type EpisodeProbePrediction = {
  trace_id?: string;
  episode_id?: string;
  task_id?: string;
  split?: string;
  target_name?: string;
  target_value?: string | boolean | number | null;
  actual?: string | boolean | number | null;
  predicted?: string | boolean | number | null;
  prediction_value?: string | boolean | number | null;
  confidence?: number | null;
  correct?: boolean | null;
  model?: string;
  feature?: string;
  layer?: number | null;
  policy_call_index?: number | null;
  timestep?: number | null;
  target_timestep?: number | null;
  generation_step?: string | number | null;
  model_site_id?: string;
  token_space_id?: string;
  eval_split?: string;
  primary_metric?: string;
};

export type EpisodeProbeSummary = {
  artifact_id: string;
  name: string;
  target?: string | null;
  metrics: Record<string, unknown>;
  best_result: Record<string, unknown>;
  target_distribution: Record<string, unknown>;
  episode_summary: {
    actual?: string | boolean | number | null;
    predicted?: string | boolean | number | null;
    confidence?: number | null;
    correct?: boolean | null;
    correct_rate?: number | null;
    all_cell_correct_rate?: number | null;
    all_cell_mean_confidence?: number | null;
    best_feature?: string;
    best_model?: string;
    best_row?: EpisodeProbePrediction;
  };
  rows: EpisodeProbePrediction[];
  row_count: number;
  available: boolean;
};

export type EpisodeProbesResponse = {
  trace_id: string;
  probes: EpisodeProbeSummary[];
  available_count: number;
  total: number;
};

export type ActivationSite = {
  name: string;
  site_id?: string;
  module?: string;
  layer?: number | null;
  tensor_type?: string | null;
  token_kind?: string | null;
  family?: string | null;
  role?: string | null;
  segment?: string | null;
  materialization?: string | null;
  exactness?: string | null;
  token_space_id?: string | null;
  query_token_space_id?: string | null;
  key_token_space_id?: string | null;
  parent_site_id?: string | null;
  summary_type?: string | null;
  capture_family?: string | null;
  view_kind?: string | null;
  capture_role?: string | null;
  default_view?: boolean | null;
  derived_from?: string[];
  derivation?: string | null;
  axes?: string[];
  shape?: number[];
  dtype?: string;
  metadata?: Record<string, unknown>;
};

export type RuntimeCollection = {
  id: string;
  label: string;
  kind: string;
  materialized: boolean;
  aggregation: string;
  members: {
    layer?: number | null;
    component?: string;
    site_name: string;
  }[];
};

export type ArchitectureNode = {
  id: string;
  label: string;
  kind: string;
  stage?: string;
  layer?: number | null;
  captured?: boolean;
};

export type ArchitectureEdge = {
  id: string;
  kind: string;
  source: string;
  target: string;
  layer?: number | null;
  source_sites?: string[];
  target_site_family?: string;
  source_token_space?: string;
  query_token_space?: string;
  key_token_space?: string;
  runtime_collection?: string;
  materialized?: boolean;
};

export type ArchitectureMetadata = {
  nodes?: ArchitectureNode[];
  edges?: ArchitectureEdge[];
};

export type ActivationSitesResponse = {
  sites: ActivationSite[];
  runtime_collections?: RuntimeCollection[];
  architecture?: ArchitectureMetadata;
};

export type ActivationFeatureRow = {
  index: number;
  value: number;
};

export type ActivationSliceResponse = {
  name: string;
  selected?: PolicyCall | null;
  axes?: string[];
  shape?: number[];
  feature_count: number;
  feature: number;
  feature_value?: number | null;
  clip_percent?: number;
  clip?: {
    enabled?: boolean;
    lower?: number | null;
    upper?: number | null;
    kept?: number;
    total?: number;
  };
  top_abs: ActivationFeatureRow[];
  reason?: string;
};

export type ImageTokenCameraMap = {
  values: number[][];
  token_start?: number;
  token_end?: number;
  active_tokens?: number | null;
  min?: number | null;
  max?: number | null;
};

export type ImageTokenMapResponse = {
  available: boolean;
  name?: string;
  feature?: number;
  feature_count?: number;
  call?: PolicyCall;
  source?: string;
  grid_size?: number;
  grid_height?: number;
  grid_width?: number;
  patches_per_image?: number;
  image_tokens?: number;
  text_tokens?: number;
  image_slots?: number;
  maps?: Record<string, ImageTokenCameraMap>;
  note?: string;
  reason?: string;
};

export type ObjectCameraOverlayObject = {
  object_index: number;
  object_name: string;
  object_kind?: string;
  body_id?: number | null;
  body_name?: string;
  site_name?: string;
  source?: string;
  position_world?: number[];
  geometry_center_world?: number[] | null;
  quaternion_xyzw?: number[] | null;
  bbox?: {
    x0?: number | null;
    y0?: number | null;
    x1?: number | null;
    y1?: number | null;
    raw_x0?: number | null;
    raw_y0?: number | null;
    raw_x1?: number | null;
    raw_y1?: number | null;
    center_x?: number | null;
    center_y?: number | null;
    center_pixel_x?: number | null;
    center_pixel_y?: number | null;
    in_frame?: boolean;
  } | null;
  pixel_x?: number | null;
  pixel_y?: number | null;
  x?: number | null;
  y?: number | null;
  depth?: number | null;
  in_frame?: boolean;
  approximate?: boolean;
  projection_kind?: string;
};

export type ObjectCameraOverlayResponse = {
  available: boolean;
  camera?: string;
  calibration_camera?: string;
  timestep?: number;
  width?: number;
  height?: number;
  include_sites?: boolean;
  approximate?: boolean;
  projection_kind?: string;
  visible_count?: number;
  objects?: ObjectCameraOverlayObject[];
  note?: string;
  reason?: string;
  detail?: string;
};

export type AttentionMapResponse = {
  available: boolean;
  kind?: string;
  call?: PolicyCall;
  generation_step?: number;
  site?: string;
  source?: string;
  head?: number | null;
  head_mode?: "average" | "selected";
  query_token?: number | null;
  query_mode?: "average" | "selected";
  grid_size?: number;
  grid_height?: number;
  grid_width?: number;
  patches_per_image?: number;
  image_tokens?: number;
  text_tokens?: number;
  image_slots?: number;
  coarse?: AttentionCoarse;
  maps?: Record<string, ImageTokenCameraMap>;
  note?: string;
  reason?: string;
};

export type PromptAttentionResponse = {
  available: boolean;
  call?: PolicyCall;
  generation_step?: number;
  kind?: string;
  name?: string;
  head?: number | null;
  head_mode?: "average" | "selected";
  query_token?: number | null;
  query_mode?: "average" | "selected";
  feature?: number;
  feature_count?: number;
  attention_site?: string | null;
  prompt?: string;
  active_text_tokens?: number;
  allocated_text_slots?: number;
  expert_coarse?: AttentionCoarse | null;
  top_text_tokens?: PromptTokenAttention[];
  prompt_tokens?: PromptTokenAttention[];
  top_image_patches?: ImagePatchAttention[];
  reason?: string;
  detail?: string;
};

export type SelectedPatch = {
  camera: string;
  row: number;
  col: number;
};

export type PatchFeaturesResponse = {
  available: boolean;
  name?: string;
  call?: PolicyCall;
  camera?: string;
  patch_row?: number;
  patch_col?: number;
  token_index?: number;
  feature?: number;
  feature_value?: number | null;
  feature_rank_by_abs?: number | null;
  feature_count?: number;
  top_abs?: ActivationFeatureRow[];
  top_positive?: ActivationFeatureRow[];
  top_negative?: ActivationFeatureRow[];
  reason?: string;
};

export type ExpertTokenActivationsResponse = {
  available: boolean;
  name?: string;
  call?: PolicyCall;
  generation_step?: number;
  feature?: number;
  feature_count?: number;
  values?: number[];
  min?: number | null;
  max?: number | null;
  note?: string;
  reason?: string;
};

export type ActionVectorSummary = {
  source?: string;
  dim?: number;
  norm?: number | null;
  top_abs?: ActivationFeatureRow[];
};

export type AttentionCoarse = {
  image?: number | null;
  prompt?: number | null;
  action_suffix?: number | null;
};

export type PromptTokenAttention = {
  local_index: number;
  prefix_index?: number;
  token_id?: number | string | null;
  token_piece?: string | null;
  attention: number;
};

export type ImagePatchAttention = {
  camera: string;
  row: number;
  col: number;
  token_index?: number;
  attention: number;
};

export type ExpertTokenDetailsResponse = {
  available: boolean;
  name?: string;
  call?: PolicyCall;
  generation_step?: number;
  token_index?: number;
  token_count?: number;
  feature?: number;
  feature_value?: number | null;
  feature_rank_by_abs?: number | null;
  feature_count?: number;
  top_abs?: ActivationFeatureRow[];
  attention_site?: string | null;
  attention_coarse?: AttentionCoarse | null;
  top_prompt_tokens?: PromptTokenAttention[];
  prompt_tokens?: PromptTokenAttention[];
  top_image_patches?: ImagePatchAttention[];
  maps?: Record<string, ImageTokenCameraMap>;
  action?: ActionVectorSummary | null;
  note?: string;
  reason?: string;
};

export type DatasetDiagnostics = {
  fingerprint?: string;
  stale?: boolean;
  latest?: Record<string, unknown> | null;
};
