export type DatasetEpisode = {
  trace_id: string;
  episode_id: string;
  task_id?: string | null;
  prompt?: string | null;
  model_id?: string | null;
  env_id?: string | null;
  robot_id?: string | null;
  outcome?: string | null;
  length?: number | null;
  schema_version?: string | null;
  metadata?: Record<string, unknown>;
};

export type DatasetPayload = {
  root: string;
  episodes: DatasetEpisode[];
  activation_sites: number;
  artifacts: {
    total: number;
    counts: Record<string, number>;
  };
  workbench?: unknown;
};

export type EpisodeArtifactRecord = Record<string, unknown> & {
  artifact_id?: string;
  artifact_type?: string;
  name?: string;
  group_id?: string;
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
