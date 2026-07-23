import type { ResearchRunRecord } from "./researchRuns";

export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

export type AxisValues = Record<string, unknown>;

export type StorageRef = {
  format: "zarr" | "parquet" | "jpeg" | "mp4" | string;
  uri: string;
  relative_to: "dataset" | "bundle" | string;
  chunks: number[];
  compression?: string | null;
};

export type AxisSpec = {
  name: string;
  kind: string;
  label: string;
  unit?: string | null;
  aliases: string[];
  values: unknown[];
  alignments: string[];
};

export type LensArraySpec = {
  array_id: string;
  kind: "tensor" | "table" | "image_sequence" | "video" | "artifact_array";
  label: string;
  storage: StorageRef;
  dims: string[];
  shape: number[];
  dtype?: string | null;
  coords: Record<string, unknown>;
  provenance: Record<string, unknown>;
  summary: Record<string, unknown>;
};

export type TableSpec = {
  table_id: string;
  label: string;
  storage: StorageRef;
  columns: string[];
  row_count: number;
  provenance: Record<string, unknown>;
};

export type ImageFrameSpec = {
  frame_id: string;
  trace_id: string;
  episode_id: string;
  camera: string;
  storage: StorageRef;
  dims: string[];
  shape: number[];
  dtype?: string | null;
  frame_count: number;
  uri_template?: string | null;
  provenance: Record<string, unknown>;
};

export type MediaSpec = {
  media_id: string;
  kind: string;
  label: string;
  storage: StorageRef;
  dims: string[];
  shape: number[];
  provenance: Record<string, unknown>;
};

export type ModelSiteSpec = {
  site_id: string;
  module: string;
  site_type: string;
  axes: string[];
  layer?: number | null;
  token_kind?: string | null;
  tensor_type?: string | null;
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
  refs: Record<string, unknown>;
  summary: Record<string, unknown>;
  shape: number[];
  source_trace_count: number;
};

export type UnitRef = {
  kind: string;
  site_id?: string | null;
  index?: number | null;
  name?: string | null;
  metadata: Record<string, unknown>;
};

export type SelectionState = {
  selection_id: string;
  axis_values: AxisValues;
  unit_refs: UnitRef[];
  cohort_refs: string[];
  source_panel_id?: string | null;
  intent: string;
};

export type PanelRecipe = {
  panel_type: string;
  label: string;
  accepts: Record<string, unknown>;
  emits: string[];
  responds_to: string[];
  preferred_axes: Record<string, string>;
};

export type PanelRegistryEntry = {
  panel_type: string;
  recipe: PanelRecipe;
  selection_axes: string[];
  renderer: string;
  workflow_families: string[];
};

export type WorkflowPreset = {
  workflow_id: string;
  label: string;
  enabled: boolean;
  panels: string[];
  primary_axes: string[];
  outputs?: string[];
  [key: string]: unknown;
};

export type AnalysisRunSpec = {
  run_id: string;
  workflow: string;
  inputs: Record<string, unknown>;
  outputs: string[];
  provenance: Record<string, unknown>;
};

export type CohortSpec = {
  cohort_id: string;
  label: string;
  definition: Record<string, unknown>;
  filters: Record<string, unknown>;
  members: Record<string, string[]>;
  provenance: Record<string, unknown>;
};

export type SavedWorkspace = {
  workspace_id: string;
  dataset_id: string;
  panels: Record<string, unknown>[];
  selection?: SelectionState | null;
  cohorts: string[];
  analysis_runs: string[];
};

export type WorkbenchManifest = {
  schema_version: string;
  dataset_id: string;
  axes: Record<string, AxisSpec>;
  lens_arrays: LensArraySpec[];
  tables: TableSpec[];
  image_frames: ImageFrameSpec[];
  media: MediaSpec[];
  model_sites: ModelSiteSpec[];
  panel_registry: Record<string, PanelRegistryEntry>;
  panel_recipes: PanelRecipe[];
  workflow_presets: WorkflowPreset[];
  overlay_score_types: Record<string, unknown>[];
  graph_edge_types: Record<string, unknown>[];
  cohorts: CohortSpec[];
  analysis_runs: AnalysisRunSpec[];
  intervention_runs: Record<string, unknown>[];
  research_runs: ResearchRunRecord[];
  saved_workspaces: SavedWorkspace[];
  contract_validation: {
    valid: boolean;
    [key: string]: unknown;
  };
};

export type LensArraySlice = {
  array: LensArraySpec;
  selection: AxisValues;
  resolved_slices: unknown[];
  shape: number[];
  dtype: string;
  summary: Record<string, unknown>;
  truncated: boolean;
  values?: unknown;
  preview?: unknown;
  preview_shape?: number[];
};

export type ResolvedSelection = {
  selection: SelectionState;
  episodes: Record<string, unknown>[];
  examples: Record<string, Record<string, unknown>[]>;
  lens_arrays: LensArraySpec[];
  model_sites: ModelSiteSpec[];
  suggested_panels: PanelRecipe[];
  provenance: Record<string, unknown>;
  valid_references: Record<string, unknown>;
  target_object_cell?: Record<string, unknown>;
  action_stabilization_cell?: Record<string, unknown>;
};

export type SavedCohortResponse = {
  cohort?: CohortSpec;
  cohorts?: CohortSpec[];
};

export type SavedWorkspaceResponse = {
  workspace?: SavedWorkspace;
  workspaces?: SavedWorkspace[];
};
