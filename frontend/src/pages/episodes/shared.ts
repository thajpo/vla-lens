import type {
  ActivationSite,
  AttentionMapResponse,
  ExpertTokenDetailsResponse,
  ImageTokenMapResponse,
} from "../../types/dataset";

export type InspectorContext = "vlm" | "expert" | "attention" | "other";
export const ACTIVATION_CLIP_OPTIONS = [0, 1, 5, 10, 20] as const;
export const TOP_CHANNEL_COUNT_OPTIONS = [8, 12, 24, 48, 96] as const;
export const EPISODE_PROBE_RESULT_LIMIT = 40;
export const EMPTY_ACTIVATION_SITES: ActivationSite[] = [];
export const DEFAULT_METRIC_X_KEY = "__metric_x__";
export const DEFAULT_METRIC_ORDER = [
  "action_norm",
  "eef_speed",
  "rewards",
  "gripper_open_signal",
  "generation_delta",
  "generation_start",
  "generation_end",
  "eef_x",
  "eef_y",
  "eef_z",
] as const;

export type MetricPlotConfig = {
  id: string;
  xKey: string;
  yKey: string;
};
export type EpisodePlotTab = "episode" | "probes";
export type PipelineFamily = "input" | "vlm" | "handoff" | "expert" | "action" | "other";
export type InspectionMode = "features" | "attention" | "computation" | "saved_state" | "advanced";
export type PipelineSiteChoice = {
  group: CaptureGroupId;
  id: string;
  label: string;
  mode: InspectionMode;
  site: ActivationSite;
};
export type CaptureGroupId = "features" | "attention" | "mlp" | "saved_state" | "action" | "other";
export type PipelineNode = {
  id: string;
  label: string;
  sublabel: string;
  family: PipelineFamily;
  captured: boolean;
  sites: ActivationSite[];
  allSites: ActivationSite[];
  choices: PipelineSiteChoice[];
  rawChoices: PipelineSiteChoice[];
};
export type PipelineStage = {
  id: string;
  label: string;
  family: PipelineFamily;
  nodes: PipelineNode[];
};
export type PipelineDiagramNode = {
  node: PipelineNode;
  stageId: string;
  x: number;
  y: number;
  width: number;
  height: number;
};
export type PipelineDiagramBand = {
  className: string;
  id: string;
  label: string;
  x: number;
  y: number;
  width: number;
  height: number;
};
export type PipelineDiagramArrow = {
  className: string;
  id: string;
  label?: string;
  labelAnchor?: "start" | "middle" | "end";
  labelX?: number;
  labelY?: number;
  path: string;
};
export type PipelineDiagramPort = {
  className: string;
  id: string;
  label?: string;
  radius?: number;
  textAnchor?: "start" | "middle" | "end";
  x: number;
  y: number;
};
export type PipelineDiagramLayout = {
  arrows: PipelineDiagramArrow[];
  bands: PipelineDiagramBand[];
  height: number;
  nodes: PipelineDiagramNode[];
  ports: PipelineDiagramPort[];
  width: number;
};
export type CameraOverlayPayload = Pick<
  ImageTokenMapResponse | ExpertTokenDetailsResponse | AttentionMapResponse,
  "available" | "maps" | "note" | "reason"
> & {
  attention_site?: string | null;
  feature?: number;
  generation_step?: number;
  head?: number | null;
  head_mode?: string;
  kind?: string;
  name?: string;
  query_mode?: string;
  query_token?: number | null;
  site?: string;
  token_index?: number;
};
export type CameraOverlayStatus = {
  detail?: string;
  isStale: boolean;
  isUpdating: boolean;
  label: string;
  mode: InspectorContext;
};
export type ProbeLayerRef = {
  actual?: string | boolean | number | null;
  artifactId: string;
  confidence?: number | null;
  correct?: boolean | null;
  default?: boolean;
  layer: number | null;
  modelSiteId?: string;
  name: string;
  policyCall: number | null;
  predicted?: string | boolean | number | null;
  selected?: boolean;
  target?: string | null;
  trained?: boolean;
};
export type ProbeTone = "correct" | "incorrect" | "unscored";
