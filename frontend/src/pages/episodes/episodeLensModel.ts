import type {
  EpisodeInspectorSelection,
  EpisodeLensView,
  LensInspectorRanking,
  LensReadoutSummary,
  ProbeSourceSite,
  ProbeFeatureContribution,
  ProbeSiteReadout,
  TimelineLensAnnotation,
} from "../../types/dataset";
import type { ProbeLayerRef } from "./shared";

export type LensRankingMode = "probe_contribution" | "raw_activation";

export type LensFeatureRow = {
  detail?: string | null;
  direction?: string | null;
  index: number;
  label?: string | null;
  title?: string | null;
  value: number;
};

export function lensDefaultApplicationKey(view?: EpisodeLensView | null): string {
  if (!view?.lens?.artifact_id || !view?.episode?.trace_id) {
    return "";
  }
  return `${view.lens.artifact_id}:${view.episode.trace_id}`;
}

export function probeLayerReferencesFromLensView(
  view?: EpisodeLensView | null,
): ProbeLayerRef[] {
  if (!view || view.family !== "probe_suite") {
    return [];
  }
  const readout = view.readout ?? undefined;
  const displayName = view.lens.display_name || view.lens.artifact_id;
  return lensPipelineMarks(view)
    .filter((annotation) => annotation.model_site_id && annotation.trained)
    .map((annotation) => ({
      actual: readout?.actual,
      artifactId: view.lens.artifact_id,
      confidence: readout?.confidence ?? null,
      correct: readout?.correct ?? (readout?.verdict === "correct" ? true : undefined),
      default: Boolean(annotation.default),
      layer: annotation.layer ?? null,
      modelSiteId: annotation.model_site_id,
      name: displayName,
      policyCall: view.resolved_selection?.policy_call_index ?? null,
      predicted: readout?.predicted,
      selected: Boolean(annotation.selected),
      target: String(view.lens.spec?.prediction ?? displayName),
      trained: Boolean(annotation.trained),
    }));
}

export function probeSiteReadoutFromLensView(
  view?: EpisodeLensView | null,
): ProbeSiteReadout | null {
  if (!view || view.family !== "probe_suite") {
    return null;
  }
  const payload = view.view as { site_readout?: ProbeSiteReadout };
  return payload.site_readout ?? null;
}

export function lensPipelineMarks(view?: EpisodeLensView | null) {
  return view?.inspector?.pipeline_marks ?? view?.annotations?.pipeline ?? [];
}

export function lensTimelineMarks(view?: EpisodeLensView | null): TimelineLensAnnotation[] {
  return view?.inspector?.timeline_marks ?? view?.annotations?.timeline ?? [];
}

export function lensInspectorRankings(view?: EpisodeLensView | null): LensInspectorRanking[] {
  return view?.inspector?.rankings ?? [];
}

export function probeSourceSitesFromLensView(
  view?: EpisodeLensView | null,
): ProbeSourceSite[] {
  if (!view || view.family !== "probe_suite") {
    return [];
  }
  const payload = view.view as { source_scope?: { sites?: ProbeSourceSite[] } };
  return payload.source_scope?.sites ?? [];
}

export function lensFeatureRows(
  view: EpisodeLensView | null | undefined,
  rankingMode: LensRankingMode,
): LensFeatureRow[] {
  const siteReadout = probeSiteReadoutFromLensView(view);
  if (!siteReadout) {
    return [];
  }
  if (rankingMode === "raw_activation") {
    return (siteReadout.raw_activation_ranking ?? []).map((row) => ({
      detail: null,
      direction: row.activation >= 0 ? "positive" : "negative",
      index: row.feature,
      label: `#${row.rank}`,
      title: `Raw activation for feature ${row.feature}`,
      value: row.activation,
    }));
  }
  if (!siteReadout.probe_contribution_ranking_available) {
    return [];
  }
  return siteReadout.feature_contributors.map((row) => ({
    detail: null,
    direction: row.direction,
    index: row.feature,
    label: row.sign_label ?? null,
    title: contributorTitle(row),
    value: row.contribution ?? row.normalized_activation ?? row.activation ?? 0,
  }));
}

export function lensReadoutLine(readout?: LensReadoutSummary | null): string {
  if (!readout) {
    return "";
  }
  return [
    verdictLabel(readout.verdict),
    readout.confidence === null || readout.confidence === undefined
      ? ""
      : `confidence ${readout.confidence.toFixed(3)}`,
    readout.split ? `${humanLabel(String(readout.split))} split` : "",
  ].filter(Boolean).join(" · ");
}

export function probeTrainingLine(view?: EpisodeLensView | null): string {
  if (!view || view.family !== "probe_suite") {
    return "";
  }
  const payload = view.view as {
    probe?: {
      training_spec?: {
        layers?: number[];
        model?: string | null;
        policy_calls?: unknown;
        probe_type?: string | null;
      };
    };
  };
  const spec = payload.probe?.training_spec;
  if (!spec) {
    return "";
  }
  return [
    spec.layers?.length ? `Trained on L${spec.layers.join(", L")}` : "",
    policyCallsLabel(spec.policy_calls),
  ].filter(Boolean).join(" · ");
}

export function lensSelectionPatch(
  selection?: EpisodeInspectorSelection | null,
): Partial<EpisodeInspectorSelection> {
  if (!selection) {
    return {};
  }
  return {
    feature: selection.feature,
    layer: selection.layer,
    mode: selection.mode ?? "features",
    model_site_id: selection.model_site_id,
    policy_call_index: selection.policy_call_index,
    timestep: selection.timestep,
  };
}

export function shouldApplyLensDefault(
  view: EpisodeLensView | null | undefined,
  appliedKey: string,
): boolean {
  const key = lensDefaultApplicationKey(view);
  return Boolean(view?.available && view.recommended_selection && key && key !== appliedKey);
}

function contributorTitle(row: ProbeFeatureContribution): string | null {
  const parts = [
    "Contribution = normalized activation × probe weight",
    row.weight === null || row.weight === undefined ? "" : `weight ${row.weight.toFixed(3)}`,
    row.normalized_activation === null || row.normalized_activation === undefined
      ? ""
      : `normalized activation ${row.normalized_activation.toFixed(3)}`,
    row.activation === null || row.activation === undefined
      ? ""
      : `raw activation ${row.activation.toFixed(3)}`,
  ];
  return parts.filter(Boolean).join(" · ") || null;
}

function policyCallsLabel(value: unknown): string {
  if (value === "all") {
    return "all policy calls";
  }
  if (Array.isArray(value) && value.length) {
    const numbers = value.filter((item): item is number => typeof item === "number");
    if (!numbers.length) {
      return "";
    }
    const sorted = [...numbers].sort((a, b) => a - b);
    const contiguous = sorted.every((item, index) => index === 0 || item === sorted[index - 1] + 1);
    return contiguous && sorted.length > 2
      ? `policy calls ${sorted[0]}-${sorted[sorted.length - 1]}`
      : `policy calls ${sorted.join(", ")}`;
  }
  if (value && typeof value === "object" && "start" in value && "end" in value) {
    const range = value as { start?: unknown; end?: unknown };
    return typeof range.start === "number" && typeof range.end === "number"
      ? `policy calls ${range.start}-${range.end}`
      : "";
  }
  return "";
}

function verdictLabel(value: string): string {
  const normalized = String(value || "").replaceAll("_", " ");
  if (normalized === "high conf wrong") {
    return "High-conf wrong";
  }
  return humanLabel(normalized);
}

function humanLabel(value: string): string {
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}
