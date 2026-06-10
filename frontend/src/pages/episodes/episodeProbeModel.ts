import type {
  EpisodeProbePrediction,
  EpisodeProbeSummary,
  EpisodeProbesResponse,
} from "../../types/dataset";
import { EPISODE_PROBE_RESULT_LIMIT, type ProbeLayerRef, type ProbeTone } from "./shared";
import { formatLayerNumber, formatMaybeNumber, labelFromSnake } from "./formatters";

export type EpisodeProbeUsage = {
  detail: string;
  label: string;
  tone: "heldout" | "scored" | "train" | "unknown";
};

export type EpisodeProbeMembership = {
  detail: string;
  label: string;
  tone: "test" | "train" | "unknown" | "validation";
};

export type ProbeAuditLine = {
  detail: string;
  value: string;
};

export function sortEpisodeProbesByInterestingness(probes: EpisodeProbeSummary[]): EpisodeProbeSummary[] {
  return [...probes].sort((left, right) => {
    const delta = episodeProbeInterestingness(right) - episodeProbeInterestingness(left);
    if (delta !== 0) {
      return delta;
    }
    return String(left.target ?? left.name).localeCompare(String(right.target ?? right.name));
  });
}

export function ensureSelectedProbeVisible(
  visible: EpisodeProbeSummary[],
  sorted: EpisodeProbeSummary[],
  selectedArtifactId: string,
): EpisodeProbeSummary[] {
  if (!selectedArtifactId || visible.some((probe) => probe.artifact_id === selectedArtifactId)) {
    return visible;
  }
  const selected = sorted.find((probe) => probe.artifact_id === selectedArtifactId);
  return selected ? [...visible.slice(0, Math.max(0, EPISODE_PROBE_RESULT_LIMIT - 1)), selected] : visible;
}

export function episodeProbeInterestingness(probe: EpisodeProbeSummary): number {
  const predicted = probe.episode_summary.predicted;
  const actual = probe.episode_summary.actual;
  const correct = probeDisplayedCorrect(predicted, actual, probe.episode_summary.correct);
  const confidence = episodeProbeNumber(probe.episode_summary.confidence);
  const membership = probeEpisodeMembership(probe, probe.rows);
  const temporalRows = episodeProbeTemporalRows(probe.rows);
  const predictions = uniqueStrings(
    temporalRows
      .map((row) => formatProbeValue(row.predicted ?? row.prediction_value))
      .filter((value) => value !== "-"),
  );
  let score = 0;
  if (!probe.available) score -= 120;
  if (membership.tone === "train") score -= 260;
  if (membership.tone === "validation") score += 80;
  if (membership.tone === "test") score += 110;
  if (correct === false) score += 360;
  if (correct === null) score += 80;
  if (confidence !== null) {
    score += (1 - Math.abs(confidence - 0.5) * 2) * 110;
    if (confidence >= 0.95 && correct === true) score -= 40;
  }
  if (temporalRows.length > 1) score += Math.min(120, temporalRows.length * 10);
  if (predictions.length > 1) score += 160;
  if (String(probe.target ?? "").includes("outcome")) score += 35;
  return score;
}

export function episodeProbeUsage(
  probe: EpisodeProbeSummary | undefined,
  rows: EpisodeProbeSummary["rows"],
): EpisodeProbeUsage {
  if (!probe) {
    return { detail: "-", label: "unknown", tone: "unknown" };
  }
  const candidateRows = [
    probe.episode_summary.best_row,
    ...rows,
  ].filter(Boolean) as EpisodeProbeSummary["rows"];
  const splits = uniqueStrings(
    candidateRows
      .map((row) => normalizeProbeSplit(row.split))
      .filter((value): value is string => Boolean(value)),
  );
  const categories = uniqueStrings(
    splits
      .map(probeSplitCategory)
      .filter((value): value is Exclude<EpisodeProbeMembership["tone"], "unknown"> => value !== "unknown"),
  );
  const detail = categories.length ? `episode split ${categories.join(", ")}` : "episode split missing";
  if (candidateRows.some((row) => String(row.eval_split ?? "") === "on_demand_episode")) {
    return {
      detail: categories.length ? `scored on demand · ${detail}` : "scored on demand · episode split missing",
      label: "episode scored",
      tone: "scored",
    };
  }
  if (categories.includes("train")) {
    return { detail, label: "train", tone: "train" };
  }
  if (categories.includes("test")) {
    return { detail, label: "test", tone: "heldout" };
  }
  if (categories.includes("validation")) {
    return { detail, label: "validation", tone: "heldout" };
  }
  if (!probe.available) {
    return { detail: "not scored", label: "split missing", tone: "unknown" };
  }
  return { detail, label: "split missing", tone: "unknown" };
}

export function probeEpisodeMembership(
  probe: EpisodeProbeSummary | undefined,
  rows: EpisodeProbeSummary["rows"],
): EpisodeProbeMembership {
  if (!probe) {
    return { detail: "Split unknown", label: "split missing", tone: "unknown" };
  }
  const candidateRows = [
    probe.episode_summary.best_row,
    ...rows,
  ].filter(Boolean) as EpisodeProbeSummary["rows"];
  const splits = uniqueStrings(
    candidateRows
      .map((row) => normalizeProbeSplit(row.split))
      .filter((value): value is string => Boolean(value)),
  );
  const categories = uniqueStrings(
    splits
      .map(probeSplitCategory)
      .filter((value): value is Exclude<EpisodeProbeMembership["tone"], "unknown"> => value !== "unknown"),
  );
  if (categories.includes("train")) {
    return {
      detail: "this episode is in the probe training split",
      label: "train episode",
      tone: "train",
    };
  }
  if (categories.includes("test")) {
    return {
      detail: "this episode is in the probe test split",
      label: "test episode",
      tone: "test",
    };
  }
  if (categories.includes("validation")) {
    return {
      detail: "this episode is in the probe validation split",
      label: "validation episode",
      tone: "validation",
    };
  }
  return {
    detail: splits.length ? `unrecognized split ${splits.join(", ")}` : "Split unknown",
    label: "split missing",
    tone: "unknown",
  };
}

export function normalizeProbeSplit(value: unknown): string | null {
  const text = String(value ?? "").trim().toLowerCase();
  if (!text || text === "nan" || text === "none" || text === "null") {
    return null;
  }
  return text.replaceAll("-", "_");
}

export function probeSplitCategory(split: string | null): EpisodeProbeMembership["tone"] {
  const text = String(split ?? "").trim().toLowerCase().replaceAll("-", "_");
  if (!text) {
    return "unknown";
  }
  if (text === "train" || text === "training") {
    return "train";
  }
  if (text === "test" || text.startsWith("test_")) {
    return "test";
  }
  if (
    text === "val" ||
    text === "valid" ||
    text === "validation" ||
    text.startsWith("val_") ||
    text.includes("heldout") ||
    text.includes("held_out")
  ) {
    return "validation";
  }
  return "unknown";
}

export function probeQuestionLabel(probe: EpisodeProbeSummary | undefined): string {
  const target = String(probe?.target ?? "").trim();
  const labels: Record<string, string> = {
    first_moved_is_target: "Was the first moved object the target?",
    outcome: "How did this episode end?",
    target_moved: "Did the target object move?",
  };
  if (target && labels[target]) {
    return labels[target];
  }
  if (target) {
    return labelFromSnake(target);
  }
  return probe?.name || "Probe result";
}

export function probeTrajectoryAudit(
  rows: EpisodeProbePrediction[],
  bestRow?: EpisodeProbePrediction,
): ProbeAuditLine {
  if (!rows.length) {
    return {
      detail: "No probe predictions were returned for this episode.",
      value: "not scored",
    };
  }
  const calls = uniqueStrings(
    rows
      .map((row) => episodeProbeNumber(row.policy_call_index))
      .filter((value): value is number => value !== null)
      .map(String),
  );
  const predictions = uniqueStrings(
    rows
      .map((row) => formatProbeValue(row.predicted ?? row.prediction_value))
      .filter((value) => value !== "-"),
  );
  const firstConfident = rows.find((row) => {
    const confidence = episodeProbeNumber(row.confidence);
    return confidence !== null && confidence >= 0.75;
  });
  const bestCall = episodeProbeNumber(bestRow?.policy_call_index);
  const firstConfidentLabel = firstConfident ? probeCallTimestepLabel(firstConfident) : "";
  const value = calls.length > 1
    ? `${calls.length} calls`
    : bestCall === null
      ? "1 prediction"
      : `call ${bestCall}`;
  const detail = [
    firstConfidentLabel ? `first confident prediction at ${firstConfidentLabel}` : "no high-confidence call",
    predictions.length > 1
      ? `prediction changes across calls: ${predictions.join(" -> ")}`
      : predictions[0]
        ? `stable ${predictions[0]}`
        : "",
  ].filter(Boolean).join("; ");
  return {
    detail: detail || "One prediction available for this episode.",
    value,
  };
}

export function probeSiteAudit(
  row: EpisodeProbePrediction | undefined,
  ref: ProbeLayerRef | undefined,
  fallback: string,
): ProbeAuditLine {
  const source = probeSourceLabel(row, ref, fallback);
  const details = [
    ref?.layer === null || ref?.layer === undefined ? "" : `layer ${formatLayerNumber(ref.layer)}`,
    ref?.policyCall === null || ref?.policyCall === undefined ? "" : `policy call ${ref.policyCall}`,
    probeFeatureLabel(row, fallback),
  ].filter(Boolean);
  return {
    detail: details.join(" · ") || "Using the selected probe input.",
    value: source,
  };
}

export function probeReliabilityAudit(
  probe: EpisodeProbeSummary | undefined,
  usage: EpisodeProbeUsage,
): ProbeAuditLine {
  if (!probe?.available) {
    return {
      detail: "This episode has no usable predictions.",
      value: "unscored",
    };
  }
  const score = episodeProbeNumber(probe.metrics.best_score);
  const delta = episodeProbeNumber(probe.metrics.best_delta);
  const detail = [
    usage.detail,
    score === null ? "" : `global ${formatMaybeNumber(score)}`,
    delta === null ? "" : `delta ${formatSignedProbeNumber(delta)}`,
  ].filter(Boolean).join(" · ");
  return {
    detail,
    value: usage.label,
  };
}

export function probeCallTimestepLabel(row: EpisodeProbePrediction): string {
  const call = episodeProbeNumber(row.policy_call_index);
  const timestep = episodeProbeNumber(row.timestep);
  return [
    call === null ? "prediction" : `call ${call}`,
    timestep === null ? "" : `t~${timestep}`,
  ].filter(Boolean).join(" / ");
}

export function probeToneFromCorrect(correct: boolean | null | undefined, available: boolean | undefined): ProbeTone {
  if (!available) {
    return "unscored";
  }
  return correct === false ? "incorrect" : correct === true ? "correct" : "unscored";
}

export function probeDisplayedCorrect(
  predicted: unknown,
  actual: unknown,
  fallback: boolean | null | undefined,
): boolean | null {
  const predictedText = normalizedProbeValue(predicted);
  const actualText = normalizedProbeValue(actual);
  if (predictedText && actualText) {
    return predictedText === actualText;
  }
  return fallback === undefined ? null : fallback;
}

export function normalizedProbeValue(value: unknown): string {
  return String(value ?? "").trim().toLowerCase();
}

export function probeRowTone(row: EpisodeProbePrediction): ProbeTone {
  if (row.correct === null || row.correct === undefined) {
    return "unscored";
  }
  return row.correct ? "correct" : "incorrect";
}

export function episodeProbeTemporalRows(rows: EpisodeProbePrediction[]): EpisodeProbePrediction[] {
  return [...rows].sort((a, b) => {
    const aCall = episodeProbeNumber(a.policy_call_index);
    const bCall = episodeProbeNumber(b.policy_call_index);
    if (aCall !== null && bCall !== null && aCall !== bCall) {
      return aCall - bCall;
    }
    const aTimestep = episodeProbeNumber(a.timestep);
    const bTimestep = episodeProbeNumber(b.timestep);
    if (aTimestep !== null && bTimestep !== null && aTimestep !== bTimestep) {
      return aTimestep - bTimestep;
    }
    return String(a.model_site_id ?? a.feature ?? "").localeCompare(String(b.model_site_id ?? b.feature ?? ""));
  });
}

export function probeTemporalLabel(rows: EpisodeProbePrediction[]): string {
  if (!rows.length) {
    return "No probe predictions were returned for this episode.";
  }
  const calls = uniqueStrings(
    rows
      .map((row) => episodeProbeNumber(row.policy_call_index))
      .filter((value): value is number => value !== null)
      .map((value) => String(value)),
  );
  if (calls.length > 1) {
    return `${calls.length} calls`;
  }
  const row = rows[0];
  const timestep = episodeProbeNumber(row.timestep);
  return [
    calls[0] ? `call ${calls[0]}` : "one prediction",
    timestep === null ? "" : `t~${timestep}`,
  ].filter(Boolean).join(" · ");
}

export function probeSourceLabel(
  row: EpisodeProbePrediction | undefined,
  ref: ProbeLayerRef | undefined,
  fallback: string,
): string {
  if (row?.model_site_id) {
    return humanizeModelSite(row.model_site_id);
  }
  if (ref?.modelSiteId) {
    return humanizeModelSite(ref.modelSiteId);
  }
  if (ref?.layer !== null && ref?.layer !== undefined) {
    return `Expert layer ${formatLayerNumber(ref.layer)}`;
  }
  return fallback || "selected input";
}

export function probeModelLabel(
  row: EpisodeProbePrediction | undefined,
  probe: EpisodeProbeSummary | undefined,
): string {
  const model = String(row?.model || probe?.metrics.best_model || "").trim();
  const metric = String(probe?.metrics.best_primary_metric || row?.primary_metric || "").trim();
  return [model ? `${model} probe` : "probe", metric].filter(Boolean).join(" · ");
}

export function probeFeatureLabel(row: EpisodeProbePrediction | undefined, fallback: string): string {
  const feature = String(row?.feature || fallback || "").trim();
  if (!feature) {
    return "Probe input unavailable.";
  }
  if (feature === "selected model_sites") {
    return "Uses the selected activation tensor at each policy call.";
  }
  if (feature.includes("policy_call_index") || feature.includes("layer=")) {
    return feature.replaceAll("_", " ");
  }
  return feature;
}

export function humanizeModelSite(siteId: string): string {
  const layerMatch = siteId.match(/layers\.([0-9.]+)/);
  if (layerMatch) {
    return `Expert layer ${formatLayerNumber(Number(layerMatch[1]))}`;
  }
  if (siteId.includes("action_head.input")) {
    return "Action head input";
  }
  if (siteId.includes("action_head.output")) {
    return "Action head output";
  }
  if (siteId.includes("image_features")) {
    return "Image feature input";
  }
  return siteId.replace(/^pi05\./, "").replaceAll(".", " ");
}

export function formatProbeComparison(predicted: unknown, actual: unknown): string {
  return `pred ${formatProbeValue(predicted)} / truth ${formatProbeValue(actual)}`;
}

export type EpisodeProbeCell = {
  layer: string;
  policyCall: string;
  value: number;
  count: number;
};


export function episodeProbeCells(
  rows: EpisodeProbeSummary["rows"],
  metric: "confidence" | "correct",
): EpisodeProbeCell[] {
  const buckets = new Map<string, { layer: string; policyCall: string; total: number; count: number }>();
  for (const row of rows) {
    const layer = row.layer ?? layerFromFeature(row.feature) ?? "all";
    const policyCall = row.policy_call_index ?? policyCallFromFeature(row.feature) ?? "all";
    const value =
      metric === "correct"
        ? row.correct === null || row.correct === undefined
          ? null
          : row.correct
            ? 1
            : 0
        : episodeProbeNumber(row.confidence);
    if (value === null) {
      continue;
    }
    const key = `${layer}:${policyCall}`;
    const bucket = buckets.get(key) ?? {
      layer: String(layer),
      policyCall: String(policyCall),
      total: 0,
      count: 0,
    };
    bucket.total += value;
    bucket.count += 1;
    buckets.set(key, bucket);
  }
  return [...buckets.values()].map((bucket) => ({
    layer: bucket.layer,
    policyCall: bucket.policyCall,
    value: bucket.total / Math.max(1, bucket.count),
    count: bucket.count,
  }));
}

export function layerFromFeature(feature: string | undefined): number | null {
  const match = feature?.match(/layer=([0-9.]+)/);
  return match ? Number(match[1]) : null;
}

export function policyCallFromFeature(feature: string | undefined): number | null {
  const match = feature?.match(/policy_call_index=([0-9.]+)/);
  return match ? Number(match[1]) : null;
}

export function selectedEpisodeProbe(
  probes: EpisodeProbesResponse | undefined,
  artifactId: string,
): EpisodeProbeSummary | undefined {
  const allProbes = probes?.probes ?? [];
  const availableProbes = allProbes.filter((probe) => probe.available);
  return (
    availableProbes.find((probe) => probe.artifact_id === artifactId) ??
    allProbes.find((probe) => probe.artifact_id === artifactId) ??
    availableProbes[0] ??
    allProbes[0]
  );
}

export function probeLayerReferences(probes?: EpisodeProbesResponse): ProbeLayerRef[] {
  return (probes?.probes ?? []).map((probe) => {
    const row = probe.episode_summary.best_row;
    const bestFeature =
      probe.episode_summary.best_feature ||
      row?.feature ||
      (probe.metrics.best_feature === undefined ? "" : String(probe.metrics.best_feature));
    const rowLayer = episodeProbeNumber(row?.layer);
    const rowPolicyCall = episodeProbeNumber(row?.policy_call_index);
    return {
      actual: probe.episode_summary.actual,
      artifactId: probe.artifact_id,
      confidence: episodeProbeNumber(probe.episode_summary.confidence),
      correct: probe.episode_summary.correct,
      layer: rowLayer ?? layerFromFeature(bestFeature),
      modelSiteId: row?.model_site_id,
      name: probe.name,
      policyCall: rowPolicyCall ?? policyCallFromFeature(bestFeature),
      predicted: probe.episode_summary.predicted,
      target: probe.target,
    };
  }).filter((ref) => ref.layer !== null || ref.policyCall !== null);
}

export function episodeProbeNumber(value: unknown): number | null {
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) ? number : null;
}

export function formatSignedProbeNumber(value: number | null): string {
  return value === null ? "-" : `${value >= 0 ? "+" : ""}${value.toFixed(4)}`;
}

export function formatProbeValue(value: unknown): string {
  return value === null || value === undefined || value === "" ? "-" : String(value);
}

export function formatProbeCorrect(value: boolean | null | undefined): string {
  if (value === null || value === undefined) {
    return "unscored";
  }
  return value ? "correct" : "incorrect";
}

export function episodeProbeValueLabel(value: number): string {
  return value <= 1 ? value.toFixed(2) : value.toFixed(1);
}

export function episodeProbeColor(value: number | null, range: { min: number; max: number }): string {
  if (value === null) {
    return "#eef2f7";
  }
  const min = Number.isFinite(range.min) ? range.min : 0;
  const max = Number.isFinite(range.max) ? range.max : 1;
  const ratio = Math.max(0, Math.min(1, (value - min) / Math.max(max - min, 1e-6)));
  return `rgba(47, 111, 127, ${0.12 + ratio * 0.58})`;
}

export function uniqueStrings(values: string[]): string[] {
  return Array.from(new Set(values)).sort((left, right) =>
    left.localeCompare(right, undefined, { numeric: true }),
  );
}
