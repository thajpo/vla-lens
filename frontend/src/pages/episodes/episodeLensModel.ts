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
import type {
  EvidenceClaimLevel,
  EvidencePinSavePayload,
  ModelLocusRef,
  ProbeEvidenceBundle,
  ResearchSelectionState,
} from "../../types/probeEvidence";
import {
  primitivesByKind,
  probeEpisodeLensAdapter,
  selectContributionClaimLevel,
  selectCurrentMomentEvidence,
  selectUnavailableReasons,
} from "../../types/probeEvidence.ts";
import {
  contributionCaveat,
  contributionFeatureLabel,
  humanizeProbeText,
  modelLocusDisplayLabel,
  probeEvidenceDisplaySpec,
} from "../probeDisplayCopy";
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
export type ProbeEvidenceSelectionArgs = {
  activeTraceId: string;
  currentTimestep: number;
  initialLensRunId?: string;
  initialResearchSelection?: ResearchSelectionState;
  policyCallIndex?: number;
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

export function probeEvidenceSelection(
  bundle: ProbeEvidenceBundle | undefined,
  {
    activeTraceId,
    currentTimestep,
    initialLensRunId,
    initialResearchSelection,
    policyCallIndex,
  }: ProbeEvidenceSelectionArgs,
): ResearchSelectionState | null {
  if (!bundle) {
    return null;
  }
  const defaultSelection = probeEpisodeLensAdapter.defaultSelection(bundle);
  const initialMatches =
    initialResearchSelection?.lens_id === bundle.artifact.lens_id &&
    (!initialResearchSelection.lens_run_id ||
      initialResearchSelection.lens_run_id === bundle.run.lens_run_id) &&
    (!initialLensRunId || initialLensRunId === bundle.run.lens_run_id);
  const base = initialMatches ? initialResearchSelection : defaultSelection;
  const selection = {
    ...base,
    dataset_id: base.dataset_id ?? bundle.run.dataset_id,
    episode_id: (activeTraceId || base.episode_id) ?? null,
    lens_id: bundle.artifact.lens_id,
    lens_run_id: bundle.run.lens_run_id,
    policy_call: policyCallIndex ?? base.policy_call ?? null,
    timestep: currentTimestep ?? base.timestep ?? null,
  };
  return {
    ...selection,
    model_locus: selection.model_locus ?? modelLocusForResearchSelection(bundle, selection),
  };
}

export function probeLayerReferencesFromEvidenceBundle(
  bundle: ProbeEvidenceBundle | undefined,
  selection: ResearchSelectionState | null,
): ProbeLayerRef[] {
  if (!bundle || !selection) {
    return [];
  }
  const readout = probeEvidenceReadout(bundle, selection);
  return probeEpisodeLensAdapter.pipelineAnnotations(bundle, selection)
    .filter((annotation) => annotation.model_locus.model_site_id || annotation.model_locus.layer !== undefined)
    .map((annotation) => ({
      actual: readout.actual,
      artifactId: bundle.artifact.lens_id,
      confidence: readout.confidence ?? null,
      correct: readout.correct,
      default: modelLocusMatchesSelection(annotation.model_locus, selection),
      layer: annotation.model_locus.layer ?? null,
      modelSiteId: annotation.model_locus.model_site_id ?? annotation.model_locus.module ?? `layer-${annotation.model_locus.layer ?? "unknown"}`,
      name: bundle.artifact.name,
      policyCall: selection.policy_call ?? annotation.policy_call ?? null,
      predicted: readout.predicted,
      selected: modelLocusMatchesSelection(annotation.model_locus, selection),
      target: bundle.artifact.target ?? bundle.artifact.name,
      trained: annotation.source === "model_locus",
    }));
}

export function probeEvidenceTimelineMarks(
  bundle: ProbeEvidenceBundle | undefined,
  selection: ResearchSelectionState | null,
): TimelineLensAnnotation[] {
  if (!bundle) {
    return [];
  }
  return probeEpisodeLensAdapter.timelineRows(bundle)
    .filter((row) => !selection?.episode_id || row.episode_id === selection.episode_id)
    .map((row) => ({
      kind: row.source,
      label: row.ranking ? humanLabel(String(row.ranking)) : row.source.replaceAll("_", " "),
      policy_call_index: row.policy_call ?? null,
      selected: Boolean(selection) && timelineRowMatchesSelection(row, selection),
      timestep: row.timestep ?? null,
      value: row.score ?? row.confidence ?? row.prediction ?? null,
      verdict: row.source === "failure_case" ? "wrong" : undefined,
    }));
}

export function probeEvidenceSiteReadoutFromBundle(
  bundle: ProbeEvidenceBundle | undefined,
  selection: ResearchSelectionState | null,
  selectedSiteName: string,
): ProbeSiteReadout | null {
  if (!bundle || !selection) {
    return null;
  }
  const annotations = probeEpisodeLensAdapter.pipelineAnnotations(bundle, selection);
  const locusAnnotation = annotations.find((annotation) => {
    const site = annotation.model_locus.model_site_id ?? annotation.model_locus.module ?? "";
    return !selectedSiteName || site === selectedSiteName;
  }) ?? annotations[0];
  const modelSiteId = locusAnnotation?.model_locus.model_site_id ?? locusAnnotation?.model_locus.module ?? null;
  if (!modelSiteId) {
    return null;
  }
  const rows = probeEpisodeLensAdapter.channelRanking(bundle, selection);
  const unavailable = selectUnavailableReasons(bundle, { panel_id: "contribution" })[0] ??
    selectUnavailableReasons(bundle, { capability: "contribution_breakdown" })[0];
  return {
    available: true,
    default_feature: featureFromContributionKey(rows[0]?.key),
    feature_contributors: rows.map((row) => ({
      contribution: row.value,
      direction: row.sign ?? "unknown",
      feature: featureFromContributionKey(row.key) ?? row.rank,
      label: contributionLabel(row.label, row.claim_level),
      rank: row.rank,
      sign_label: claimLevelLabel(row.claim_level),
    })),
    feature_contributors_available: rows.length > 0,
    feature_contributors_unavailable_reason: rows.length ? null : unavailable?.message ?? "Contribution breakdown unavailable for this probe.",
    intervention_seed_available: false,
    layer: locusAnnotation?.model_locus.layer ?? null,
    model_site_id: modelSiteId,
    normalization: "probe evidence bundle",
    policy_call_index: selection.policy_call ?? null,
    probe_contribution_ranking_available: rows.length > 0,
    ranking_basis: selectContributionClaimLevel(bundle, selection) ?? "numeric_only",
    raw_activation_ranking_available: true,
    site_readout_available: true,
    temporal_readout_available: true,
    timestep: selection.timestep ?? null,
    units: "contribution",
  };
}

export function probeEvidenceFeatureRows(
  bundle: ProbeEvidenceBundle | undefined,
  selection: ResearchSelectionState | null,
  rankingMode: LensRankingMode = "probe_contribution",
): LensFeatureRow[] {
  if (!bundle || !selection || rankingMode === "raw_activation") {
    return [];
  }
  return probeEpisodeLensAdapter.channelRanking(bundle, selection).map((row) => ({
    detail: contributionCaveat(row.claim_level),
    direction: row.sign ?? "unknown",
    index: featureFromContributionKey(row.key) ?? row.rank,
    label: contributionFeatureLabel(row.key, row.rank),
    title: "Probe-weighted contribution",
    value: row.value,
  }));
}

export function probeEvidenceReadout(
  bundle: ProbeEvidenceBundle,
  selection: ResearchSelectionState,
): LensReadoutSummary {
  const current = selectCurrentMomentEvidence(bundle, selection);
  const prediction = current.predictions[0];
  const failure = current.failure_moments[0];
  const ranked = current.ranked_moments[0];
  const correct = prediction?.correct ?? null;
  const confidence = prediction?.confidence ?? ranked?.confidence ?? ranked?.score ?? null;
  return {
    actual: prediction?.label ?? ranked?.label ?? failure?.label ?? null,
    confidence,
    correct,
    model_site_id: current.model_loci[0]?.locus.model_site_id ?? null,
    policy_call_index: prediction?.policy_call ?? selection.policy_call ?? null,
    predicted: prediction?.prediction ?? ranked?.prediction ?? failure?.prediction ?? null,
    score: ranked?.score ?? failure?.score ?? null,
    split: prediction?.split ?? null,
    timestep: prediction?.timestep ?? selection.timestep ?? null,
    verdict: correct === true ? "correct" : correct === false ? "wrong" : confidence === null ? "unknown" : "ambiguous",
  };
}

export function probeEvidenceSpec(bundle?: ProbeEvidenceBundle): Record<string, string> {
  if (!bundle) {
    return {};
  }
  const spec = probeEvidenceDisplaySpec(bundle);
  return {
    input: spec.input.value,
    objective: spec.objective.value,
    output: spec.output.value,
    prediction: spec.prediction.value,
  };
}

export function probeEvidenceCallouts(bundle?: ProbeEvidenceBundle): string[] {
  return (bundle?.unavailable ?? []).map((reason) => reason.message).slice(0, 3);
}

export function probeEvidencePinPayload(
  bundle: ProbeEvidenceBundle,
  selection: ResearchSelectionState,
  note = "",
  options: { feature?: number | null; modelSiteId?: string | null } = {},
): EvidencePinSavePayload {
  const current = selectCurrentMomentEvidence(bundle, selection);
  const readout = probeEvidenceReadout(bundle, selection);
  const contribution = current.contributions[0];
  const selectedModelSiteId = options.modelSiteId || selection.model_locus?.model_site_id || current.model_loci[0]?.locus.model_site_id || null;
  const selectedFeatureId = typeof options.feature === "number" ? `dim_${options.feature}` : selection.feature_id ?? null;
  const primitive_kind = current.failure_moments[0] ||
    selection.ranking === "false_positive" ||
    selection.ranking === "false_negative"
    ? "failure_case"
    : contribution
    ? "contribution"
    : current.predictions[0]
      ? "prediction"
      : current.ranked_moments[0]
        ? "ranked_moments"
        : current.model_loci[0]
          ? "model_locus"
          : current.score_series[0]
            ? "score_series"
            : null;
  return {
    label: bundle.artifact.name,
    note,
    selection: {
      ...selection,
      feature_id: selectedFeatureId,
      model_locus: selectedModelSiteId ? { ...selection.model_locus, model_site_id: selectedModelSiteId } : selection.model_locus,
    },
    evidence: {
      claim_level: selectContributionClaimLevel(bundle, selection),
      confidence: readout.confidence ?? null,
      model_site_id: selectedModelSiteId,
      prediction: readout.predicted ?? null,
      primitive_kind,
      score: readout.score ?? null,
      selected_contributor: selectedFeatureId,
    },
  };
}

export function probeSourceSitesFromEvidenceBundle(
  bundle: ProbeEvidenceBundle | undefined,
  selection: ResearchSelectionState | null,
): ProbeSourceSite[] {
  if (!bundle || !selection) {
    return [];
  }
  return probeEpisodeLensAdapter.pipelineAnnotations(bundle, selection)
    .filter((annotation) => annotation.source === "model_locus")
    .map((annotation) => {
      const modelSiteId = annotation.model_locus.model_site_id ?? annotation.model_locus.module ?? `layer-${annotation.model_locus.layer ?? "unknown"}`;
      return {
        available: true,
        default: modelLocusMatchesSelection(annotation.model_locus, selection),
        label: annotation.label ? humanizeProbeText(annotation.label) : modelLocusDisplayLabel(annotation.model_locus),
        layer: annotation.model_locus.layer ?? null,
        model_site_id: modelSiteId,
        selected: modelLocusMatchesSelection(annotation.model_locus, selection),
        short_label: annotation.model_locus.layer === null || annotation.model_locus.layer === undefined
          ? modelSiteId
          : `L${annotation.model_locus.layer}`,
        trained: true,
      };
    });
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

function modelLocusMatchesSelection(
  locus: { model_site_id?: string | null; layer?: number | null },
  selection: ResearchSelectionState,
): boolean {
  if (selection.model_locus?.model_site_id && locus.model_site_id) {
    return selection.model_locus.model_site_id === locus.model_site_id;
  }
  if (selection.model_locus?.layer !== undefined && selection.model_locus?.layer !== null) {
    return selection.model_locus.layer === locus.layer;
  }
  return false;
}

function timelineRowMatchesSelection(
  row: { policy_call?: number | null; timestep?: number | null },
  selection: ResearchSelectionState | null,
): boolean {
  if (!selection) {
    return false;
  }
  if (
    selection.policy_call !== undefined &&
    selection.policy_call !== null &&
    row.policy_call !== undefined &&
    row.policy_call !== null &&
    selection.policy_call !== row.policy_call
  ) {
    return false;
  }
  if (
    selection.timestep !== undefined &&
    selection.timestep !== null &&
    row.timestep !== undefined &&
    row.timestep !== null &&
    selection.timestep !== row.timestep
  ) {
    return false;
  }
  return true;
}

function modelLocusForResearchSelection(
  bundle: ProbeEvidenceBundle,
  selection: ResearchSelectionState,
): ModelLocusRef | null {
  return primitivesByKind(bundle, "model_locus").find((primitive) => {
    if (selection.episode_id && primitive.episode_id && primitive.episode_id !== selection.episode_id) {
      return false;
    }
    if (
      selection.policy_call !== undefined &&
      selection.policy_call !== null &&
      primitive.policy_call !== undefined &&
      primitive.policy_call !== null &&
      selection.policy_call !== primitive.policy_call
    ) {
      return false;
    }
    if (
      selection.timestep !== undefined &&
      selection.timestep !== null &&
      primitive.timestep !== undefined &&
      primitive.timestep !== null &&
      selection.timestep !== primitive.timestep
    ) {
      return false;
    }
    return true;
  })?.locus ?? null;
}

function featureFromContributionKey(value?: string | null): number | null {
  const match = String(value ?? "").match(/(?:dim|feature|sae|head)_([0-9]+)/);
  if (!match) {
    return null;
  }
  const parsed = Number(match[1]);
  return Number.isFinite(parsed) ? parsed : null;
}

function contributionLabel(label: string | null | undefined, claimLevel: EvidenceClaimLevel): string | null {
  if (!label) {
    return claimLevel === "numeric_only" ? null : contributionCaveat(claimLevel);
  }
  return claimLevel === "numeric_only" ? `${humanizeProbeText(label)} (numeric)` : humanizeProbeText(label);
}

function claimLevelLabel(value: EvidenceClaimLevel): string {
  const labels: Record<EvidenceClaimLevel, string> = {
    grouped_model_locus: "grouped model locus",
    human_labeled_feature: "human-labeled feature",
    numeric_only: "numeric only",
    semantic_hypothesis: "semantic hypothesis",
  };
  return labels[value];
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
