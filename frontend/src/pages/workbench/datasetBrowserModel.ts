import type { ArtifactRecord, DatasetEpisode, ProbeDatasetIndex, ProbeEpisodeIndex, ProbeStudy, ProbeStudyReadout } from "../../types/dataset";
import { researchCopy } from "../../copy/researchCopy.ts";
import type {
  ContributionEvidence,
  LensProvenanceEvidence,
  ModelLocusEvidence,
  ProbeEvidenceBundle,
  RankedMoment,
  RankingKind,
  ResearchSelectionState,
  ScoreSeriesEvidence,
} from "../../types/probeEvidence";
import type { EpisodeOpenContext } from "./types";
import {
  conciseModelSiteLabel,
  contributionCaveat,
  humanizeProbeText,
  probeEvidenceDisplaySpec,
} from "../probeDisplayCopy.ts";

export type CoverageRow = {
  benchmark: string;
  tasks: string;
  seeds: string;
  episodes: number;
  outcomes: string;
  profiles: string;
};

export type ProbeCohortPreset =
  | "all"
  | "needs_review"
  | "heldout_wrong"
  | "confident_wrong"
  | "heldout_scored"
  | "train_sanity";
export type ProbeReviewStats = {
  confidentWrong: number;
  heldoutScored: number;
  heldoutWrong: number;
  highConfidence: number;
  scored: number;
  test: number;
  train: number;
  unscored: number;
  validation: number;
  wrong: number;
};
export type ProbeRankRecord = {
  probe: ProbeDatasetIndex;
  reasons: string[];
  score: number;
  stats: ProbeReviewStats;
};
export type ProbeTrainingDetailRow = {
  detail?: string;
  label: string;
  value: string;
};
export type ProbeTrainingDetails = {
  rows: ProbeTrainingDetailRow[];
  unavailable: boolean;
};
export type ProbeLensSpec = {
  input: ProbeTrainingDetailRow;
  objective: ProbeTrainingDetailRow;
  output: ProbeTrainingDetailRow;
  prediction: ProbeTrainingDetailRow;
};
export type ProbeEvidenceRankedRow = {
  context: EpisodeOpenContext | undefined;
  episode?: DatasetEpisode;
  moment: RankedMoment;
  predictionLabel: string;
  provenanceBadges: string[];
  ranking: RankingKind;
  resultLabel: string;
  scoreLabel: string;
  selected: boolean;
  splitLabel: string;
  timeLabel: string;
};
export type ProbeEvidenceEpisodeCue = {
  context: EpisodeOpenContext | undefined;
  markerLabel: string;
  scoreLabel: string;
  timelinePercent: number | null;
};
export type ProbeLensTone = "credible" | "limited" | "debug" | "unknown" | "danger";
export type ProbeLensMetricChip = {
  detail?: string;
  label: string;
  tone: ProbeLensTone;
  value: string;
};
export type ProbeSplitChartRow = {
  detail?: string;
  highConfWrong: number | null;
  id: "test" | "train" | "validation";
  label: string;
  scored: number;
  total: number;
  wrong: number | null;
};
export type ProbeSplitBarSegments = {
  correctCount: number;
  correctLeft: number;
  correctWidth: number;
  hasErrorCounts: boolean;
  highConfWrongCount: number;
  highConfWrongLeft: number;
  highConfWrongWidth: number;
  unknownCount: number;
  unknownWidth: number;
  wrongLeft: number;
  wrongOnlyCount: number;
  wrongWidth: number;
};
export type ProbeCalibrationBucket = {
  accuracy: number | null;
  avgConfidence: number | null;
  confidenceSum: number;
  correct: number;
  label: string;
  total: number;
};
export type ProbeConfusionRow = {
  correct: number;
  label: string;
  total: number;
  unknown: number;
  wrong: number;
};
export type ProbeContributorSummary = {
  detail: string;
  key: string;
  label: string;
  tone: "positive" | "negative" | "neutral";
  value: string;
};
export type ProbeLensWorkbenchViewModel = {
  mechanism: {
    basis: string;
    contributors: ProbeContributorSummary[];
    missing: string[];
    modelSite: string;
    output: string;
    temporal: string;
  };
  metrics: ProbeLensMetricChip[];
  spec: ProbeLensSpec;
  title: string;
  verdict: {
    detail: string;
    headline: string;
    label: string;
    tone: ProbeLensTone;
  };
};
export type ProbeEpisodeInspectionReason = {
  detail: string;
  label: string;
  timelinePercent: number | null;
  tone: "good" | "warning" | "danger" | "muted" | "selected";
};
export type ProbeReadoutFilterMode =
  | "useful"
  | "test"
  | "validation"
  | "train"
  | "selected_layer"
  | "primary_target"
  | "high_score"
  | "all";

export const PROBE_LIST_LIMIT = 80;
export const EVIDENCE_EPISODE_LIMIT = 12;
export const PROBE_READOUT_FILTER_MODES: ProbeReadoutFilterMode[] = [
  "useful",
  "test",
  "validation",
  "train",
  "selected_layer",
  "primary_target",
  "high_score",
  "all",
];
export const PROBE_READOUT_FILTER_LABELS: Record<ProbeReadoutFilterMode, string> = {
  useful: "High usefulness",
  test: "Test",
  validation: "Validation",
  train: "Train",
  selected_layer: "Selected layer",
  primary_target: "Target match",
  high_score: "High score",
  all: "All trained probes",
};
export const PROBE_SPLIT_FILTERS = ["train", "validation", "test"] as const;
export const PROBE_SPLIT_FILTER_LABELS: Record<string, string> = {
  test: "Test",
  train: "Train",
  validation: "Validation",
};
export const PROBE_PREDICTION_FILTERS = [
  "scored",
  "unscored",
  "correct",
  "incorrect",
  "high_confidence",
  "low_confidence",
] as const;
export const PROBE_PREDICTION_FILTER_LABELS: Record<string, string> = {
  correct: "Correct",
  high_confidence: "High conf.",
  incorrect: "Incorrect",
  low_confidence: "Low conf.",
  scored: "Scored",
  unscored: "Unscored",
};
export const COHORT_PRESETS: Array<{ id: ProbeCohortPreset; label: string }> = [
  { id: "all", label: "All" },
  { id: "needs_review", label: "Needs review" },
  { id: "heldout_wrong", label: "Validation/Test wrong" },
  { id: "confident_wrong", label: "High-conf wrong" },
  { id: "heldout_scored", label: "Validation/Test scored" },
  { id: "train_sanity", label: "Train-only check" },
];

export function filterProbeStudyReadouts(
  readouts: ProbeStudyReadout[],
  filterMode: ProbeReadoutFilterMode,
  study?: ProbeStudy,
): ProbeStudyReadout[] {
  const orderedReadouts = sortProbeStudyReadouts(readouts, study);
  if (filterMode === "all") {
    return orderedReadouts;
  }
  if (filterMode === "useful") {
    return highUsefulnessProbeReadouts(orderedReadouts, study);
  }
  return orderedReadouts.filter((readout) => matchesProbeReadoutFilter(readout, filterMode, study));
}

function sortProbeStudyReadouts(
  readouts: ProbeStudyReadout[],
  study?: ProbeStudy,
): ProbeStudyReadout[] {
  return [...readouts].sort((left, right) =>
    compareDescending(probeReadoutUsefulness(left, study), probeReadoutUsefulness(right, study))
    || compareProbeReadoutStable(left, right));
}

function highUsefulnessProbeReadouts(
  orderedReadouts: ProbeStudyReadout[],
  study?: ProbeStudy,
): ProbeStudyReadout[] {
  const selectedLayer = orderedReadouts.filter((readout) => readout.is_selected_layer);
  if (selectedLayer.length) {
    return selectedLayer;
  }
  const heldoutTarget = orderedReadouts.filter((readout) =>
    isProbeReadoutPrimaryTarget(readout, study) && ["test", "validation"].includes(probeReadoutSplit(readout)));
  if (heldoutTarget.length) {
    return heldoutTarget;
  }
  return orderedReadouts.slice(0, Math.min(5, orderedReadouts.length));
}

function matchesProbeReadoutFilter(
  readout: ProbeStudyReadout,
  filterMode: Exclude<ProbeReadoutFilterMode, "all" | "useful">,
  study?: ProbeStudy,
): boolean {
  if (filterMode === "test" || filterMode === "validation" || filterMode === "train") {
    return probeReadoutSplit(readout) === filterMode;
  }
  if (filterMode === "selected_layer") {
    return Boolean(readout.is_selected_layer);
  }
  if (filterMode === "primary_target") {
    return isProbeReadoutPrimaryTarget(readout, study);
  }
  return probeReadoutScoreValue(readout) >= 0.8;
}

function probeReadoutUsefulness(readout: ProbeStudyReadout, study?: ProbeStudy): number {
  const split = probeReadoutSplit(readout);
  const score = probeReadoutScoreValue(readout);
  return (
    (isProbeReadoutPrimaryTarget(readout, study) ? 100 : 0) +
    (readout.is_selected_layer ? 80 : 0) +
    (split === "test" ? 70 : split === "validation" ? 55 : split === "train" ? 20 : 0) +
    (readout.is_test_split ? 25 : 0) +
    (readout.is_selection_split ? 15 : 0) +
    score * 50 +
    Math.min(10, Number(readout.policy_call_count ?? readout.row_count ?? 0) / 100)
  );
}

function isProbeReadoutPrimaryTarget(readout: ProbeStudyReadout, study?: ProbeStudy): boolean {
  return Boolean(readout.is_primary_target || (study?.target && readout.target === study.target));
}

function probeReadoutSplit(readout: ProbeStudyReadout): string {
  return canonicalProbeSplitCategory(readout.split_category || readout.split);
}

function probeReadoutScoreValue(readout: ProbeStudyReadout): number {
  return typeof readout.balanced_accuracy === "number" && Number.isFinite(readout.balanced_accuracy)
    ? readout.balanced_accuracy
    : -1;
}

function compareDescending(left: number, right: number): number {
  return right - left;
}

function compareProbeReadoutStable(left: ProbeStudyReadout, right: ProbeStudyReadout): number {
  return String(left.readout_id || "").localeCompare(String(right.readout_id || ""));
}

export function matchesProbeFilters(
  episode: DatasetEpisode,
  probe: ProbeDatasetIndex | undefined,
  cohortPreset: ProbeCohortPreset,
  splitFilter: string,
  predictionFilter: string,
): boolean {
  if (!probe) {
    return splitFilter === "all" && predictionFilter === "all" && cohortPreset === "all";
  }
  const record = probeRecordForEpisode(probe, episode);
  if (!record) {
    return (
      splitFilter === "all" &&
      ["all", "unscored"].includes(predictionFilter) &&
      ["all"].includes(cohortPreset)
    );
  }
  if (!matchesProbeCohortPreset(record, cohortPreset)) {
    return false;
  }
  if (splitFilter !== "all" && canonicalProbeSplitCategory(record.split_category) !== splitFilter) {
    return false;
  }
  if (predictionFilter === "all") {
    return true;
  }
  if (predictionFilter === "scored") {
    return record.available;
  }
  if (predictionFilter === "unscored") {
    return !record.available;
  }
  if (predictionFilter === "correct") {
    return record.correct === true;
  }
  if (predictionFilter === "incorrect") {
    return record.correct === false;
  }
  if (predictionFilter === "high_confidence") {
    return Number(record.confidence) >= 0.8;
  }
  if (predictionFilter === "low_confidence") {
    return record.available && Number(record.confidence) < 0.8;
  }
  return true;
}

export function matchesProbeCohortPreset(record: ProbeEpisodeIndex | undefined, preset: ProbeCohortPreset): boolean {
  if (preset === "all") {
    return true;
  }
  if (!record) {
    return false;
  }
  const split = canonicalProbeSplitCategory(record.split_category);
  const heldout = split === "validation" || split === "test";
  const confidence = probeConfidenceValue(record);
  const confident = confidence !== null && confidence >= 0.8;
  if (preset === "needs_review") {
    return Boolean(record.available && (heldout || record.correct === false || confidence === null || confidence < 0.65));
  }
  if (preset === "heldout_wrong") {
    return Boolean(heldout && record.correct === false);
  }
  if (preset === "confident_wrong") {
    return Boolean(record.correct === false && confident);
  }
  if (preset === "heldout_scored") {
    return Boolean(heldout && record.available);
  }
  if (preset === "train_sanity") {
    return split === "train";
  }
  return true;
}

export function probeRecordForEpisode(
  probe: ProbeDatasetIndex | undefined,
  episode: DatasetEpisode,
): ProbeEpisodeIndex | undefined {
  return episode.probe_record ?? probe?.by_trace?.[episode.trace_id];
}

export function episodeOpenContextForProbe(
  probe: ProbeDatasetIndex | undefined,
  episode: DatasetEpisode,
): EpisodeOpenContext | undefined {
  if (!probe) {
    return undefined;
  }
  const record = probeRecordForEpisode(probe, episode);
  return {
    fromCohort: true,
    policyCall: record?.policy_call_index ?? policyCallFromProbeFeature(record?.feature ?? probe.best_feature ?? ""),
    probeId: probe.artifact_id,
  };
}

export function episodeOpenContextForProbeMoment(
  probe: ProbeDatasetIndex | undefined,
  bundle: ProbeEvidenceBundle | undefined,
  moment: RankedMoment,
  ranking: RankingKind,
  episode?: DatasetEpisode,
  datasetId?: string,
): EpisodeOpenContext | undefined {
  if (!probe || !bundle) {
    return episode ? episodeOpenContextForProbe(probe, episode) : undefined;
  }
  const selection = researchSelectionForMoment(bundle, moment, ranking, datasetId);
  const locus = modelLocusForSelection(bundle, selection);
  const feature = numberFromContributionKey(selection.feature_id);
  return {
    feature,
    fromCohort: true,
    lensRunId: bundle.run.lens_run_id,
    policyCall: selection.policy_call ?? policyCallFromProbeFeature(probe.best_feature ?? ""),
    probeId: probe.artifact_id,
    rankingMode: "probe_contribution",
    researchSelection: selection,
    siteName: locus?.locus.model_site_id ?? locus?.locus.module ?? "",
  };
}

export function researchSelectionForMoment(
  bundle: ProbeEvidenceBundle,
  moment: RankedMoment,
  ranking: RankingKind,
  datasetId?: string,
): ResearchSelectionState {
  return {
    dataset_id: activeDatasetId(bundle, datasetId),
    episode_id: moment.episode_id,
    lens_id: bundle.artifact.lens_id,
    lens_run_id: bundle.run.lens_run_id,
    policy_call: moment.policy_call ?? null,
    ranking,
    timestep: moment.timestep ?? null,
  };
}

export function probeEvidenceRankedRows({
  bundle,
  episodes,
  limit = 12,
  probe,
  selected,
  selectedDatasetId,
}: {
  bundle?: ProbeEvidenceBundle;
  episodes: DatasetEpisode[];
  limit?: number;
  probe?: ProbeDatasetIndex;
  selected?: ResearchSelectionState | null;
  selectedDatasetId?: string;
}): ProbeEvidenceRankedRow[] {
  if (!bundle || !probe) {
    return [];
  }
  const episodeById = episodeLookup(episodes);
  const provenanceBadges = probeEvidenceProvenanceBadges(bundle);
  const rowsByRanking = new Map<RankingKind, ProbeEvidenceRankedRow[]>();
  for (const ranking of ["top", "bottom", "uncertain"] as RankingKind[]) {
    const rankingRows: ProbeEvidenceRankedRow[] = [];
    for (const moment of rankedEvidenceMoments(bundle, ranking)) {
      const episode = episodeById.get(moment.episode_id);
      const record = episode ? probeRecordForEpisode(probe, episode) : probe.by_trace?.[moment.episode_id];
      const selection = researchSelectionForMoment(bundle, moment, ranking, selectedDatasetId);
      rankingRows.push({
        context: episodeOpenContextForProbeMoment(
          probe,
          bundle,
          moment,
          ranking,
          episode,
          selectedDatasetId,
        ),
        episode,
        moment,
        predictionLabel: probePredictionMomentLabel(moment, record),
        provenanceBadges,
        ranking,
        resultLabel: probeResultLabel(record),
        scoreLabel: moment.score === null || moment.score === undefined ? "-" : moment.score.toFixed(3),
        selected: researchSelectionsEqual(selection, selected),
        splitLabel: probeSplitLabel(record?.split_category, record?.split),
        timeLabel: probeMomentTimeLabel(moment),
      });
    }
    rowsByRanking.set(ranking, rankingRows);
  }
  return balancedRankedRows(rowsByRanking, limit);
}

export function probeEvidenceCueForEpisode(
  bundle: ProbeEvidenceBundle | undefined,
  episode: DatasetEpisode,
  probe?: ProbeDatasetIndex,
  selectedDatasetId?: string,
): ProbeEvidenceEpisodeCue | undefined {
  if (!bundle) {
    return undefined;
  }
  const rankingOrder = ["top", "uncertain", "bottom"] as RankingKind[];
  for (const ranking of rankingOrder) {
    const moment = rankedEvidenceMoments(bundle, ranking).find(
      (item) => item.episode_id === episode.trace_id || item.episode_id === episode.episode_id,
    );
    if (!moment) {
      continue;
    }
    const timelinePercent = momentTimelinePercent(bundle, moment);
    return {
      context: probe
        ? episodeOpenContextForProbeMoment(probe, bundle, moment, ranking, episode, selectedDatasetId)
        : undefined,
      markerLabel: `${rankingLabel(ranking)} ${probeMomentTimeLabel(moment)}`,
      scoreLabel: moment.score === null || moment.score === undefined ? "score -" : `score ${moment.score.toFixed(3)}`,
      timelinePercent,
    };
  }
  return undefined;
}

export function probeEvidenceContextForEpisode(
  probe: ProbeDatasetIndex | undefined,
  bundle: ProbeEvidenceBundle | undefined,
  episode: DatasetEpisode,
  selectedDatasetId?: string,
): EpisodeOpenContext | undefined {
  const cue = probeEvidenceCueForEpisode(bundle, episode, probe, selectedDatasetId);
  return cue?.context ?? episodeOpenContextForProbe(probe, episode);
}

export function probeEpisodeInspectionReason(
  probe: ProbeDatasetIndex,
  bundle: ProbeEvidenceBundle | undefined,
  episode: DatasetEpisode,
  selectedDatasetId?: string,
): ProbeEpisodeInspectionReason {
  const cue = probeEvidenceCueForEpisode(bundle, episode, probe, selectedDatasetId);
  if (cue) {
    const tone = cue.markerLabel.toLowerCase().includes("uncertain")
      ? "warning"
      : cue.markerLabel.toLowerCase().includes("bottom")
        ? "muted"
        : "selected";
    return {
      detail: cue.scoreLabel,
      label: cue.markerLabel,
      timelinePercent: cue.timelinePercent,
      tone,
    };
  }
  const record = probeRecordForEpisode(probe, episode);
  const confidence = record ? probeConfidenceValue(record) : null;
  if (!record?.available) {
    return {
      detail: "No probe score for this episode",
      label: "Unscored",
      timelinePercent: null,
      tone: "muted",
    };
  }
  if (record.correct === false && confidence !== null && confidence >= 0.8) {
    return {
      detail: formatDatasetProbeConfidence(confidence),
      label: "High-conf wrong",
      timelinePercent: null,
      tone: "danger",
    };
  }
  if (record.correct === false) {
    return {
      detail: formatDatasetProbeConfidence(confidence),
      label: "Wrong prediction",
      timelinePercent: null,
      tone: "warning",
    };
  }
  return {
    detail: formatDatasetProbeConfidence(confidence),
    label: "Scored episode",
    timelinePercent: null,
    tone: "good",
  };
}

export function probeLensWorkbenchModel({
  artifact,
  bundle,
  probe,
  totalEpisodes = 0,
}: {
  artifact?: ArtifactRecord;
  bundle?: ProbeEvidenceBundle;
  probe: ProbeDatasetIndex;
  totalEpisodes?: number;
}): ProbeLensWorkbenchViewModel {
  const spec = probeEvidenceSpec(bundle) ?? probeLensSpec(probe, artifact);
  const stats = probeReviewStats(probe);
  const indexedTotal = Object.keys(probe.by_trace ?? {}).length || stats.scored + stats.unscored || totalEpisodes;
  const heldoutTotal = stats.validation + stats.test;
  const heldoutScored = stats.heldoutScored || Math.min(stats.scored, heldoutTotal);
  const wrongRate = stats.scored ? stats.wrong / stats.scored : null;
  const verdict = probeLensVerdict(stats, heldoutScored, wrongRate);
  return {
    mechanism: {
      basis: bundle ? labelFromSnake(bundle.geometry.input_basis) : spec.input.value,
      contributors: bundle ? topContributorSummaries(bundle, 5) : [],
      missing: (bundle?.unavailable ?? []).map((reason) => reason.message).filter(Boolean).slice(0, 3),
      modelSite: bundle ? modelLocusSummary(bundle) || "Probe input unavailable" : probe.best_feature || probe.best_model || "Probe input unavailable",
      output: bundle ? labelFromSnake(bundle.geometry.output_kind) : spec.output.value,
      temporal: bundle ? labelFromSnake(bundle.geometry.temporal_scope) : "Episode",
    },
    metrics: probeLensMetricChips(probe, stats, indexedTotal, heldoutScored, wrongRate),
    spec,
    title: probe.name,
    verdict,
  };
}

export function probeEvidenceSpec(bundle: ProbeEvidenceBundle | undefined): ProbeLensSpec | undefined {
  if (!bundle) {
    return undefined;
  }
  return probeEvidenceDisplaySpec(bundle);
}

export function probeEvidenceProvenanceBadges(bundle: ProbeEvidenceBundle | undefined): string[] {
  if (!bundle) {
    return [];
  }
  const provenance = evidencePrimitivesByKind(bundle, "provenance")[0] as LensProvenanceEvidence | undefined;
  const fields = provenance?.fields ?? {};
  return [
    stringValue(fields.Input) || labelFromSnake(bundle.geometry.input_basis),
    stringValue(fields.Objective),
    modelLocusSummary(bundle),
  ].filter(Boolean).slice(0, 3);
}

function probeLensVerdict(
  stats: ProbeReviewStats,
  heldoutScored: number,
  wrongRate: number | null,
): ProbeLensWorkbenchViewModel["verdict"] {
  if (!stats.scored) {
    return {
      detail: researchCopy.probeVerdict.noScores.detail,
      headline: researchCopy.probeVerdict.noScores.headline,
      label: researchCopy.probeVerdict.noScores.label,
      tone: "unknown",
    };
  }
  if (!heldoutScored) {
    return {
      detail: researchCopy.probeVerdict.trainOnly.detail,
      headline: researchCopy.probeVerdict.trainOnly.headline,
      label: researchCopy.probeVerdict.trainOnly.label,
      tone: "debug",
    };
  }
  if (stats.confidentWrong > 0 || (wrongRate !== null && wrongRate >= 0.25)) {
    return {
      detail: researchCopy.probeVerdict.reviewFailures.detail,
      headline: researchCopy.probeVerdict.reviewFailures.headline,
      label: researchCopy.probeVerdict.reviewFailures.label,
      tone: "limited",
    };
  }
  return {
    detail: researchCopy.probeVerdict.heldoutAvailable.detail,
    headline: researchCopy.probeVerdict.heldoutAvailable.headline,
    label: researchCopy.probeVerdict.heldoutAvailable.label,
    tone: "credible",
  };
}

function probeLensMetricChips(
  probe: ProbeDatasetIndex,
  stats: ProbeReviewStats,
  indexedTotal: number,
  heldoutScored: number,
  wrongRate: number | null,
): ProbeLensMetricChip[] {
  const chips: ProbeLensMetricChip[] = [
    {
      detail: "episodes scored by this probe",
      label: "Scored",
      tone: stats.scored ? "credible" : "unknown",
      value: indexedTotal ? `${formatInteger(stats.scored)}/${formatInteger(indexedTotal)}` : formatInteger(stats.scored),
    },
    {
      detail: `${formatInteger(stats.validation)} validation · ${formatInteger(stats.test)} test`,
      label: "Validation/Test",
      tone: heldoutScored ? "credible" : "debug",
      value: formatInteger(heldoutScored),
    },
    {
      detail: wrongRate === null ? "No error rate yet" : `${Math.round(wrongRate * 100)}% of scored`,
      label: "Wrong",
      tone: stats.wrong ? "limited" : "credible",
      value: formatInteger(stats.wrong),
    },
    {
      detail: "highest priority review cases",
      label: "High-conf wrong",
      tone: stats.confidentWrong ? "danger" : "credible",
      value: formatInteger(stats.confidentWrong),
    },
  ];
  if (typeof probe.best_score === "number" && Number.isFinite(probe.best_score)) {
    chips.push({
      detail: probe.best_model || "selected activation tensor",
      label: "Metric",
      tone: "unknown",
      value: probe.best_score.toFixed(3),
    });
  }
  return chips;
}

function topContributorSummaries(bundle: ProbeEvidenceBundle, limit: number): ProbeContributorSummary[] {
  const bestByKey = new Map<string, { item: ContributionEvidence["items"][number]; primitive: ContributionEvidence }>();
  for (const primitive of evidencePrimitivesByKind(bundle, "contribution") as ContributionEvidence[]) {
    for (const item of primitive.items) {
      const current = bestByKey.get(item.key);
      if (!current || Math.abs(item.value) > Math.abs(current.item.value)) {
        bestByKey.set(item.key, { item, primitive });
      }
    }
  }
  return [...bestByKey.values()]
    .sort((left, right) => Math.abs(right.item.value) - Math.abs(left.item.value))
    .slice(0, limit)
    .map(({ item, primitive }) => ({
      detail: compactJoin(
        [
          contributionCaveat(primitive.claim_level),
          modelLocusRefLabel(item.model_locus),
        ],
        " · ",
      ),
      key: item.key,
      label: item.label || item.description || item.key,
      tone: item.value > 0 ? "positive" : item.value < 0 ? "negative" : "neutral",
      value: formatSignedNumber(item.value),
    }));
}

function modelLocusRefLabel(locus: ModelLocusEvidence["locus"] | null | undefined): string {
  if (!locus) {
    return "";
  }
  if (locus.model_site_id) {
    return conciseModelSiteLabel(locus.model_site_id);
  }
  if (locus.module) {
    return conciseModelSiteLabel(locus.module);
  }
  if (locus.layer !== null && locus.layer !== undefined) {
    return `layer ${locus.layer}`;
  }
  return "";
}

export function probeCoverageRows(
  episodes: DatasetEpisode[],
  probe: ProbeDatasetIndex,
): Array<{ label: string; value: number | string }> {
  let scored = 0;
  let correct = 0;
  let incorrect = 0;
  const splits = new Map<string, number>();
  for (const episode of episodes) {
    const record = probeRecordForEpisode(probe, episode);
    const category = canonicalProbeSplitCategory(record?.split_category);
    splits.set(category, (splits.get(category) ?? 0) + 1);
    if (record?.available) {
      scored += 1;
    }
    if (record?.correct === true) {
      correct += 1;
    }
    if (record?.correct === false) {
      incorrect += 1;
    }
  }
  const splitLabel = [...splits.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, count]) => `${probeSplitLabel(key)} ${count}`)
    .join(" / ");
  return [
    { label: "visible scored", value: scored },
    { label: "visible correct", value: correct },
    { label: "visible wrong", value: incorrect },
    { label: "visible splits", value: splitLabel || "-" },
  ];
}

export function rankProbesForReview(probes: ProbeDatasetIndex[]): ProbeRankRecord[] {
  return probes
    .map((probe) => {
      const stats = probeReviewStats(probe);
      const score = probeReviewScore(probe, stats);
      return { probe, reasons: probeReviewReasons(probe, stats), score, stats };
    })
    .sort((left, right) => {
      const delta = right.score - left.score;
      if (delta !== 0) {
        return delta;
      }
      return left.probe.name.localeCompare(right.probe.name);
    });
}

export function filterProbeChoices(probes: ProbeRankRecord[], query: string): ProbeRankRecord[] {
  const needle = query.trim().toLowerCase();
  if (!needle) {
    return probes;
  }
  return probes.filter(({ probe }) =>
    [
      probe.name,
      probe.target,
      probe.best_feature,
      probe.best_model,
      probeQuestionLabel(probe),
    ]
      .filter(Boolean)
      .join(" ")
      .toLowerCase()
      .includes(needle),
  );
}

export function probeReviewStats(probe: ProbeDatasetIndex): ProbeReviewStats {
  const stats: ProbeReviewStats = {
    confidentWrong: 0,
    heldoutScored: 0,
    heldoutWrong: 0,
    highConfidence: 0,
    scored: 0,
    test: 0,
    train: 0,
    unscored: 0,
    validation: 0,
    wrong: 0,
  };
  if (probe.review_stats) {
    return {
      confidentWrong: numericProbeStat(probe.review_stats.confidentWrong),
      heldoutScored: numericProbeStat(probe.review_stats.heldoutScored),
      heldoutWrong: numericProbeStat(probe.review_stats.heldoutWrong),
      highConfidence: numericProbeStat(probe.review_stats.highConfidence),
      scored: numericProbeStat(probe.review_stats.scored),
      test: numericProbeStat(probe.review_stats.test),
      train: numericProbeStat(probe.review_stats.train),
      unscored: numericProbeStat(probe.review_stats.unscored),
      validation: numericProbeStat(probe.review_stats.validation),
      wrong: numericProbeStat(probe.review_stats.wrong),
    };
  }
  for (const record of Object.values(probe.by_trace ?? {})) {
    const split = canonicalProbeSplitCategory(record.split_category);
    if (split === "train") stats.train += 1;
    if (split === "validation") stats.validation += 1;
    if (split === "test") stats.test += 1;
    if (record.available) {
      stats.scored += 1;
    } else {
      stats.unscored += 1;
    }
    const heldout = split === "validation" || split === "test";
    if (record.available && heldout) {
      stats.heldoutScored += 1;
    }
    const confidence = probeConfidenceValue(record);
    if (confidence !== null && confidence >= 0.8) {
      stats.highConfidence += 1;
    }
    if (record.correct === false) {
      stats.wrong += 1;
      if (heldout) {
        stats.heldoutWrong += 1;
      }
      if (confidence !== null && confidence >= 0.8) {
        stats.confidentWrong += 1;
      }
    }
  }
  return stats;
}

function numericProbeStat(value: number | undefined): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

export function probeSplitChartRows(probe: ProbeDatasetIndex): ProbeSplitChartRow[] {
  const rows = new Map<"test" | "train" | "validation", ProbeSplitChartRow>([
    ["train", { highConfWrong: 0, id: "train" as const, label: "Train", scored: 0, total: 0, wrong: 0 }],
    [
      "validation",
      { highConfWrong: 0, id: "validation" as const, label: "Validation", scored: 0, total: 0, wrong: 0 },
    ],
    ["test", { highConfWrong: 0, id: "test" as const, label: "Test", scored: 0, total: 0, wrong: 0 }],
  ]);
  if (probe.review_stats_by_split) {
    for (const [split, stats] of Object.entries(probe.review_stats_by_split)) {
      const splitKey = canonicalProbeSplitCategory(split);
      if (splitKey !== "train" && splitKey !== "validation" && splitKey !== "test") {
        continue;
      }
      const row = rows.get(splitKey);
      if (!row) {
        continue;
      }
      row.total = numericProbeStat(stats.total);
      row.scored = numericProbeStat(stats.scored);
      row.wrong = numericProbeStat(stats.wrong);
      row.highConfWrong = numericProbeStat(stats.highConfWrong);
    }
    return [...rows.values()];
  }
  const records = Object.values(probe.by_trace ?? {});
  if (!records.length) {
    for (const row of rows.values()) {
      const total = Number(probe.split_summary[row.id] ?? 0);
      row.total = Number.isFinite(total) ? total : 0;
      row.scored = row.total;
      row.wrong = null;
      row.highConfWrong = null;
    }
    return [...rows.values()];
  }
  for (const record of records) {
    const split = canonicalProbeSplitCategory(record.split_category);
    if (split !== "train" && split !== "validation" && split !== "test") {
      continue;
    }
    const row = rows.get(split);
    if (!row) {
      continue;
    }
    row.total += 1;
    if (record.available) {
      row.scored += 1;
    }
    if (record.correct === false) {
      row.wrong = (row.wrong ?? 0) + 1;
      if ((probeConfidenceValue(record) ?? 0) >= 0.8) {
        row.highConfWrong = (row.highConfWrong ?? 0) + 1;
      }
    }
  }
  return [...rows.values()];
}

export function probeSplitBarSegments(row: ProbeSplitChartRow): ProbeSplitBarSegments {
  const total = Math.max(0, row.total);
  const scored = Math.min(Math.max(0, row.scored), total);
  const wrongCount = row.wrong;
  if (wrongCount === null) {
    return {
      correctCount: 0,
      correctLeft: 0,
      correctWidth: 0,
      hasErrorCounts: false,
      highConfWrongCount: 0,
      highConfWrongLeft: 0,
      highConfWrongWidth: 0,
      unknownCount: scored,
      unknownWidth: percentOf(scored, total),
      wrongLeft: 0,
      wrongOnlyCount: 0,
      wrongWidth: 0,
    };
  }
  const wrong = Math.min(Math.max(0, wrongCount), scored);
  const highConfWrong = Math.min(Math.max(0, row.highConfWrong ?? 0), wrong);
  const wrongOnly = Math.min(Math.max(0, wrong - highConfWrong), scored - highConfWrong);
  const correct = Math.max(0, scored - wrongOnly - highConfWrong);
  const correctWidth = percentOf(correct, total);
  const wrongWidth = percentOf(wrongOnly, total);
  const highConfWrongWidth = percentOf(highConfWrong, total);
  return {
    correctCount: correct,
    correctLeft: 0,
    correctWidth,
    hasErrorCounts: true,
    highConfWrongCount: highConfWrong,
    highConfWrongLeft: correctWidth + wrongWidth,
    highConfWrongWidth,
    unknownCount: 0,
    unknownWidth: 0,
    wrongLeft: correctWidth,
    wrongOnlyCount: wrongOnly,
    wrongWidth,
  };
}

export function probeCalibrationRows(records: ProbeEpisodeIndex[]): ProbeCalibrationBucket[] {
  const rows: ProbeCalibrationBucket[] = [
    { accuracy: null, avgConfidence: null, confidenceSum: 0, correct: 0, label: "0-.20", total: 0 },
    { accuracy: null, avgConfidence: null, confidenceSum: 0, correct: 0, label: ".20-.40", total: 0 },
    { accuracy: null, avgConfidence: null, confidenceSum: 0, correct: 0, label: ".40-.60", total: 0 },
    { accuracy: null, avgConfidence: null, confidenceSum: 0, correct: 0, label: ".60-.80", total: 0 },
    { accuracy: null, avgConfidence: null, confidenceSum: 0, correct: 0, label: ".80-1", total: 0 },
  ];
  for (const record of records) {
    if (
      !record.available ||
      typeof record.confidence !== "number" ||
      !Number.isFinite(record.confidence) ||
      (record.correct !== true && record.correct !== false)
    ) {
      continue;
    }
    const index = Math.max(0, Math.min(4, Math.floor(record.confidence * 5)));
    const row = rows[index];
    row.total += 1;
    row.confidenceSum += record.confidence;
    if (record.correct === true) {
      row.correct += 1;
    }
  }
  return rows.map((row) => ({
    ...row,
    accuracy: row.total ? row.correct / row.total : null,
    avgConfidence: row.total ? row.confidenceSum / row.total : null,
  }));
}

export function probeConfusionRows(records: ProbeEpisodeIndex[]): ProbeConfusionRow[] {
  const rows = new Map<string, ProbeConfusionRow>();
  for (const record of records) {
    if (record.predicted === null || record.predicted === undefined || record.actual === null || record.actual === undefined) {
      continue;
    }
    const label = `${displayProbeValue(record.predicted)} -> ${displayProbeValue(record.actual)}`;
    const row = rows.get(label) ?? { correct: 0, label, total: 0, unknown: 0, wrong: 0 };
    row.total += 1;
    if (record.correct === false) row.wrong += 1;
    else if (record.correct === true) row.correct += 1;
    else row.unknown += 1;
    rows.set(label, row);
  }
  return [...rows.values()]
    .sort((left, right) => right.total - left.total || right.wrong - left.wrong || left.label.localeCompare(right.label))
    .slice(0, 8);
}

export function probeResultChartRows(probe: ProbeDatasetIndex): Array<{
  active: (activePredictionFilter: string) => boolean;
  apply: (
    onPredictionFilterChange: (value: string) => void,
    onCohortPresetChange: (preset: ProbeCohortPreset) => void,
  ) => void;
  id: string;
  label: string;
  total: number;
  value: number;
}> {
  const stats = probeReviewStats(probe);
  const total = Object.keys(probe.by_trace ?? {}).length || stats.scored + stats.unscored || 1;
  return [
    {
      active: (active) => active === "scored",
      apply: (onPredictionFilterChange) => onPredictionFilterChange("scored"),
      id: "scored",
      label: "Scored",
      total,
      value: stats.scored,
    },
    {
      active: (active) => active === "incorrect",
      apply: (onPredictionFilterChange) => onPredictionFilterChange("incorrect"),
      id: "wrong",
      label: "Wrong",
      total,
      value: stats.wrong,
    },
    {
      active: () => false,
      apply: (_onPredictionFilterChange, onCohortPresetChange) => onCohortPresetChange("confident_wrong"),
      id: "confident_wrong",
      label: "High-conf wrong",
      total,
      value: stats.confidentWrong,
    },
    {
      active: (active) => active === "high_confidence",
      apply: (onPredictionFilterChange) => onPredictionFilterChange("high_confidence"),
      id: "high_confidence",
      label: "High confidence",
      total,
      value: stats.highConfidence,
    },
    {
      active: (active) => active === "unscored",
      apply: (onPredictionFilterChange) => onPredictionFilterChange("unscored"),
      id: "unscored",
      label: "Unscored",
      total,
      value: stats.unscored,
    },
  ];
}

export function countProbeRecords(
  probe: ProbeDatasetIndex,
  predicate: (record: ProbeEpisodeIndex) => boolean,
): number {
  return Object.values(probe.by_trace ?? {}).filter(predicate).length;
}

export function probeReviewScore(probe: ProbeDatasetIndex, stats: ProbeReviewStats): number {
  const bestDelta = typeof probe.best_delta === "number" && Number.isFinite(probe.best_delta) ? probe.best_delta : 0;
  const bestScore = typeof probe.best_score === "number" && Number.isFinite(probe.best_score) ? probe.best_score : 0;
  return (
    stats.heldoutWrong * 80 +
    stats.confidentWrong * 70 +
    stats.heldoutScored * 8 +
    stats.wrong * 12 +
    stats.validation * 1.5 +
    stats.test * 2 +
    bestDelta * 120 +
    bestScore * 12 -
    Math.max(0, stats.train - stats.heldoutScored) * 0.2
  );
}

export function probeReviewReasons(probe: ProbeDatasetIndex, stats: ProbeReviewStats): string[] {
  const reasons: string[] = [];
  if (stats.confidentWrong) {
    reasons.push(`${stats.confidentWrong} high-conf wrong`);
  }
  if (stats.heldoutWrong) {
    reasons.push(`${stats.heldoutWrong} validation/test wrong`);
  }
  if (stats.heldoutScored) {
    reasons.push(`${stats.heldoutScored} validation/test scored`);
  }
  if (!stats.heldoutScored && stats.train) {
    reasons.push("training split only");
  }
  if (typeof probe.best_delta === "number" && Number.isFinite(probe.best_delta)) {
    reasons.push(`delta ${formatSignedNumber(probe.best_delta)}`);
  }
  return reasons.slice(0, 3).length ? reasons.slice(0, 3) : ["no scored episodes yet"];
}

export function rankEpisodesForProbe(episodes: DatasetEpisode[], probe: ProbeDatasetIndex): DatasetEpisode[] {
  return [...episodes].sort((left, right) => {
    const delta =
      probeEpisodeInterestScore(probeRecordForEpisode(probe, right)) -
      probeEpisodeInterestScore(probeRecordForEpisode(probe, left));
    if (delta !== 0) {
      return delta;
    }
    return episodeTitle(left).localeCompare(episodeTitle(right));
  });
}

export function probeEpisodeInterestScore(record: ProbeEpisodeIndex | undefined): number {
  if (!record) {
    return -40;
  }
  const split = canonicalProbeSplitCategory(record.split_category);
  const confidence = probeConfidenceValue(record);
  let score = 0;
  if (split === "train") score -= 90;
  if (split === "validation") score += 80;
  if (split === "test") score += 110;
  if (!record.available) score -= split === "train" ? 20 : 5;
  if (record.correct === false) score += 260;
  if (record.correct === null || record.correct === undefined) score += record.available ? 45 : 0;
  if (confidence !== null && record.correct === false && confidence >= 0.8) score += 140;
  if (confidence !== null && record.correct === true && confidence >= 0.95 && split === "train") score -= 40;
  if (confidence !== null && confidence >= 0.45 && confidence <= 0.65) score += 35;
  score += Math.min(40, Number(record.row_count ?? 0) * 4);
  return score;
}

export function probeEpisodeInterestLabel(record: ProbeEpisodeIndex | undefined): string {
  if (!record) {
    return "not scored by this probe";
  }
  const split = probeSplitLabel(record.split_category, record.split);
  if (!record.available) {
    return `${split} · unscored`;
  }
  if (record.correct === false && (probeConfidenceValue(record) ?? 0) >= 0.8) {
    return `${split} · high-conf wrong`;
  }
  if (record.correct === false) {
    return `${split} · wrong`;
  }
  if (record.correct === true) {
    return `${split} · correct`;
  }
  return `${split} · scored`;
}

export function probeMomentTimeLabel(moment: RankedMoment): string {
  if (moment.policy_call !== null && moment.policy_call !== undefined) {
    return `call ${moment.policy_call}`;
  }
  if (moment.timestep !== null && moment.timestep !== undefined) {
    return `timestep ${moment.timestep}`;
  }
  if (moment.frame_idx !== null && moment.frame_idx !== undefined) {
    return `frame ${moment.frame_idx}`;
  }
  return "episode";
}

export function rankingLabel(ranking: RankingKind): string {
  const labels: Record<RankingKind, string> = {
    bottom: "Low",
    false_negative: "False negative",
    false_positive: "False positive",
    largest_delta: "Delta",
    top: "Top",
    uncertain: "Uncertain",
  };
  return labels[ranking] ?? labelFromSnake(ranking);
}

export function probeResultLabel(record: ProbeEpisodeIndex | undefined): string {
  if (!record) {
    return "no probe record";
  }
  if (!record.available) {
    return "unscored";
  }
  const confidence = formatDatasetProbeConfidence(record.confidence);
  if (record.correct === false) {
    return `wrong · ${confidence}`;
  }
  if (record.correct === true) {
    return `correct · ${confidence}`;
  }
  return `scored · ${confidence}`;
}

export function probeTrustLabel(stats: ProbeReviewStats | undefined): string {
  if (!stats) {
    return "Select probe";
  }
  if (stats.heldoutScored > 0) {
    return "validation/test scores";
  }
  if (stats.train > 0) {
    return "training split only";
  }
  return "scores unavailable";
}

export function probeTrustDetail(stats: ProbeReviewStats | undefined): string {
  if (!stats) {
    return "No probe is selected";
  }
  if (stats.heldoutScored > 0) {
    return `${stats.validation} validation / ${stats.test} test episodes, ${stats.heldoutScored} scored`;
  }
  if (stats.train > 0) {
    return `${stats.train} training episodes; no held-out scores yet`;
  }
  return "No split labels or probe scores returned";
}

export function probeScoredCohortDetail(stats: ProbeReviewStats, rankedEpisodeTotal: number): string {
  if (rankedEpisodeTotal > 0 && rankedEpisodeTotal !== stats.scored) {
    return `${stats.scored} scored / ${rankedEpisodeTotal} ranked episodes`;
  }
  return `${stats.scored} scored episodes`;
}

export function probeTrainingDetails(
  probe: ProbeDatasetIndex,
  artifact?: ArtifactRecord,
): ProbeTrainingDetails {
  const method = recordValue(artifact?.method);
  const input = recordValue(method?.input);
  const selector = recordValue(input?.selector) ?? recordValue(artifact?.selector);
  const target = recordValue(method?.target);
  const probeMethod = recordValue(method?.probe);
  const split = recordValue(method?.split);
  const evaluation = recordValue(method?.evaluation);
  const examples = recordValue(method?.examples);
  const display = recordValue(artifact?.display);

  const rows = [
    {
      detail: featureInputDetail(input, selector, display),
      label: "X",
      value: featureInputSummary(selector, input, probe),
    },
    {
      detail: targetDetail(target),
      label: "Y",
      value: targetSummary(target, probe),
    },
    {
      detail: modelObjectiveDetail(probeMethod),
      label: "Objective",
      value: modelObjectiveLabel(probeMethod, target),
    },
    {
      detail: splitDetail(method, split, evaluation, probeMethod),
      label: "Split",
      value: splitSummary(method, split),
    },
    {
      detail: metricDetail(evaluation),
      label: "Metric",
      value: metricSummary(evaluation),
    },
    {
      detail: examplesDetail(examples),
      label: "Training data",
      value: examplesSummary(examples, input, display),
    },
    {
      detail: probe.best_feature || "No selected feature recorded",
      label: "Probe input",
      value: probe.best_model || "Probe input unavailable",
    },
    {
      detail: "Training/evaluation metrics stay fixed; compatible new episodes can be scored from saved weights.",
      label: "Frozen",
      value: "frozen metrics + refreshable scores",
    },
  ];

  return {
    rows: rows.filter((row) => row.value && row.value !== "-"),
    unavailable: !method,
  };
}

export function probeLensSpec(probe: ProbeDatasetIndex, artifact?: ArtifactRecord): ProbeLensSpec {
  const method = recordValue(artifact?.method);
  const input = recordValue(method?.input);
  const selector = recordValue(input?.selector) ?? recordValue(artifact?.selector);
  const target = recordValue(method?.target);
  const probeMethod = recordValue(method?.probe);
  const evaluation = recordValue(method?.evaluation);
  return {
    input: {
      detail: simpleInputDetail(selector, input),
      label: "Input",
      value: simpleInputLabel(selector, input, probe),
    },
    objective: {
      detail: simpleObjectiveDetail(evaluation),
      label: "Objective",
      value: simpleObjectiveLabel(probeMethod, target),
    },
    output: {
      detail: simpleOutputDetail(target, probe),
      label: "Output",
      value: simpleOutputLabel(probeMethod, target),
    },
    prediction: {
      detail: simplePredictionDetail(target),
      label: "Prediction",
      value: simplePredictionLabel(target, probe),
    },
  };
}

export function probeConfidenceValue(record: ProbeEpisodeIndex): number | null {
  return typeof record.confidence === "number" && Number.isFinite(record.confidence)
    ? record.confidence
    : null;
}

export function percentOf(value: number, total: number): number {
  if (!total) {
    return 0;
  }
  return Math.max(0, Math.min(100, (value / total) * 100));
}

export function policyCallFromProbeFeature(feature: string | null | undefined): number | null {
  const match = String(feature ?? "").match(/policy_call_index=([0-9.]+)/);
  if (!match) {
    return null;
  }
  const parsed = Number(match[1]);
  return Number.isFinite(parsed) ? parsed : null;
}

export function probeQuestionLabel(probe: ProbeDatasetIndex): string {
  const target = String(probe.target ?? "").trim();
  const labels: Record<string, string> = {
    first_moved_is_target: "Was the first moved object the target?",
    outcome: "How did this episode end?",
    target_contacted: "Did the target object get contacted?",
    target_moved: "Did the target object move?",
  };
  if (target && labels[target]) {
    return labels[target];
  }
  return target ? labelFromSnake(target) : "Probe result";
}

function simplePredictionLabel(target: Record<string, unknown> | undefined, probe: ProbeDatasetIndex): string {
  const name = stringValue(target?.name) || stringValue(target?.resolved_column) || probe.target || "";
  const labels: Record<string, string> = {
    first_moved_is_target: "First moved object is target",
    outcome: "Episode outcome",
    target_contacted: "Target contacted",
    target_moved: "Target moved",
  };
  return labels[name] ?? (name ? labelFromSnake(name) : "Probe result");
}

function simplePredictionDetail(target: Record<string, unknown> | undefined): string {
  const source = stringValue(target?.source);
  if (source === "row") {
    return "episode label";
  }
  return source ? `${labelFromSnake(source)} label` : "";
}

function simpleInputLabel(
  selector: Record<string, unknown> | undefined,
  input: Record<string, unknown> | undefined,
  probe: ProbeDatasetIndex,
): string {
  const moduleName = stringValue(selector?.module) || firstString(listValue(input?.model_site_ids)) || probe.best_model || "";
  const tensor = stringValue(selector?.tensor_type);
  const prefix = moduleName.includes("expert") ? "Expert" : moduleName.includes("policy") ? "Policy" : "";
  if (tensor.includes("hidden")) {
    return compactJoin([prefix, "hidden states"], " ");
  }
  if (tensor) {
    return labelFromSnake(tensor);
  }
  return moduleName ? humanModuleLabel(moduleName) : "Activation features";
}

function simpleInputDetail(
  selector: Record<string, unknown> | undefined,
  input: Record<string, unknown> | undefined,
): string {
  const selection = recordValue(input?.selection);
  const tokenKind = stringValue(selector?.token_kind) || stringValue(selection?.token_kind);
  return compactJoin(
    [
      tokenKind ? `${tokenKind} tokens` : "",
      shortLayerLabel(selector?.layers),
      stringValue(selector?.generation_step) === "final" || stringValue(selection?.generation_step) === "final"
        ? "final step"
        : "",
    ],
    " · ",
  );
}

function simpleOutputLabel(
  probeMethod: Record<string, unknown> | undefined,
  target: Record<string, unknown> | undefined,
): string {
  const bestState = recordValue(probeMethod?.best_model_state);
  const classes = listValue(bestState?.classes);
  const kind = (stringValue(target?.kind) || stringValue(probeMethod?.type) || stringValue(bestState?.probe_type)).toLowerCase();
  if (kind.includes("regression") || kind === "continuous") {
    return "Number";
  }
  if (classes.length && classes.length <= 3) {
    return classes.join(" / ");
  }
  if (kind.includes("classification")) {
    return "Class label";
  }
  return kind ? labelFromSnake(kind) : "Prediction";
}

function simpleOutputDetail(target: Record<string, unknown> | undefined, probe: ProbeDatasetIndex): string {
  const name = stringValue(target?.resolved_column) || stringValue(target?.name) || probe.target || "";
  return name ? labelFromSnake(name) : "";
}

function simpleObjectiveLabel(
  probeMethod: Record<string, unknown> | undefined,
  target: Record<string, unknown> | undefined,
): string {
  const objective = modelObjectiveLabel(probeMethod, target).replace(" classification", "");
  return objective.replace(/^\w/, (letter) => letter.toUpperCase());
}

function simpleObjectiveDetail(evaluation: Record<string, unknown> | undefined): string {
  const metric = metricSummary(evaluation);
  return metric === "evaluation metric" ? "" : `metric: ${metric}`;
}

function humanModuleLabel(moduleName: string): string {
  if (moduleName.includes("expert")) {
    return "Expert activations";
  }
  return moduleName.replace(/^pi05\./, "").replaceAll(".", " ");
}

function shortLayerLabel(value: unknown): string {
  const layers = listValue(value);
  if (!layers.length) {
    return "";
  }
  if (layers.length > 5) {
    return `${layers.length} layers`;
  }
  return `layers ${rangeOrList(layers)}`;
}

function featureInputSummary(
  selector: Record<string, unknown> | undefined,
  input: Record<string, unknown> | undefined,
  probe: ProbeDatasetIndex,
): string {
  const selection = recordValue(input?.selection);
  const moduleName =
    stringValue(selector?.module) ||
    stringValue(selector?.name) ||
    firstString(listValue(input?.model_site_ids)) ||
    probe.best_model ||
    "activation features";
  const layers = listLabel(selector?.layers, "layers");
  const tensor = stringValue(selector?.tensor_type);
  const tokenKind = stringValue(selector?.token_kind) || stringValue(selection?.token_kind);
  const policyCalls = listLabel(selector?.policy_calls ?? selection?.policy_calls, "policy calls");
  return compactJoin(
    [
      moduleName,
      layers,
      tensor,
      tokenKind ? `${tokenKind} tokens` : "",
      policyCalls,
    ],
    " · ",
  );
}

function featureInputDetail(
  input: Record<string, unknown> | undefined,
  selector: Record<string, unknown> | undefined,
  display: Record<string, unknown> | undefined,
): string {
  const selection = recordValue(input?.selection);
  const featureDim = numberValue(input?.feature_dim) ?? numberValue(display?.feature_dim);
  const shape = listValue(input?.feature_shape).join(" x ");
  return compactJoin(
    [
      stringValue(selector?.generation_step) || stringValue(selection?.generation_step),
      stringValue(input?.pooling) || stringValue(selector?.reduce_tokens),
      featureDim ? `${formatInteger(featureDim)} dims` : "",
      stringValue(input?.dtype),
      shape ? `shape ${shape}` : "",
    ],
    " · ",
  );
}

function targetSummary(target: Record<string, unknown> | undefined, probe: ProbeDatasetIndex): string {
  const name = stringValue(target?.name) || stringValue(target?.resolved_column) || probe.target || "probe target";
  const kind = stringValue(target?.kind);
  return compactJoin([name, kind], " · ");
}

function targetDetail(target: Record<string, unknown> | undefined): string {
  const selector = recordValue(target?.selector);
  return compactJoin(
    [
      stringValue(target?.source) ? `source ${stringValue(target?.source)}` : "",
      stringValue(target?.resolved_column) ? `column ${stringValue(target?.resolved_column)}` : "",
      stringValue(selector?.object) ? `object ${stringValue(selector?.object)}` : "",
    ],
    " · ",
  );
}

function modelObjectiveLabel(
  probeMethod: Record<string, unknown> | undefined,
  target: Record<string, unknown> | undefined,
): string {
  const bestState = recordValue(probeMethod?.best_model_state);
  const model = (stringValue(probeMethod?.primary_model) || stringValue(bestState?.model)).toLowerCase();
  const kind = (
    stringValue(target?.kind) ||
    stringValue(probeMethod?.type) ||
    stringValue(bestState?.probe_type)
  ).toLowerCase();
  const regression = kind.includes("regression") || kind === "continuous";
  if (model.includes("linear")) {
    return regression ? "ridge regression" : "logistic regression classification";
  }
  if (model.includes("mlp")) {
    return regression ? "MLP regression" : "MLP classification";
  }
  if (model) {
    return regression ? `${model} regression` : `${model} classification`;
  }
  return regression ? "regression probe" : "classification probe";
}

function modelObjectiveDetail(probeMethod: Record<string, unknown> | undefined): string {
  const models = listValue(probeMethod?.models);
  return compactJoin(
    [
      stringValue(probeMethod?.library),
      stringValue(probeMethod?.trained_on_split) ? `trained on ${stringValue(probeMethod?.trained_on_split)}` : "",
      models.length ? `candidates ${models.join(", ")}` : "",
    ],
    " · ",
  );
}

function splitSummary(
  method: Record<string, unknown> | undefined,
  split: Record<string, unknown> | undefined,
): string {
  return (
    stringValue(split?.kind) ||
    stringValue(split?.strategy) ||
    stringValue(split?.split_kind) ||
    (listValue(method?.eval_values).length ? "heldout evaluation" : "split metadata")
  );
}

function splitDetail(
  method: Record<string, unknown> | undefined,
  split: Record<string, unknown> | undefined,
  evaluation: Record<string, unknown> | undefined,
  probeMethod: Record<string, unknown> | undefined,
): string {
  const evalValues = listValue(split?.eval_values).length
    ? listValue(split?.eval_values)
    : listValue(method?.eval_values).length
      ? listValue(method?.eval_values)
      : listValue(evaluation?.eval_splits);
  return compactJoin(
    [
      stringValue(split?.train_value) || stringValue(probeMethod?.trained_on_split)
        ? `train ${stringValue(split?.train_value) || stringValue(probeMethod?.trained_on_split)}`
        : "",
      evalValues.length ? `eval ${evalValues.join(", ")}` : "",
      stringValue(split?.selection_value) ||
      stringValue(method?.selection_value) ||
      stringValue(evaluation?.selection_split)
        ? `select ${
            stringValue(split?.selection_value) ||
            stringValue(method?.selection_value) ||
            stringValue(evaluation?.selection_split)
          }`
        : "",
    ],
    " · ",
  );
}

function metricSummary(evaluation: Record<string, unknown> | undefined): string {
  const metric = stringValue(evaluation?.primary_metric) || stringValue(evaluation?.metric) || "evaluation metric";
  return metric.replaceAll("_", " ");
}

function metricDetail(evaluation: Record<string, unknown> | undefined): string {
  return compactJoin(
    [
      stringValue(evaluation?.primary_split) ? `primary ${stringValue(evaluation?.primary_split)}` : "",
      stringValue(evaluation?.selection_split) ? `selected on ${stringValue(evaluation?.selection_split)}` : "",
      stringValue(evaluation?.aggregation),
      stringValue(evaluation?.grain) ? `${stringValue(evaluation?.grain)} grain` : "",
    ],
    " · ",
  );
}

function examplesSummary(
  examples: Record<string, unknown> | undefined,
  input: Record<string, unknown> | undefined,
  display: Record<string, unknown> | undefined,
): string {
  const rowCount =
    numberValue(examples?.count) ??
    numberValue(display?.row_count) ??
    numberValue(input?.feature_shape, 0);
  const featureDim = numberValue(input?.feature_dim) ?? numberValue(display?.feature_dim);
  return compactJoin(
    [
      rowCount ? `${formatInteger(rowCount)} records` : "",
      featureDim ? `${formatInteger(featureDim)} features` : "",
    ],
    " · ",
  ) || "input metadata unavailable";
}

function examplesDetail(examples: Record<string, unknown> | undefined): string {
  const countBySplit = recordValue(examples?.count_by_split);
  if (!countBySplit) {
    return "";
  }
  return Object.entries(countBySplit)
    .map(([split, count]) => `${split} ${formatInteger(numberValue(count) ?? 0)}`)
    .join(" · ");
}

function recordValue(value: unknown): Record<string, unknown> | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return undefined;
  }
  return value as Record<string, unknown>;
}

function stringValue(value: unknown): string {
  if (typeof value === "string") {
    return value.trim();
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return "";
}

function episodeLookup(episodes: DatasetEpisode[]): Map<string, DatasetEpisode> {
  const out = new Map<string, DatasetEpisode>();
  for (const episode of episodes) {
    out.set(episode.trace_id, episode);
    out.set(episode.episode_id, episode);
  }
  return out;
}

function activeDatasetId(bundle: ProbeEvidenceBundle, selectedDatasetId?: string): string {
  return selectedDatasetId && selectedDatasetId !== "all" ? selectedDatasetId : bundle.run.dataset_id;
}

function balancedRankedRows(
  rowsByRanking: Map<RankingKind, ProbeEvidenceRankedRow[]>,
  limit: number,
): ProbeEvidenceRankedRow[] {
  const rankings = ["top", "bottom", "uncertain"] as RankingKind[];
  const visible: ProbeEvidenceRankedRow[] = [];
  const perRanking = Math.max(1, Math.floor(limit / rankings.length));
  for (const ranking of rankings) {
    visible.push(...(rowsByRanking.get(ranking) ?? []).slice(0, perRanking));
  }
  if (visible.length >= limit) {
    return visible.slice(0, limit);
  }
  for (const ranking of rankings) {
    const existing = new Set(visible.map(rankedRowKey));
    for (const row of rowsByRanking.get(ranking) ?? []) {
      if (visible.length >= limit) {
        return visible;
      }
      if (!existing.has(rankedRowKey(row))) {
        visible.push(row);
        existing.add(rankedRowKey(row));
      }
    }
  }
  return visible;
}

function rankedRowKey(row: ProbeEvidenceRankedRow): string {
  return [
    row.ranking,
    row.moment.episode_id,
    row.moment.timestep ?? "",
    row.moment.policy_call ?? "",
  ].join("|");
}

function probePredictionMomentLabel(
  moment: RankedMoment,
  record: ProbeEpisodeIndex | undefined,
): string {
  const prediction = moment.prediction ?? record?.predicted;
  const label = moment.label ?? record?.actual;
  if (prediction !== null && prediction !== undefined && label !== null && label !== undefined) {
    return `${String(prediction)} / ${String(label)}`;
  }
  if (prediction !== null && prediction !== undefined) {
    return String(prediction);
  }
  return record?.available ? "prediction available" : "prediction missing";
}

function researchSelectionsEqual(
  left: ResearchSelectionState,
  right?: ResearchSelectionState | null,
): boolean {
  if (!right) {
    return false;
  }
  return (
    left.lens_id === right.lens_id &&
    left.lens_run_id === right.lens_run_id &&
    left.episode_id === right.episode_id &&
    left.ranking === right.ranking &&
    (left.timestep ?? null) === (right.timestep ?? null) &&
    (left.policy_call ?? null) === (right.policy_call ?? null)
  );
}

function modelLocusForSelection(
  bundle: ProbeEvidenceBundle,
  selection: ResearchSelectionState,
): ModelLocusEvidence | undefined {
  const currentLoci = evidencePrimitivesByKind(bundle, "model_locus").filter(
    (primitive): primitive is ModelLocusEvidence =>
      primitive.kind === "model_locus" && evidenceMatchesSelection(primitive, selection),
  );
  return (
    currentLoci[0] ??
    (evidencePrimitivesByKind(bundle, "model_locus")[0] as ModelLocusEvidence | undefined)
  );
}

function modelLocusSummary(bundle: ProbeEvidenceBundle): string {
  const locus = (evidencePrimitivesByKind(bundle, "model_locus")[0] as ModelLocusEvidence | undefined)?.locus;
  if (!locus) {
    return "";
  }
  return compactJoin(
    [
      conciseModelSiteLabel(locus.model_site_id ?? locus.module),
      locus.layer === null || locus.layer === undefined ? "" : `layer ${locus.layer}`,
    ],
    " · ",
  );
}

function numberFromContributionKey(value: string | null | undefined): number | null {
  const match = String(value ?? "").match(/(?:dim|feature|sae)_([0-9]+)/);
  if (!match) {
    return null;
  }
  const parsed = Number(match[1]);
  return Number.isFinite(parsed) ? parsed : null;
}

function momentTimelinePercent(bundle: ProbeEvidenceBundle, moment: RankedMoment): number | null {
  const series = evidencePrimitivesByKind(bundle, "score_series").find(
    (primitive): primitive is ScoreSeriesEvidence =>
      primitive.kind === "score_series" &&
      primitive.episode_id === moment.episode_id &&
      (primitive.time_axis === "policy_call" || primitive.time_axis === "timestep"),
  );
  const position = moment.policy_call ?? moment.timestep;
  const length = series?.values_ref.shape?.[0];
  if (position === null || position === undefined || !length || length <= 1) {
    return null;
  }
  return percentOf(position, length - 1);
}

function evidencePrimitivesByKind(
  bundle: ProbeEvidenceBundle,
  kind: string,
): ProbeEvidenceBundle["primitives"] {
  return bundle.primitives.filter((primitive) => primitive.kind === kind);
}

function rankedEvidenceMoments(
  bundle: ProbeEvidenceBundle,
  ranking: RankingKind,
  limit?: number,
): RankedMoment[] {
  const primitive = bundle.primitives.find(
    (item) => item.kind === "ranked_moments" && item.ranking === ranking,
  );
  const moments = primitive?.kind === "ranked_moments" ? primitive.moments : [];
  return limit === undefined ? moments : moments.slice(0, limit);
}

function evidenceMatchesSelection(
  evidence: ModelLocusEvidence,
  selection: ResearchSelectionState,
): boolean {
  if (selection.episode_id && evidence.episode_id && evidence.episode_id !== selection.episode_id) {
    return false;
  }
  if (
    selection.policy_call !== null &&
    selection.policy_call !== undefined &&
    evidence.policy_call !== null &&
    evidence.policy_call !== undefined &&
    evidence.policy_call !== selection.policy_call
  ) {
    return false;
  }
  if (
    selection.timestep !== null &&
    selection.timestep !== undefined &&
    evidence.timestep !== null &&
    evidence.timestep !== undefined &&
    evidence.timestep !== selection.timestep
  ) {
    return false;
  }
  return true;
}

function numberValue(value: unknown, arrayIndex?: number): number | undefined {
  const candidate = arrayIndex !== undefined && Array.isArray(value) ? value[arrayIndex] : value;
  if (typeof candidate === "number" && Number.isFinite(candidate)) {
    return candidate;
  }
  return undefined;
}

function listValue(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.map((item) => stringValue(item)).filter(Boolean);
  }
  const item = stringValue(value);
  return item ? [item] : [];
}

function listLabel(value: unknown, label: string): string {
  const items = listValue(value);
  if (!items.length) {
    return "";
  }
  return `${label} ${rangeOrList(items)}`;
}

function rangeOrList(items: string[]): string {
  const numbers = items.map((item) => Number(item));
  const allNumbers = numbers.every((item) => Number.isInteger(item));
  if (allNumbers && numbers.length > 2) {
    const sorted = [...numbers].sort((left, right) => left - right);
    const consecutive = sorted.every((item, index) => index === 0 || item === sorted[index - 1] + 1);
    if (consecutive) {
      return `${sorted[0]}-${sorted[sorted.length - 1]}`;
    }
  }
  return items.join(", ");
}

function firstString(items: string[]): string {
  return items[0] ?? "";
}

function compactJoin(parts: Array<string | null | undefined>, separator: string): string {
  return parts
    .map((part) => String(part ?? "").trim())
    .filter(Boolean)
    .join(separator);
}

function formatInteger(value: number): string {
  return Math.round(value).toLocaleString("en-US");
}

export function labelFromSnake(value: string): string {
  return humanizeProbeText(value);
}

function displayProbeValue(value: string | boolean | number): string {
  return String(value).replaceAll("_", " ");
}

export function formatSignedNumber(value: number): string {
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(3)}`;
}

export function probeSplitLabel(category?: string | null, fallback?: string | null): string {
  const key = canonicalProbeSplitCategory(category);
  if (key && key !== "unknown") {
    return PROBE_SPLIT_FILTER_LABELS[key] ?? key;
  }
  return fallback ? String(fallback) : "Split missing";
}

export function canonicalProbeSplitCategory(category?: string | null): string {
  const key = String(category || "").trim().toLowerCase().replaceAll("-", "_");
  if (key === "train" || key === "training") {
    return "train";
  }
  if (key === "test" || key.startsWith("test_")) {
    return "test";
  }
  if (
    key === "validation" ||
    key === "valid" ||
    key === "val" ||
    key.startsWith("val_") ||
    key.includes("heldout") ||
    key.includes("held_out")
  ) {
    return "validation";
  }
  return "unknown";
}

export function formatDatasetProbeConfidence(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? `conf ${value.toFixed(2)}` : "conf -";
}

export function datasetCoverageRows(episodes: DatasetEpisode[]): CoverageRow[] {
  const groups = new Map<
    string,
    {
      tasks: Set<string>;
      seeds: Set<string>;
      episodes: number;
      outcomes: Map<string, number>;
      profiles: Set<string>;
    }
  >();
  for (const episode of episodes) {
    const benchmark = episodeBenchmark(episode) || "unknown";
    const group =
      groups.get(benchmark) ??
      {
        tasks: new Set<string>(),
        seeds: new Set<string>(),
        episodes: 0,
        outcomes: new Map<string, number>(),
        profiles: new Set<string>(),
      };
    if (episode.task_id) {
      group.tasks.add(String(episode.task_id));
    }
    const seed = episodeSeed(episode);
    if (seed) {
      group.seeds.add(seed);
    }
    const outcome = String(episode.outcome || "unknown");
    group.outcomes.set(outcome, (group.outcomes.get(outcome) ?? 0) + 1);
    const profile = episodeProfile(episode);
    if (profile) {
      group.profiles.add(profile);
    }
    group.episodes += 1;
    groups.set(benchmark, group);
  }
  return [...groups.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([benchmark, group]) => ({
      benchmark,
      tasks: sortedSetLabel(group.tasks),
      seeds: sortedSetLabel(group.seeds),
      episodes: group.episodes,
      outcomes: [...group.outcomes.entries()]
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([name, count]) => `${name}: ${count}`)
        .join(", "),
      profiles: sortedSetLabel(group.profiles),
    }));
}

export function episodeTitle(episode: DatasetEpisode): string {
  return String(episode.prompt || episode.episode_id || episode.trace_id);
}

export function episodeDatasetId(episode: DatasetEpisode): string {
  return metadataString(episode, "dataset_id");
}

export function episodeBenchmark(episode: DatasetEpisode): string {
  return metadataString(episode, "benchmark") || String(episode.env_id || "");
}

export function episodeProfile(episode: DatasetEpisode): string {
  return metadataString(episode, "capture_profile");
}

export function episodeSeed(episode: DatasetEpisode): string {
  return metadataString(episode, "seed");
}

export function metadataString(episode: DatasetEpisode, key: string): string {
  const value = episode.metadata?.[key];
  if (value === undefined || value === null) {
    return "";
  }
  return String(value);
}

export function uniqueValues(values: Array<string | null | undefined>): string[] {
  return [...new Set(values.filter((value): value is string => Boolean(value)))].sort((left, right) =>
    left.localeCompare(right),
  );
}

export function sortedSetLabel(values: Set<string>): string {
  if (!values.size) {
    return "-";
  }
  return [...values].sort((left, right) => left.localeCompare(right)).join(", ");
}

export function shortTrace(traceId: string): string {
  return traceId.length > 44 ? `${traceId.slice(0, 24)}...${traceId.slice(-12)}` : traceId;
}
