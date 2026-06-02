import type { DatasetEpisode, ProbeDatasetIndex, ProbeEpisodeIndex } from "../../types/dataset";
import type { EpisodeOpenContext } from "./types";

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

export const PROBE_LIST_LIMIT = 80;
export const EVIDENCE_EPISODE_LIMIT = 12;
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
  { id: "heldout_wrong", label: "Heldout wrong" },
  { id: "confident_wrong", label: "High-conf wrong" },
  { id: "heldout_scored", label: "Heldout scored" },
  { id: "train_sanity", label: "Train sanity" },
];

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
    if (record.correct === false) {
      stats.wrong += 1;
      if (heldout) {
        stats.heldoutWrong += 1;
      }
      const confidence = probeConfidenceValue(record);
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

export function probeSplitChartRows(probe: ProbeDatasetIndex): Array<{
  highConfWrong: number;
  id: "test" | "train" | "validation";
  label: string;
  scored: number;
  total: number;
  wrong: number;
}> {
  const rows = new Map([
    ["train", { highConfWrong: 0, id: "train" as const, label: "Train", scored: 0, total: 0, wrong: 0 }],
    [
      "validation",
      { highConfWrong: 0, id: "validation" as const, label: "Validation", scored: 0, total: 0, wrong: 0 },
    ],
    ["test", { highConfWrong: 0, id: "test" as const, label: "Test", scored: 0, total: 0, wrong: 0 }],
  ]);
  const records = Object.values(probe.by_trace ?? {});
  if (!records.length) {
    for (const row of rows.values()) {
      const total = Number(probe.split_summary[row.id] ?? 0);
      row.total = Number.isFinite(total) ? total : 0;
      row.scored = row.total;
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
      row.wrong += 1;
      if ((probeConfidenceValue(record) ?? 0) >= 0.8) {
        row.highConfWrong += 1;
      }
    }
  }
  return [...rows.values()];
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
      value: countProbeRecords(probe, (record) => (probeConfidenceValue(record) ?? 0) >= 0.8),
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
    reasons.push(`${stats.heldoutWrong} heldout wrong`);
  }
  if (stats.heldoutScored) {
    reasons.push(`${stats.heldoutScored} heldout scored`);
  }
  if (!stats.heldoutScored && stats.train) {
    reasons.push("train-heavy sanity check");
  }
  if (typeof probe.best_delta === "number" && Number.isFinite(probe.best_delta)) {
    reasons.push(`delta ${formatSignedNumber(probe.best_delta)}`);
  }
  return reasons.slice(0, 3).length ? reasons.slice(0, 3) : ["no scored cohort yet"];
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
    return "not indexed";
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
    return "heldout evidence";
  }
  if (stats.train > 0) {
    return "train sanity";
  }
  return "coverage missing";
}

export function probeTrustDetail(stats: ProbeReviewStats | undefined): string {
  if (!stats) {
    return "No probe is selected";
  }
  if (stats.heldoutScored > 0) {
    return `${stats.validation} validation / ${stats.test} test episodes, ${stats.heldoutScored} scored`;
  }
  if (stats.train > 0) {
    return `${stats.train} train episodes; useful mainly for debugging the probe`;
  }
  return "No split metadata or scored rows were returned";
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
    target_moved: "Did the target object move?",
  };
  if (target && labels[target]) {
    return labels[target];
  }
  return target ? labelFromSnake(target) : "Probe readout";
}

export function labelFromSnake(value: string): string {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
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
