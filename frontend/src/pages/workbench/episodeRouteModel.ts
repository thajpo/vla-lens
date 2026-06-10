import type { ResearchSelectionState } from "../../types/probeEvidence";
import type { InspectionMode } from "../episodes/shared";

export type EpisodeRouteState = {
  datasetId?: string;
  feature?: number;
  fromCohort: boolean;
  inspectionMode?: InspectionMode;
  lensRunId?: string;
  policyCall?: number;
  probeId: string;
  rankingMode?: string;
  researchSelection?: ResearchSelectionState;
  siteName: string;
  timestep?: number;
  traceId: string;
};

export function emptyEpisodeRouteState(): EpisodeRouteState {
  return {
    fromCohort: false,
    inspectionMode: undefined,
    lensRunId: undefined,
    rankingMode: "",
    probeId: "",
    researchSelection: undefined,
    siteName: "",
    timestep: undefined,
    traceId: "",
  };
}

export function parseEpisodeRoute(value: string): EpisodeRouteState {
  const queryStart = value.indexOf("?");
  const tracePart = queryStart >= 0 ? value.slice(0, queryStart) : value;
  const query = queryStart >= 0 ? value.slice(queryStart + 1) : "";
  const params = new URLSearchParams(query);
  const policyCall = numberParam(params.get("call"));
  const feature = numberParam(params.get("feature"));
  const contributor = params.get("contributor") || "";
  const timestep = numberParam(params.get("timestep"));
  const datasetId = params.get("dataset_id") ?? "";
  const lensRunId = params.get("lens_run_id") ?? "";
  const probeId = params.get("probe_id") ?? params.get("probe") ?? "";
  const ranking = params.get("rank") ?? "";
  const siteName = params.get("site") ?? "";
  const traceId = decodeURIComponent(tracePart);
  const researchSelection = probeId ||
    lensRunId ||
    datasetId ||
    ranking ||
    siteName ||
    typeof feature === "number" ||
    contributor ||
    typeof timestep === "number" ||
    typeof policyCall === "number"
    ? {
        dataset_id: datasetId || undefined,
        episode_id: traceId || undefined,
        feature_id: contributor || (typeof feature === "number" ? `dim_${feature}` : undefined),
        lens_id: probeId || undefined,
        lens_run_id: lensRunId || undefined,
        model_locus: siteName ? { model_site_id: siteName } : undefined,
        policy_call: policyCall ?? null,
        ranking: parseRankingKind(ranking),
        timestep: timestep ?? null,
      }
    : undefined;
  return {
    datasetId: datasetId || undefined,
    feature,
    fromCohort: params.get("from") === "cohort",
    inspectionMode: parseInspectionMode(params.get("mode")),
    lensRunId: lensRunId || undefined,
    policyCall: policyCall ?? undefined,
    probeId,
    rankingMode: params.get("ranking") ?? "",
    researchSelection,
    siteName,
    timestep,
    traceId,
  };
}

export function buildEpisodeHash(route: EpisodeRouteState): string {
  const params = new URLSearchParams();
  if (route.probeId) {
    params.set("probe_id", route.probeId);
  }
  if (route.lensRunId) {
    params.set("lens_run_id", route.lensRunId);
  }
  if (route.datasetId) {
    params.set("dataset_id", route.datasetId);
  }
  if (typeof route.policyCall === "number") {
    params.set("call", String(route.policyCall));
  }
  if (typeof route.timestep === "number") {
    params.set("timestep", String(route.timestep));
  }
  if (route.researchSelection?.ranking) {
    params.set("rank", route.researchSelection.ranking);
  }
  if (typeof route.feature === "number") {
    params.set("feature", String(route.feature));
  }
  if (route.rankingMode) {
    params.set("ranking", route.rankingMode);
  }
  if (route.inspectionMode && route.inspectionMode !== "features") {
    params.set("mode", route.inspectionMode);
  }
  if (route.siteName) {
    params.set("site", route.siteName);
  }
  if (route.fromCohort) {
    params.set("from", "cohort");
  }
  const query = params.toString();
  return `#episode/${encodeURIComponent(route.traceId)}${query ? `?${query}` : ""}`;
}

export function episodeRouteKey(route: EpisodeRouteState): string {
  return [
    route.traceId,
    route.probeId,
    route.lensRunId ?? "",
    route.datasetId ?? "",
    route.policyCall ?? "",
    route.timestep ?? "",
    route.researchSelection?.ranking ?? "",
    route.feature ?? "",
    route.inspectionMode ?? "",
    route.rankingMode ?? "",
    route.siteName,
    route.fromCohort ? "cohort" : "",
  ].join("|");
}

export function parseInspectionMode(value: string | null | undefined): InspectionMode | undefined {
  if (
    value === "features" ||
    value === "attention" ||
    value === "computation" ||
    value === "saved_state" ||
    value === "advanced"
  ) {
    return value;
  }
  return undefined;
}

function numberParam(value: string | null): number | undefined {
  if (value === null || value.trim() === "") {
    return undefined;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function parseRankingKind(value: string | null | undefined): ResearchSelectionState["ranking"] {
  if (
    value === "top" ||
    value === "bottom" ||
    value === "uncertain" ||
    value === "false_positive" ||
    value === "false_negative" ||
    value === "largest_delta"
  ) {
    return value;
  }
  return undefined;
}
