import { useCallback, useEffect, useState } from "react";
import { AppShell, type AppPage } from "../components/layout/AppShell";
import { ProbeSuitePreset } from "../components/workflows/ProbeSuitePreset";
import { EvidencePage } from "./EvidencePage";
import { EpisodesPage } from "./EpisodesPage";
import { DatasetBrowser } from "./workbench/DatasetBrowser";
import { useWorkbenchStore } from "../store/workbenchStore";
import type { InterventionLabSeed } from "../types/interventions";
import type { ResearchSelectionState } from "../types/probeEvidence";
import type { InspectionMode } from "./episodes/shared";
import type { EpisodeOpenContext } from "./workbench/types";

type EpisodeRouteState = {
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

export function WorkbenchPage() {
  const initialRoute = initialPage();
  const [activePage, setActivePage] = useState<AppPage>(initialRoute.page);
  const [evidenceRunId, setEvidenceRunId] = useState(initialRoute.evidenceRunId);
  const [interventionSeed, setInterventionSeed] = useState<InterventionLabSeed | undefined>(undefined);
  const [episodeTraceId, setEpisodeTraceId] = useState(initialRoute.traceId);
  const [episodeRouteState, setEpisodeRouteState] = useState<EpisodeRouteState>(initialRoute.episodeState);
  const {
    activeRunId,
    setActiveRunId,
  } = useWorkbenchStore();

  useEffect(() => {
    function syncRoute() {
      const route = initialPage();
      setActivePage(route.page);
      setEvidenceRunId(route.evidenceRunId);
      setEpisodeTraceId(route.traceId);
      setEpisodeRouteState(route.episodeState);
    }
    window.addEventListener("hashchange", syncRoute);
    return () => window.removeEventListener("hashchange", syncRoute);
  }, []);

  const handleEvidenceRunChange = useCallback((runId: string) => {
    setActivePage("evidence");
    setEvidenceRunId(runId);
    window.history.replaceState(null, "", buildEvidenceHash(runId));
  }, []);

  return (
    <AppShell
      activePage={activePage}
      onPageChange={handlePageChange}
    >
      {activePage === "dataset" ? (
        <DatasetBrowser onOpenEpisode={handleOpenDatasetEpisode} />
      ) : null}
      {activePage === "episode" ? (
        <EpisodesPage
          cohortReturnHref={episodeRouteState.fromCohort ? "#dataset" : undefined}
          initialFeature={episodeRouteState.feature}
          initialInspectionMode={episodeRouteState.inspectionMode}
          initialLensRunId={episodeRouteState.lensRunId}
          initialLensRankingMode={episodeRouteState.rankingMode}
          initialPolicyCall={episodeRouteState.policyCall}
          initialProbeArtifactId={episodeRouteState.probeId}
          initialResearchSelection={episodeRouteState.researchSelection}
          initialSiteName={episodeRouteState.siteName}
          initialTimestep={episodeRouteState.timestep}
          initialTraceId={episodeTraceId}
          key={episodeRouteKey(episodeRouteState)}
          onSendToIntervention={handleSendToIntervention}
          onTraceChange={handleEpisodeTraceChange}
        />
      ) : null}
      {activePage === "probes" ? (
        <ProbeSuitePreset
          activeRunId={activeRunId}
          onRunChange={setActiveRunId}
        />
      ) : null}
      {activePage === "evidence" ? (
        <EvidencePage
          interventionSeed={interventionSeed}
          selectedRunId={evidenceRunId}
          onRunChange={handleEvidenceRunChange}
        />
      ) : null}
    </AppShell>
  );

  function handlePageChange(page: AppPage) {
    setActivePage(page);
    if (page === "dataset") {
      setEpisodeTraceId("");
      setEpisodeRouteState(emptyEpisodeRouteState());
    }
    if (page === "evidence") {
      setEvidenceRunId("");
      setInterventionSeed(undefined);
    }
    window.history.replaceState(null, "", `#${page}`);
  }

  function handleOpenDatasetEpisode(traceId: string, context: EpisodeOpenContext = {}) {
    const nextRoute = {
      datasetId: context.researchSelection?.dataset_id ?? undefined,
      fromCohort: Boolean(context.fromCohort),
      feature: typeof context.feature === "number" ? context.feature : undefined,
      inspectionMode: parseInspectionMode(context.inspectionMode),
      lensRunId: context.lensRunId ?? context.researchSelection?.lens_run_id ?? undefined,
      policyCall: typeof context.policyCall === "number" ? context.policyCall : undefined,
      probeId: context.probeId ?? "",
      rankingMode: context.rankingMode ?? "",
      researchSelection: context.researchSelection,
      siteName: context.siteName ?? "",
      timestep: typeof context.researchSelection?.timestep === "number"
        ? context.researchSelection.timestep
        : undefined,
      traceId,
    };
    setActivePage("episode");
    setEpisodeTraceId(traceId);
    setEpisodeRouteState(nextRoute);
    window.history.replaceState(null, "", buildEpisodeHash(nextRoute));
  }

  function handleEpisodeTraceChange(traceId: string, context: EpisodeOpenContext = {}) {
    const nextRoute = {
      datasetId: context.researchSelection?.dataset_id ?? episodeRouteState.datasetId,
      fromCohort: context.fromCohort ?? episodeRouteState.fromCohort,
      feature: typeof context.feature === "number" ? context.feature : episodeRouteState.feature,
      inspectionMode: parseInspectionMode(context.inspectionMode) ?? episodeRouteState.inspectionMode,
      lensRunId: context.lensRunId ?? context.researchSelection?.lens_run_id ?? episodeRouteState.lensRunId,
      policyCall: typeof context.policyCall === "number" ? context.policyCall : episodeRouteState.policyCall,
      probeId: context.probeId ?? episodeRouteState.probeId,
      rankingMode: context.rankingMode ?? episodeRouteState.rankingMode,
      researchSelection: context.researchSelection ?? episodeRouteState.researchSelection,
      siteName: context.siteName ?? episodeRouteState.siteName,
      timestep: typeof context.researchSelection?.timestep === "number"
        ? context.researchSelection.timestep
        : episodeRouteState.timestep,
      traceId,
    };
    setActivePage("episode");
    setEpisodeTraceId(traceId);
    setEpisodeRouteState(nextRoute);
    window.history.replaceState(null, "", buildEpisodeHash(nextRoute));
  }

  function handleSendToIntervention(seed: InterventionLabSeed) {
    setActivePage("evidence");
    setEvidenceRunId("");
    setInterventionSeed(seed);
    window.history.replaceState(null, "", "#evidence");
  }

}

function initialPage(): {
  episodeState: EpisodeRouteState;
  evidenceRunId: string;
  page: AppPage;
  traceId: string;
} {
  const page = window.location.hash.replace("#", "");
  if (page.startsWith("episode/")) {
    const episodeState = parseEpisodeRoute(page.slice("episode/".length));
    return {
      episodeState,
      evidenceRunId: "",
      page: "episode",
      traceId: episodeState.traceId,
    };
  }
  if (page.startsWith("dataset/")) {
    const episodeState = parseEpisodeRoute(page.slice("dataset/".length));
    return {
      episodeState,
      evidenceRunId: "",
      page: "episode",
      traceId: episodeState.traceId,
    };
  }
  if (page.startsWith("evidence/")) {
    return {
      episodeState: emptyEpisodeRouteState(),
      evidenceRunId: decodeURIComponent(page.slice("evidence/".length)),
      page: "evidence",
      traceId: "",
    };
  }
  if (page === "dataset" || page === "evidence" || page === "probes") {
    return { episodeState: emptyEpisodeRouteState(), evidenceRunId: "", page, traceId: "" };
  }
  if (page === "episode" || page === "episodes") {
    return {
      episodeState: emptyEpisodeRouteState(),
      evidenceRunId: "",
      page: "episode",
      traceId: "",
    };
  }
  return { episodeState: emptyEpisodeRouteState(), evidenceRunId: "", page: "dataset", traceId: "" };
}

function emptyEpisodeRouteState(): EpisodeRouteState {
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

function parseEpisodeRoute(value: string): EpisodeRouteState {
  const queryStart = value.indexOf("?");
  const tracePart = queryStart >= 0 ? value.slice(0, queryStart) : value;
  const query = queryStart >= 0 ? value.slice(queryStart + 1) : "";
  const params = new URLSearchParams(query);
  const policyCall = numberParam(params.get("call"));
  const feature = numberParam(params.get("feature"));
  const timestep = numberParam(params.get("timestep"));
  const datasetId = params.get("dataset_id") ?? "";
  const lensRunId = params.get("lens_run_id") ?? "";
  const probeId = params.get("probe_id") ?? params.get("probe") ?? "";
  const ranking = params.get("rank") ?? "";
  const traceId = decodeURIComponent(tracePart);
  const researchSelection = probeId || lensRunId || datasetId || ranking || typeof timestep === "number" || typeof policyCall === "number"
    ? {
        dataset_id: datasetId || undefined,
        episode_id: traceId || undefined,
        lens_id: probeId || undefined,
        lens_run_id: lensRunId || undefined,
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
    siteName: params.get("site") ?? "",
    timestep,
    traceId,
  };
}

function buildEpisodeHash(route: EpisodeRouteState): string {
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

function buildEvidenceHash(runId: string): string {
  return runId ? `#evidence/${encodeURIComponent(runId)}` : "#evidence";
}

function episodeRouteKey(route: EpisodeRouteState): string {
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

function parseInspectionMode(value: string | null | undefined): InspectionMode | undefined {
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
