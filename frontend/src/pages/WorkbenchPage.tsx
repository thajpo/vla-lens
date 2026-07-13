import { useCallback, useEffect, useState } from "react";
import { AppShell, type AppPage } from "../components/layout/AppShell";
import { ProbeSuitePreset } from "../components/workflows/ProbeSuitePreset";
import { InterventionsPage } from "./EvidencePage";
import { EpisodesPage } from "./EpisodesPage";
import { DatasetBrowser } from "./workbench/DatasetBrowser";
import { useWorkbenchStore } from "../store/workbenchStore";
import type { InterventionLabSeed } from "../types/interventions";
import type { EpisodeOpenContext } from "./workbench/types";
import {
  buildEpisodeHash,
  emptyEpisodeRouteState,
  episodeRouteKey,
  parseInspectionMode,
  type EpisodeRouteState,
} from "./workbench/episodeRouteModel";
import {
  buildInterventionsHash,
  parseWorkbenchHash,
} from "./workbench/workbenchRouteModel";

export function WorkbenchPage() {
  const initialRoute = initialPage();
  const [activePage, setActivePage] = useState<AppPage>(initialRoute.page);
  const [interventionRunId, setInterventionRunId] = useState(initialRoute.interventionRunId);
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
      setInterventionRunId(route.interventionRunId);
      setEpisodeTraceId(route.traceId);
      setEpisodeRouteState(route.episodeState);
    }
    window.addEventListener("hashchange", syncRoute);
    return () => window.removeEventListener("hashchange", syncRoute);
  }, []);

  const handleEvidenceRunChange = useCallback((runId: string) => {
    setActivePage("interventions");
    setInterventionRunId(runId);
    window.history.replaceState(null, "", buildInterventionsHash(runId));
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
      {activePage === "interventions" ? (
        <InterventionsPage
          interventionSeed={interventionSeed}
          selectedRunId={interventionRunId}
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
    if (page === "interventions") {
      setInterventionRunId("");
      setInterventionSeed(undefined);
    }
    window.history.replaceState(null, "", page === "interventions" ? buildInterventionsHash() : `#${page}`);
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
    setActivePage("interventions");
    setInterventionRunId("");
    setInterventionSeed(seed);
    window.history.replaceState(null, "", buildInterventionsHash());
  }

}

function initialPage(): {
  episodeState: EpisodeRouteState;
  interventionRunId: string;
  page: AppPage;
  traceId: string;
} {
  return parseWorkbenchHash(window.location.hash);
}
