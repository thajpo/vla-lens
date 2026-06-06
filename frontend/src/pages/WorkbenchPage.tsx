import { useCallback, useEffect, useState } from "react";
import { AppShell, type AppPage } from "../components/layout/AppShell";
import { ProbeSuitePreset } from "../components/workflows/ProbeSuitePreset";
import { EvidencePage } from "./EvidencePage";
import { EpisodesPage } from "./EpisodesPage";
import { DatasetBrowser } from "./workbench/DatasetBrowser";
import { useWorkbenchStore } from "../store/workbenchStore";
import type { InterventionLabSeed } from "../types/interventions";
import type { EpisodeOpenContext } from "./workbench/types";

type EpisodeRouteState = {
  fromCohort: boolean;
  policyCall?: number;
  probeId: string;
  siteName: string;
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
          initialPolicyCall={episodeRouteState.policyCall}
          initialProbeArtifactId={episodeRouteState.probeId}
          initialSiteName={episodeRouteState.siteName}
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
      fromCohort: Boolean(context.fromCohort),
      policyCall: typeof context.policyCall === "number" ? context.policyCall : undefined,
      probeId: context.probeId ?? "",
      siteName: context.siteName ?? "",
      traceId,
    };
    setActivePage("episode");
    setEpisodeTraceId(traceId);
    setEpisodeRouteState(nextRoute);
    window.history.replaceState(null, "", buildEpisodeHash(nextRoute));
  }

  function handleEpisodeTraceChange(traceId: string, context: EpisodeOpenContext = {}) {
    const nextRoute = {
      fromCohort: Boolean(context.fromCohort),
      policyCall: typeof context.policyCall === "number" ? context.policyCall : undefined,
      probeId: context.probeId ?? "",
      siteName: context.siteName ?? "",
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
    probeId: "",
    siteName: "",
    traceId: "",
  };
}

function parseEpisodeRoute(value: string): EpisodeRouteState {
  const queryStart = value.indexOf("?");
  const tracePart = queryStart >= 0 ? value.slice(0, queryStart) : value;
  const query = queryStart >= 0 ? value.slice(queryStart + 1) : "";
  const params = new URLSearchParams(query);
  const policyCall = numberParam(params.get("call"));
  return {
    fromCohort: params.get("from") === "cohort",
    policyCall: policyCall ?? undefined,
    probeId: params.get("probe_id") ?? params.get("probe") ?? "",
    siteName: params.get("site") ?? "",
    traceId: decodeURIComponent(tracePart),
  };
}

function buildEpisodeHash(route: EpisodeRouteState): string {
  const params = new URLSearchParams();
  if (route.probeId) {
    params.set("probe_id", route.probeId);
  }
  if (typeof route.policyCall === "number") {
    params.set("call", String(route.policyCall));
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
    route.policyCall ?? "",
    route.siteName,
    route.fromCohort ? "cohort" : "",
  ].join("|");
}

function numberParam(value: string | null): number | undefined {
  if (value === null || value.trim() === "") {
    return undefined;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}
