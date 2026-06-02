import { useEffect, useState } from "react";
import { AppShell, type AppPage } from "../components/layout/AppShell";
import { ProbeSuitePreset } from "../components/workflows/ProbeSuitePreset";
import { EpisodesPage } from "./EpisodesPage";
import { DatasetBrowser } from "./workbench/DatasetBrowser";
import { useWorkbenchStore } from "../store/workbenchStore";
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
      setEpisodeTraceId(route.traceId);
      setEpisodeRouteState(route.episodeState);
    }
    window.addEventListener("hashchange", syncRoute);
    return () => window.removeEventListener("hashchange", syncRoute);
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
          onTraceChange={handleEpisodeTraceChange}
        />
      ) : null}
      {activePage === "probes" ? (
        <ProbeSuitePreset
          activeRunId={activeRunId}
          onRunChange={setActiveRunId}
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

}

function initialPage(): { episodeState: EpisodeRouteState; page: AppPage; traceId: string } {
  const page = window.location.hash.replace("#", "");
  if (page.startsWith("episode/")) {
    const episodeState = parseEpisodeRoute(page.slice("episode/".length));
    return { episodeState, page: "episode", traceId: episodeState.traceId };
  }
  if (page.startsWith("dataset/")) {
    const episodeState = parseEpisodeRoute(page.slice("dataset/".length));
    return { episodeState, page: "episode", traceId: episodeState.traceId };
  }
  if (page === "dataset" || page === "probes") {
    return { episodeState: emptyEpisodeRouteState(), page, traceId: "" };
  }
  if (page === "episode" || page === "episodes") {
    return { episodeState: emptyEpisodeRouteState(), page: "episode", traceId: "" };
  }
  return { episodeState: emptyEpisodeRouteState(), page: "dataset", traceId: "" };
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
