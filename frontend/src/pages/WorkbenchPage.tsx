import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchWorkbench } from "../api/workbench";
import { AppShell, type AppPage } from "../components/layout/AppShell";
import { LeftRail } from "../components/layout/LeftRail";
import { ProbeSuitePreset } from "../components/workflows/ProbeSuitePreset";
import { TargetObjectEncodingPreset } from "../components/workflows/TargetObjectEncodingPreset";
import { EpisodesPage } from "./EpisodesPage";
import { ArtifactsSummary, BackendUnavailable, DatasetBrowser } from "./workbench/DatasetBrowser";
import { useWorkbenchStore } from "../store/workbenchStore";
import type { WorkbenchManifest } from "../types/workbench";
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
  const needsWorkbench = activePage === "workbench" || activePage === "artifacts";
  const workbench = useQuery({
    queryKey: ["workbench"],
    queryFn: fetchWorkbench,
    enabled: needsWorkbench,
    staleTime: 60_000,
  });
  const {
    activeWorkflowId,
    activeRunId,
    activeMetric,
    activeTokenKind,
    setActiveWorkflowId,
    setActiveRunId,
    setActiveMetric,
    setActiveTokenKind,
  } = useWorkbenchStore();

  const manifest = workbench.data;
  const workflows = manifest?.workflow_presets ?? [];
  const runs = manifest?.analysis_runs.filter((run) => run.workflow === activeWorkflowId) ?? [];
  const selectedArray = manifest?.lens_arrays.find(
    (array) => array.array_id === `artifact.${activeRunId || runs[0]?.run_id}.${activeMetric}`,
  );
  const tokenCoords = selectedArray?.coords.token_kind;
  const tokenKinds = Array.isArray(tokenCoords) ? tokenCoords.map(String) : [];

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
      {needsWorkbench && workbench.isLoading ? (
        <div className="app-message">Loading workbench...</div>
      ) : null}
      {needsWorkbench && workbench.isError ? (
        <BackendUnavailable />
      ) : null}
      {needsWorkbench && manifest ? (
        <ActivePage
          page={activePage}
          manifest={manifest}
          workflows={workflows}
          runs={runs}
          activeWorkflowId={activeWorkflowId}
          activeRunId={activeRunId || runs[0]?.run_id || ""}
          activeMetric={activeMetric}
          activeTokenKind={activeTokenKind || tokenKinds[0] || ""}
          tokenKinds={tokenKinds}
          onWorkflowChange={setActiveWorkflowId}
          onRunChange={setActiveRunId}
          onMetricChange={setActiveMetric}
          onTokenKindChange={setActiveTokenKind}
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
  if (page === "dataset" || page === "probes" || page === "workbench" || page === "artifacts") {
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
    probeId: params.get("probe") ?? "",
    siteName: params.get("site") ?? "",
    traceId: decodeURIComponent(tracePart),
  };
}

function buildEpisodeHash(route: EpisodeRouteState): string {
  const params = new URLSearchParams();
  if (route.probeId) {
    params.set("probe", route.probeId);
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
  return `#dataset/${encodeURIComponent(route.traceId)}${query ? `?${query}` : ""}`;
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

type ActivePageProps = {
  page: AppPage;
  manifest: WorkbenchManifest;
  workflows: WorkbenchManifest["workflow_presets"];
  runs: WorkbenchManifest["analysis_runs"];
  activeWorkflowId: string;
  activeRunId: string;
  activeMetric: string;
  activeTokenKind: string;
  tokenKinds: string[];
  onWorkflowChange: (workflowId: string) => void;
  onRunChange: (runId: string) => void;
  onMetricChange: (metric: string) => void;
  onTokenKindChange: (tokenKind: string) => void;
};

function ActivePage({
  page,
  manifest,
  workflows,
  runs,
  activeWorkflowId,
  activeRunId,
  activeMetric,
  activeTokenKind,
  tokenKinds,
  onWorkflowChange,
  onRunChange,
  onMetricChange,
  onTokenKindChange,
}: ActivePageProps) {
  if (page === "episode") {
    return null;
  }
  if (page === "artifacts") {
    return <ArtifactsSummary manifest={manifest} />;
  }
  return (
    <div className="workbench-shell">
      <LeftRail
        workflows={workflows}
        runs={runs}
        activeWorkflowId={activeWorkflowId}
        activeRunId={activeRunId}
        activeMetric={activeMetric}
        activeTokenKind={activeTokenKind}
        tokenKinds={tokenKinds}
        onWorkflowChange={onWorkflowChange}
        onRunChange={onRunChange}
        onMetricChange={onMetricChange}
        onTokenKindChange={onTokenKindChange}
      />
      {activeWorkflowId === "target_object_encoding" ? (
        <TargetObjectEncodingPreset manifest={manifest} />
      ) : activeWorkflowId === "probe_suite" ? (
        <ProbeSuitePreset
          activeRunId={activeRunId}
          onRunChange={onRunChange}
        />
      ) : (
        <div className="workflow-empty">
          <h1>{activeWorkflowId}</h1>
          <p>This workflow preset is not implemented in the React workbench yet.</p>
        </div>
      )}
    </div>
  );
}
