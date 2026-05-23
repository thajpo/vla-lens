import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ArrowRight, Search, Target } from "lucide-react";
import { fetchDataset, fetchDatasetDiagnostics, fetchProbeIndex } from "../api/dataset";
import { fetchWorkbench } from "../api/workbench";
import { AppShell, type AppPage } from "../components/layout/AppShell";
import { LeftRail } from "../components/layout/LeftRail";
import { ProbeSuitePreset } from "../components/workflows/ProbeSuitePreset";
import { TargetObjectEncodingPreset } from "../components/workflows/TargetObjectEncodingPreset";
import { EpisodesPage } from "./EpisodesPage";
import { useWorkbenchStore } from "../store/workbenchStore";
import type { DatasetEpisode, ProbeDatasetIndex, ProbeEpisodeIndex } from "../types/dataset";
import type { WorkbenchManifest } from "../types/workbench";

type EpisodeOpenContext = {
  fromCohort?: boolean;
  policyCall?: number | null;
  probeId?: string;
  siteName?: string;
};

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

  function handleEpisodeTraceChange(traceId: string) {
    const nextRoute = { ...emptyEpisodeRouteState(), traceId };
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

function DatasetBrowser({
  onOpenEpisode,
}: {
  onOpenEpisode: (traceId: string, context?: EpisodeOpenContext) => void;
}) {
  const dataset = useQuery({
    queryKey: ["dataset"],
    queryFn: fetchDataset,
    staleTime: 30_000,
  });
  const diagnostics = useQuery({
    queryKey: ["dataset-diagnostics"],
    queryFn: fetchDatasetDiagnostics,
  });
  const probeIndex = useQuery({
    queryKey: ["probe-index"],
    queryFn: fetchProbeIndex,
    staleTime: 60_000,
  });
  const episodes = useMemo(() => dataset.data?.episodes ?? [], [dataset.data?.episodes]);
  const probes = useMemo(() => probeIndex.data?.probes ?? [], [probeIndex.data?.probes]);
  const [query, setQuery] = useState("");
  const [datasetFilter, setDatasetFilter] = useState("all");
  const [benchmarkFilter, setBenchmarkFilter] = useState("all");
  const [taskFilter, setTaskFilter] = useState("all");
  const [outcomeFilter, setOutcomeFilter] = useState("all");
  const [profileFilter, setProfileFilter] = useState("all");
  const [probeFilter, setProbeFilter] = useState("all");
  const [probeQuery, setProbeQuery] = useState("");
  const [probeCohortPreset, setProbeCohortPreset] = useState<ProbeCohortPreset>("all");
  const [probeSplitFilter, setProbeSplitFilter] = useState("all");
  const [probePredictionFilter, setProbePredictionFilter] = useState("all");
  const selectedProbe = useMemo(
    () => probes.find((probe) => probe.artifact_id === probeFilter),
    [probeFilter, probes],
  );
  const rankedProbes = useMemo(() => rankProbesForReview(probes), [probes]);
  const visibleProbeChoices = useMemo(
    () => filterProbeChoices(rankedProbes, probeQuery).slice(0, PROBE_LIST_LIMIT),
    [probeQuery, rankedProbes],
  );

  const datasetIds = useMemo(() => uniqueValues(episodes.map(episodeDatasetId)), [episodes]);
  const benchmarks = useMemo(() => uniqueValues(episodes.map(episodeBenchmark)), [episodes]);
  const tasks = useMemo(() => uniqueValues(episodes.map((episode) => episode.task_id)), [episodes]);
  const outcomes = useMemo(
    () => uniqueValues(episodes.map((episode) => episode.outcome)),
    [episodes],
  );
  const profiles = useMemo(() => uniqueValues(episodes.map(episodeProfile)), [episodes]);
  const filteredEpisodes = useMemo(
    () =>
      episodes.filter((episode) => {
        const searchText = [
          episode.trace_id,
          episode.episode_id,
          episode.task_id,
          episode.prompt,
          episode.outcome,
          episodeDatasetId(episode),
          episodeBenchmark(episode),
          episodeProfile(episode),
          episodeSeed(episode),
        ]
          .filter(Boolean)
          .join(" ")
          .toLowerCase();
        const needle = query.trim().toLowerCase();
        return (
          (datasetFilter === "all" || episodeDatasetId(episode) === datasetFilter) &&
          (benchmarkFilter === "all" || episodeBenchmark(episode) === benchmarkFilter) &&
          (taskFilter === "all" || episode.task_id === taskFilter) &&
          (outcomeFilter === "all" || episode.outcome === outcomeFilter) &&
          (profileFilter === "all" || episodeProfile(episode) === profileFilter) &&
          matchesProbeFilters(
            episode,
            selectedProbe,
            probeCohortPreset,
            probeSplitFilter,
            probePredictionFilter,
          ) &&
          (!needle || searchText.includes(needle))
        );
      }),
    [
      datasetFilter,
      benchmarkFilter,
      episodes,
      outcomeFilter,
      probeCohortPreset,
      probePredictionFilter,
      probeSplitFilter,
      profileFilter,
      query,
      selectedProbe,
      taskFilter,
    ],
  );
  const coverageRows = useMemo(() => datasetCoverageRows(filteredEpisodes), [filteredEpisodes]);
  const probeVisibleRows = useMemo(
    () => (selectedProbe ? probeCoverageRows(filteredEpisodes, selectedProbe) : []),
    [filteredEpisodes, selectedProbe],
  );
  const rankedEpisodes = useMemo(
    () => (selectedProbe ? rankEpisodesForProbe(filteredEpisodes, selectedProbe) : filteredEpisodes),
    [filteredEpisodes, selectedProbe],
  );
  const evidenceEpisodes = useMemo(
    () => (selectedProbe ? rankEpisodesForProbe(episodes, selectedProbe).slice(0, EVIDENCE_EPISODE_LIMIT) : []),
    [episodes, selectedProbe],
  );

  return (
    <main className="dataset-browser-page">
      <header className="dataset-browser-header">
        <div>
          <h1>Dataset</h1>
          <p>{dataset.data?.root ?? "Dataset"}</p>
        </div>
        <div className={diagnostics.data?.stale ? "diagnostic-chip stale" : "diagnostic-chip"}>
          {diagnostics.data?.stale ? "Diagnostics stale" : "Diagnostics current"}
        </div>
      </header>

      <div className="dataset-browser-metrics">
        <Metric label="Episodes" value={episodes.length} />
        <Metric label="Visible" value={filteredEpisodes.length} />
        <Metric label="Datasets" value={datasetIds.length} />
        <Metric label="Benchmarks" value={benchmarks.length} />
        <Metric label="Tasks" value={tasks.length} />
        <Metric
          label="Model Sites"
          value={dataset.data?.activation_sites ?? 0}
        />
        <Metric label="Probes" value={probes.length} />
      </div>

      <section className="dataset-browser-controls" aria-label="Dataset filters">
        <div className="dataset-search">
          <Search size={15} />
          <input
            aria-label="Search dataset episodes"
            placeholder="Search trace, task, dataset, prompt"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
        </div>
        <FilterSelect
          label="Dataset"
          value={datasetFilter}
          values={datasetIds}
          onChange={setDatasetFilter}
        />
        <FilterSelect
          label="Benchmark"
          value={benchmarkFilter}
          values={benchmarks}
          onChange={setBenchmarkFilter}
        />
        <FilterSelect label="Task" value={taskFilter} values={tasks} onChange={setTaskFilter} />
        <FilterSelect
          label="Outcome"
          value={outcomeFilter}
          values={outcomes}
          onChange={setOutcomeFilter}
        />
        <FilterSelect
          label="Profile"
          value={profileFilter}
          values={profiles}
          onChange={setProfileFilter}
        />
      </section>

      {dataset.isLoading ? <div className="app-message">Loading dataset...</div> : null}
      {dataset.isError ? <div className="empty-state">Dataset API unavailable.</div> : null}
      {probeIndex.isError ? <div className="empty-state compact">Probe index unavailable.</div> : null}

      <section className="dataset-probe-workbench" aria-label="Probe evidence and cohort mining">
        <div className="probe-workbench-toolbar">
          <div className="dataset-search probe-search">
            <Search size={15} />
            <input
              aria-label="Search probes"
              placeholder="Search probes, targets, sites"
              value={probeQuery}
              onChange={(event) => setProbeQuery(event.target.value)}
            />
          </div>
          <FilterSelect
            disabled={!selectedProbe}
            label="Episode Split"
            value={probeSplitFilter}
            values={PROBE_SPLIT_FILTERS}
            labels={PROBE_SPLIT_FILTER_LABELS}
            onChange={(value) => {
              setProbeCohortPreset("all");
              setProbeSplitFilter(value);
            }}
          />
          <FilterSelect
            disabled={!selectedProbe}
            label="Probe Result"
            value={probePredictionFilter}
            values={PROBE_PREDICTION_FILTERS}
            labels={PROBE_PREDICTION_FILTER_LABELS}
            onChange={(value) => {
              setProbeCohortPreset("all");
              setProbePredictionFilter(value);
            }}
          />
          <button
            className="text-command"
            disabled={!selectedProbe}
            type="button"
            onClick={() => {
              setProbeFilter("all");
              setProbeCohortPreset("all");
              setProbeSplitFilter("all");
              setProbePredictionFilter("all");
            }}
          >
            Clear probe
          </button>
        </div>
        <div className="probe-workbench-grid">
          <RankedProbeList
            probes={visibleProbeChoices}
            selectedProbeId={selectedProbe?.artifact_id ?? ""}
            total={rankedProbes.length}
            onProbeSelect={(artifactId) => {
              setProbeFilter(artifactId);
              setProbeCohortPreset("all");
              setProbeSplitFilter("all");
              setProbePredictionFilter("all");
            }}
          />
          <ProbeEvidencePanel
            activePredictionFilter={probePredictionFilter}
            activeSplitFilter={probeSplitFilter}
            episodes={evidenceEpisodes}
            probe={selectedProbe}
            summaryRows={probeVisibleRows}
            cohortPreset={probeCohortPreset}
            onCohortPresetChange={(preset) => {
              setProbeCohortPreset(preset);
              setProbeSplitFilter("all");
              setProbePredictionFilter("all");
            }}
            onOpenEpisode={onOpenEpisode}
            onPredictionFilterChange={(value) => {
              setProbeCohortPreset("all");
              setProbePredictionFilter(value);
            }}
            onSplitFilterChange={(value) => {
              setProbeCohortPreset("all");
              setProbeSplitFilter(value);
            }}
          />
        </div>
      </section>

      <section className="dataset-browser-grid">
        <div className="dataset-browser-panel">
          <header>
            <h2>Episodes</h2>
            <span>{filteredEpisodes.length}</span>
          </header>
          <div className="dataset-episode-table-wrap">
            <table className="compact-table dataset-episode-table">
              <thead>
                <tr>
                  <th>Episode</th>
                  <th>Dataset</th>
                  <th>Benchmark</th>
                  <th>Task</th>
                  <th>Seed</th>
                  <th>Outcome</th>
                  <th>Steps</th>
                  {selectedProbe ? <th>Probe</th> : null}
                  <th />
                </tr>
              </thead>
              <tbody>
                {rankedEpisodes.map((episode) => (
                  <tr key={episode.trace_id}>
                    <td>
                      <button
                        className="dataset-episode-link"
                        type="button"
                        onClick={() => onOpenEpisode(episode.trace_id, episodeOpenContextForProbe(selectedProbe, episode))}
                      >
                        <span>{episodeTitle(episode)}</span>
                        <small>{shortTrace(episode.trace_id)}</small>
                      </button>
                    </td>
                    <td>{episodeDatasetId(episode) || "-"}</td>
                    <td>{episodeBenchmark(episode) || "-"}</td>
                    <td>{episode.task_id ?? "-"}</td>
                    <td>{episodeSeed(episode) || "-"}</td>
                    <td>
                      <span className={`outcome-pill ${episode.outcome ?? "unknown"}`}>
                        {episode.outcome ?? "unknown"}
                      </span>
                    </td>
                    <td>{episode.length ?? "-"}</td>
                    {selectedProbe ? (
                      <td>
                        <ProbeEpisodeBadge record={probeRecordForEpisode(selectedProbe, episode)} />
                      </td>
                    ) : null}
                    <td>
                      <button
                        className="icon-command"
                        type="button"
                        title="Open episode"
                        aria-label={`Open ${episode.trace_id}`}
                        onClick={() => onOpenEpisode(episode.trace_id, episodeOpenContextForProbe(selectedProbe, episode))}
                      >
                        <ArrowRight size={16} />
                      </button>
                    </td>
                  </tr>
                ))}
                {!filteredEpisodes.length ? (
                  <tr>
                    <td colSpan={selectedProbe ? 9 : 8}>No episodes match the current filters.</td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
        </div>

        <div className="dataset-browser-panel">
          <header>
            <h2>Coverage</h2>
            <span>{coverageRows.length}</span>
          </header>
          <table className="compact-table">
            <thead>
              <tr>
                <th>Benchmark</th>
                <th>Tasks</th>
                <th>Seeds</th>
                <th>Episodes</th>
                <th>Outcomes</th>
                <th>Profiles</th>
              </tr>
            </thead>
            <tbody>
              {coverageRows.map((row) => (
                <tr key={row.benchmark}>
                  <td>{row.benchmark}</td>
                  <td>{row.tasks}</td>
                  <td>{row.seeds}</td>
                  <td>{row.episodes}</td>
                  <td>{row.outcomes}</td>
                  <td>{row.profiles}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}

function FilterSelect({
  disabled = false,
  labels,
  label,
  value,
  values,
  onChange,
}: {
  disabled?: boolean;
  labels?: Record<string, string>;
  label: string;
  value: string;
  values: readonly string[];
  onChange: (value: string) => void;
}) {
  return (
    <label className="dataset-filter">
      {label}
      <select disabled={disabled} value={value} onChange={(event) => onChange(event.target.value)}>
        <option value="all">All</option>
        {values.map((item) => (
          <option key={item} value={item}>
            {labels?.[item] ?? item}
          </option>
        ))}
      </select>
    </label>
  );
}

function RankedProbeList({
  probes,
  selectedProbeId,
  total,
  onProbeSelect,
}: {
  probes: ProbeRankRecord[];
  selectedProbeId: string;
  total: number;
  onProbeSelect: (artifactId: string) => void;
}) {
  return (
    <div className="ranked-probe-panel">
      <header>
        <div>
          <span>Ranked probes</span>
          <strong>{probes.length}/{total}</strong>
        </div>
        <small>least useful: train-only sanity checks</small>
      </header>
      <div className="ranked-probe-list" aria-label="Ranked probes">
        {probes.map(({ probe, reasons, stats }) => (
          <button
            className={probe.artifact_id === selectedProbeId ? "active" : ""}
            key={probe.artifact_id}
            type="button"
            onClick={() => onProbeSelect(probe.artifact_id)}
          >
            <span>{probeQuestionLabel(probe)}</span>
            <strong>{probe.name}</strong>
            <small>
              {reasons.join(" · ")} · {stats.scored} scored / {stats.heldoutScored} heldout
            </small>
          </button>
        ))}
        {!probes.length ? <div className="ranked-probe-empty">No probes match the search.</div> : null}
      </div>
    </div>
  );
}

function ProbeEvidencePanel({
  activePredictionFilter,
  activeSplitFilter,
  cohortPreset,
  episodes,
  probe,
  summaryRows,
  onCohortPresetChange,
  onOpenEpisode,
  onPredictionFilterChange,
  onSplitFilterChange,
}: {
  activePredictionFilter: string;
  activeSplitFilter: string;
  cohortPreset: ProbeCohortPreset;
  episodes: DatasetEpisode[];
  probe?: ProbeDatasetIndex;
  summaryRows: Array<{ label: string; value: number | string }>;
  onCohortPresetChange: (preset: ProbeCohortPreset) => void;
  onOpenEpisode: (traceId: string, context?: EpisodeOpenContext) => void;
  onPredictionFilterChange: (value: string) => void;
  onSplitFilterChange: (value: string) => void;
}) {
  const stats = probe ? probeReviewStats(probe) : undefined;
  return (
    <div className="probe-evidence-panel">
      {probe ? (
        <>
          <header>
            <div>
              <span>Probe evidence</span>
              <strong>{probe.name}</strong>
              <small>{probeQuestionLabel(probe)}</small>
            </div>
            <div className="probe-evidence-site">
              <Target size={14} />
              <span>{probe.best_feature || "site missing"}</span>
            </div>
          </header>
          <div className="probe-evidence-summary">
            <ProbeEvidenceFact label="Trust" value={probeTrustLabel(stats)} detail={probeTrustDetail(stats)} />
            <ProbeEvidenceFact
              label="Heldout review"
              value={`${stats?.heldoutWrong ?? 0} wrong`}
              detail={`${stats?.heldoutScored ?? 0} validation/test scored`}
            />
            <ProbeEvidenceFact
              label="Model site"
              value={probe.best_model || "model missing"}
              detail={probe.best_feature || "No mapped feature was returned"}
            />
            <ProbeEvidenceFact
              label="Visible cohort"
              value={String(summaryRows.find((row) => row.label === "visible scored")?.value ?? 0)}
              detail={String(summaryRows.find((row) => row.label === "visible splits")?.value ?? "-")}
            />
          </div>
          <ProbeEvidencePlot
            activePredictionFilter={activePredictionFilter}
            activeSplitFilter={activeSplitFilter}
            probe={probe}
            onCohortPresetChange={onCohortPresetChange}
            onPredictionFilterChange={onPredictionFilterChange}
            onSplitFilterChange={onSplitFilterChange}
          />
          <div className="probe-cohort-presets" aria-label="Probe cohort presets">
            {COHORT_PRESETS.map((preset) => (
              <button
                className={cohortPreset === preset.id ? "active" : ""}
                key={preset.id}
                type="button"
                onClick={() => onCohortPresetChange(preset.id)}
              >
                {preset.label}
              </button>
            ))}
          </div>
          <div className="probe-evidence-episodes">
            <header>
              <span>Episodes to inspect</span>
              <small>ranked by split, errors, and confidence</small>
            </header>
            <div className="probe-evidence-episode-list">
              {episodes.map((episode) => {
                const record = probe.by_trace[episode.trace_id];
                return (
                  <button
                    key={episode.trace_id}
                    type="button"
                    onClick={() => onOpenEpisode(episode.trace_id, episodeOpenContextForProbe(probe, episode))}
                  >
                    <span>{probeEpisodeInterestLabel(record)}</span>
                    <strong>{episodeTitle(episode)}</strong>
                    <small>
                      {probeSplitLabel(record?.split_category, record?.split)} · {probeResultLabel(record)} ·{" "}
                      {episodeBenchmark(episode) || "benchmark missing"} / task {episode.task_id ?? "-"}
                    </small>
                  </button>
                );
              })}
              {!episodes.length ? <div className="ranked-probe-empty">No episodes available for this probe.</div> : null}
            </div>
          </div>
        </>
      ) : (
        <div className="probe-evidence-empty">
          <span>Probe evidence</span>
          <strong>Select a ranked probe</strong>
          <p>
            Pick a probe to mine cohorts by split, correctness, confidence, and episode outcome.
          </p>
        </div>
      )}
    </div>
  );
}

function ProbeEvidenceFact({
  detail,
  label,
  value,
}: {
  detail: string;
  label: string;
  value: string;
}) {
  return (
    <div className="probe-evidence-fact">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  );
}

function ProbeEvidencePlot({
  activePredictionFilter,
  activeSplitFilter,
  probe,
  onCohortPresetChange,
  onPredictionFilterChange,
  onSplitFilterChange,
}: {
  activePredictionFilter: string;
  activeSplitFilter: string;
  probe: ProbeDatasetIndex;
  onCohortPresetChange: (preset: ProbeCohortPreset) => void;
  onPredictionFilterChange: (value: string) => void;
  onSplitFilterChange: (value: string) => void;
}) {
  const splitRows = probeSplitChartRows(probe);
  const resultRows = probeResultChartRows(probe);
  return (
    <div className="probe-evidence-plot" aria-label="Probe evidence plot">
      <div className="probe-evidence-plot-column">
        <header>
          <span>Split map</span>
          <small>click to filter episodes</small>
        </header>
        <div className="probe-split-bars">
          {splitRows.map((row) => (
            <button
              className={activeSplitFilter === row.id ? "active" : ""}
              key={row.id}
              type="button"
              onClick={() => onSplitFilterChange(activeSplitFilter === row.id ? "all" : row.id)}
            >
              <span>{row.label}</span>
              <strong>{row.scored}/{row.total}</strong>
              <i className="probe-evidence-bar">
                <b style={{ width: `${percentOf(row.scored, row.total)}%` }} />
                <em style={{ width: `${percentOf(row.wrong, row.total)}%` }} />
              </i>
              <small>{row.wrong} wrong · {row.highConfWrong} high-conf wrong</small>
            </button>
          ))}
        </div>
      </div>
      <div className="probe-evidence-plot-column">
        <header>
          <span>Result map</span>
          <small>click to choose a cohort</small>
        </header>
        <div className="probe-result-bars">
          {resultRows.map((row) => (
            <button
              className={row.active(activePredictionFilter) ? "active" : ""}
              key={row.id}
              type="button"
              onClick={() => row.apply(onPredictionFilterChange, onCohortPresetChange)}
            >
              <span>{row.label}</span>
              <strong>{row.value}</strong>
              <i className="probe-evidence-bar single">
                <b style={{ width: `${percentOf(row.value, row.total)}%` }} />
              </i>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function ArtifactsSummary({ manifest }: { manifest: WorkbenchManifest }) {
  return (
    <main className="summary-page">
      <h1>Artifacts</h1>
      <table className="compact-table">
        <thead>
          <tr>
            <th>Run</th>
            <th>Workflow</th>
            <th>Outputs</th>
          </tr>
        </thead>
        <tbody>
          {manifest.analysis_runs.map((run) => (
            <tr key={run.run_id}>
              <td>{run.run_id}</td>
              <td>{run.workflow}</td>
              <td>{run.outputs.join(", ")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </main>
  );
}

function BackendUnavailable() {
  return (
    <main className="summary-page">
      <h1>Backend unavailable</h1>
      <p>
        The React dev server is running, but the VLA-lens Python backend is not listening on
        127.0.0.1:8765.
      </p>
      <pre>uv run python scripts/serve_vla_lens_dashboard.py runs/pi05_real_20_vlatraces --port 8765</pre>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div className="summary-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

type CoverageRow = {
  benchmark: string;
  tasks: string;
  seeds: string;
  episodes: number;
  outcomes: string;
  profiles: string;
};

type ProbeCohortPreset =
  | "all"
  | "needs_review"
  | "heldout_wrong"
  | "confident_wrong"
  | "heldout_scored"
  | "train_sanity";
type ProbeReviewStats = {
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
type ProbeRankRecord = {
  probe: ProbeDatasetIndex;
  reasons: string[];
  score: number;
  stats: ProbeReviewStats;
};

const PROBE_LIST_LIMIT = 80;
const EVIDENCE_EPISODE_LIMIT = 12;
const PROBE_SPLIT_FILTERS = ["train", "validation", "test"] as const;
const PROBE_SPLIT_FILTER_LABELS: Record<string, string> = {
  test: "Test",
  train: "Train",
  validation: "Validation",
};
const PROBE_PREDICTION_FILTERS = [
  "scored",
  "unscored",
  "correct",
  "incorrect",
  "high_confidence",
  "low_confidence",
] as const;
const PROBE_PREDICTION_FILTER_LABELS: Record<string, string> = {
  correct: "Correct",
  high_confidence: "High conf.",
  incorrect: "Incorrect",
  low_confidence: "Low conf.",
  scored: "Scored",
  unscored: "Unscored",
};
const COHORT_PRESETS: Array<{ id: ProbeCohortPreset; label: string }> = [
  { id: "all", label: "All" },
  { id: "needs_review", label: "Needs review" },
  { id: "heldout_wrong", label: "Heldout wrong" },
  { id: "confident_wrong", label: "High-conf wrong" },
  { id: "heldout_scored", label: "Heldout scored" },
  { id: "train_sanity", label: "Train sanity" },
];

function ProbeEpisodeBadge({ record }: { record?: ProbeEpisodeIndex }) {
  if (!record) {
    return <span className="probe-episode-badge muted">Select probe</span>;
  }
  const tone = record.correct === true ? "correct" : record.correct === false ? "incorrect" : "";
  return (
    <span className={["probe-episode-badge", tone].filter(Boolean).join(" ")}>
      <strong>{probeSplitLabel(record.split_category, record.split)}</strong>
      <small>
        {record.available
          ? `${record.correct === null || record.correct === undefined ? "scored" : record.correct ? "correct" : "wrong"} · ${formatDatasetProbeConfidence(record.confidence)}`
          : "unscored"}
      </small>
    </span>
  );
}

function matchesProbeFilters(
  episode: DatasetEpisode,
  probe: ProbeDatasetIndex | undefined,
  cohortPreset: ProbeCohortPreset,
  splitFilter: string,
  predictionFilter: string,
): boolean {
  if (!probe) {
    return splitFilter === "all" && predictionFilter === "all" && cohortPreset === "all";
  }
  const record = probe.by_trace[episode.trace_id];
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

function matchesProbeCohortPreset(record: ProbeEpisodeIndex | undefined, preset: ProbeCohortPreset): boolean {
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

function probeRecordForEpisode(
  probe: ProbeDatasetIndex | undefined,
  episode: DatasetEpisode,
): ProbeEpisodeIndex | undefined {
  return probe?.by_trace[episode.trace_id];
}

function episodeOpenContextForProbe(
  probe: ProbeDatasetIndex | undefined,
  episode: DatasetEpisode,
): EpisodeOpenContext | undefined {
  if (!probe) {
    return undefined;
  }
  const record = probe.by_trace[episode.trace_id];
  return {
    fromCohort: true,
    policyCall: policyCallFromProbeFeature(record?.feature ?? probe.best_feature ?? ""),
    probeId: probe.artifact_id,
  };
}

function probeCoverageRows(
  episodes: DatasetEpisode[],
  probe: ProbeDatasetIndex,
): Array<{ label: string; value: number | string }> {
  let scored = 0;
  let correct = 0;
  let incorrect = 0;
  const splits = new Map<string, number>();
  for (const episode of episodes) {
    const record = probe.by_trace[episode.trace_id];
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

function rankProbesForReview(probes: ProbeDatasetIndex[]): ProbeRankRecord[] {
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

function filterProbeChoices(probes: ProbeRankRecord[], query: string): ProbeRankRecord[] {
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

function probeReviewStats(probe: ProbeDatasetIndex): ProbeReviewStats {
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
  for (const record of Object.values(probe.by_trace)) {
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

function probeSplitChartRows(probe: ProbeDatasetIndex): Array<{
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
  for (const record of Object.values(probe.by_trace)) {
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

function probeResultChartRows(probe: ProbeDatasetIndex): Array<{
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
  const total = Object.keys(probe.by_trace).length || 1;
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

function countProbeRecords(
  probe: ProbeDatasetIndex,
  predicate: (record: ProbeEpisodeIndex) => boolean,
): number {
  return Object.values(probe.by_trace).filter(predicate).length;
}

function probeReviewScore(probe: ProbeDatasetIndex, stats: ProbeReviewStats): number {
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

function probeReviewReasons(probe: ProbeDatasetIndex, stats: ProbeReviewStats): string[] {
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

function rankEpisodesForProbe(episodes: DatasetEpisode[], probe: ProbeDatasetIndex): DatasetEpisode[] {
  return [...episodes].sort((left, right) => {
    const delta =
      probeEpisodeInterestScore(probe.by_trace[right.trace_id]) -
      probeEpisodeInterestScore(probe.by_trace[left.trace_id]);
    if (delta !== 0) {
      return delta;
    }
    return episodeTitle(left).localeCompare(episodeTitle(right));
  });
}

function probeEpisodeInterestScore(record: ProbeEpisodeIndex | undefined): number {
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

function probeEpisodeInterestLabel(record: ProbeEpisodeIndex | undefined): string {
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

function probeResultLabel(record: ProbeEpisodeIndex | undefined): string {
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

function probeTrustLabel(stats: ProbeReviewStats | undefined): string {
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

function probeTrustDetail(stats: ProbeReviewStats | undefined): string {
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

function probeConfidenceValue(record: ProbeEpisodeIndex): number | null {
  return typeof record.confidence === "number" && Number.isFinite(record.confidence)
    ? record.confidence
    : null;
}

function percentOf(value: number, total: number): number {
  if (!total) {
    return 0;
  }
  return Math.max(0, Math.min(100, (value / total) * 100));
}

function policyCallFromProbeFeature(feature: string | null | undefined): number | null {
  const match = String(feature ?? "").match(/policy_call_index=([0-9.]+)/);
  if (!match) {
    return null;
  }
  const parsed = Number(match[1]);
  return Number.isFinite(parsed) ? parsed : null;
}

function probeQuestionLabel(probe: ProbeDatasetIndex): string {
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

function labelFromSnake(value: string): string {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatSignedNumber(value: number): string {
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(3)}`;
}

function probeSplitLabel(category?: string | null, fallback?: string | null): string {
  const key = canonicalProbeSplitCategory(category);
  if (key && key !== "unknown") {
    return PROBE_SPLIT_FILTER_LABELS[key] ?? key;
  }
  return fallback ? String(fallback) : "Split missing";
}

function canonicalProbeSplitCategory(category?: string | null): string {
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

function formatDatasetProbeConfidence(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? `conf ${value.toFixed(2)}` : "conf -";
}

function datasetCoverageRows(episodes: DatasetEpisode[]): CoverageRow[] {
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

function episodeTitle(episode: DatasetEpisode): string {
  return String(episode.prompt || episode.episode_id || episode.trace_id);
}

function episodeDatasetId(episode: DatasetEpisode): string {
  return metadataString(episode, "dataset_id");
}

function episodeBenchmark(episode: DatasetEpisode): string {
  return metadataString(episode, "benchmark") || String(episode.env_id || "");
}

function episodeProfile(episode: DatasetEpisode): string {
  return metadataString(episode, "capture_profile");
}

function episodeSeed(episode: DatasetEpisode): string {
  return metadataString(episode, "seed");
}

function metadataString(episode: DatasetEpisode, key: string): string {
  const value = episode.metadata?.[key];
  if (value === undefined || value === null) {
    return "";
  }
  return String(value);
}

function uniqueValues(values: Array<string | null | undefined>): string[] {
  return [...new Set(values.filter((value): value is string => Boolean(value)))].sort((left, right) =>
    left.localeCompare(right),
  );
}

function sortedSetLabel(values: Set<string>): string {
  if (!values.size) {
    return "-";
  }
  return [...values].sort((left, right) => left.localeCompare(right)).join(", ");
}

function shortTrace(traceId: string): string {
  return traceId.length > 44 ? `${traceId.slice(0, 24)}...${traceId.slice(-12)}` : traceId;
}
