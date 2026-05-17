import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ArrowRight, Search } from "lucide-react";
import { cachedDatasetSnapshot, fetchDataset, fetchDatasetDiagnostics } from "../api/dataset";
import { fetchWorkbench } from "../api/workbench";
import { AppShell, type AppPage } from "../components/layout/AppShell";
import { LeftRail } from "../components/layout/LeftRail";
import { TargetObjectEncodingPreset } from "../components/workflows/TargetObjectEncodingPreset";
import { EpisodesPage } from "./EpisodesPage";
import { useWorkbenchStore } from "../store/workbenchStore";
import type { DatasetEpisode } from "../types/dataset";
import type { WorkbenchManifest } from "../types/workbench";

export function WorkbenchPage() {
  const initialRoute = initialPage();
  const [activePage, setActivePage] = useState<AppPage>(initialRoute.page);
  const [episodeTraceId, setEpisodeTraceId] = useState(initialRoute.traceId);
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
  const runs =
    manifest?.analysis_runs.filter((run) => run.workflow === "target_object_encoding") ?? [];
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
          initialTraceId={episodeTraceId}
          onTraceChange={handleEpisodeTraceChange}
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
    }
    window.history.replaceState(null, "", `#${page}`);
  }

  function handleOpenDatasetEpisode(traceId: string) {
    setActivePage("episode");
    setEpisodeTraceId(traceId);
    window.history.replaceState(null, "", `#dataset/${encodeURIComponent(traceId)}`);
  }

  function handleEpisodeTraceChange(traceId: string) {
    setActivePage("episode");
    setEpisodeTraceId(traceId);
    window.history.replaceState(null, "", `#dataset/${encodeURIComponent(traceId)}`);
  }

}

function initialPage(): { page: AppPage; traceId: string } {
  const page = window.location.hash.replace("#", "");
  if (page.startsWith("episode/")) {
    return { page: "episode", traceId: decodeURIComponent(page.slice("episode/".length)) };
  }
  if (page.startsWith("dataset/")) {
    return { page: "episode", traceId: decodeURIComponent(page.slice("dataset/".length)) };
  }
  if (page === "dataset" || page === "workbench" || page === "artifacts") {
    return { page, traceId: "" };
  }
  if (page === "episode" || page === "episodes") {
    return { page: "episode", traceId: "" };
  }
  return { page: "dataset", traceId: "" };
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
  onOpenEpisode: (traceId: string) => void;
}) {
  const dataset = useQuery({
    queryKey: ["dataset"],
    queryFn: fetchDataset,
    initialData: cachedDatasetSnapshot,
    staleTime: 30_000,
  });
  const diagnostics = useQuery({
    queryKey: ["dataset-diagnostics"],
    queryFn: fetchDatasetDiagnostics,
  });
  const episodes = useMemo(() => dataset.data?.episodes ?? [], [dataset.data?.episodes]);
  const [query, setQuery] = useState("");
  const [datasetFilter, setDatasetFilter] = useState("all");
  const [benchmarkFilter, setBenchmarkFilter] = useState("all");
  const [taskFilter, setTaskFilter] = useState("all");
  const [outcomeFilter, setOutcomeFilter] = useState("all");
  const [profileFilter, setProfileFilter] = useState("all");

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
          (!needle || searchText.includes(needle))
        );
      }),
    [datasetFilter, benchmarkFilter, episodes, outcomeFilter, profileFilter, query, taskFilter],
  );
  const coverageRows = useMemo(() => datasetCoverageRows(filteredEpisodes), [filteredEpisodes]);

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
                  <th />
                </tr>
              </thead>
              <tbody>
                {filteredEpisodes.map((episode) => (
                  <tr key={episode.trace_id}>
                    <td>
                      <button
                        className="dataset-episode-link"
                        type="button"
                        onClick={() => onOpenEpisode(episode.trace_id)}
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
                    <td>
                      <button
                        className="icon-command"
                        type="button"
                        title="Open episode"
                        aria-label={`Open ${episode.trace_id}`}
                        onClick={() => onOpenEpisode(episode.trace_id)}
                      >
                        <ArrowRight size={16} />
                      </button>
                    </td>
                  </tr>
                ))}
                {!filteredEpisodes.length ? (
                  <tr>
                    <td colSpan={8}>No episodes match the current filters.</td>
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
  label,
  value,
  values,
  onChange,
}: {
  label: string;
  value: string;
  values: string[];
  onChange: (value: string) => void;
}) {
  return (
    <label className="dataset-filter">
      {label}
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        <option value="all">All</option>
        {values.map((item) => (
          <option key={item} value={item}>
            {item}
          </option>
        ))}
      </select>
    </label>
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
