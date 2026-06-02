import { useDeferredValue, useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ArrowRight, Search, Target } from "lucide-react";
import {
  type EpisodePageParams,
  fetchDataset,
  fetchDatasetDiagnostics,
  fetchEpisodesPage,
  fetchProbeEvidence,
  fetchProbeIndex,
} from "../../api/dataset";
import type {
  DatasetEpisode,
  EpisodeFacetValue,
  ProbeDatasetIndex,
  ProbeEpisodeIndex,
} from "../../types/dataset";
import type { WorkbenchManifest } from "../../types/workbench";
import { datasetBrowserCapabilityGates } from "../capabilityGating";
import {
  COHORT_PRESETS,
  EVIDENCE_EPISODE_LIMIT,
  PROBE_LIST_LIMIT,
  PROBE_PREDICTION_FILTER_LABELS,
  PROBE_PREDICTION_FILTERS,
  PROBE_SPLIT_FILTER_LABELS,
  PROBE_SPLIT_FILTERS,
  datasetCoverageRows,
  episodeBenchmark,
  episodeDatasetId,
  episodeOpenContextForProbe,
  episodeSeed,
  episodeTitle,
  filterProbeChoices,
  formatDatasetProbeConfidence,
  percentOf,
  probeCoverageRows,
  probeEpisodeInterestLabel,
  probeQuestionLabel,
  probeRecordForEpisode,
  probeResultChartRows,
  probeResultLabel,
  probeReviewStats,
  probeSplitChartRows,
  probeSplitLabel,
  probeTrustDetail,
  probeTrustLabel,
  rankProbesForReview,
  shortTrace,
  type ProbeCohortPreset,
  type ProbeRankRecord,
} from "./datasetBrowserModel";
import type { EpisodeOpenContext } from "./types";

export function DatasetBrowser({
  onOpenEpisode,
}: {
  onOpenEpisode: (traceId: string, context?: EpisodeOpenContext) => void;
}) {
  const diagnostics = useQuery({
    queryKey: ["dataset-diagnostics"],
    queryFn: fetchDatasetDiagnostics,
  });
  const datasetFingerprint = diagnostics.data?.fingerprint;
  const datasetCacheReady = diagnostics.isFetched || diagnostics.isError;
  const datasetIdentityKey = datasetFingerprint
    ? `fingerprint:${datasetFingerprint}`
    : datasetCacheReady
      ? "unknown"
      : "pending";
  const dataset = useQuery({
    queryKey: ["dataset", datasetIdentityKey],
    queryFn: () => fetchDataset({ fingerprint: datasetFingerprint }),
    enabled: datasetCacheReady,
    staleTime: 30_000,
  });
  const { hasProbeArtifacts } = datasetBrowserCapabilityGates(dataset.data?.capabilities?.flags);
  const probeIndex = useQuery({
    queryKey: ["probe-index"],
    queryFn: fetchProbeIndex,
    enabled: hasProbeArtifacts,
    staleTime: 60_000,
  });
  const probes = useMemo(
    () => (hasProbeArtifacts ? probeIndex.data?.probes ?? [] : []),
    [hasProbeArtifacts, probeIndex.data?.probes],
  );
  const [query, setQuery] = useState("");
  const [datasetFilter, setDatasetFilter] = useState("all");
  const [benchmarkFilter, setBenchmarkFilter] = useState("all");
  const [taskFilter, setTaskFilter] = useState("all");
  const [outcomeFilter, setOutcomeFilter] = useState("all");
  const [profileFilter, setProfileFilter] = useState("all");
  const [probeFilter, setProbeFilter] = useState("all");
  const [probeQuery, setProbeQuery] = useState("");
  const deferredQuery = useDeferredValue(query);
  const deferredProbeQuery = useDeferredValue(probeQuery);
  const [probeCohortPreset, setProbeCohortPreset] = useState<ProbeCohortPreset>("all");
  const [probeSplitFilter, setProbeSplitFilter] = useState("all");
  const [probePredictionFilter, setProbePredictionFilter] = useState("all");
  const [pageOffset, setPageOffset] = useState(0);
  const selectedProbe = useMemo(
    () => probes.find((probe) => probe.artifact_id === probeFilter),
    [probeFilter, probes],
  );
  const episodePageParams = useMemo<EpisodePageParams>(
    () => ({
      benchmark: benchmarkFilter,
      dataset_id: datasetFilter,
      limit: 100,
      offset: pageOffset,
      outcome: outcomeFilter,
      profile: profileFilter,
      probe_cohort_preset: probeCohortPreset,
      probe_id: selectedProbe?.artifact_id,
      probe_prediction: probePredictionFilter,
      probe_split: probeSplitFilter,
      q: deferredQuery,
      task_id: taskFilter,
    }),
    [
      benchmarkFilter,
      datasetFilter,
      deferredQuery,
      outcomeFilter,
      pageOffset,
      probeCohortPreset,
      probePredictionFilter,
      probeSplitFilter,
      profileFilter,
      selectedProbe?.artifact_id,
      taskFilter,
    ],
  );
  const episodePage = useQuery({
    queryKey: ["episodes", datasetIdentityKey, episodePageParams],
    queryFn: ({ signal }) => fetchEpisodesPage(episodePageParams, signal),
    enabled: datasetCacheReady,
    staleTime: 15_000,
  });
  useEffect(() => {
    setPageOffset(0);
  }, [
    benchmarkFilter,
    datasetFilter,
    deferredQuery,
    outcomeFilter,
    probeCohortPreset,
    probePredictionFilter,
    probeSplitFilter,
    profileFilter,
    selectedProbe?.artifact_id,
    taskFilter,
  ]);
  const rankedProbes = useMemo(() => rankProbesForReview(probes), [probes]);
  const visibleProbeChoices = useMemo(
    () => filterProbeChoices(rankedProbes, deferredProbeQuery).slice(0, PROBE_LIST_LIMIT),
    [deferredProbeQuery, rankedProbes],
  );
  const episodes = useMemo(() => episodePage.data?.episodes ?? [], [episodePage.data?.episodes]);
  const datasetIds = facetValues(episodePage.data?.facets.dataset_id);
  const benchmarks = facetValues(episodePage.data?.facets.benchmark);
  const tasks = facetValues(episodePage.data?.facets.task_id);
  const outcomes = facetValues(episodePage.data?.facets.outcome);
  const profiles = facetValues(episodePage.data?.facets.profile);
  const coverageRows = useMemo(() => datasetCoverageRows(episodes), [episodes]);
  const probeVisibleRows = useMemo(
    () => (selectedProbe ? probeCoverageRows(episodes, selectedProbe) : []),
    [episodes, selectedProbe],
  );
  const evidenceParams = useMemo<EpisodePageParams>(
    () => ({
      limit: EVIDENCE_EPISODE_LIMIT,
      probe_cohort_preset: probeCohortPreset,
      probe_prediction: probePredictionFilter,
      probe_split: probeSplitFilter,
    }),
    [probeCohortPreset, probePredictionFilter, probeSplitFilter],
  );
  const probeEvidence = useQuery({
    queryKey: ["probe-evidence", selectedProbe?.artifact_id, evidenceParams],
    queryFn: ({ signal }) => fetchProbeEvidence(selectedProbe?.artifact_id ?? "", evidenceParams, signal),
    enabled: Boolean(selectedProbe),
    staleTime: 15_000,
  });
  const totalEpisodes = dataset.data?.episode_count ?? episodePage.data?.total ?? 0;
  const visibleTotal = episodePage.data?.total ?? 0;
  const pageStart = visibleTotal ? pageOffset + 1 : 0;
  const pageEnd = Math.min(pageOffset + (episodePage.data?.limit ?? 100), visibleTotal);

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
        <Metric label="Episodes" value={totalEpisodes} />
        <Metric label="Visible" value={visibleTotal} />
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

      {!datasetCacheReady || dataset.isLoading ? <div className="app-message">Loading dataset...</div> : null}
      {dataset.isError ? <div className="empty-state">Dataset API unavailable.</div> : null}
      {episodePage.isError ? <div className="empty-state compact">Episode index unavailable.</div> : null}
      {episodePage.isFetching ? <div className="app-message compact">Loading episodes...</div> : null}
      {hasProbeArtifacts && probeIndex.isError ? (
        <div className="empty-state compact">Probe index unavailable.</div>
      ) : null}

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
            episodes={probeEvidence.data?.episodes ?? []}
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
            <span>
              {pageStart}-{pageEnd} / {visibleTotal}
            </span>
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
                {episodes.map((episode) => (
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
                {!episodes.length ? (
                  <tr>
                    <td colSpan={selectedProbe ? 9 : 8}>No episodes match the current filters.</td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
          <div className="dataset-pagination">
            <button
              className="text-command"
              disabled={pageOffset <= 0}
              type="button"
              onClick={() => setPageOffset(Math.max(0, pageOffset - 100))}
            >
              Previous
            </button>
            <button
              className="text-command"
              disabled={episodePage.data?.next_offset === null || episodePage.data?.next_offset === undefined}
              type="button"
              onClick={() => setPageOffset(episodePage.data?.next_offset ?? pageOffset)}
            >
              Next
            </button>
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
                const record = probeRecordForEpisode(probe, episode);
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

export function ArtifactsSummary({ manifest }: { manifest: WorkbenchManifest }) {
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

export function BackendUnavailable() {
  return (
    <main className="summary-page">
      <h1>Backend unavailable</h1>
      <p>
        The React dev server is running, but the VLA-lens Python backend is not listening on
        127.0.0.1:8765.
      </p>
      <pre>uv run python scripts/serve_vla_lens_dashboard.py runs/vla_lens_demo --port 8765</pre>
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

function facetValues(values: EpisodeFacetValue[] | undefined): string[] {
  return (values ?? []).map((item) => item.value).filter(Boolean);
}

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
