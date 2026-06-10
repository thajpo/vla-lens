import { useDeferredValue, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ArrowRight, Search } from "lucide-react";
import {
  fetchArtifacts,
  fetchDiscoveryArtifactFamilies,
  type DiscoveryArtifactEpisodeParams,
  type EpisodePageParams,
  fetchDiscoveryArtifactEpisodes,
  fetchDataset,
  fetchDatasetDiagnostics,
  fetchEpisodesPage,
  fetchProbeEvidenceBundle,
  fetchProbeIndex,
} from "../../api/dataset";
import type {
  ArtifactRecord,
  DatasetEpisode,
  DiscoveryArtifactEpisodesResponse,
  DiscoveryArtifactFamily,
  EpisodeFacetValue,
  ProbeDatasetIndex,
  ProbeEpisodeIndex,
} from "../../types/dataset";
import type { WorkbenchManifest } from "../../types/workbench";
import type { ProbeEvidenceBundle, ResearchSelectionState } from "../../types/probeEvidence";
import { datasetBrowserCapabilityGates } from "../capabilityGating";
import {
  COHORT_PRESETS,
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
  formatDatasetProbeConfidence,
  percentOf,
  probeEvidenceContextForEpisode,
  probeEpisodeInspectionReason,
  probeEvidenceCueForEpisode,
  probeEvidenceRankedRows,
  probeEvidenceSpec,
  probeLensWorkbenchModel,
  probeRecordForEpisode,
  probeResultChartRows,
  probeLensSpec,
  probeSplitChartRows,
  probeSplitLabel,
  rankingLabel,
  shortTrace,
  type ProbeCohortPreset,
  type ProbeEvidenceRankedRow,
  type ProbeLensWorkbenchViewModel,
} from "./datasetBrowserModel";
import type { EpisodeOpenContext } from "./types";

const NO_LENS_ID = "none";
const SORT_OPTIONS = ["episode_index", "lens_interest", "task_id", "outcome", "length", "trace_id"] as const;
const SORT_LABELS: Record<string, string> = {
  episode_index: "Episode order",
  lens_interest: "Lens interest",
  length: "Steps",
  outcome: "Outcome",
  task_id: "Task",
  trace_id: "Trace ID",
};
const PROBE_COHORT_FILTERS = COHORT_PRESETS.map((preset) => preset.id);
const PROBE_COHORT_FILTER_LABELS: Record<string, string> = Object.fromEntries(
  COHORT_PRESETS.map((preset) => [preset.id, preset.label]),
);

type DatasetLens = {
  artifact?: ArtifactRecord;
  artifactId: string;
  artifactType: string;
  family?: DiscoveryArtifactFamily;
  name: string;
  probe?: ProbeDatasetIndex;
};

export function DatasetBrowser({
  onOpenEpisode,
}: {
  onOpenEpisode: (traceId: string, context?: EpisodeOpenContext) => void;
}) {
  const dataset = useQuery({
    queryKey: ["dataset"],
    queryFn: () => fetchDataset(),
    staleTime: 30_000,
  });
  const datasetFingerprint = dataset.data?.index?.dataset_fingerprint;
  const datasetIdentityKey = datasetFingerprint
    ? `fingerprint:${datasetFingerprint}`
    : dataset.isFetched
      ? "unknown"
      : "pending";
  const { hasProbeArtifacts } = datasetBrowserCapabilityGates(dataset.data?.capabilities?.flags);
  const discoveryFamilies = useQuery({
    queryKey: ["discovery-artifact-families"],
    queryFn: fetchDiscoveryArtifactFamilies,
    enabled: dataset.isFetched,
    staleTime: 60_000,
  });
  const artifactIndex = useQuery({
    queryKey: ["artifacts", datasetIdentityKey],
    queryFn: fetchArtifacts,
    enabled: dataset.isFetched,
    staleTime: 60_000,
  });
  const probeIndex = useQuery({
    queryKey: ["probe-index"],
    queryFn: fetchProbeIndex,
    enabled: dataset.isFetched && hasProbeArtifacts,
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
  const [selectedLensId, setSelectedLensId] = useState(NO_LENS_ID);
  const [sortMode, setSortMode] = useState("episode_index");
  const deferredQuery = useDeferredValue(query);
  const [probeCohortPreset, setProbeCohortPreset] = useState<ProbeCohortPreset>("all");
  const [probeSplitFilter, setProbeSplitFilter] = useState("all");
  const [probePredictionFilter, setProbePredictionFilter] = useState("all");
  const [researchSelection, setResearchSelection] = useState<ResearchSelectionState | null>(null);
  const [pageOffset, setPageOffset] = useState(0);
  const familyByType = useMemo(
    () => new Map((discoveryFamilies.data?.families ?? []).map((family) => [family.artifact_type, family])),
    [discoveryFamilies.data?.families],
  );
  const artifactsById = useMemo(
    () => new Map((artifactIndex.data?.artifacts ?? []).map((artifact) => [String(artifact.artifact_id ?? ""), artifact])),
    [artifactIndex.data?.artifacts],
  );
  const lenses = useMemo(
    () => discoveryLenses({
      artifacts: artifactIndex.data?.artifacts ?? [],
      artifactsById,
      familyByType,
      probes,
    }),
    [artifactIndex.data?.artifacts, artifactsById, familyByType, probes],
  );
  const selectedLens = useMemo(
    () => lenses.find((lens) => lens.artifactId === selectedLensId),
    [lenses, selectedLensId],
  );
  const selectedProbe = selectedLens?.probe;
  const activeLensArtifactId = selectedLens?.artifactId ?? "";
  const probeEvidenceBundle = useQuery({
    queryKey: [
      "probe-evidence-bundle",
      datasetIdentityKey,
      datasetFilter,
      selectedProbe?.artifact_id ?? NO_LENS_ID,
    ],
    queryFn: ({ signal }) =>
      fetchProbeEvidenceBundle(
        selectedProbe?.artifact_id ?? "",
        { dataset_id: datasetFilter, limit: 50 },
        signal,
      ),
    enabled: dataset.isFetched && Boolean(selectedProbe),
    staleTime: 60_000,
  });
  const activeProbeEvidenceBundle = selectedProbe ? probeEvidenceBundle.data : undefined;
  const episodePageParams = useMemo<DiscoveryArtifactEpisodeParams>(
    () => ({
      benchmark: benchmarkFilter,
      cohort_preset: selectedProbe ? probeCohortPreset : undefined,
      dataset_id: datasetFilter,
      limit: 100,
      offset: pageOffset,
      outcome: outcomeFilter,
      prediction: selectedProbe ? probePredictionFilter : undefined,
      profile: profileFilter,
      q: deferredQuery,
      rank_by: selectedLens && sortMode === "lens_interest" ? "interest" : undefined,
      sort: sortMode === "lens_interest" ? undefined : sortMode,
      split: selectedProbe ? probeSplitFilter : undefined,
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
      selectedLens,
      selectedProbe,
      sortMode,
      taskFilter,
    ],
  );
  const episodePage = useQuery({
    queryKey: ["episodes", datasetIdentityKey, activeLensArtifactId || NO_LENS_ID, episodePageParams],
    queryFn: ({ signal }) => selectedLens
      ? fetchDiscoveryArtifactEpisodes(activeLensArtifactId, episodePageParams, signal)
      : fetchEpisodesPage(episodePageParams as EpisodePageParams, signal),
    enabled: dataset.isFetched,
    staleTime: 15_000,
  });
  const discoveryPayload = discoveryEpisodePayload(episodePage.data);
  const activeLensUnavailable = Boolean(selectedLens && discoveryPayload?.available === false);
  const activeLensReason = selectedLens ? discoveryPayload?.reason ?? "" : "";
  const activeLensFamilyLabel = selectedLens ? familyLabel(selectedLens.artifactType) : "Episode index";
  const diagnostics = useQuery({
    queryKey: ["dataset-diagnostics", datasetIdentityKey],
    queryFn: fetchDatasetDiagnostics,
    enabled: episodePage.isFetched || episodePage.isError,
    staleTime: 60_000,
  });
  const episodes = useMemo(() => episodePage.data?.episodes ?? [], [episodePage.data?.episodes]);
  const datasetIds = facetValues(episodePage.data?.facets.dataset_id);
  const benchmarks = facetValues(episodePage.data?.facets.benchmark);
  const tasks = facetValues(episodePage.data?.facets.task_id);
  const outcomes = facetValues(episodePage.data?.facets.outcome);
  const profiles = facetValues(episodePage.data?.facets.profile);
  const coverageRows = useMemo(() => datasetCoverageRows(episodes), [episodes]);
  const rankedEvidenceRows = useMemo(
    () => probeEvidenceRankedRows({
      bundle: activeProbeEvidenceBundle,
      episodes,
      probe: selectedProbe,
      selected: researchSelection,
      selectedDatasetId: datasetFilter,
    }),
    [activeProbeEvidenceBundle, datasetFilter, episodes, researchSelection, selectedProbe],
  );
  const probeWorkbench = useMemo(
    () => selectedProbe
      ? probeLensWorkbenchModel({
          artifact: selectedLens?.artifact,
          bundle: activeProbeEvidenceBundle,
          probe: selectedProbe,
          totalEpisodes: dataset.data?.episode_count ?? episodePage.data?.total ?? 0,
        })
      : undefined,
    [
      activeProbeEvidenceBundle,
      dataset.data?.episode_count,
      episodePage.data?.total,
      selectedLens?.artifact,
      selectedProbe,
    ],
  );
  const totalEpisodes = dataset.data?.episode_count ?? episodePage.data?.total ?? 0;
  const visibleTotal = episodePage.data?.total ?? 0;
  const pageStart = visibleTotal ? pageOffset + 1 : 0;
  const pageEnd = Math.min(pageOffset + (episodePage.data?.limit ?? 100), visibleTotal);
  const activeFilterCount = [
    datasetFilter,
    benchmarkFilter,
    taskFilter,
    outcomeFilter,
    profileFilter,
    selectedProbe ? probeCohortPreset : "all",
    selectedProbe ? probeSplitFilter : "all",
    selectedProbe ? probePredictionFilter : "all",
  ].filter((value) => value !== "all").length;
  const resetPage = () => setPageOffset(0);
  const selectLens = (lensId: string) => {
    setSelectedLensId(lensId);
    setSortMode(lensId === NO_LENS_ID ? "episode_index" : "lens_interest");
    setProbeCohortPreset("all");
    setProbeSplitFilter("all");
    setProbePredictionFilter("all");
    setResearchSelection(null);
    resetPage();
  };
  const openProbeMoment = (row: ProbeEvidenceRankedRow) => {
    const selection = row.context?.researchSelection ?? null;
    setResearchSelection(selection);
    onOpenEpisode(row.episode?.trace_id ?? row.moment.episode_id, row.context);
  };
  const openDatasetEpisode = (episode: DatasetEpisode) => {
    const evidenceContext = probeEvidenceContextForEpisode(
      selectedProbe,
      activeProbeEvidenceBundle,
      episode,
      datasetFilter,
    );
    if (evidenceContext?.researchSelection) {
      setResearchSelection(evidenceContext.researchSelection);
      onOpenEpisode(episode.trace_id, evidenceContext);
      return;
    }
    onOpenEpisode(episode.trace_id, episodeOpenContextForProbe(selectedProbe, episode));
  };

  return (
    <main className="dataset-browser-page">
      <header className="dataset-browser-header">
        <div>
          <h1>Dataset</h1>
          <p>{dataset.data?.root ?? "Dataset"}</p>
        </div>
        {diagnostics.data?.stale ? <div className="diagnostic-chip stale">Diagnostics stale</div> : null}
      </header>

      <section className="dataset-lens-bar" aria-label="Dataset lens">
        <LensSelector
          lenses={lenses}
          selectedLensId={selectedLensId}
          onLensChange={selectLens}
        />
        {selectedLens ? <span>{activeLensFamilyLabel}</span> : null}
      </section>

      {selectedProbe && probeWorkbench ? (
        <ProbeLensWorkbench
          activePredictionFilter={probePredictionFilter}
          activeSplitFilter={probeSplitFilter}
          evidenceIsError={probeEvidenceBundle.isError}
          evidenceIsLoading={probeEvidenceBundle.isFetching}
          model={probeWorkbench}
          probe={selectedProbe}
          rankedRows={rankedEvidenceRows}
          unavailable={activeProbeEvidenceBundle?.unavailable ?? []}
          onCohortPresetChange={(preset) => {
            setProbeCohortPreset(preset);
            setProbeSplitFilter("all");
            setProbePredictionFilter("all");
            resetPage();
          }}
          onOpenMoment={openProbeMoment}
          onPredictionFilterChange={(value) => {
            setProbeCohortPreset("all");
            setProbePredictionFilter(value);
            resetPage();
          }}
          onSplitFilterChange={(value) => {
            setProbeCohortPreset("all");
            setProbeSplitFilter(value);
            resetPage();
          }}
        />
      ) : selectedLens ? (
        <LensEvidencePanel lens={selectedLens} ranking={discoveryPayload} />
      ) : null}

      <section className="dataset-table-controls" aria-label="Episode filters and sorting">
        <div className="dataset-search">
          <Search size={15} />
          <input
            aria-label="Search dataset episodes"
            placeholder="Search trace, task, dataset, prompt"
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
              resetPage();
            }}
          />
        </div>
        <FilterSelect
          includeAll={false}
          label="Sort"
          value={sortMode}
          values={selectedLens ? SORT_OPTIONS : SORT_OPTIONS.filter((option) => option !== "lens_interest")}
          labels={SORT_LABELS}
          onChange={(value) => {
            setSortMode(value);
            resetPage();
          }}
        />
        <FilterSelect
          label="Dataset"
          value={datasetFilter}
          values={datasetIds}
          onChange={(value) => {
            setDatasetFilter(value);
            resetPage();
          }}
        />
        <FilterSelect
          label="Benchmark"
          value={benchmarkFilter}
          values={benchmarks}
          onChange={(value) => {
            setBenchmarkFilter(value);
            resetPage();
          }}
        />
        <FilterSelect
          label="Task"
          value={taskFilter}
          values={tasks}
          onChange={(value) => {
            setTaskFilter(value);
            resetPage();
          }}
        />
        <FilterSelect
          label="Outcome"
          value={outcomeFilter}
          values={outcomes}
          onChange={(value) => {
            setOutcomeFilter(value);
            resetPage();
          }}
        />
        <FilterSelect
          label="Profile"
          value={profileFilter}
          values={profiles}
          onChange={(value) => {
            setProfileFilter(value);
            resetPage();
          }}
        />
        {selectedProbe ? (
          <>
            <FilterSelect
              label="Probe Cohort"
              value={probeCohortPreset}
              values={PROBE_COHORT_FILTERS}
              labels={PROBE_COHORT_FILTER_LABELS}
              onChange={(value) => {
                setProbeCohortPreset(value as ProbeCohortPreset);
                setProbeSplitFilter("all");
                setProbePredictionFilter("all");
                resetPage();
              }}
            />
            <FilterSelect
              label="Probe Split"
              value={probeSplitFilter}
              values={PROBE_SPLIT_FILTERS}
              labels={PROBE_SPLIT_FILTER_LABELS}
              onChange={(value) => {
                setProbeCohortPreset("all");
                setProbeSplitFilter(value);
                resetPage();
              }}
            />
            <FilterSelect
              label="Probe Result"
              value={probePredictionFilter}
              values={PROBE_PREDICTION_FILTERS}
              labels={PROBE_PREDICTION_FILTER_LABELS}
              onChange={(value) => {
                setProbeCohortPreset("all");
                setProbePredictionFilter(value);
                resetPage();
              }}
            />
          </>
        ) : null}
        <button
          className="text-command"
          disabled={!selectedLens && activeFilterCount === 0 && sortMode === "episode_index"}
          type="button"
          onClick={() => {
            setDatasetFilter("all");
            setBenchmarkFilter("all");
            setTaskFilter("all");
            setOutcomeFilter("all");
            setProfileFilter("all");
            setSortMode("episode_index");
            selectLens(NO_LENS_ID);
          }}
        >
          Reset
        </button>
      </section>

      {dataset.isLoading ? <div className="app-message">Loading dataset...</div> : null}
      {dataset.isError ? <div className="empty-state">Dataset API unavailable.</div> : null}
      {episodePage.isError ? (
        <div className="empty-state compact">
          {selectedLens
            ? "Lens ranking API unavailable. Restart the Python dashboard server so it picks up the discovery artifact routes."
            : "Episode index unavailable."}
        </div>
      ) : null}
      {activeLensUnavailable ? (
        <div className="empty-state compact">
          {activeLensReason || "This lens cannot rank episodes yet."}
        </div>
      ) : null}
      {episodePage.isFetching ? <div className="app-message compact">Loading episodes...</div> : null}
      {hasProbeArtifacts && probeIndex.isError ? (
        <div className="empty-state compact">Probe index unavailable.</div>
      ) : null}

      <section className="dataset-browser-grid">
        <div className="dataset-browser-panel">
          <header>
            <h2>{selectedProbe ? "Episodes through probe" : selectedLens ? "Episodes through lens" : "Episodes"}</h2>
            <span>
              {pageStart}-{pageEnd} / {visibleTotal}
            </span>
          </header>
          <div className="dataset-episode-table-wrap">
            <table className="compact-table dataset-episode-table">
              <thead>
                {selectedProbe ? (
                  <tr>
                    <th>Episode</th>
                    <th>Probe readout</th>
                    <th>Why inspect</th>
                    <th>Split</th>
                    <th>Outcome</th>
                    <th />
                  </tr>
                ) : (
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
                )}
              </thead>
              <tbody>
                {episodes.map((episode) => selectedProbe ? (
                  <ProbeEpisodeTableRow
                    bundle={activeProbeEvidenceBundle}
                    datasetFilter={datasetFilter}
                    episode={episode}
                    key={episode.trace_id}
                    probe={selectedProbe}
                    onOpenEpisode={openDatasetEpisode}
                  />
                ) : (
                  <DatasetEpisodeTableRow
                    episode={episode}
                    key={episode.trace_id}
                    onOpenEpisode={openDatasetEpisode}
                  />
                ))}
                {!episodes.length ? (
                  <tr>
                    <td colSpan={selectedProbe ? 6 : 8}>No episodes match the current filters.</td>
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

        <details className="dataset-details-drawer">
          <summary>
            Dataset details
            <span>{totalEpisodes} episodes · {probes.length} probes · {dataset.data?.activation_sites ?? 0} model sites</span>
          </summary>
          <div className="dataset-browser-metrics">
            <Metric label="Episodes" value={totalEpisodes} />
            <Metric label="Visible" value={visibleTotal} />
            <Metric label="Datasets" value={datasetIds.length} />
            <Metric label="Benchmarks" value={benchmarks.length} />
            <Metric label="Tasks" value={tasks.length} />
            <Metric label="Model Sites" value={dataset.data?.activation_sites ?? 0} />
            <Metric label="Probes" value={probes.length} />
          </div>
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
        </details>
      </section>
    </main>
  );
}

function ProbeLensWorkbench({
  activePredictionFilter,
  activeSplitFilter,
  evidenceIsError,
  evidenceIsLoading,
  model,
  probe,
  rankedRows,
  unavailable,
  onCohortPresetChange,
  onOpenMoment,
  onPredictionFilterChange,
  onSplitFilterChange,
}: {
  activePredictionFilter: string;
  activeSplitFilter: string;
  evidenceIsError: boolean;
  evidenceIsLoading: boolean;
  model: ProbeLensWorkbenchViewModel;
  probe: ProbeDatasetIndex;
  rankedRows: ProbeEvidenceRankedRow[];
  unavailable: ProbeEvidenceBundle["unavailable"];
  onCohortPresetChange: (preset: ProbeCohortPreset) => void;
  onOpenMoment: (row: ProbeEvidenceRankedRow) => void;
  onPredictionFilterChange: (value: string) => void;
  onSplitFilterChange: (value: string) => void;
}) {
  return (
    <section className="probe-lens-workbench" aria-label="Selected probe lens workbench">
      <div className="probe-lens-head">
        <span>Selected probe lens</span>
        <h2>{model.title}</h2>
        <p>{model.verdict.detail}</p>
        <div className="probe-lens-specs">
          <ProbeEvidenceFact
            label={model.spec.prediction.label}
            value={model.spec.prediction.value}
            detail={model.spec.prediction.detail}
          />
          <ProbeEvidenceFact
            label={model.spec.input.label}
            value={model.spec.input.value}
            detail={model.spec.input.detail}
          />
          <ProbeEvidenceFact
            label={model.spec.output.label}
            value={model.spec.output.value}
            detail={model.spec.output.detail}
          />
          <ProbeEvidenceFact
            label={model.spec.objective.label}
            value={model.spec.objective.value}
            detail={model.spec.objective.detail}
          />
        </div>
      </div>
      <aside className={`probe-lens-verdict ${model.verdict.tone}`}>
        <span>{model.verdict.label}</span>
        <strong>{model.verdict.headline}</strong>
        <small>{model.subtitle}</small>
        <div className="probe-lens-metrics">
          {model.metrics.map((metric) => (
            <div className={`probe-lens-metric ${metric.tone}`} key={metric.label}>
              <span>{metric.label}</span>
              <strong>{metric.value}</strong>
              {metric.detail ? <small>{metric.detail}</small> : null}
            </div>
          ))}
        </div>
      </aside>
      <ProbeMechanismCard model={model} />
      <ProbeSummaryVisual
        activePredictionFilter={activePredictionFilter}
        activeSplitFilter={activeSplitFilter}
        probe={probe}
        onCohortPresetChange={onCohortPresetChange}
        onPredictionFilterChange={onPredictionFilterChange}
        onSplitFilterChange={onSplitFilterChange}
      />
      <ProbeRankedEvidencePanel
        isError={evidenceIsError}
        isLoading={evidenceIsLoading}
        rows={rankedRows}
        unavailable={unavailable}
        onOpenMoment={onOpenMoment}
      />
    </section>
  );
}

function ProbeMechanismCard({ model }: { model: ProbeLensWorkbenchViewModel }) {
  return (
    <section className="probe-mechanism-card" aria-label="Probe mechanism summary">
      <header>
        <span>Where it reads</span>
        <strong>{model.mechanism.modelSite}</strong>
      </header>
      <div className="probe-mechanism-tags">
        <span>{model.mechanism.basis}</span>
        <span>{model.mechanism.temporal}</span>
        <span>{model.mechanism.output}</span>
      </div>
      {model.mechanism.contributors.length ? (
        <div className="probe-contributor-list">
          {model.mechanism.contributors.map((item) => (
            <div className={`probe-contributor ${item.tone}`} key={item.key}>
              <span>{item.value}</span>
              <strong>{item.label}</strong>
              <small>{item.detail}</small>
            </div>
          ))}
        </div>
      ) : (
        <p>Contributor details are not available in the dataset-level bundle yet. Open an evidence moment for feature-level readout.</p>
      )}
      {model.mechanism.missing.length ? (
        <details className="probe-mechanism-missing">
          <summary>Unavailable evidence</summary>
          <ul>
            {model.mechanism.missing.map((message) => (
              <li key={message}>{message}</li>
            ))}
          </ul>
        </details>
      ) : null}
    </section>
  );
}

function DatasetEpisodeTableRow({
  episode,
  onOpenEpisode,
}: {
  episode: DatasetEpisode;
  onOpenEpisode: (episode: DatasetEpisode) => void;
}) {
  return (
    <tr>
      <td>
        <button
          className="dataset-episode-link"
          type="button"
          onClick={() => onOpenEpisode(episode)}
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
          onClick={() => onOpenEpisode(episode)}
        >
          <ArrowRight size={16} />
        </button>
      </td>
    </tr>
  );
}

function ProbeEpisodeTableRow({
  bundle,
  datasetFilter,
  episode,
  probe,
  onOpenEpisode,
}: {
  bundle?: ProbeEvidenceBundle;
  datasetFilter: string;
  episode: DatasetEpisode;
  probe: ProbeDatasetIndex;
  onOpenEpisode: (episode: DatasetEpisode) => void;
}) {
  const cue = probeEvidenceCueForEpisode(bundle, episode, probe, datasetFilter);
  const record = probeRecordForEpisode(probe, episode);
  const reason = probeEpisodeInspectionReason(probe, bundle, episode, datasetFilter);
  const meta = [
    episodeBenchmark(episode),
    episode.task_id ? `task ${episode.task_id}` : "",
    episodeSeed(episode) ? `seed ${episodeSeed(episode)}` : "",
    episode.length ? `${episode.length} steps` : "",
  ].filter(Boolean).join(" · ");
  return (
    <tr className={`probe-episode-row ${reason.tone}`}>
      <td>
        <button
          className="dataset-episode-link probe-oriented"
          type="button"
          onClick={() => onOpenEpisode(episode)}
        >
          <span>{episodeTitle(episode)}</span>
          <small>{meta || shortTrace(episode.trace_id)}</small>
        </button>
      </td>
      <td>
        <ProbeEpisodeBadge
          cue={cue}
          record={record}
        />
      </td>
      <td>
        <div className={`probe-inspect-reason ${reason.tone}`}>
          <strong>{reason.label}</strong>
          <small>{reason.detail}</small>
          {reason.timelinePercent !== null ? (
            <i className="probe-mini-timeline">
              <b style={{ left: `${reason.timelinePercent}%` }} />
            </i>
          ) : null}
        </div>
      </td>
      <td>
        <span className="probe-split-chip">{probeSplitLabel(record?.split_category, record?.split)}</span>
      </td>
      <td>
        <span className={`outcome-pill ${episode.outcome ?? "unknown"}`}>
          {episode.outcome ?? "unknown"}
        </span>
      </td>
      <td>
        <button
          className="icon-command"
          type="button"
          title="Open episode at probe evidence"
          aria-label={`Open ${episode.trace_id}`}
          onClick={() => onOpenEpisode(episode)}
        >
          <ArrowRight size={16} />
        </button>
      </td>
    </tr>
  );
}

function FilterSelect({
  disabled = false,
  includeAll = true,
  labels,
  label,
  value,
  values,
  onChange,
}: {
  disabled?: boolean;
  includeAll?: boolean;
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
        {includeAll ? <option value="all">All</option> : null}
        {values.map((item) => (
          <option key={item} value={item}>
            {labels?.[item] ?? item}
          </option>
        ))}
      </select>
    </label>
  );
}

function LensSelector({
  lenses,
  selectedLensId,
  onLensChange,
}: {
  lenses: DatasetLens[];
  selectedLensId: string;
  onLensChange: (lensId: string) => void;
}) {
  const groups = lensGroups(lenses);
  return (
    <section className="dataset-lens-strip" aria-label="Dataset lens">
      <label className="dataset-lens-select">
        <span>Dataset Lens</span>
        <select value={selectedLensId} onChange={(event) => onLensChange(event.target.value)}>
          <option value={NO_LENS_ID}>None - episode index</option>
          {groups.map((group) => (
            <optgroup key={group.family} label={group.label}>
              {group.lenses.map((lens) => (
                <option key={lens.artifactId} value={lens.artifactId}>
                  {lensDisplayName(lens)}{lens.artifactType === "probe_suite" ? "" : " - ranking pending"}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
      </label>
    </section>
  );
}

function LensEvidencePanel({
  lens,
  ranking,
}: {
  lens?: DatasetLens;
  ranking?: DiscoveryArtifactEpisodesResponse;
}) {
  if (!lens) {
    return (
      <div className="probe-evidence-empty">
        <span>Dataset lens</span>
        <strong>No lens selected</strong>
        <p>Choose a probe or another discovery artifact above to rank episodes through that signal.</p>
      </div>
    );
  }
  return (
    <div className="probe-evidence-empty">
      <span>{familyLabel(lens.artifactType)}</span>
      <strong>{lensDisplayName(lens)}</strong>
      <p>
        {ranking?.available === false
          ? ranking.reason || "This artifact family is registered, but episode ranking is not implemented yet."
          : "This lens is selected for dataset ranking. Detailed side-panel evidence is available for probe lenses first."}
      </p>
    </div>
  );
}

export function SelectedProbeLensSummary({
  artifact,
  bundle,
  probe,
}: {
  artifact?: ArtifactRecord;
  bundle?: ProbeEvidenceBundle;
  probe: ProbeDatasetIndex;
}) {
  const spec = probeEvidenceSpec(bundle) ?? probeLensSpec(probe, artifact);
  return (
    <section className="selected-lens-summary" aria-label="Selected lens summary">
      <div className="selected-lens-title">
        <span>Probe</span>
        <strong>{probe.name}</strong>
        {bundle ? <small>{bundle.run.dataset_id} · {bundle.run.status}</small> : null}
      </div>
      <div className="selected-lens-facts">
        <ProbeEvidenceFact
          label={spec.prediction.label}
          value={spec.prediction.value}
          detail={spec.prediction.detail}
        />
        <ProbeEvidenceFact
          label={spec.input.label}
          value={spec.input.value}
          detail={spec.input.detail}
        />
        <ProbeEvidenceFact
          label={spec.output.label}
          value={spec.output.value}
          detail={spec.output.detail}
        />
        <ProbeEvidenceFact
          label={spec.objective.label}
          value={spec.objective.value}
          detail={spec.objective.detail}
        />
      </div>
    </section>
  );
}

function ProbeRankedEvidencePanel({
  isError,
  isLoading,
  rows,
  unavailable,
  onOpenMoment,
}: {
  isError: boolean;
  isLoading: boolean;
  rows: ProbeEvidenceRankedRow[];
  unavailable: ProbeEvidenceBundle["unavailable"];
  onOpenMoment: (row: ProbeEvidenceRankedRow) => void;
}) {
  if (isLoading) {
    return <div className="app-message compact">Loading probe evidence...</div>;
  }
  if (isError) {
    return <div className="empty-state compact">Probe evidence bundle unavailable.</div>;
  }
  if (!rows.length) {
    const message = unavailable[0]?.message ?? "No ranked probe moments were returned for this dataset.";
    return <div className="empty-state compact">{message}</div>;
  }
  return (
    <section className="probe-ranked-evidence" aria-label="Ranked probe evidence">
      <header>
        <div>
          <span>Ranked evidence</span>
          <strong>Top, low, and uncertain moments</strong>
        </div>
        <small>{rows.length} moments from the selected probe run</small>
      </header>
      <div className="probe-ranked-list">
        {rows.map((row) => (
          <button
            className={["probe-ranked-row", row.selected ? "active" : ""].filter(Boolean).join(" ")}
            key={`${row.ranking}:${row.moment.episode_id}:${row.moment.timestep ?? ""}:${row.moment.policy_call ?? ""}`}
            type="button"
            onClick={() => onOpenMoment(row)}
          >
            <span className={`probe-rank-tag ${row.ranking}`}>{rankingLabel(row.ranking)}</span>
            <strong>{row.episode ? episodeTitle(row.episode) : row.moment.episode_id}</strong>
            <span>{row.timeLabel}</span>
            <span>{row.scoreLabel}</span>
            <span>{row.predictionLabel}</span>
            <small>{row.splitLabel} · {row.resultLabel}</small>
            <em>{row.provenanceBadges.join(" · ")}</em>
          </button>
        ))}
      </div>
    </section>
  );
}

function ProbeEvidenceFact({
  detail,
  label,
  value,
}: {
  detail?: string;
  label: string;
  value: string;
}) {
  return (
    <div className="probe-evidence-fact">
      <span>{label}</span>
      <strong>{value}</strong>
      {detail ? <small>{detail}</small> : null}
    </div>
  );
}

function ProbeSummaryVisual({
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
    <section className="probe-evidence-plot probe-summary-visual" aria-label="Probe summary">
      <div className="probe-evidence-plot-column">
        <header>
          <span>Indexed split map</span>
          <small className="probe-map-legend">
            <span className="wrong">wrong</span>
            <span className="high-conf-wrong">high-conf wrong</span>
          </small>
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
                <em className="wrong" style={{ width: `${percentOf(row.wrong ?? 0, row.total)}%` }} />
                <em className="high-conf-wrong" style={{ width: `${percentOf(row.highConfWrong ?? 0, row.total)}%` }} />
              </i>
              <small>
                {row.total === 0
                  ? "no episodes"
                  : row.wrong === null
                  ? "error counts unavailable"
                  : `${Math.max(0, row.scored - row.wrong)} correct · ${row.wrong} wrong · ${
                      row.highConfWrong ?? 0
                    } high-conf wrong`}
              </small>
            </button>
          ))}
        </div>
      </div>
      <div className="probe-evidence-plot-column">
        <header>
          <span>Indexed result map</span>
          <small>compatible cohort counts</small>
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
    </section>
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

function discoveryLenses({
  artifacts,
  artifactsById,
  familyByType,
  probes,
}: {
  artifacts: ArtifactRecord[];
  artifactsById: Map<string, ArtifactRecord>;
  familyByType: Map<string, DiscoveryArtifactFamily>;
  probes: ProbeDatasetIndex[];
}): DatasetLens[] {
  const seen = new Set<string>();
  const lenses: DatasetLens[] = probes.map((probe) => {
    const artifact = artifactsById.get(probe.artifact_id);
    seen.add(probe.artifact_id);
    return {
      artifact,
      artifactId: probe.artifact_id,
      artifactType: "probe_suite",
      family: familyByType.get("probe_suite"),
      name: probe.name,
      probe,
    };
  });
  for (const artifact of artifacts) {
    const artifactId = String(artifact.artifact_id ?? "");
    const artifactType = String(artifact.artifact_type ?? "");
    if (!artifactId || seen.has(artifactId) || !familyByType.has(artifactType)) {
      continue;
    }
    seen.add(artifactId);
    lenses.push({
      artifact,
      artifactId,
      artifactType,
      family: familyByType.get(artifactType),
      name: String(artifact.name ?? artifactId),
    });
  }
  return lenses.sort((left, right) => {
    const familyDelta = familyLabel(left.artifactType).localeCompare(familyLabel(right.artifactType));
    if (familyDelta !== 0) {
      return familyDelta;
    }
    return lensDisplayName(left).localeCompare(lensDisplayName(right));
  });
}

function lensGroups(lenses: DatasetLens[]): Array<{ family: string; label: string; lenses: DatasetLens[] }> {
  const groups = new Map<string, DatasetLens[]>();
  for (const lens of lenses) {
    const key = lens.artifactType;
    groups.set(key, [...(groups.get(key) ?? []), lens]);
  }
  return [...groups.entries()]
    .map(([family, items]) => ({ family, label: familyLabel(family), lenses: items }))
    .sort((left, right) => left.label.localeCompare(right.label));
}

function lensDisplayName(lens: DatasetLens): string {
  return lens.name || String(lens.artifact?.name ?? lens.artifactId);
}

function familyLabel(artifactType: string): string {
  const labels: Record<string, string> = {
    activation_cluster: "Activation Cluster",
    attention_edge: "Attention Edge",
    attention_map: "Attention Map",
    contrast_direction: "Contrast Direction",
    crosscoder_feature: "Crosscoder",
    probe_suite: "Probe",
    sae_feature: "SAE",
    transcoder_feature: "Transcoder",
  };
  return labels[artifactType] ?? artifactType.replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function discoveryEpisodePayload(value: unknown): DiscoveryArtifactEpisodesResponse | undefined {
  if (
    value &&
    typeof value === "object" &&
    "artifact" in value &&
    "family" in value &&
    "available" in value
  ) {
    return value as DiscoveryArtifactEpisodesResponse;
  }
  return undefined;
}

function ProbeEpisodeBadge({
  cue,
  record,
}: {
  cue?: ReturnType<typeof probeEvidenceCueForEpisode>;
  record?: ProbeEpisodeIndex;
}) {
  if (!record) {
    return (
      <span className="probe-episode-badge muted">
        <strong>Not scored</strong>
        <small>{cue?.markerLabel ?? "No probe row"}</small>
        {cue ? <ProbeTimelineCue cue={cue} /> : null}
      </span>
    );
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
      {cue ? (
        <>
          <small>{cue.markerLabel} · {cue.scoreLabel}</small>
          <ProbeTimelineCue cue={cue} />
        </>
      ) : null}
    </span>
  );
}

function ProbeTimelineCue({ cue }: { cue: NonNullable<ReturnType<typeof probeEvidenceCueForEpisode>> }) {
  return (
    <i className="probe-mini-timeline" aria-label={cue.markerLabel}>
      <b style={{ left: `${cue.timelinePercent ?? 50}%` }} />
    </i>
  );
}
