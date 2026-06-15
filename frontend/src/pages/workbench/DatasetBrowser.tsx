import { useDeferredValue, useMemo, useState, type CSSProperties, type PointerEvent as ReactPointerEvent } from "react";
import { useQuery } from "@tanstack/react-query";
import { ArrowRight, Search } from "lucide-react";
import { InfoIconTrigger, InlineInfoText } from "../../components/ui/InfoHover";
import { infoTextCard } from "../../components/ui/infoHoverModel";
import {
  fetchArtifacts,
  fetchDiscoveryArtifactFamilies,
  type DiscoveryArtifactEpisodeParams,
  type EpisodePageParams,
  type ProbeStudyEpisodeParams,
  fetchDiscoveryArtifactEpisodes,
  fetchDataset,
  fetchDatasetDiagnostics,
  fetchEpisodesPage,
  fetchProbeEvidenceBundle,
  fetchProbeIndex,
  fetchProbeStudies,
  fetchProbeStudyEpisodes,
} from "../../api/dataset";
import type {
  ArtifactRecord,
  DatasetEpisode,
  DiscoveryArtifactEpisodesResponse,
  DiscoveryArtifactFamily,
  EpisodeFacetValue,
  ProbeDatasetIndex,
  ProbeEpisodeIndex,
  ProbeStudy,
  ProbeStudyEpisodeSummary,
  ProbeStudyEpisodesResponse,
  ProbeStudyReadout,
} from "../../types/dataset";
import { researchCopy } from "../../copy/researchCopy";
import type { WorkbenchManifest } from "../../types/workbench";
import type { ProbeEvidenceBundle } from "../../types/probeEvidence";
import { datasetBrowserCapabilityGates } from "../capabilityGating";
import {
  COHORT_PRESETS,
  PROBE_PREDICTION_FILTER_LABELS,
  PROBE_PREDICTION_FILTERS,
  PROBE_READOUT_FILTER_LABELS,
  PROBE_READOUT_FILTER_MODES,
  PROBE_SPLIT_FILTER_LABELS,
  canonicalProbeSplitCategory,
  compactProbeMetricValue,
  compactProbeLayerLabel,
  compactProbeReadoutLabel,
  datasetCoverageRows,
  episodeBenchmark,
  episodeDatasetId,
  episodeOpenContextForProbe,
  episodeSeed,
  episodeTitle,
  formatDatasetProbeConfidence,
  percentOf,
  probeCalibrationRows,
  probeConfusionRows,
  probeEvidenceContextForEpisode,
  probeEvidenceCueForEpisode,
  probeEpisodeInspectionReason,
  probeFamilyHoverModel,
  probeLensWorkbenchModel,
  probeRecordForEpisode,
  probeReadoutHoverModel,
  probeResultChartRows,
  probeSplitBarSegments,
  probeSplitChartRows,
  probeSplitLabel,
  probeTargetDisplayLabel,
  shortTrace,
  filterProbeStudyReadouts,
  trainedProbeDisplayId,
  type ProbeCohortPreset,
  type ProbeCalibrationBucket,
  type ProbeConfusionRow,
  type ProbeLensSpec,
  type ProbeLensMetricChip,
  type ProbeMetadataCard,
  type ProbeReadoutFilterMode,
  type ProbeSplitChartRow,
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
  lensId: string;
  name: string;
  probe?: ProbeDatasetIndex;
  probeStudy?: ProbeStudy;
};

type ProbeSummarySplitRow = ProbeSplitChartRow;

type ProbeReadoutPerformanceRow = {
  detail: string;
  id: "test" | "train" | "validation";
  label: string;
  readout?: ProbeStudyReadout;
  score: number | null;
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
  const probeStudies = useQuery({
    queryKey: ["probe-studies", datasetIdentityKey],
    queryFn: fetchProbeStudies,
    enabled: dataset.isFetched && hasProbeArtifacts,
    staleTime: 60_000,
  });
  const probes = useMemo(
    () => (hasProbeArtifacts ? probeIndex.data?.probes ?? [] : []),
    [hasProbeArtifacts, probeIndex.data?.probes],
  );
  const studies = useMemo(
    () => (hasProbeArtifacts ? probeStudies.data?.studies ?? [] : []),
    [hasProbeArtifacts, probeStudies.data?.studies],
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
  const [selectedProbeReadoutId, setSelectedProbeReadoutId] = useState("");
  const [probeReadoutFilterMode, setProbeReadoutFilterMode] = useState<ProbeReadoutFilterMode>("useful");
  const [pageOffset, setPageOffset] = useState(0);
  const [probeLeftColumnWidth, setProbeLeftColumnWidth] = useState(940);
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
      studies,
    }),
    [artifactIndex.data?.artifacts, artifactsById, familyByType, studies],
  );
  const selectedLens = useMemo(
    () => lenses.find((lens) => lens.lensId === selectedLensId),
    [lenses, selectedLensId],
  );
  const selectedProbe = selectedLens?.probe;
  const activeProbeStudy = selectedLens?.probeStudy;
  const activeProbeReadouts = useMemo(
    () => activeProbeStudy?.readouts ?? [],
    [activeProbeStudy?.readouts],
  );
  const filteredProbeReadouts = useMemo(
    () => filterProbeStudyReadouts(activeProbeReadouts, probeReadoutFilterMode, activeProbeStudy),
    [activeProbeReadouts, activeProbeStudy, probeReadoutFilterMode],
  );
  const selectedProbeReadout = useMemo(
    () => selectedProbe
      ? filteredProbeReadouts.find((readout) => readout.readout_id === selectedProbeReadoutId)
        ?? filteredProbeReadouts[0]
      : undefined,
    [selectedProbe, selectedProbeReadoutId, filteredProbeReadouts],
  );
  const probePageStyle = selectedProbe
    ? ({ "--probe-left-width": `${probeLeftColumnWidth}px` } as CSSProperties)
    : undefined;
  const activeLensArtifactId = selectedLens?.artifactId ?? "";
  const canFetchProbeEvidenceBundle = Boolean(selectedProbe && !selectedLens?.probeStudy);
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
    enabled: dataset.isFetched && canFetchProbeEvidenceBundle,
    staleTime: 60_000,
  });
  const activeProbeEvidenceBundle = selectedProbe && canFetchProbeEvidenceBundle
    ? probeEvidenceBundle.data
    : undefined;
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
  const probeStudyEpisodeParams = useMemo<ProbeStudyEpisodeParams>(
    () => ({
      benchmark: benchmarkFilter,
      cohort_preset: probeCohortPreset,
      dataset_id: datasetFilter,
      layer: selectedProbeReadout?.layer,
      limit: 100,
      offset: pageOffset,
      outcome: outcomeFilter,
      prediction: probePredictionFilter,
      profile: profileFilter,
      q: deferredQuery,
      sort: sortMode === "lens_interest" ? "probe_interest" : sortMode,
      split: selectedProbeReadout?.split,
      task_id: taskFilter,
      target: selectedProbeReadout?.target,
    }),
    [
      benchmarkFilter,
      datasetFilter,
      deferredQuery,
      outcomeFilter,
      pageOffset,
      probeCohortPreset,
      probePredictionFilter,
      profileFilter,
      selectedProbeReadout,
      sortMode,
      taskFilter,
    ],
  );
  const useProbeReadoutEpisodes = Boolean(selectedProbe && selectedProbeReadout);
  const waitingForProbeReadouts = Boolean(selectedProbe && probeStudies.isLoading);
  const readoutFilterHasNoMatches = Boolean(
    selectedProbe && activeProbeStudy && activeProbeReadouts.length > 0 && !selectedProbeReadout,
  );
  const episodePage = useQuery({
    queryKey: [
      "episodes",
      datasetIdentityKey,
      activeLensArtifactId || NO_LENS_ID,
      selectedProbeReadout?.readout_id ?? "artifact",
      useProbeReadoutEpisodes ? probeStudyEpisodeParams : episodePageParams,
    ],
    queryFn: ({ signal }) => useProbeReadoutEpisodes
      ? fetchProbeStudyEpisodes(activeLensArtifactId, probeStudyEpisodeParams, signal)
      : selectedLens
      ? fetchDiscoveryArtifactEpisodes(activeLensArtifactId, episodePageParams, signal)
      : fetchEpisodesPage(episodePageParams as EpisodePageParams, signal),
    enabled: dataset.isFetched && !waitingForProbeReadouts && !readoutFilterHasNoMatches,
    staleTime: 15_000,
  });
  const discoveryPayload = discoveryEpisodePayload(episodePage.data);
  const probeReadoutPayload = probeStudyEpisodePayload(episodePage.data);
  const activeProbeReadoutPayload = probeReadoutPayloadMatches(probeReadoutPayload, selectedProbeReadout)
    ? probeReadoutPayload
    : undefined;
  const activeLensUnavailable = Boolean(selectedLens && discoveryPayload?.available === false);
  const activeLensReason = selectedLens ? discoveryPayload?.reason ?? "" : "";
  const activeReadoutReason = selectedProbe && activeProbeReadoutPayload?.available === false
    ? activeProbeReadoutPayload.reason
    : "";
  const activeProbeRowsAvailable = activeProbeReadoutPayload?.available === true;
  const activeLensFamilyLabel = selectedLens ? familyLabel(selectedLens.artifactType) : researchCopy.labels.episodeOrder;
  const diagnostics = useQuery({
    queryKey: ["dataset-diagnostics", datasetIdentityKey],
    queryFn: fetchDatasetDiagnostics,
    enabled: episodePage.isFetched || episodePage.isError,
    staleTime: 60_000,
  });
  const activeEpisodePageData = readoutFilterHasNoMatches
    ? undefined
    : useProbeReadoutEpisodes
    ? activeProbeReadoutPayload
    : episodePage.data;
  const episodes = useMemo(() => activeEpisodePageData?.episodes ?? [], [activeEpisodePageData?.episodes]);
  const datasetIds = facetValues(activeEpisodePageData?.facets.dataset_id);
  const benchmarks = facetValues(activeEpisodePageData?.facets.benchmark);
  const tasks = facetValues(activeEpisodePageData?.facets.task_id);
  const outcomes = facetValues(activeEpisodePageData?.facets.outcome);
  const profiles = facetValues(activeEpisodePageData?.facets.profile);
  const coverageRows = useMemo(() => datasetCoverageRows(episodes), [episodes]);
  const probeWorkbench = useMemo(
    () => selectedProbe
      ? probeLensWorkbenchModel({
          artifact: selectedLens?.probeStudy ? undefined : selectedLens?.artifact,
          bundle: activeProbeEvidenceBundle,
          probe: selectedProbe,
          totalEpisodes: dataset.data?.episode_count ?? activeEpisodePageData?.total ?? 0,
        })
      : undefined,
    [
      activeProbeEvidenceBundle,
      activeEpisodePageData?.total,
      dataset.data?.episode_count,
      selectedLens?.artifact,
      selectedLens?.probeStudy,
      selectedProbe,
    ],
  );
  const probeAnalysis = useMemo(
    () => selectedProbe
      ? probeDatasetAnalysisModel(
          selectedProbe,
          episodes,
          activeProbeEvidenceBundle,
          datasetFilter,
          useProbeReadoutEpisodes ? readoutRecordsForEpisodes(episodes) : undefined,
        )
      : undefined,
    [activeProbeEvidenceBundle, datasetFilter, episodes, selectedProbe, useProbeReadoutEpisodes],
  );
  const totalEpisodes = dataset.data?.episode_count ?? activeEpisodePageData?.total ?? 0;
  const visibleTotal = activeEpisodePageData?.total ?? 0;
  const pageStart = visibleTotal ? pageOffset + 1 : 0;
  const pageEnd = Math.min(pageOffset + (activeEpisodePageData?.limit ?? 100), visibleTotal);
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
    setSelectedProbeReadoutId("");
    setProbeReadoutFilterMode("useful");
    resetPage();
  };
  const openDatasetEpisode = (episode: DatasetEpisode) => {
    const evidenceContext = probeEvidenceContextForEpisode(
      selectedProbe,
      activeProbeEvidenceBundle,
      episode,
      datasetFilter,
    );
    if (evidenceContext?.researchSelection) {
      onOpenEpisode(episode.trace_id, evidenceContext);
      return;
    }
    onOpenEpisode(episode.trace_id, episodeOpenContextForProbe(selectedProbe, episode));
  };
  const resizeProbeColumns = (event: ReactPointerEvent<HTMLButtonElement>) => {
    event.preventDefault();
    const startX = event.clientX;
    const startWidth = probeLeftColumnWidth;
    const onPointerMove = (moveEvent: PointerEvent) => {
      setProbeLeftColumnWidth(Math.max(620, Math.min(1280, startWidth + moveEvent.clientX - startX)));
    };
    const onPointerUp = () => {
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
    };
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
  };

  return (
    <main className={`dataset-browser-page ${selectedProbe ? "probe-mode" : ""}`} style={probePageStyle}>
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
          selectedLensId={selectedLens?.lensId ?? NO_LENS_ID}
          onLensChange={selectLens}
        />
        {selectedLens && !selectedProbe ? <span>{activeLensFamilyLabel}</span> : null}
      </section>

      {selectedProbe && probeWorkbench ? (
        <ProbeLensWorkbench
          activePredictionFilter={probePredictionFilter}
          activeSplitFilter={selectedProbeReadout?.split_category ?? probeSplitFilter}
          readout={selectedProbeReadout}
          readoutEpisodeSummary={activeProbeRowsAvailable ? activeProbeReadoutPayload?.summary : undefined}
          readoutEpisodeTotal={activeProbeRowsAvailable ? visibleTotal : undefined}
          readoutUnavailableReason={activeReadoutReason}
          readouts={activeProbeReadouts}
          model={probeWorkbench}
          probe={selectedProbe}
          study={activeProbeStudy}
          onCohortPresetChange={(preset) => {
            setProbeCohortPreset(preset);
            setProbeSplitFilter("all");
            setProbePredictionFilter("all");
            resetPage();
          }}
          onPredictionFilterChange={(value) => {
            setProbeCohortPreset("all");
            setProbePredictionFilter(value);
            resetPage();
          }}
          onSplitFilterChange={(value) => {
            setProbeCohortPreset("all");
            let readoutFilterMode = probeReadoutFilterMode;
            if (value === "test" || value === "validation" || value === "train") {
              readoutFilterMode = value;
            }
            const splitReadouts = filterProbeStudyReadouts(
              activeProbeReadouts,
              readoutFilterMode,
              activeProbeStudy,
            );
            const nextReadout = selectedProbeReadout
              ? probeReadoutForSplitCategory(splitReadouts, selectedProbeReadout, value)
              : undefined;
            if (nextReadout) {
              const nextSplitFilter = canonicalProbeSplitCategory(nextReadout.split_category || nextReadout.split);
              setSelectedProbeReadoutId(nextReadout.readout_id);
              setProbeSplitFilter(nextReadout.split_category ?? "all");
              if (nextSplitFilter === "test" || nextSplitFilter === "validation" || nextSplitFilter === "train") {
                setProbeReadoutFilterMode(nextSplitFilter);
              }
            } else {
              setProbeSplitFilter(value);
            }
            resetPage();
          }}
        />
      ) : selectedLens ? (
        <LensEvidencePanel lens={selectedLens} ranking={discoveryPayload} />
      ) : null}

      {selectedProbe && activeProbeStudy ? (
        <ProbeReadoutNavigator
          readouts={filteredProbeReadouts}
          selectedReadout={selectedProbeReadout}
          filterMode={probeReadoutFilterMode}
          study={activeProbeStudy}
          totalReadoutCount={activeProbeReadouts.length}
          unavailableReason={activeReadoutReason}
          onReadoutChange={(readoutId) => {
            const nextReadout = activeProbeReadouts.find((readout) => readout.readout_id === readoutId);
            setSelectedProbeReadoutId(readoutId);
            setProbeSplitFilter(nextReadout?.split_category ?? "all");
            setProbeCohortPreset("all");
            resetPage();
          }}
          onFilterChange={(filterMode) => {
            const nextReadouts = filterProbeStudyReadouts(activeProbeReadouts, filterMode, activeProbeStudy);
            const nextReadout = nextReadouts.find((readout) => readout.readout_id === selectedProbeReadoutId)
              ?? nextReadouts[0];
            setProbeReadoutFilterMode(filterMode);
            if (nextReadout) {
              setProbeSplitFilter(nextReadout.split_category ?? "all");
              setSelectedProbeReadoutId(nextReadout.readout_id);
            }
            setProbeCohortPreset("all");
            resetPage();
          }}
        />
      ) : null}

      <section className={`dataset-table-controls ${selectedProbe ? "probe-mode" : ""}`} aria-label="Episode filters and sorting">
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
        {selectedProbe ? (
          <>
            <FilterSelect
              label="Review"
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
              label="Result"
              value={probePredictionFilter}
              values={PROBE_PREDICTION_FILTERS}
              labels={PROBE_PREDICTION_FILTER_LABELS}
              onChange={(value) => {
                setProbeCohortPreset("all");
                setProbePredictionFilter(value);
                resetPage();
              }}
            />
            <details className="dataset-refine-cohort">
              <summary>{researchCopy.labels.refineEpisodes}</summary>
              <div>
                <FilterSelect
                  includeAll={false}
                  label="Sort"
                  value={sortMode}
                  values={SORT_OPTIONS}
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
              </div>
            </details>
          </>
        ) : (
          <>
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
          </>
        )}
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
            ? "Lens ranking API unavailable. Restart the Python dashboard server so it picks up the episode ranking endpoint."
            : "Episode list unavailable."}
        </div>
      ) : null}
      {activeLensUnavailable ? (
        <div className="empty-state compact">
          {activeLensReason || "This lens cannot rank episodes yet."}
        </div>
      ) : null}
      {episodePage.isFetching ? <div className="app-message compact">Loading episodes...</div> : null}
      {hasProbeArtifacts && probeIndex.isError ? (
        <div className="empty-state compact">Probe list unavailable.</div>
      ) : null}
      {selectedProbe ? (
        <button
          className="probe-column-resizer"
          type="button"
          aria-label="Resize probe dataset columns"
          onPointerDown={resizeProbeColumns}
        >
          <span />
        </button>
      ) : null}

      <section className={`dataset-browser-grid ${selectedProbe ? "probe-analysis-grid" : ""}`}>
        <div className="dataset-browser-panel">
          <header>
            <h2>{selectedProbe ? "Episodes to inspect" : selectedLens ? "Episodes through lens" : "Episodes"}</h2>
            <span>
              {pageStart}-{pageEnd} / {visibleTotal}
            </span>
          </header>
          <div className="dataset-episode-table-wrap">
            <table className={`compact-table dataset-episode-table ${selectedProbe ? "probe-evidence-table" : ""}`}>
              <thead>
                {selectedProbe ? (
                  <tr>
                    <th>Episode</th>
                    <th>Probe result</th>
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
              disabled={activeEpisodePageData?.next_offset === null || activeEpisodePageData?.next_offset === undefined}
              type="button"
              onClick={() => setPageOffset(activeEpisodePageData?.next_offset ?? pageOffset)}
            >
              Next
            </button>
          </div>
        </div>

        {selectedProbe && probeAnalysis ? <ProbeDatasetAnalysisPanel model={probeAnalysis} /> : null}

        <details className="dataset-details-drawer">
          <summary>
            Dataset details
            <span>{totalEpisodes} episodes · {probes.length} probes · {dataset.data?.activation_sites ?? 0} {researchCopy.labels.activationSources}</span>
          </summary>
          <div className="dataset-browser-metrics">
            <Metric label="Episodes" value={totalEpisodes} />
            <Metric label="Visible" value={visibleTotal} />
            <Metric label="Datasets" value={datasetIds.length} />
            <Metric label="Benchmarks" value={benchmarks.length} />
            <Metric label="Tasks" value={tasks.length} />
            <Metric label="Activation sources" value={dataset.data?.activation_sites ?? 0} />
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
  model,
  probe,
  readout,
  readoutEpisodeSummary,
  readoutEpisodeTotal,
  readoutUnavailableReason,
  readouts,
  study,
  onCohortPresetChange,
  onPredictionFilterChange,
  onSplitFilterChange,
}: {
  activePredictionFilter: string;
  activeSplitFilter: string;
  model: ProbeLensWorkbenchViewModel;
  probe: ProbeDatasetIndex;
  readout?: ProbeStudyReadout;
  readoutEpisodeSummary?: ProbeStudyEpisodeSummary;
  readoutEpisodeTotal?: number;
  readoutUnavailableReason?: string;
  readouts?: ProbeStudyReadout[];
  study?: ProbeStudy;
  onCohortPresetChange: (preset: ProbeCohortPreset) => void;
  onPredictionFilterChange: (value: string) => void;
  onSplitFilterChange: (value: string) => void;
}) {
  const metricChips = readout
    ? probeReadoutMetricChips(readout, readoutEpisodeSummary, readoutEpisodeTotal)
    : model.metrics;
  const spec = study
    ? probeStudySpec(study, model)
    : model.spec;
  const verdict = readout
    ? probeReadoutVerdict(readout, readoutEpisodeSummary, readoutUnavailableReason)
    : model.verdict;
  const familyHover = study ? probeFamilyHoverModel(study) : undefined;
  return (
    <section className="probe-lens-workbench" aria-label="Selected probe lens workbench">
      <div className="probe-lens-head">
        {familyHover ? <InfoIconTrigger card={familyHover} className="info-hover-corner" /> : null}
        <span>{study ? "Selected probe family" : "Selected probe lens"}</span>
        <h2>{study?.name ?? model.title}</h2>
        <p>{study?.question_label || verdict.detail}</p>
        <div className="probe-lens-specs">
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
      </div>
      <aside className={`probe-lens-verdict ${verdict.tone}`}>
        <span>{verdict.label}</span>
        <strong>{verdict.headline}</strong>
        {verdict.detail ? <small>{verdict.detail}</small> : null}
        <div className="probe-lens-metrics">
          {metricChips.map((metric) => (
            <div className={`probe-lens-metric ${metric.tone}`} key={metric.label}>
              <span><InlineInfoText card={metricHoverModel(metric)} label={metric.label} /></span>
              <strong>{metric.value}</strong>
            </div>
          ))}
        </div>
      </aside>
      <ProbeSummaryVisual
        activePredictionFilter={activePredictionFilter}
        activeSplitFilter={activeSplitFilter}
        probe={probe}
        readout={readout}
        readoutEpisodeSummary={readoutEpisodeSummary}
        readoutUnavailableReason={readoutUnavailableReason}
        readouts={readouts}
        onCohortPresetChange={onCohortPresetChange}
        onPredictionFilterChange={onPredictionFilterChange}
        onSplitFilterChange={onSplitFilterChange}
      />
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
  const record = probeRecordForEpisode(probe, episode);
  const reason = probeEpisodeInspectionReason(probe, bundle, episode, datasetFilter);
  const meta = [
    episode.task_id ? `Task ${episode.task_id}` : "",
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
        <ProbeEpisodeBadge record={record} />
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
  const selectorLabel = lenses.length > 0 && lenses.every((lens) => lens.artifactType === "probe_suite")
    ? "Probe family"
    : "Lens artifact";
  return (
    <section className="dataset-lens-strip" aria-label="Dataset lens">
      <label className="dataset-lens-select">
        <span>{selectorLabel}</span>
        <select value={selectedLensId} onChange={(event) => onLensChange(event.target.value)}>
          <option value={NO_LENS_ID}>None - {researchCopy.labels.episodeOrder}</option>
          {groups.map((group) => (
            <optgroup key={group.family} label={group.label}>
              {group.lenses.map((lens) => (
                <option key={lens.lensId} value={lens.lensId}>
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

function ProbeReadoutNavigator({
  readouts,
  selectedReadout,
  filterMode,
  study,
  totalReadoutCount,
  unavailableReason,
  onReadoutChange,
  onFilterChange,
}: {
  readouts: ProbeStudyReadout[];
  selectedReadout?: ProbeStudyReadout;
  filterMode: ProbeReadoutFilterMode;
  study: ProbeStudy;
  totalReadoutCount: number;
  unavailableReason?: string;
  onReadoutChange: (readoutId: string) => void;
  onFilterChange: (filterMode: ProbeReadoutFilterMode) => void;
}) {
  if (!totalReadoutCount) {
    return (
      <section className="probe-readout-navigator" aria-label="Trained probe scope">
        <div>
          <span>Trained probe</span>
          <strong>No trained probes</strong>
        </div>
      </section>
    );
  }
  const activeReadout = selectedReadout ?? readouts[0];
  const readoutHover = activeReadout ? probeReadoutHoverModel(activeReadout, study) : undefined;
  const warningHover = activeReadout && unavailableReason
    ? readoutWarningHoverModel(activeReadout, study, unavailableReason)
    : undefined;
  return (
    <section className="probe-readout-navigator" aria-label="Trained probe scope">
      <div className="probe-readout-controls">
        <label className="probe-readout-picker">
          <span>Trained probe</span>
          <select
            value={activeReadout?.readout_id ?? ""}
            onChange={(event) => onReadoutChange(event.target.value)}
            disabled={!readouts.length}
          >
            {readouts.length ? (
              readouts.map((readout) => (
                <option key={readout.readout_id} value={readout.readout_id}>
                  {compactProbeReadoutLabel(readout, study)}
                </option>
              ))
            ) : (
              <option value="">
                No trained probes match
              </option>
            )}
          </select>
        </label>
        <label className="probe-readout-filter">
          <span>Filter</span>
          <select
            value={filterMode}
            onChange={(event) => onFilterChange(event.target.value as ProbeReadoutFilterMode)}
          >
            {PROBE_READOUT_FILTER_MODES.map((mode) => (
              <option key={mode} value={mode}>
                {PROBE_READOUT_FILTER_LABELS[mode]}
              </option>
            ))}
          </select>
        </label>
      </div>
      {activeReadout ? (
        <>
          <dl
            className="probe-readout-scope-facts"
          >
            {readoutHover ? <InfoIconTrigger card={readoutHover} className="info-hover-corner" /> : null}
            <div>
              <dt>
                <InlineInfoText
                  card={infoTextCard("Target", "The label this trained probe predicts from activations.")}
                  label="Target"
                />
              </dt>
              <dd>{probeTargetDisplayLabel(activeReadout.target || study.target || "")}</dd>
            </div>
            <div>
              <dt>
                <InlineInfoText
                  card={infoTextCard("Layer", "The activation layer used as the probe input.")}
                  label="Layer"
                />
              </dt>
              <dd>{compactProbeLayerLabel(activeReadout.layer)}</dd>
            </div>
            <div>
              <dt>
                <InlineInfoText
                  card={infoTextCard("Split", "The train, validation, or test split this trained probe was evaluated on.")}
                  label="Split"
                />
              </dt>
              <dd>{probeSplitLabel(activeReadout.split_category, activeReadout.split)}</dd>
            </div>
            <div>
              <dt>
                <InlineInfoText
                  card={infoTextCard("Rows", "Policy-call rows available to this trained probe before episode aggregation.")}
                  label="Rows"
                />
              </dt>
              <dd>{activeReadout.policy_call_count ?? activeReadout.row_count ?? "-"}</dd>
            </div>
            <div>
              <dt>
                <InlineInfoText
                  card={infoTextCard("Balanced acc.", "Balanced accuracy for this target, layer, and split.")}
                  label="Balanced acc."
                />
              </dt>
              <dd>{compactProbeMetricValue("BA", activeReadout.balanced_accuracy)}</dd>
            </div>
            <div>
              <dt>
                <InlineInfoText
                  card={infoTextCard("ID", "Stable identifier for debugging or sharing this exact trained probe.")}
                  label="ID"
                />
              </dt>
              <dd><ProbeIdCopy value={trainedProbeDisplayId(activeReadout, study)} /></dd>
            </div>
          </dl>
          {warningHover ? (
            <small className="probe-readout-warning">
              Aggregate only
              <InfoIconTrigger card={warningHover} />
            </small>
          ) : null}
        </>
      ) : (
        <p className="probe-readout-empty">No trained probes match this filter.</p>
      )}
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
        <p>Choose a lens above to rank episodes.</p>
      </div>
    );
  }
  return (
    <div className="probe-evidence-empty">
      <span>{familyLabel(lens.artifactType)}</span>
      <strong>{lensDisplayName(lens)}</strong>
      <p>
        {ranking?.available === false
          ? ranking.reason || "Episode ranking is not available for this lens yet."
          : "Probe details are shown after selecting a probe."}
      </p>
    </div>
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
  const visibleDetail = detail && detail.trim().toLowerCase() !== value.trim().toLowerCase()
    ? detail
    : "";
  return (
    <div className="probe-evidence-fact">
      <span>{label}</span>
      <strong>{value}</strong>
      {visibleDetail ? <small>{visibleDetail}</small> : null}
    </div>
  );
}

function probeStudySpec(study: ProbeStudy, fallback: ProbeLensWorkbenchViewModel): ProbeLensSpec {
  return {
    input: {
      ...fallback.spec.input,
      value: study.input || fallback.spec.input.value,
    },
    objective: {
      ...fallback.spec.objective,
      value: study.objective || fallback.spec.objective.value,
    },
    output: {
      ...fallback.spec.output,
      value: study.output || fallback.spec.output.value,
    },
    prediction: {
      ...fallback.spec.prediction,
      detail: study.question_label || fallback.spec.prediction.detail,
      value: study.prediction || study.target || fallback.spec.prediction.value,
    },
  };
}

function ProbeSummaryVisual({
  activePredictionFilter,
  activeSplitFilter,
  probe,
  readout,
  readoutEpisodeSummary,
  readoutUnavailableReason,
  readouts,
  onCohortPresetChange,
  onPredictionFilterChange,
  onSplitFilterChange,
}: {
  activePredictionFilter: string;
  activeSplitFilter: string;
  probe: ProbeDatasetIndex;
  readout?: ProbeStudyReadout;
  readoutEpisodeSummary?: ProbeStudyEpisodeSummary;
  readoutUnavailableReason?: string;
  readouts?: ProbeStudyReadout[];
  onCohortPresetChange: (preset: ProbeCohortPreset) => void;
  onPredictionFilterChange: (value: string) => void;
  onSplitFilterChange: (value: string) => void;
}) {
  const splitRows: ProbeSummarySplitRow[] = readout
    ? probeReadoutSplitChartRows(readouts ?? [], readout, readoutEpisodeSummary)
    : probeSplitChartRows(probe);
  const showAggregatePerformance = Boolean(readout && !readoutEpisodeSummary);
  const performanceRows = readout
    ? probeReadoutPerformanceRows(readouts ?? [], readout)
    : [];
  const resultRows = readout && !showAggregatePerformance
    ? probeReadoutResultChartRows(readoutEpisodeSummary)
    : probeResultChartRows(probe);
  const splitSegments = splitRows.map((row) => [row.id, probeSplitBarSegments(row)] as const);
  const segmentsBySplit = new Map(splitSegments);
  const hasSplitErrorCounts = splitSegments.some(([, segments]) => segments.hasErrorCounts);
  const hasUnknownSplitRows = splitSegments.some(
    ([, segments]) => !segments.hasErrorCounts && segments.unknownCount > 0,
  );
  return (
    <section className="probe-evidence-plot probe-summary-visual" aria-label="Probe summary">
      <div className="probe-evidence-plot-column">
        <header>
          <span>{readout ? "Probe split rows" : researchCopy.labels.splitCoverage}</span>
          <small className="probe-map-legend">
            {hasSplitErrorCounts ? (
              <>
                <span className="correct">correct</span>
                <span className="wrong">
                  <InlineInfoText
                    card={infoTextCard("Other wrong", "Wrong rows that are not high-confidence wrong.")}
                    label="other wrong"
                  />
                </span>
                <span className="high-conf-wrong">
                  <InlineInfoText
                    card={infoTextCard("High-conf wrong", "Wrong rows where the probe was highly confident in the wrong prediction.")}
                    label="high-conf wrong"
                  />
                </span>
              </>
            ) : null}
            {hasUnknownSplitRows ? (
              <span className="unknown">
                <InlineInfoText
                  card={infoTextCard("Rows only", "Correct and wrong counts are unavailable for these aggregate split rows.")}
                  label="rows only"
                />
              </span>
            ) : null}
          </small>
        </header>
        <div className="probe-split-bars">
          {splitRows.map((row) => {
            const segments = segmentsBySplit.get(row.id) ?? probeSplitBarSegments(row);
            return (
              <button
                className={activeSplitFilter === row.id ? "active" : ""}
                key={row.id}
                type="button"
                onClick={() => onSplitFilterChange(activeSplitFilter === row.id ? "all" : row.id)}
              >
                <span>{row.label}</span>
                <strong>{probeSplitRowCountLabel(row, segments.hasErrorCounts)}</strong>
                <i
                  className={`probe-evidence-bar segmented${segments.hasErrorCounts ? "" : " unknown"}`}
                  title={probeSplitCountDetail(row)}
                >
                  <b
                    className={segments.hasErrorCounts && segments.correctCount > 0 ? "visible-segment" : undefined}
                    style={{ flexBasis: `${segments.correctWidth}%`, width: `${segments.correctWidth}%` }}
                  />
                  <em
                    className={`wrong${segments.wrongOnlyCount > 0 ? " visible-segment" : ""}`}
                    style={{ flexBasis: `${segments.wrongWidth}%`, width: `${segments.wrongWidth}%` }}
                  />
                  <em
                    className={`high-conf-wrong${segments.highConfWrongCount > 0 ? " visible-segment" : ""}`}
                    style={{ flexBasis: `${segments.highConfWrongWidth}%`, width: `${segments.highConfWrongWidth}%` }}
                  />
                  {!segments.hasErrorCounts ? (
                    <b
                      className={`unknown${segments.unknownCount > 0 ? " visible-segment" : ""}`}
                      style={{ flexBasis: `${segments.unknownWidth}%`, width: `${segments.unknownWidth}%` }}
                    />
                  ) : null}
                </i>
                <small>
                  {row.detail
                    ? row.detail
                    : row.total === 0
                    ? "no episodes"
                    : row.wrong === null
                    ? "error counts unavailable"
                    : probeSplitCountDetail(row)}
                </small>
              </button>
            );
          })}
        </div>
      </div>
      <div className="probe-evidence-plot-column">
        <header>
          <span>
            {showAggregatePerformance
              ? "Probe performance"
              : readout
              ? "Probe results"
              : researchCopy.labels.resultCoverage}
          </span>
          <small>{showAggregatePerformance ? "aggregate metrics" : readout ? "visible episodes" : "episode counts"}</small>
        </header>
        {showAggregatePerformance ? (
          <>
            {readoutUnavailableReason ? (
              <small className="probe-aggregate-note">{humanizeProbeReadoutReason(readoutUnavailableReason)}</small>
            ) : null}
            <div className="probe-split-bars readout-performance-bars">
              {performanceRows.map((row) => (
                <button
                  className={activeSplitFilter === row.id ? "active" : ""}
                  disabled={!row.readout}
                  key={row.id}
                  type="button"
                  onClick={() => onSplitFilterChange(activeSplitFilter === row.id ? "all" : row.id)}
                >
                  <span>{row.label}</span>
                  <strong>{row.score === null ? "-" : row.score.toFixed(3)}</strong>
                  <i className="probe-evidence-bar single">
                    <b style={{ width: `${percentOf(row.score ?? 0, 1)}%` }} />
                  </i>
                  <small>{row.detail}</small>
                </button>
              ))}
            </div>
          </>
        ) : (
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
        )}
      </div>
    </section>
  );
}

type ProbeAnalysisCountRow = {
  accuracy: number | null;
  correct: number;
  highConfWrong: number;
  label: string;
  scored: number;
  total: number;
  unscored: number;
  unknown: number;
  wrong: number;
};

type ProbeConfidenceBucket = {
  correct: number;
  highConfWrong: number;
  label: string;
  total: number;
  unknown: number;
  wrong: number;
};

type ProbeScatterPoint = {
  confidence: number;
  episodeLength: number;
  label: string;
  tone: "correct" | "wrong" | "unknown";
  x: number;
  y: number;
};

type ProbeRollingPoint = {
  accuracy: number;
  correct: number;
  index: number;
  label: string;
  scored: number;
  wrong: number;
};

type ProbeTemporalRow = {
  label: string;
  marker: string;
  position: number;
  score: string;
  tone: "selected" | "warning" | "muted";
};

type ProbeCohortRow = {
  detail: string;
  label: string;
  tone: "correct" | "wrong" | "muted" | "warning";
};

type ProbeCohortGroup = {
  label: string;
  rows: ProbeCohortRow[];
};

type ProbeDatasetAnalysisModel = {
  calibrationRows: ProbeCalibrationBucket[];
  confidenceBuckets: ProbeConfidenceBucket[];
  cohortGroups: ProbeCohortGroup[];
  confusionRows: ProbeConfusionRow[];
  lengthScorePoints: ProbeScatterPoint[];
  outcomeRows: ProbeAnalysisCountRow[];
  rollingAccuracy: ProbeRollingPoint[];
  taskRows: ProbeAnalysisCountRow[];
  temporalRows: ProbeTemporalRow[];
};

function ProbeDatasetAnalysisPanel({ model }: { model: ProbeDatasetAnalysisModel }) {
  return (
    <aside className="probe-analysis-panel" aria-label="Probe dataset analysis">
      <section className="probe-analysis-card">
        <InfoIconTrigger
          card={infoTextCard("Score distribution", "Confidence bins for scored probe rows. Green is correct, orange is wrong, gray is unknown.")}
          className="info-hover-corner"
        />
        <header>
          <span>Score distribution</span>
          <small>confidence bins</small>
        </header>
        <div className="probe-analysis-bars">
          {model.confidenceBuckets.map((bucket) => (
            <div
              className="probe-analysis-bar-row"
              key={bucket.label}
            >
              <InlineInfoText card={confidenceBucketTooltip(bucket)} label={bucket.label} />
              <i>
                <b className="correct" style={{ width: `${percentOf(bucket.correct, bucket.total)}%` }} />
                <b className="wrong" style={{ width: `${percentOf(bucket.wrong, bucket.total)}%` }} />
                <b className="unknown" style={{ width: `${percentOf(bucket.unknown, bucket.total)}%` }} />
              </i>
              <strong>{bucket.total}</strong>
            </div>
          ))}
        </div>
      </section>

      {model.rollingAccuracy.length ? (
        <section className="probe-analysis-card">
          <InfoIconTrigger
            card={infoTextCard("Accuracy over episode order", "Rolling probe accuracy over the visible scored episode order. Bars summarize recent correct vs wrong probe results.")}
            className="info-hover-corner"
          />
          <header>
            <span>Accuracy over episode order</span>
            <small>rolling window</small>
          </header>
          <div className="probe-rolling-plot" aria-label="Rolling probe accuracy">
            {model.rollingAccuracy.map((point) => (
              <i
                key={`${point.label}-${point.index}`}
              >
                <b style={{ height: `${percentOf(point.accuracy, 1)}%` }} />
              </i>
            ))}
          </div>
        </section>
      ) : null}

      {model.calibrationRows.some((row) => row.total > 0) ? (
        <section className="probe-analysis-card">
          <InfoIconTrigger
            card={infoTextCard("Calibration", "Compares observed accuracy with probe confidence in each confidence bin.")}
            className="info-hover-corner"
          />
          <header>
            <span>Calibration</span>
            <small>accuracy by confidence</small>
          </header>
          <div className="probe-calibration-list">
            {model.calibrationRows.map((row) => (
              <div
                className="probe-calibration-row"
                key={row.label}
              >
                <InlineInfoText card={calibrationRowTooltip(row)} label={row.label} />
                <i>
                  <b className="accuracy" style={{ width: `${percentOf(row.accuracy ?? 0, 1)}%` }} />
                  <em style={{ left: `${percentOf(row.avgConfidence ?? 0, 1)}%` }} />
                </i>
                <strong>{row.accuracy === null ? "-" : `${Math.round(row.accuracy * 100)}%`}</strong>
              </div>
            ))}
          </div>
        </section>
      ) : null}

      {model.lengthScorePoints.length ? (
        <section className="probe-analysis-card">
          <InfoIconTrigger
            card={infoTextCard("Confidence vs episode length", "Each dot is a visible scored episode. Horizontal position is episode length; vertical position is probe confidence.")}
            className="info-hover-corner"
          />
          <header>
            <span>Confidence vs episode length</span>
            <small>visible scored episodes</small>
          </header>
          <div className="probe-scatter-plot" aria-label="Confidence against episode length">
            {model.lengthScorePoints.map((point, index) => (
              <i
                className={point.tone}
                key={`${point.label}-${index}`}
                style={{ left: `${point.x}%`, bottom: `${point.y}%` }}
              />
            ))}
          </div>
        </section>
      ) : null}

      <ProbeSliceCard title="Error by task" subtitle="visible episodes" rows={model.taskRows} />
      <ProbeSliceCard title="Error by outcome" subtitle="visible episodes" rows={model.outcomeRows} />

      {model.temporalRows.length ? (
        <section className="probe-analysis-card">
          <InfoIconTrigger
            card={infoTextCard("Temporal evidence", "Ranked probe evidence projected onto episode progress and policy-call position.")}
            className="info-hover-corner"
          />
          <header>
            <span>Temporal evidence</span>
            <small>ranked policy calls</small>
          </header>
          <div className="probe-temporal-map">
            {model.temporalRows.map((row, index) => (
              <div
                className={`probe-temporal-row ${row.tone}`}
                key={`${row.label}-${row.marker}-${index}`}
              >
                <InlineInfoText card={temporalRowTooltip(row)} label={row.label} />
                <i><b style={{ left: `${row.position}%` }} /></i>
                <strong>{row.marker}</strong>
              </div>
            ))}
          </div>
        </section>
      ) : null}

      {model.confusionRows.length ? (
        <section className="probe-analysis-card">
          <InfoIconTrigger
            card={infoTextCard("Prediction vs label", "Rows are predicted-label to true-label pairs. Green is correct, orange is wrong, gray is unknown.")}
            className="info-hover-corner"
          />
          <header>
            <span>Prediction vs label</span>
            <small>all indexed records</small>
          </header>
          <div className="probe-confusion-list">
            {model.confusionRows.map((row) => (
              <div
                key={row.label}
              >
                <InlineInfoText card={confusionRowTooltip(row)} label={row.label} />
                <i>
                  <b className="correct" style={{ width: `${percentOf(row.correct, row.total)}%` }} />
                  <b className="wrong" style={{ width: `${percentOf(row.wrong, row.total)}%` }} />
                  <b className="unknown" style={{ width: `${percentOf(row.unknown, row.total)}%` }} />
                </i>
                <strong>{row.total}</strong>
              </div>
            ))}
          </div>
        </section>
      ) : null}

      {model.cohortGroups.length ? (
        <section className="probe-analysis-card">
          <header>
            <span>Review cohorts</span>
            <small>visible episodes</small>
          </header>
          <div className="probe-cohort-grid">
            {model.cohortGroups.map((group) => (
              <section key={group.label}>
                <span>{group.label}</span>
                {group.rows.map((row, index) => (
                  <div className={`probe-cohort-row ${row.tone}`} key={`${group.label}-${row.label}-${index}`}>
                    <strong>{row.label}</strong>
                    <small>{row.detail}</small>
                  </div>
                ))}
              </section>
            ))}
          </div>
        </section>
      ) : null}
    </aside>
  );
}

function ProbeSliceCard({
  rows,
  subtitle,
  title,
}: {
  rows: ProbeAnalysisCountRow[];
  subtitle: string;
  title: string;
}) {
  return (
    <section className="probe-analysis-card">
      <InfoIconTrigger
        card={infoTextCard(title, "Visible episodes grouped by this slice. Bars separate correct, wrong, high-confidence wrong, and unknown labels.")}
        className="info-hover-corner"
      />
      <header>
        <span>{title}</span>
        <small>{subtitle}</small>
      </header>
      <div className="probe-slice-list">
        {rows.length ? rows.map((row, index) => (
          <div
            className="probe-slice-row"
            key={`${row.label}-${index}`}
          >
            <InlineInfoText card={sliceRowTooltip(row, title)} label={row.label} />
            <i>
              <b className="correct" style={{ width: `${percentOf(row.correct, row.total)}%` }} />
              <b className="wrong" style={{ width: `${percentOf(Math.max(0, row.wrong - row.highConfWrong), row.total)}%` }} />
              <b className="high-conf-wrong" style={{ width: `${percentOf(row.highConfWrong, row.total)}%` }} />
              <b className="unknown" style={{ width: `${percentOf(row.unknown, row.total)}%` }} />
            </i>
            <strong>{row.wrong}/{row.scored}</strong>
          </div>
        )) : <p>No visible scored episodes.</p>}
      </div>
    </section>
  );
}

function metricHoverModel(metric: ProbeLensMetricChip): ProbeMetadataCard {
  return {
    groups: [
      {
        lines: [
          { label: "Value", value: metric.value },
          ...(metric.detail ? [{ label: "Meaning", value: metric.detail }] : []),
        ],
        title: "Metric",
      },
    ],
    title: metric.label,
  };
}

function readoutWarningHoverModel(
  readout: ProbeStudyReadout,
  study: ProbeStudy,
  unavailableReason: string,
): ProbeMetadataCard {
  return {
    groups: [
      {
        lines: [
          { label: "Reason", value: humanizeProbeReadoutReason(unavailableReason) },
          { label: "Current readout", value: compactProbeReadoutLabel(readout, study) },
          { label: "Aggregate metrics", value: "still shown" },
        ],
        title: "Episode drilldown",
      },
    ],
    subtitle: "Episode table rows are unavailable for this trained probe.",
    title: "Aggregate only",
  };
}

function confidenceBucketTooltip(bucket: ProbeConfidenceBucket): ProbeMetadataCard {
  return {
    groups: [
      {
        lines: [
          { label: "Rows", value: formatReadoutInteger(bucket.total) },
          { label: "Correct", value: formatTooltipShare(bucket.correct, bucket.total) },
          { label: "Wrong", value: formatTooltipShare(bucket.wrong, bucket.total) },
          { label: "High-conf wrong", value: formatTooltipShare(bucket.highConfWrong, bucket.total) },
          { label: "Unknown labels", value: formatTooltipShare(bucket.unknown, bucket.total) },
        ],
        title: "Confidence bin",
      },
    ],
    title: `${bucket.label} confidence`,
  };
}

function calibrationRowTooltip(row: ProbeCalibrationBucket): ProbeMetadataCard {
  return {
    groups: [
      {
        lines: [
          { label: "Rows", value: formatReadoutInteger(row.total) },
          { label: "Correct", value: formatTooltipShare(row.correct, row.total) },
          { label: "Accuracy", value: row.accuracy === null ? "-" : `${Math.round(row.accuracy * 100)}%` },
          { label: "Avg confidence", value: row.avgConfidence === null ? "-" : row.avgConfidence.toFixed(3) },
        ],
        title: "Calibration",
      },
    ],
    title: `${row.label} confidence`,
  };
}

function temporalRowTooltip(row: ProbeTemporalRow): ProbeMetadataCard {
  return {
    groups: [
      {
        lines: [
          { label: "Marker", value: row.marker },
          { label: "Position", value: `${Math.round(row.position)}%` },
          { label: "Score", value: row.score },
          { label: "Episode", value: row.label },
        ],
        title: "Timeline",
      },
    ],
    title: row.label,
  };
}

function confusionRowTooltip(row: ProbeConfusionRow): ProbeMetadataCard {
  return {
    groups: [
      {
        lines: [
          { label: "Rows", value: formatReadoutInteger(row.total) },
          { label: "Correct", value: formatTooltipShare(row.correct, row.total) },
          { label: "Wrong", value: formatTooltipShare(row.wrong, row.total) },
          { label: "Unknown", value: formatTooltipShare(row.unknown, row.total) },
        ],
        title: "Prediction vs label",
      },
    ],
    title: row.label,
  };
}

function sliceRowTooltip(row: ProbeAnalysisCountRow, title: string): ProbeMetadataCard {
  const otherWrong = Math.max(0, row.wrong - row.highConfWrong);
  return {
    groups: [
      {
        lines: [
          { label: "Episodes", value: formatReadoutInteger(row.total) },
          { label: "Scored", value: formatTooltipShare(row.scored, row.total) },
          { label: "Unscored", value: formatTooltipShare(row.unscored, row.total) },
          { label: "Correct", value: formatTooltipShare(row.correct, row.scored) },
          { label: "Wrong", value: formatTooltipShare(otherWrong, row.scored) },
          { label: "High-conf wrong", value: formatTooltipShare(row.highConfWrong, row.scored) },
          { label: "Accuracy", value: row.accuracy === null ? "-" : `${Math.round(row.accuracy * 100)}%` },
        ],
        title: "Result",
      },
    ],
    title: `${title}: ${row.label}`,
  };
}

function formatTooltipShare(value: number, total: number): string {
  if (!total) {
    return formatReadoutInteger(value);
  }
  return `${formatReadoutInteger(value)} (${Math.round(percentOf(value, total))}%)`;
}

function probeDatasetAnalysisModel(
  probe: ProbeDatasetIndex,
  episodes: DatasetEpisode[],
  bundle: ProbeEvidenceBundle | undefined,
  datasetFilter: string,
  recordsOverride?: ProbeEpisodeIndex[],
): ProbeDatasetAnalysisModel {
  const records = recordsOverride ?? Object.values(probe.by_trace ?? {});
  return {
    calibrationRows: probeCalibrationRows(records),
    confidenceBuckets: confidenceBucketRows(records),
    cohortGroups: reviewCohortGroups(probe, episodes),
    confusionRows: probeConfusionRows(records),
    lengthScorePoints: scoreLengthPoints(probe, episodes),
    outcomeRows: sliceRowsForEpisodes(probe, episodes, (episode) => episode.outcome || "Outcome missing"),
    rollingAccuracy: rollingAccuracyRows(records),
    taskRows: sliceRowsForEpisodes(probe, episodes, (episode) => episode.task_id ? `Task ${episode.task_id}` : "Task missing"),
    temporalRows: temporalEvidenceRows(bundle, probe, episodes, datasetFilter),
  };
}

function readoutRecordsForEpisodes(episodes: DatasetEpisode[]): ProbeEpisodeIndex[] {
  return episodes
    .map((episode) => episode.probe_record)
    .filter((record): record is ProbeEpisodeIndex => Boolean(record));
}

function confidenceBucketRows(records: ProbeEpisodeIndex[]): ProbeConfidenceBucket[] {
  const buckets: ProbeConfidenceBucket[] = [
    { correct: 0, highConfWrong: 0, label: "0-.20", total: 0, unknown: 0, wrong: 0 },
    { correct: 0, highConfWrong: 0, label: ".20-.40", total: 0, unknown: 0, wrong: 0 },
    { correct: 0, highConfWrong: 0, label: ".40-.60", total: 0, unknown: 0, wrong: 0 },
    { correct: 0, highConfWrong: 0, label: ".60-.80", total: 0, unknown: 0, wrong: 0 },
    { correct: 0, highConfWrong: 0, label: ".80-1", total: 0, unknown: 0, wrong: 0 },
  ];
  for (const record of records) {
    if (!record.available || typeof record.confidence !== "number" || !Number.isFinite(record.confidence)) {
      continue;
    }
    const index = Math.max(0, Math.min(4, Math.floor(record.confidence * 5)));
    const bucket = buckets[index];
    bucket.total += 1;
    if (record.correct === true) bucket.correct += 1;
    else if (record.correct === false) bucket.wrong += 1;
    else bucket.unknown += 1;
    if (record.correct === false && record.confidence >= 0.8) bucket.highConfWrong += 1;
  }
  return buckets;
}

function rollingAccuracyRows(records: ProbeEpisodeIndex[]): ProbeRollingPoint[] {
  const scored = records
    .filter((record) => record.available && record.correct !== null && record.correct !== undefined)
    .sort((left, right) => left.trace_id.localeCompare(right.trace_id));
  if (!scored.length) {
    return [];
  }
  const windowSize = Math.max(5, Math.min(25, Math.round(scored.length / 20)));
  const points = scored.map((record, index) => {
    const window = scored.slice(Math.max(0, index - windowSize + 1), index + 1);
    const correct = window.filter((item) => item.correct === true).length;
    const wrong = window.filter((item) => item.correct === false).length;
    const scoredCount = correct + wrong;
    return {
      accuracy: scoredCount ? correct / scoredCount : 0,
      correct,
      index,
      label: record.trace_id,
      scored: scoredCount,
      wrong,
    };
  });
  if (points.length <= 64) {
    return points;
  }
  const stride = Math.ceil(points.length / 64);
  return points.filter((_point, index) => index % stride === 0 || index === points.length - 1);
}

function scoreLengthPoints(probe: ProbeDatasetIndex, episodes: DatasetEpisode[]): ProbeScatterPoint[] {
  const points = episodes
    .map((episode) => {
      const record = probeRecordForEpisode(probe, episode);
      if (
        !record?.available ||
        typeof record.confidence !== "number" ||
        !Number.isFinite(record.confidence) ||
        typeof episode.length !== "number" ||
        !Number.isFinite(episode.length)
      ) {
        return null;
      }
      return { episode, record };
    })
    .filter((item): item is { episode: DatasetEpisode; record: ProbeEpisodeIndex } => Boolean(item));
  if (!points.length) {
    return [];
  }
  const lengths = points.map((point) => point.episode.length ?? 0);
  const minLength = Math.min(...lengths);
  const maxLength = Math.max(...lengths);
  const spread = Math.max(1, maxLength - minLength);
  const sampled = points.length <= 160
    ? points
    : points.filter((_point, index) => index % Math.ceil(points.length / 160) === 0);
  return sampled.map(({ episode, record }) => ({
    confidence: record.confidence ?? 0,
    episodeLength: episode.length ?? 0,
    label: episodeTitle(episode),
    tone: record.correct === true ? "correct" : record.correct === false ? "wrong" : "unknown",
    x: percentOf((episode.length ?? 0) - minLength, spread),
    y: percentOf(record.confidence ?? 0, 1),
  }));
}

function temporalEvidenceRows(
  bundle: ProbeEvidenceBundle | undefined,
  probe: ProbeDatasetIndex,
  episodes: DatasetEpisode[],
  datasetFilter: string,
): ProbeTemporalRow[] {
  if (!bundle) {
    return [];
  }
  return episodes
    .map((episode) => {
      const cue = probeEvidenceCueForEpisode(bundle, episode, probe, datasetFilter);
      if (!cue || cue.timelinePercent === null) {
        return null;
      }
      const lowerMarker = cue.markerLabel.toLowerCase();
      return {
        label: episodeTitle(episode),
        marker: cue.markerLabel,
        position: cue.timelinePercent,
        score: cue.scoreLabel,
        tone: lowerMarker.includes("uncertain")
          ? "warning"
          : lowerMarker.includes("bottom")
            ? "muted"
            : "selected",
      };
    })
    .filter((row): row is ProbeTemporalRow => Boolean(row))
    .slice(0, 14);
}

function sliceRowsForEpisodes(
  probe: ProbeDatasetIndex,
  episodes: DatasetEpisode[],
  labelForEpisode: (episode: DatasetEpisode) => string,
): ProbeAnalysisCountRow[] {
  const rows = new Map<string, ProbeAnalysisCountRow>();
  for (const episode of episodes) {
    const label = labelForEpisode(episode);
    const row = rows.get(label) ?? emptyAnalysisRow(label);
    incrementAnalysisRow(row, probeRecordForEpisode(probe, episode));
    rows.set(label, row);
  }
  return [...rows.values()]
    .map(withAccuracy)
    .sort((left, right) =>
      right.highConfWrong - left.highConfWrong ||
      right.wrong - left.wrong ||
      right.scored - left.scored ||
      right.total - left.total ||
      left.label.localeCompare(right.label),
    )
    .slice(0, 8);
}

function reviewCohortGroups(probe: ProbeDatasetIndex, episodes: DatasetEpisode[]): ProbeCohortGroup[] {
  const groups: ProbeCohortGroup[] = [
    { label: "High-conf wrong", rows: [] },
    { label: "Wrong", rows: [] },
    { label: "Uncertain", rows: [] },
    { label: "Unscored heldout", rows: [] },
    { label: "Confident correct", rows: [] },
  ];
  for (const episode of episodes) {
    const record = probeRecordForEpisode(probe, episode);
    const confidence = typeof record?.confidence === "number" && Number.isFinite(record.confidence) ? record.confidence : null;
    const split = canonicalProbeSplitCategory(record?.split_category);
    const row = {
      detail: [
        episode.task_id ? `Task ${episode.task_id}` : "",
        confidence === null ? "" : formatDatasetProbeConfidence(confidence),
      ].filter(Boolean).join(" · "),
      label: episodeTitle(episode),
      tone: "muted" as ProbeCohortRow["tone"],
    };
    if (record?.correct === false && confidence !== null && confidence >= 0.8) {
      groups[0].rows.push({ ...row, tone: "wrong" });
    } else if (record?.correct === false) {
      groups[1].rows.push({ ...row, tone: "warning" });
    } else if (record?.available && (record.correct === null || record.correct === undefined || (confidence !== null && confidence >= 0.45 && confidence <= 0.65))) {
      groups[2].rows.push({ ...row, tone: "warning" });
    } else if (!record?.available && (split === "validation" || split === "test")) {
      groups[3].rows.push(row);
    } else if (record?.correct === true && confidence !== null && confidence >= 0.9) {
      groups[4].rows.push({ ...row, tone: "correct" });
    }
  }
  return groups
    .map((group) => ({ ...group, rows: group.rows.slice(0, 4) }))
    .filter((group) => group.rows.length);
}

function emptyAnalysisRow(label: string): ProbeAnalysisCountRow {
  return { accuracy: null, correct: 0, highConfWrong: 0, label, scored: 0, total: 0, unscored: 0, unknown: 0, wrong: 0 };
}

function incrementAnalysisRow(row: ProbeAnalysisCountRow, record?: ProbeEpisodeIndex) {
  row.total += 1;
  if (!record?.available) {
    row.unscored += 1;
    return;
  }
  row.scored += 1;
  if (record.correct === true) row.correct += 1;
  else if (record.correct === false) row.wrong += 1;
  else row.unknown += 1;
  if (record.correct === false && typeof record.confidence === "number" && record.confidence >= 0.8) {
    row.highConfWrong += 1;
  }
}

function withAccuracy(row: ProbeAnalysisCountRow): ProbeAnalysisCountRow {
  return {
    ...row,
    accuracy: row.correct + row.wrong ? row.correct / (row.correct + row.wrong) : null,
  };
}

export function ArtifactsSummary({ manifest }: { manifest: WorkbenchManifest }) {
  return (
    <main className="summary-page">
      <h1>Saved analyses</h1>
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
  studies,
}: {
  artifacts: ArtifactRecord[];
  artifactsById: Map<string, ArtifactRecord>;
  familyByType: Map<string, DiscoveryArtifactFamily>;
  studies: ProbeStudy[];
}): DatasetLens[] {
  const seenLensIds = new Set<string>();
  const seenArtifactIds = new Set<string>();
  const lenses: DatasetLens[] = [];
  for (const study of studies) {
    const artifactId = String(study.artifact_id ?? "");
    const lensId = probeStudyId(study);
    if (!artifactId || !lensId || seenLensIds.has(lensId)) {
      continue;
    }
    seenLensIds.add(lensId);
    seenArtifactIds.add(artifactId);
    lenses.push({
      artifact: artifactsById.get(artifactId),
      artifactId,
      artifactType: "probe_suite",
      family: familyByType.get("probe_suite"),
      lensId,
      name: study.name,
      probe: probeFromStudy(study),
      probeStudy: study,
    });
  }
  for (const artifact of artifacts) {
    const artifactId = String(artifact.artifact_id ?? "");
    const artifactType = String(artifact.artifact_type ?? "");
    if (!artifactId || seenArtifactIds.has(artifactId) || !familyByType.has(artifactType)) {
      continue;
    }
    seenArtifactIds.add(artifactId);
    lenses.push({
      artifact,
      artifactId,
      artifactType,
      family: familyByType.get(artifactType),
      lensId: artifactId,
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

function probeStudyId(study: ProbeStudy | undefined): string {
  return study?.study_id || study?.artifact_id || "";
}

function probeFromStudy(study: ProbeStudy): ProbeDatasetIndex {
  const bestReadout = defaultProbeReadout(study.readouts, study) ?? study.readouts[0];
  const splitSummary: Record<string, number> = {};
  for (const readout of study.readouts) {
    const split = readout.split_category || readout.split || "unknown";
    const count = readout.policy_call_count ?? readout.row_count ?? 0;
    splitSummary[split] = (splitSummary[split] ?? 0) + count;
  }
  return {
    artifact_id: study.artifact_id,
    best_feature: bestReadout
      ? `${probeReadoutLayerLabel(bestReadout.layer)} / ${probeReadoutSplitLabel(bestReadout)}`
      : null,
    best_model: study.input || null,
    best_score: typeof bestReadout?.balanced_accuracy === "number" ? bestReadout.balanced_accuracy : null,
    name: study.name,
    prediction_summary: {},
    split_summary: splitSummary,
    target: study.target,
  };
}

function familyLabel(artifactType: string): string {
  const labels: Record<string, string> = {
    activation_cluster: "Activation Cluster",
    attention_edge: "Attention Edge",
    attention_map: "Attention Map",
    contrast_direction: "Contrast Direction",
    crosscoder_feature: "Crosscoder",
    probe_suite: "Probe Family",
    sae_feature: "SAE",
    transcoder_feature: "Transcoder",
  };
  return labels[artifactType] ?? artifactType.replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function defaultProbeReadout(
  readouts: ProbeStudyReadout[],
  study?: ProbeStudy,
): ProbeStudyReadout | undefined {
  if (!readouts.length) {
    return undefined;
  }
  const primary = readouts.filter((readout) =>
    readout.is_primary_target || (study?.target && readout.target === study.target),
  );
  const candidates = primary.length ? primary : readouts;
  return candidates.find((readout) => readout.is_selected_layer && readout.is_test_split)
    ?? candidates.find((readout) => readout.is_selected_layer && readout.is_selection_split)
    ?? candidates.find((readout) => readout.is_test_split)
    ?? candidates[0];
}

function probeReadoutPayloadMatches(
  payload: ProbeStudyEpisodesResponse | undefined,
  readout: ProbeStudyReadout | undefined,
): boolean {
  if (!payload || !readout) {
    return false;
  }
  return String(payload.target ?? "") === String(readout.target ?? "")
    && probeReadoutLayerKey(payload.layer) === probeReadoutLayerKey(readout.layer)
    && String(payload.split ?? "") === String(readout.split ?? "");
}

function probeReadoutVerdict(
  readout: ProbeStudyReadout,
  summary?: ProbeStudyEpisodeSummary,
  unavailableReason?: string,
): ProbeLensWorkbenchViewModel["verdict"] {
  const score = readout.balanced_accuracy;
  const splitLabel = probeReadoutSplitLabel(readout);
  const scoreLabel = probeReadoutScoreLabel(readout);
  const availabilityDetail = unavailableReason
    ? humanizeProbeReadoutReason(unavailableReason)
    : summary
      ? `${formatReadoutInteger(summary.episode_count)} episode rows available for drilldown.`
      : "Aggregate metrics are available for this trained split.";
  if (typeof score !== "number" || !Number.isFinite(score)) {
    return {
      detail: availabilityDetail,
      headline: "Score unavailable",
      label: "Unknown",
      tone: "unknown",
    };
  }
  if (score < 0.35) {
    return {
      detail: `Balanced accuracy ${scoreLabel} on ${splitLabel}. ${availabilityDetail}`,
      headline: "Weak split readout",
      label: "Low score",
      tone: "danger",
    };
  }
  if (score < 0.7) {
    return {
      detail: `Balanced accuracy ${scoreLabel} on ${splitLabel}. ${availabilityDetail}`,
      headline: "Limited split readout",
      label: "Review",
      tone: "limited",
    };
  }
  return {
    detail: `Balanced accuracy ${scoreLabel} on ${splitLabel}. ${availabilityDetail}`,
    headline: "Split score available",
    label: "Usable",
    tone: "credible",
  };
}

function probeReadoutMetricChips(
  readout: ProbeStudyReadout,
  summary?: ProbeStudyEpisodeSummary,
  episodeTotal?: number,
): ProbeLensMetricChip[] {
  const score = readout.balanced_accuracy;
  const gap = readout.train_gap_balanced_accuracy;
  const episodeCount = summary?.episode_count ?? episodeTotal;
  return [
    {
      detail: `${probeReadoutLayerLabel(readout.layer)} · ${probeReadoutSplitLabel(readout)}`,
      label: "BA",
      tone: probeReadoutScoreTone(score),
      value: compactProbeMetricValue("", score).trim(),
    },
    {
      detail: "policy-call rows in this trained probe",
      label: "Rows",
      tone: readout.policy_call_count ? "credible" : "unknown",
      value: formatReadoutInteger(readout.policy_call_count ?? readout.row_count),
    },
    {
      detail: "unique episodes in the current trained-probe set",
      label: "Episodes",
      tone: episodeCount ? "credible" : "unknown",
      value: formatReadoutInteger(episodeCount),
    },
    {
      detail: "classes with labels for this target",
      label: "Classes",
      tone: readout.class_count ? "unknown" : "debug",
      value: formatReadoutInteger(readout.class_count),
    },
    {
      detail: "train balanced accuracy minus this split score",
      label: "Gap",
      tone: typeof gap === "number" && Number.isFinite(gap) && gap > 0.2 ? "limited" : "unknown",
      value: compactProbeMetricValue("", gap).trim(),
    },
  ];
}

function probeReadoutSplitChartRows(
  readouts: ProbeStudyReadout[],
  selectedReadout: ProbeStudyReadout,
  summary?: ProbeStudyEpisodeSummary,
): ProbeSummarySplitRow[] {
  const layer = probeReadoutLayerKey(selectedReadout.layer);
  const bySplit = new Map(
    readouts
      .filter((readout) =>
        readout.target === selectedReadout.target
        && probeReadoutLayerKey(readout.layer) === layer,
      )
      .map((readout) => [readout.split_category, readout]),
  );
  return (["train", "validation", "test"] as const).map((split) => {
    const readout = bySplit.get(split);
    const splitCounts = readout?.split ? summary?.split_counts?.[readout.split] : undefined;
    const total = splitCounts?.policy_call_count ?? readout?.policy_call_count ?? readout?.row_count ?? 0;
    const wrong = splitCounts?.wrong ?? null;
    const highConfWrong = splitCounts?.high_conf_wrong ?? null;
    const correct = splitCounts?.correct ?? Math.max(0, total - (wrong ?? 0));
    const wrongOnly = wrong === null ? null : Math.max(0, wrong - (highConfWrong ?? 0));
    return {
      detail: readout
        ? wrong === null
          ? `${formatReadoutInteger(total)} policy-call rows · bal acc ${probeReadoutScoreLabel(readout)}`
          : `${formatReadoutInteger(correct)} correct · ${formatReadoutInteger(
              wrongOnly,
            )} other wrong · ${formatReadoutInteger(highConfWrong)} high-conf wrong · bal acc ${probeReadoutScoreLabel(readout)}`
        : "no trained probe",
      highConfWrong,
      id: split,
      label: PROBE_SPLIT_FILTER_LABELS[split],
      scored: total,
      total,
      wrong,
    };
  });
}

function probeReadoutPerformanceRows(
  readouts: ProbeStudyReadout[],
  selectedReadout: ProbeStudyReadout,
): ProbeReadoutPerformanceRow[] {
  const layer = probeReadoutLayerKey(selectedReadout.layer);
  const bySplit = new Map(
    readouts
      .filter((readout) =>
        readout.target === selectedReadout.target
        && probeReadoutLayerKey(readout.layer) === layer,
      )
      .map((readout) => [readout.split_category, readout]),
  );
  return (["train", "validation", "test"] as const).map((split) => {
    const readout = bySplit.get(split);
    return {
      detail: readout
        ? `${formatReadoutInteger(readout.policy_call_count ?? readout.row_count)} policy calls · ${formatReadoutInteger(
            readout.class_count,
          )} labels · train gap ${formatReadoutMetric(readout.train_gap_balanced_accuracy)}`
        : "no trained probe",
      id: split,
      label: PROBE_SPLIT_FILTER_LABELS[split],
      readout,
      score: typeof readout?.balanced_accuracy === "number" && Number.isFinite(readout.balanced_accuracy)
        ? readout.balanced_accuracy
        : null,
    };
  });
}

function probeSplitCountDetail(row: ProbeSummarySplitRow): string {
  if (row.total === 0) {
    return "no episodes";
  }
  if (row.wrong === null) {
    return `${formatReadoutInteger(row.scored)} rows; correctness counts unavailable`;
  }
  const segments = probeSplitBarSegments(row);
  return `${formatReadoutInteger(segments.correctCount)} correct · ${formatReadoutInteger(
    segments.wrongOnlyCount,
  )} other wrong · ${formatReadoutInteger(segments.highConfWrongCount)} high-conf wrong`;
}

function probeSplitRowCountLabel(row: ProbeSummarySplitRow, hasErrorCounts: boolean): string {
  return hasErrorCounts
    ? `${formatReadoutInteger(row.scored)}/${formatReadoutInteger(row.total)}`
    : `${formatReadoutInteger(row.scored)} rows`;
}

function probeReadoutResultChartRows(summary?: ProbeStudyEpisodeSummary): Array<{
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
  const total = Math.max(summary?.episode_count ?? 0, 1);
  const scored = summary?.scored ?? 0;
  const wrong = summary?.wrong ?? 0;
  const highConfidence = summary?.high_confidence ?? 0;
  const highConfWrong = summary?.high_conf_wrong ?? 0;
  const unscored = summary?.unscored ?? 0;
  return [
    {
      active: (active) => active === "scored",
      apply: (onPredictionFilterChange) => onPredictionFilterChange("scored"),
      id: "scored",
      label: "Scored",
      total,
      value: scored,
    },
    {
      active: (active) => active === "incorrect",
      apply: (onPredictionFilterChange) => onPredictionFilterChange("incorrect"),
      id: "wrong",
      label: "Wrong",
      total,
      value: wrong,
    },
    {
      active: () => false,
      apply: (_onPredictionFilterChange, onCohortPresetChange) => onCohortPresetChange("confident_wrong"),
      id: "confident_wrong",
      label: "High-conf wrong",
      total,
      value: highConfWrong,
    },
    {
      active: (active) => active === "high_confidence",
      apply: (onPredictionFilterChange) => onPredictionFilterChange("high_confidence"),
      id: "high_confidence",
      label: "High confidence",
      total,
      value: highConfidence,
    },
    {
      active: (active) => active === "unscored",
      apply: (onPredictionFilterChange) => onPredictionFilterChange("unscored"),
      id: "unscored",
      label: "Unscored",
      total,
      value: unscored,
    },
  ];
}

function probeReadoutForSplitCategory(
  readouts: ProbeStudyReadout[],
  current: ProbeStudyReadout,
  splitCategory: string,
): ProbeStudyReadout | undefined {
  if (splitCategory === "all") {
    return current;
  }
  const currentLayer = probeReadoutLayerKey(current.layer);
  return readouts.find((readout) =>
    readout.target === current.target
    && probeReadoutLayerKey(readout.layer) === currentLayer
    && readout.split_category === splitCategory,
  ) ?? readouts.find((readout) =>
    readout.target === current.target && readout.split_category === splitCategory,
  );
}

function ProbeIdCopy({ value }: { value: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  };
  return (
    <button className="probe-id-copy" onClick={copy} title="Copy trained probe ID" type="button">
      <code>{value}</code>
      <span>{copied ? "Copied" : "Copy"}</span>
    </button>
  );
}

function humanizeProbeReadoutReason(reason: string): string {
  return [
    "active_manipulated_object",
    "active_receptacle_object",
    "next_manipulated_object",
    "task_phase",
  ].reduce(
    (text, target) => text.replaceAll(target, probeTargetDisplayLabel(target)),
    reason,
  );
}

function probeReadoutLayerLabel(layer: ProbeStudyReadout["layer"]): string {
  return layer === null || layer === undefined || layer === "" ? "all layers" : `layer ${layer}`;
}

function probeReadoutLayerKey(layer: ProbeStudyReadout["layer"]): string {
  return layer === null || layer === undefined ? "" : String(layer);
}

function probeReadoutSplitLabel(readout: ProbeStudyReadout): string {
  const category = readout.split_category ? PROBE_SPLIT_FILTER_LABELS[readout.split_category] ?? readout.split_category : "";
  if (!readout.split) {
    return category || "split missing";
  }
  return category ? `${category} / ${readout.split}` : readout.split;
}

function probeReadoutScoreLabel(readout: ProbeStudyReadout): string {
  return typeof readout.balanced_accuracy === "number" && Number.isFinite(readout.balanced_accuracy)
    ? readout.balanced_accuracy.toFixed(3)
    : "-";
}

function probeReadoutScoreTone(score: ProbeStudyReadout["balanced_accuracy"]): ProbeLensMetricChip["tone"] {
  if (typeof score !== "number" || !Number.isFinite(score)) {
    return "unknown";
  }
  if (score >= 0.7) {
    return "credible";
  }
  if (score >= 0.45) {
    return "limited";
  }
  return "danger";
}

function formatReadoutInteger(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? String(Math.round(value)) : "-";
}

function formatReadoutMetric(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "-";
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

function probeStudyEpisodePayload(value: unknown): ProbeStudyEpisodesResponse | undefined {
  if (
    value &&
    typeof value === "object" &&
    "artifact_id" in value &&
    "available" in value &&
    "episodes" in value
  ) {
    return value as ProbeStudyEpisodesResponse;
  }
  return undefined;
}

function ProbeEpisodeBadge({ record }: { record?: ProbeEpisodeIndex }) {
  if (!record) {
    return (
      <span className="probe-episode-badge muted">
        <strong>Unscored</strong>
        <small>No probe score</small>
      </span>
    );
  }
  const tone = record.correct === true ? "correct" : record.correct === false ? "incorrect" : "";
  const result = !record.available
    ? "Unscored"
    : record.correct === true
      ? "Correct"
      : record.correct === false
        ? "Wrong"
        : "Scored";
  return (
    <span className={["probe-episode-badge", tone].filter(Boolean).join(" ")}>
      <strong>{result}</strong>
      <small>{record.available ? formatDatasetProbeConfidence(record.confidence) : "No score"}</small>
    </span>
  );
}
