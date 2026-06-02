import {
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
  useCallback,
  useMemo,
  useRef,
  useState,
} from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Layers3 } from "lucide-react";
import {
  fetchActivationSites,
  fetchDataset,
  fetchEpisode,
  fetchEpisodeAnnotation,
  fetchEpisodeNeighbors,
  fetchEpisodeInteractions,
  fetchEpisodeMetrics,
  fetchEpisodeProbes,
  fetchEpisodesPage,
  fetchPolicyCalls,
  saveEpisodeAnnotation,
} from "../api/dataset";
import type { SelectedPatch } from "../types/dataset";
import type { WorkbenchManifest } from "../types/workbench";
import { episodeCapabilityGates, episodeQueryGates } from "./capabilityGating";
import { EpisodeNavigationBar } from "./episodes/EpisodeNavigation";
import {
  ActivationSitePanel,
  InspectorDebugSections,
} from "./episodes/InspectorPanels";
import { EpisodeStageView } from "./episodes/EpisodeStageView";
import {
  camerasFromManifest,
  frameVersionKey,
} from "./episodes/episodeData";
import {
  attentionSiteForSite,
  axisCountForSite,
  isAttentionSite,
} from "./episodes/siteModel";
import {
  type EpisodePlotTab,
  type InspectionMode,
} from "./episodes/shared";
import { useEpisodeInspectorModel } from "./episodes/useEpisodeInspectorModel";
import {
  useAdjacentEpisodePrefetch,
  useEpisodePlayback,
  useOverlayPrefetch,
} from "./episodes/useEpisodePrefetch";
import { useEpisodeRouteContext } from "./episodes/useEpisodeRouteContext";

type EpisodesPageProps = {
  cohortReturnHref?: string;
  initialPolicyCall?: number;
  initialProbeArtifactId?: string;
  initialSiteName?: string;
  manifest?: WorkbenchManifest;
  initialTraceId?: string;
  onTraceChange?: (traceId: string, context?: EpisodeTraceChangeContext) => void;
};

type EpisodeTraceChangeContext = {
  fromCohort?: boolean;
  policyCall?: number | null;
  probeId?: string;
  siteName?: string;
};

const EMPTY_ARTIFACTS: Record<string, unknown>[] = [];
const EMPTY_GENERATION_VALUES: (number | null)[][] = [];


export function EpisodesPage({
  cohortReturnHref,
  initialPolicyCall,
  initialProbeArtifactId = "",
  initialSiteName = "",
  manifest,
  initialTraceId = "",
  onTraceChange,
}: EpisodesPageProps) {
  const queryClient = useQueryClient();
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
  const capabilities = episodeCapabilityGates(dataset.data?.capabilities?.flags);
  const {
    hasPolicyCalls,
    hasModelSites,
    hasTokenSpaces,
    hasImageTokenMaps,
    hasAttentionMaps,
    hasActionGeneration,
    hasProbeArtifacts,
  } = capabilities;

  const firstEpisodePage = useQuery({
    queryKey: ["episodes-first", datasetIdentityKey],
    queryFn: ({ signal }) => fetchEpisodesPage({ limit: 1 }, signal),
    enabled: dataset.isFetched && !initialTraceId,
    staleTime: 30_000,
  });
  const firstEpisode = firstEpisodePage.data?.episodes[0];
  const [timestep, setTimestep] = useState(0);
  const [isPlayingFrames, setIsPlayingFrames] = useState(false);
  const [playbackFps, setPlaybackFps] = useState(10);
  const [showObjectOverlay, setShowObjectOverlay] = useState(true);
  const [showAttentionOverlay, setShowAttentionOverlay] = useState(true);
  const [selectedSiteName, setSelectedSiteName] = useState(initialSiteName);
  const [inspectionMode, setInspectionMode] = useState<InspectionMode>("features");
  const [attentionHead, setAttentionHead] = useState<number | null>(null);
  const [attentionQueryToken, setAttentionQueryToken] = useState<number | null>(null);
  const [feature, setFeature] = useState(0);
  const [activationClipPercent, setActivationClipPercent] = useState(0);
  const [topChannelCount, setTopChannelCount] = useState(12);
  const [inspectorWidthPct, setInspectorWidthPct] = useState(38);
  const [selectedPatch, setSelectedPatch] = useState<SelectedPatch | null>(null);
  const [selectedExpertToken, setSelectedExpertToken] = useState<number | null>(null);
  const [selectedPromptTokenIndex, setSelectedPromptTokenIndex] = useState<number | null>(null);
  const [generationStep, setGenerationStep] = useState(0);
  const [episodePlotTab, setEpisodePlotTab] = useState<EpisodePlotTab>("probes");
  const [selectedProbeArtifactId, setSelectedProbeArtifactId] = useState(initialProbeArtifactId);
  const appliedRouteContextRef = useRef("");

  const activeTraceId = initialTraceId || firstEpisode?.trace_id || "";
  const handlePatchSelect = useCallback((patch: SelectedPatch | null) => {
    setSelectedPatch((current) => {
      if (
        patch &&
        current?.camera === patch.camera &&
        current.row === patch.row &&
        current.col === patch.col
      ) {
        return null;
      }
      return patch;
    });
  }, [setSelectedPatch]);
  const episodeNeighbors = useQuery({
    queryKey: ["episode-neighbors", activeTraceId],
    queryFn: () => fetchEpisodeNeighbors(activeTraceId),
    enabled: Boolean(activeTraceId),
    staleTime: 30_000,
  });
  const previousTraceId = episodeNeighbors.data?.previous_trace_id ?? undefined;
  const nextTraceId = episodeNeighbors.data?.next_trace_id ?? undefined;
  const navigateEpisode = useCallback((traceId: string | undefined) => {
    if (!traceId) {
      return;
    }
    setIsPlayingFrames(false);
    setTimestep(0);
    setSelectedPatch(null);
    setSelectedExpertToken(null);
    setSelectedPromptTokenIndex(null);
    onTraceChange?.(traceId);
  }, [
    onTraceChange,
    setIsPlayingFrames,
    setSelectedExpertToken,
    setSelectedPatch,
    setSelectedPromptTokenIndex,
    setTimestep,
  ]);
  const handlePromptTokenSelect = useCallback((tokenIndex: number | null) => {
    setSelectedPromptTokenIndex((current) => (
      tokenIndex === null || current === tokenIndex ? null : tokenIndex
    ));
  }, [setSelectedPromptTokenIndex]);
  const navigatePreviousEpisode = useCallback(
    () => navigateEpisode(previousTraceId),
    [navigateEpisode, previousTraceId],
  );
  const navigateNextEpisode = useCallback(
    () => navigateEpisode(nextTraceId),
    [navigateEpisode, nextTraceId],
  );
  const activeCounterfactualPair = useMemo(
    () =>
      (dataset.data?.counterfactual_pairs ?? []).find((pair) =>
        pair.members.some((member) => member.trace_id === activeTraceId),
      ),
    [activeTraceId, dataset.data?.counterfactual_pairs],
  );
  const episodeDetail = useQuery({
    queryKey: ["episode", activeTraceId],
    queryFn: () => fetchEpisode(activeTraceId),
    enabled: Boolean(activeTraceId),
  });
  const selectedEpisode = episodeDetail.data ?? firstEpisode;
  const selectedEpisodeIndex = selectedEpisode?.episode_index ?? -1;
  const episodeCount = dataset.data?.episode_count ?? firstEpisodePage.data?.total ?? 0;
  const episodeAnnotation = useQuery({
    queryKey: ["episode-annotation", activeTraceId],
    queryFn: () => fetchEpisodeAnnotation(activeTraceId),
    enabled: Boolean(activeTraceId),
    staleTime: 60_000,
  });
  const saveAnnotation = useMutation({
    mutationFn: saveEpisodeAnnotation,
    onSuccess: (payload) => {
      queryClient.setQueryData(["episode-annotation", payload.annotation.trace_id], payload);
    },
  });
  const { mutate: mutateEpisodeAnnotation } = saveAnnotation;
  const handleSaveAnnotation = useCallback(
    (annotation: Parameters<typeof saveEpisodeAnnotation>[0]) => mutateEpisodeAnnotation(annotation),
    [mutateEpisodeAnnotation],
  );
  const initialQueryGates = episodeQueryGates(capabilities, {
    activeTraceId,
    activeSelectedProbeArtifactId: "",
    activeSelectedSiteName: "",
    activeCall: undefined,
    attentionSiteName: "",
    effectiveSelectedExpertToken: null,
    expertTokenSiteName: "",
    inspectorContext: "",
    selectedSiteHasFeatures: false,
  });
  const policyCalls = useQuery({
    queryKey: ["policy-calls", activeTraceId],
    queryFn: () => fetchPolicyCalls(activeTraceId),
    enabled: initialQueryGates.policyCalls,
  });
  const episodeMetrics = useQuery({
    queryKey: ["episode-metrics", activeTraceId],
    queryFn: () => fetchEpisodeMetrics(activeTraceId),
    enabled: Boolean(activeTraceId && episodePlotTab === "episode"),
  });
  const episodeInteractions = useQuery({
    queryKey: ["episode-interactions", activeTraceId],
    queryFn: () => fetchEpisodeInteractions(activeTraceId),
    enabled: Boolean(activeTraceId && episodePlotTab === "episode"),
  });
  const episodeProbes = useQuery({
    queryKey: ["episode-probes", activeTraceId],
    queryFn: () => fetchEpisodeProbes(activeTraceId),
    enabled: initialQueryGates.episodeProbes,
  });
  const activationSites = useQuery({
    queryKey: ["activation-sites", activeTraceId],
    queryFn: () => fetchActivationSites(activeTraceId),
    enabled: initialQueryGates.activationSites,
  });

  const cameras = episodeDetail.data?.cameras.length
    ? episodeDetail.data.cameras
    : camerasFromManifest(manifest, activeTraceId);
  const frameCacheKey = frameVersionKey(episodeDetail.data ?? selectedEpisode, datasetFingerprint);
  const maxTimestep = Math.max(0, Number(selectedEpisode?.length ?? episodeDetail.data?.length ?? 1) - 1);
  const metrics = episodeMetrics.data?.metrics ?? [];
  const {
    activationSlice,
    activeGenerationStep,
    activeSelectedProbeArtifactId,
    activeSelectedSiteName,
    architecture,
    attentionSite,
    attentionSiteName,
    cameraOverlay,
    cameraOverlayStatus,
    clampedFeature,
    effectiveSelectedExpertToken,
    expertTokenActivations,
    expertTokenDetails,
    expertTokenSiteName,
    generation,
    generationStepCount,
    inspectorContext,
    nextPolicyCall,
    observationalComparisons,
    patchFeatures,
    policyCallList,
    promptAttention,
    promptFeatureMap,
    selectedProbe,
    selectedProbeCall,
    selectedProbePolicyCall,
    selectedProbeRef,
    selectedProbeSite,
    selectedSite,
    selectedSiteHasFeatures,
    sites,
  } = useEpisodeInspectorModel({
    activationClipPercent,
    activationSitesData: activationSites.data,
    activeTraceId,
    attentionHead,
    attentionQueryToken,
    capabilities,
    episodeProbesData: episodeProbes.data,
    feature,
    generationStep,
    inspectionMode,
    policyCallsData: policyCalls.data,
    selectedExpertToken,
    selectedPatch,
    selectedProbeArtifactId,
    selectedSiteName,
    timestep,
    topChannelCount,
  });
  const openComparisonCandidate = useCallback((traceId: string) => {
    setIsPlayingFrames(false);
    setTimestep(0);
    setSelectedPatch(null);
    setSelectedExpertToken(null);
    setSelectedPromptTokenIndex(null);
    onTraceChange?.(traceId, {
      fromCohort: true,
      policyCall: selectedProbePolicyCall,
      probeId: activeSelectedProbeArtifactId,
      siteName: selectedProbeSite?.name ?? selectedProbeRef?.modelSiteId ?? "",
    });
  }, [
    activeSelectedProbeArtifactId,
    onTraceChange,
    selectedProbePolicyCall,
    selectedProbeRef?.modelSiteId,
    selectedProbeSite?.name,
    setIsPlayingFrames,
    setSelectedExpertToken,
    setSelectedPatch,
    setSelectedPromptTokenIndex,
    setTimestep,
  ]);

  const currentTimestep = Math.min(timestep, maxTimestep);
  const handleInspectionModeChange = useCallback((mode: InspectionMode) => {
    setInspectionMode(mode);
    if (
      mode === "attention" &&
      attentionQueryToken === null &&
      axisCountForSite(attentionSiteForSite(sites, selectedSite) ?? selectedSite, "query_token") > 0
    ) {
      setAttentionQueryToken(0);
    }
  }, [attentionQueryToken, selectedSite, setAttentionQueryToken, setInspectionMode, sites]);
  const handleSiteChange = useCallback((siteName: string) => {
    const nextSite = sites.find((site) => site.name === siteName);
    const nextAttentionSite = nextSite ? attentionSiteForSite(sites, nextSite) ?? nextSite : undefined;
    setSelectedSiteName(siteName);
    setAttentionHead(null);
    setAttentionQueryToken(
      nextAttentionSite && isAttentionSite(nextAttentionSite) && axisCountForSite(nextAttentionSite, "query_token") > 0
        ? 0
        : null,
    );
    setFeature(0);
    setSelectedPatch(null);
    setSelectedExpertToken(null);
    setSelectedPromptTokenIndex(null);
    setGenerationStep(0);
  }, [
    setAttentionHead,
    setAttentionQueryToken,
    setFeature,
    setGenerationStep,
    setSelectedExpertToken,
    setSelectedPatch,
    setSelectedPromptTokenIndex,
    setSelectedSiteName,
    sites,
  ]);
  const inspectProbeSite = useCallback(() => {
    if (!selectedProbeSite) {
      return;
    }
    handleInspectionModeChange("features");
    handleSiteChange(selectedProbeSite.name);
  }, [handleInspectionModeChange, handleSiteChange, selectedProbeSite]);
  const jumpToPolicyCall = useCallback((policyCallIndex: number) => {
    const call = (policyCalls.data?.calls ?? []).find(
      (item) => Number(item.index) === Number(policyCallIndex),
    );
    if (!call) {
      return;
    }
    setIsPlayingFrames(false);
    setTimestep(Math.max(0, Math.min(maxTimestep, call.env_timestep ?? call.segment_start)));
  }, [maxTimestep, policyCalls.data?.calls, setIsPlayingFrames, setTimestep]);
  const jumpToProbeCall = useCallback(() => {
    if (selectedProbePolicyCall !== null && selectedProbePolicyCall !== undefined) {
      jumpToPolicyCall(selectedProbePolicyCall);
    }
  }, [jumpToPolicyCall, selectedProbePolicyCall]);

  useEpisodeRouteContext({
    activeSelectedProbeArtifactId,
    activeTraceId,
    activationSitesIsLoading: activationSites.isLoading,
    appliedRouteContextRef,
    initialPolicyCall,
    initialProbeArtifactId,
    initialSiteName,
    maxTimestep,
    policyCalls: policyCallList,
    selectedProbe,
    selectedProbePolicyCall,
    selectedProbeSite,
    sites,
    setAttentionHead,
    setAttentionQueryToken,
    setFeature,
    setGenerationStep,
    setInspectionMode,
    setIsPlayingFrames,
    setSelectedExpertToken,
    setSelectedPatch,
    setSelectedPromptTokenIndex,
    setSelectedSiteName,
    setTimestep,
  });
  useEpisodePlayback({
    isPlayingFrames,
    maxTimestep,
    playbackFps,
    setTimestep,
  });
  useAdjacentEpisodePrefetch({
    hasModelSites,
    hasPolicyCalls,
    nextTraceId,
    previousTraceId,
    queryClient,
  });
  useOverlayPrefetch({
    activeSelectedSiteName,
    activeTraceId,
    activeGenerationStep,
    attentionHead,
    attentionQueryToken,
    attentionSite,
    attentionSiteName,
    clampedFeature,
    hasAttentionMaps,
    hasImageTokenMaps,
    hasPolicyCalls,
    hasTokenSpaces,
    inspectorContext,
    isPlayingFrames,
    nextPolicyCall,
    queryClient,
    selectedSiteHasFeatures,
    showAttentionOverlay,
  });

  const workspaceStyle = {
    "--features-width": `${inspectorWidthPct}%`,
  } as CSSProperties;
  const preloadFrameCount = Math.min(10, Math.max(2, Math.ceil(playbackFps * 0.75)));
  const handleInspectorResize = (event: ReactPointerEvent<HTMLButtonElement>) => {
    const root = event.currentTarget.closest(".episodes-workspace");
    const rect = root?.getBoundingClientRect();
    if (!rect) {
      return;
    }
    const rightWidth = rect.right - event.clientX;
    const nextPct = (rightWidth / rect.width) * 100;
    setInspectorWidthPct(Math.max(30, Math.min(62, nextPct)));
  };
  const resetPlayback = useCallback(() => {
    setIsPlayingFrames(false);
    setTimestep(0);
  }, [setIsPlayingFrames, setTimestep]);
  const togglePlayback = useCallback(
    () => setIsPlayingFrames((value) => !value),
    [setIsPlayingFrames],
  );
  const toggleAttentionOverlay = useCallback(
    () => setShowAttentionOverlay((value) => !value),
    [setShowAttentionOverlay],
  );
  const toggleObjectOverlay = useCallback(
    () => setShowObjectOverlay((value) => !value),
    [setShowObjectOverlay],
  );
  const handleFeatureChange = useCallback((nextFeature: number) => {
    setFeature(nextFeature);
    setSelectedPatch(null);
    setSelectedExpertToken(null);
    setSelectedPromptTokenIndex(null);
  }, [setFeature, setSelectedExpertToken, setSelectedPatch, setSelectedPromptTokenIndex]);
  const handleGenerationStepChange = useCallback((nextStep: number) => {
    setGenerationStep(nextStep);
    setSelectedExpertToken(null);
    setSelectedPromptTokenIndex(null);
  }, [setGenerationStep, setSelectedExpertToken, setSelectedPromptTokenIndex]);
  const handleInteractionTimestepChange = useCallback((nextTimestep: number) => {
    setIsPlayingFrames(false);
    setTimestep(Math.max(0, Math.min(maxTimestep, nextTimestep)));
  }, [maxTimestep, setIsPlayingFrames, setTimestep]);

  return (
    <main className="episodes-workspace episode-main" style={workspaceStyle}>
      {!selectedEpisode ? <div className="empty-state">No episode selected.</div> : null}
      {selectedEpisode ? (
        <>
          <EpisodeNavigationBar
            annotation={episodeAnnotation.data?.annotation}
            cohortReturnHref={cohortReturnHref}
            episode={episodeDetail.data ?? selectedEpisode}
            episodeIndex={selectedEpisodeIndex}
            episodeCount={episodeCount}
            counterfactualPair={activeCounterfactualPair}
            hasNext={Boolean(nextTraceId)}
            hasPrevious={Boolean(previousTraceId)}
            isSavingAnnotation={saveAnnotation.isPending}
            onNext={navigateNextEpisode}
            onPrevious={navigatePreviousEpisode}
            onNavigateTrace={navigateEpisode}
            onSaveAnnotation={handleSaveAnnotation}
          />
          <EpisodeStageView
            episodePlotTab={episodePlotTab}
            onEpisodePlotTabChange={setEpisodePlotTab}
            promptAttentionStrip={{
              expertTokenDetails: expertTokenDetails.data,
              context: inspectorContext,
              prompt: (episodeDetail.data ?? selectedEpisode)?.prompt,
              promptAttention: promptAttention.data,
              promptFeatureMap: promptFeatureMap.data,
              selectedPromptToken: selectedPromptTokenIndex,
              onPromptTokenSelect: handlePromptTokenSelect,
            }}
            cameraGrid={{
              cacheKey: frameCacheKey,
              cameras,
              imageTokenMap: cameraOverlay,
              overlayStatus: cameraOverlayStatus,
              isPlaying: isPlayingFrames,
              maxTimestep,
              preloadFrameCount,
              showAttentionOverlay,
              showObjectOverlay,
              onPatchSelect: handlePatchSelect,
              selectedPatch,
              traceId: activeTraceId,
              timestep: currentTimestep,
            }}
            frameControls={{
              fps: playbackFps,
              cacheKey: frameCacheKey,
              isPlaying: isPlayingFrames,
              maxTimestep,
              policyCalls: policyCallList,
              showAttentionOverlay,
              showObjectOverlay,
              timestep: currentTimestep,
              traceId: activeTraceId,
              onFpsChange: setPlaybackFps,
              onAttentionOverlayToggle: toggleAttentionOverlay,
              onObjectOverlayToggle: toggleObjectOverlay,
              onReset: resetPlayback,
              onToggle: togglePlayback,
              onTimestepChange: setTimestep,
            }}
            metricPlot={{
              metrics,
              timestep: currentTimestep,
              onTimestepChange: setTimestep,
            }}
            interactionSummary={{
              interactions: episodeInteractions.data,
              isError: episodeInteractions.isError,
              isLoading: episodeInteractions.isLoading,
              onTimestepChange: handleInteractionTimestepChange,
            }}
            episodeProbePanel={{
              comparisons: observationalComparisons.data,
              probes: hasProbeArtifacts ? episodeProbes.data : undefined,
              selectedProbe,
              selectedProbeRef,
              isError: hasProbeArtifacts && episodeProbes.isError,
              isLoading: hasProbeArtifacts && episodeProbes.isLoading,
              isComparisonError: observationalComparisons.isError,
              isComparisonLoading: observationalComparisons.isFetching,
              canInspectProbe: Boolean(selectedProbeSite),
              canJumpToProbeCall: Boolean(selectedProbeCall),
              onOpenComparison: openComparisonCandidate,
              onInspectProbe: inspectProbeSite,
              onJumpToProbeCall: jumpToProbeCall,
              onJumpToPolicyCall: jumpToPolicyCall,
              onProbeChange: setSelectedProbeArtifactId,
            }}
          />

          <button
            aria-label="Resize model inspector"
            className="episode-column-resizer"
            type="button"
            onPointerDown={(event) => {
              event.currentTarget.setPointerCapture(event.pointerId);
              handleInspectorResize(event);
            }}
            onPointerMove={(event) => {
              if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                handleInspectorResize(event);
              }
            }}
            onPointerUp={(event) => {
              if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                event.currentTarget.releasePointerCapture(event.pointerId);
              }
            }}
            onDoubleClick={() => setInspectorWidthPct(38)}
          />

          <section className="inspector">
            <div className="head">
              <div className="inspector-title">
                <Layers3 size={17} />
                <span>Model Inspector</span>
              </div>
            </div>
            <div className="body">
              {hasModelSites ? (
                <ActivationSitePanel
                  architecture={architecture}
                  sites={sites}
                  activationSlice={activationSlice.data}
                  feature={clampedFeature}
                  cameraOverlay={cameraOverlay}
                  episodeProbes={hasProbeArtifacts ? episodeProbes.data : undefined}
                  selectedProbeArtifactId={activeSelectedProbeArtifactId}
                  patchFeatures={patchFeatures.data}
                  expertTokenActivations={expertTokenActivations.data}
                  expertTokenDetails={expertTokenDetails.data}
                  activationSliceFetching={activationSlice.isFetching}
                  activationSlicePlaceholder={activationSlice.isPlaceholderData}
                  activationClipPercent={activationClipPercent}
                  attentionHead={attentionHead}
                  attentionQueryToken={attentionQueryToken}
                  selectedPatch={selectedPatch}
                  selectedExpertToken={effectiveSelectedExpertToken}
                  selectedPromptToken={selectedPromptTokenIndex}
                  onFeatureChange={handleFeatureChange}
                  generationStep={activeGenerationStep}
                  generationStepCount={generationStepCount}
                  expertTokenSiteName={expertTokenSiteName}
                  inspectorContext={inspectorContext}
                  inspectionMode={inspectionMode}
                  onGenerationStepChange={handleGenerationStepChange}
                  onAttentionHeadChange={setAttentionHead}
                  onAttentionQueryTokenChange={setAttentionQueryToken}
                  onExpertTokenChange={setSelectedExpertToken}
                  onPatchSelect={handlePatchSelect}
                  onPromptTokenSelect={handlePromptTokenSelect}
                  onActivationClipPercentChange={setActivationClipPercent}
                  onInspectionModeChange={handleInspectionModeChange}
                  onProbeSelect={setSelectedProbeArtifactId}
                  onTopChannelCountChange={setTopChannelCount}
                  selectedSite={selectedSite}
                  selectedSiteName={activeSelectedSiteName}
                  topChannelCount={topChannelCount}
                  onSiteChange={handleSiteChange}
                />
              ) : (
                <div className="empty-state">No model-site overlay is available for this dataset.</div>
              )}
              {hasPolicyCalls || hasActionGeneration ? (
                <InspectorDebugSections
                  artifacts={episodeDetail.data?.artifacts ?? EMPTY_ARTIFACTS}
                  calls={policyCallList}
                  generationValues={generation.data?.values ?? EMPTY_GENERATION_VALUES}
                  inspectorContext={inspectorContext}
                  onTimestepChange={setTimestep}
                  timestep={currentTimestep}
                />
              ) : null}
            </div>
          </section>
        </>
      ) : null}
    </main>
  );

}
