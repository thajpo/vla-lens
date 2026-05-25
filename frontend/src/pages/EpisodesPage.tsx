import {
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Layers3 } from "lucide-react";
import {
  fetchActivationSites,
  fetchActivationSlice,
  fetchAttentionMap,
  fetchDataset,
  fetchDatasetDiagnostics,
  fetchGenerationCommitment,
  fetchEpisode,
  fetchEpisodeAnnotation,
  fetchEpisodeInteractions,
  fetchEpisodeMetrics,
  fetchEpisodeProbes,
  fetchExpertTokenActivations,
  fetchExpertTokenDetails,
  fetchImageTokenMap,
  fetchObservationalComparisons,
  fetchPatchFeatures,
  fetchPolicyCalls,
  fetchPromptAttention,
  fetchPromptFeatureMap,
  saveEpisodeAnnotation,
} from "../api/dataset";
import type { SelectedPatch } from "../types/dataset";
import type { WorkbenchManifest } from "../types/workbench";
import { CameraGrid, FramePlaybackControls } from "./episodes/CameraTimeline";
import { EpisodeNavigationBar } from "./episodes/EpisodeNavigation";
import { EpisodeProbePanel } from "./episodes/EpisodeProbePanel";
import {
  ActivationSitePanel,
  InspectorDebugSections,
  PromptAttentionStrip,
} from "./episodes/InspectorPanels";
import { InteractionSummaryPanel } from "./episodes/InteractionSummary";
import { MetricPlotPanel } from "./episodes/MetricPlots";
import {
  camerasFromManifest,
  episodesFromManifest,
  frameVersionKey,
  imageTokenMapQueryKey,
  overlayStatusForSelection,
} from "./episodes/episodeData";
import { probeLayerReferences, selectedEpisodeProbe } from "./episodes/episodeProbeModel";
import {
  attentionSiteForSite,
  axisCountForSite,
  expertTokenSiteForSite,
  generationStepCountForSite,
  inspectorContextForSite,
  isAttentionSite,
  isFeatureActivationSite,
  preferredPipelineSite,
  siteForProbeRef,
  siteUsesGenerationStep,
} from "./episodes/siteModel";
import {
  EMPTY_ACTIVATION_SITES,
  type CameraOverlayPayload,
  type EpisodePlotTab,
  type InspectionMode,
} from "./episodes/shared";

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
    queryFn: fetchDataset,
    staleTime: 30_000,
  });
  const diagnostics = useQuery({
    queryKey: ["dataset-diagnostics"],
    queryFn: fetchDatasetDiagnostics,
  });

  const episodes = dataset.data?.episodes ?? episodesFromManifest(manifest);
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

  const activeTraceId = initialTraceId || episodes[0]?.trace_id || "";
  const handlePatchSelect = (patch: SelectedPatch | null) => {
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
  };
  const selectedEpisode =
    episodes.find((episode) => episode.trace_id === activeTraceId) ?? episodes[0];
  const selectedEpisodeIndex = selectedEpisode
    ? episodes.findIndex((episode) => episode.trace_id === selectedEpisode.trace_id)
    : -1;
  const previousEpisode =
    selectedEpisodeIndex > 0 ? episodes[selectedEpisodeIndex - 1] : undefined;
  const nextEpisode =
    selectedEpisodeIndex >= 0 && selectedEpisodeIndex < episodes.length - 1
      ? episodes[selectedEpisodeIndex + 1]
      : undefined;
  const navigateEpisode = (traceId: string | undefined) => {
    if (!traceId) {
      return;
    }
    setIsPlayingFrames(false);
    setTimestep(0);
    setSelectedPatch(null);
    setSelectedExpertToken(null);
    setSelectedPromptTokenIndex(null);
    onTraceChange?.(traceId);
  };
  const handlePromptTokenSelect = (tokenIndex: number | null) => {
    setSelectedPromptTokenIndex((current) => (
      tokenIndex === null || current === tokenIndex ? null : tokenIndex
    ));
  };
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
  const policyCalls = useQuery({
    queryKey: ["policy-calls", activeTraceId],
    queryFn: () => fetchPolicyCalls(activeTraceId),
    enabled: Boolean(activeTraceId),
  });
  const episodeMetrics = useQuery({
    queryKey: ["episode-metrics", activeTraceId],
    queryFn: () => fetchEpisodeMetrics(activeTraceId),
    enabled: Boolean(activeTraceId),
  });
  const episodeInteractions = useQuery({
    queryKey: ["episode-interactions", activeTraceId],
    queryFn: () => fetchEpisodeInteractions(activeTraceId),
    enabled: Boolean(activeTraceId),
  });
  const episodeProbes = useQuery({
    queryKey: ["episode-probes", activeTraceId],
    queryFn: () => fetchEpisodeProbes(activeTraceId),
    enabled: Boolean(activeTraceId),
  });
  const generation = useQuery({
    queryKey: ["generation-commitment", activeTraceId],
    queryFn: () => fetchGenerationCommitment(activeTraceId),
    enabled: Boolean(activeTraceId),
  });
  const activationSites = useQuery({
    queryKey: ["activation-sites", activeTraceId],
    queryFn: () => fetchActivationSites(activeTraceId),
    enabled: Boolean(activeTraceId),
  });

  const cameras = episodeDetail.data?.cameras.length
    ? episodeDetail.data.cameras
    : camerasFromManifest(manifest, activeTraceId);
  const frameCacheKey = frameVersionKey(episodeDetail.data ?? selectedEpisode, diagnostics.data?.fingerprint);
  const maxTimestep = Math.max(0, Number(selectedEpisode?.length ?? episodeDetail.data?.length ?? 1) - 1);
  const metrics = episodeMetrics.data?.metrics ?? [];
  const sites = activationSites.data?.sites ?? EMPTY_ACTIVATION_SITES;
  const selectedProbe = selectedEpisodeProbe(episodeProbes.data, selectedProbeArtifactId);
  const activeSelectedProbeArtifactId = selectedProbe?.artifact_id ?? "";
  const episodeProbeRefs = probeLayerReferences(episodeProbes.data);
  const selectedProbeRef = episodeProbeRefs.find(
    (ref) => ref.artifactId === activeSelectedProbeArtifactId,
  );
  const selectedProbeSite = siteForProbeRef(sites, selectedProbeRef);
  const selectedProbePolicyCall = selectedProbeRef?.policyCall;
  const selectedProbeCall = selectedProbePolicyCall === null || selectedProbePolicyCall === undefined
    ? undefined
    : (policyCalls.data?.calls ?? []).find(
        (item) => Number(item.index) === Number(selectedProbePolicyCall),
      );
  const observationalComparisons = useQuery({
    queryKey: ["observational-comparisons", activeTraceId, activeSelectedProbeArtifactId],
    queryFn: () => fetchObservationalComparisons(activeTraceId, activeSelectedProbeArtifactId),
    enabled: Boolean(activeTraceId),
    placeholderData: keepPreviousData,
  });
  const openComparisonCandidate = (traceId: string) => {
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
  };
  const architecture = activationSites.data?.architecture;
  const defaultSite = preferredPipelineSite(sites);
  const selectedSite = sites.find((site) => site.name === selectedSiteName) ?? defaultSite;
  const inspectorContext = inspectorContextForSite(selectedSite);
  const selectedSiteHasFeatures = isFeatureActivationSite(selectedSite);
  const expertTokenSite = expertTokenSiteForSite(sites, selectedSite);
  const expertTokenSiteName = expertTokenSite?.name ?? "";
  const attentionSite = attentionSiteForSite(sites, selectedSite);
  const attentionSiteName = attentionSite?.name ?? "";
  const generationStepCount = generationStepCountForSite(selectedSite);
  const activeGenerationStep = Math.max(0, Math.min(generationStep, Math.max(0, generationStepCount - 1)));
  const activeSelectedSiteName = selectedSite?.name ?? selectedSiteName;
  const effectiveSelectedExpertToken =
    inspectorContext === "expert" &&
    inspectionMode === "features" &&
    expertTokenSiteName &&
    selectedExpertToken === null
      ? 0
      : selectedExpertToken;
  const selectedSiteUsesGenerationStep = siteUsesGenerationStep(selectedSite);
  const activeCall =
    (policyCalls.data?.calls ?? []).find(
      (call) => timestep >= call.segment_start && timestep <= call.segment_end,
    ) ?? policyCalls.data?.calls[0];
  const policyCallList = policyCalls.data?.calls ?? [];
  const activeCallPosition = activeCall
    ? policyCallList.findIndex((call) => call.index === activeCall.index)
    : -1;
  const nextPolicyCall = activeCallPosition >= 0 ? policyCallList[activeCallPosition + 1] : undefined;
  const activationSlice = useQuery({
    queryKey: [
      "activation-slice",
      activeTraceId,
      activeCall?.index,
      activeSelectedSiteName,
      feature,
      selectedSiteUsesGenerationStep ? activeGenerationStep : null,
      activationClipPercent,
      topChannelCount,
    ],
    queryFn: () =>
      fetchActivationSlice(
        activeTraceId,
        activeCall?.index ?? 0,
        activeSelectedSiteName,
        feature,
        selectedSiteUsesGenerationStep ? activeGenerationStep : undefined,
        activationClipPercent,
        topChannelCount,
      ),
    enabled: Boolean(
      activeTraceId && activeSelectedSiteName && activeCall && selectedSiteHasFeatures,
    ),
    placeholderData: keepPreviousData,
  });
  const clampedFeature = Math.max(
    0,
    Math.min(feature, Math.max(0, (activationSlice.data?.feature_count ?? 1) - 1)),
  );
  const imageTokenMap = useQuery({
    queryKey: imageTokenMapQueryKey(
      activeTraceId,
      activeCall?.index,
      activeSelectedSiteName,
      clampedFeature,
    ),
    queryFn: () =>
      fetchImageTokenMap(
        activeTraceId,
        activeCall?.index ?? 0,
        activeSelectedSiteName,
        clampedFeature,
      ),
    enabled: Boolean(
      inspectorContext === "vlm" &&
        activeTraceId &&
        activeSelectedSiteName &&
        activeCall &&
        selectedSiteHasFeatures,
    ),
    placeholderData: keepPreviousData,
    staleTime: 60_000,
  });
  const patchFeatures = useQuery({
    queryKey: [
      "patch-features",
      activeTraceId,
      activeCall?.index,
      activeSelectedSiteName,
      clampedFeature,
      selectedPatch?.camera,
      selectedPatch?.row,
      selectedPatch?.col,
    ],
    queryFn: () =>
      fetchPatchFeatures(
        activeTraceId,
        activeCall?.index ?? 0,
        activeSelectedSiteName,
        clampedFeature,
        selectedPatch as SelectedPatch,
      ),
    enabled: Boolean(
      activeTraceId &&
        activeSelectedSiteName &&
        activeCall &&
        selectedPatch &&
        selectedSiteHasFeatures,
    ),
  });
  const expertTokenActivations = useQuery({
    queryKey: [
      "expert-token-activations",
      activeTraceId,
      activeCall?.index,
      expertTokenSiteName,
      clampedFeature,
      activeGenerationStep,
    ],
    queryFn: () =>
      fetchExpertTokenActivations(
        activeTraceId,
        activeCall?.index ?? 0,
        expertTokenSiteName,
        clampedFeature,
        activeGenerationStep,
      ),
    enabled: Boolean(
      inspectorContext === "expert" && activeTraceId && activeCall && expertTokenSiteName,
    ),
  });
  const expertTokenDetails = useQuery({
    queryKey: [
      "expert-token-details",
      activeTraceId,
      activeCall?.index,
      expertTokenSiteName,
      clampedFeature,
      effectiveSelectedExpertToken,
      activeGenerationStep,
    ],
    queryFn: () =>
      fetchExpertTokenDetails(
        activeTraceId,
        activeCall?.index ?? 0,
        expertTokenSiteName,
        clampedFeature,
        effectiveSelectedExpertToken ?? 0,
        activeGenerationStep,
      ),
    enabled: Boolean(
      inspectorContext === "expert" &&
        activeTraceId &&
        activeCall &&
        expertTokenSiteName &&
        effectiveSelectedExpertToken !== null,
    ),
  });
  const attentionMap = useQuery({
    queryKey: [
      "attention-map",
      activeTraceId,
      activeCall?.index,
      inspectorContext,
      attentionSiteName,
      activeGenerationStep,
      attentionHead,
      attentionQueryToken,
    ],
    queryFn: () =>
      fetchAttentionMap(
        activeTraceId,
        activeCall?.index ?? 0,
        attentionSite?.name.includes(".vlm.") ? "vlm" : "expert",
        activeGenerationStep,
        attentionSiteName,
        attentionHead,
        attentionQueryToken,
      ),
    enabled: Boolean(
      inspectorContext === "attention" && activeTraceId && activeCall && attentionSiteName,
    ),
    placeholderData: keepPreviousData,
    staleTime: 60_000,
  });
  const promptAttentionKind = attentionSite?.name.includes(".vlm.") ? "vlm" : "expert";
  const promptAttention = useQuery({
    queryKey: [
      "prompt-attention",
      activeTraceId,
      activeCall?.index,
      promptAttentionKind,
      attentionSiteName,
      activeGenerationStep,
      attentionHead,
      attentionQueryToken,
    ],
    queryFn: () =>
      fetchPromptAttention(
        activeTraceId,
        activeCall?.index ?? 0,
        activeGenerationStep,
        promptAttentionKind,
        attentionSiteName,
        attentionHead,
        attentionQueryToken,
      ),
    enabled: Boolean(activeTraceId && activeCall && attentionSiteName),
    staleTime: 60_000,
  });
  const promptFeatureMap = useQuery({
    queryKey: [
      "prompt-feature-map",
      activeTraceId,
      activeCall?.index,
      activeSelectedSiteName,
      clampedFeature,
    ],
    queryFn: () =>
      fetchPromptFeatureMap(
        activeTraceId,
        activeCall?.index ?? 0,
        activeSelectedSiteName,
        clampedFeature,
      ),
    enabled: Boolean(
      inspectorContext === "vlm" &&
        activeTraceId &&
        activeCall &&
        activeSelectedSiteName &&
        selectedSiteHasFeatures,
    ),
    placeholderData: keepPreviousData,
  });
  const cameraOverlay: CameraOverlayPayload | undefined =
    inspectorContext === "expert"
      ? expertTokenDetails.data
      : inspectorContext === "attention"
        ? attentionMap.data
        : inspectorContext === "vlm"
          ? imageTokenMap.data
          : undefined;
  const cameraOverlayStatus = overlayStatusForSelection({
    attentionMapFetching: attentionMap.isFetching,
    attentionMapPlaceholder: attentionMap.isPlaceholderData,
    attentionHead,
    attentionQueryToken,
    expertTokenDetailsFetching: expertTokenDetails.isFetching,
    feature: clampedFeature,
    imageTokenMapFetching: imageTokenMap.isFetching,
    imageTokenMapPlaceholder: imageTokenMap.isPlaceholderData,
    inspectorContext,
    selectedExpertToken: effectiveSelectedExpertToken,
    selectedSite,
  });

  const currentTimestep = Math.min(timestep, maxTimestep);
  const handleInspectionModeChange = (mode: InspectionMode) => {
    setInspectionMode(mode);
    if (
      mode === "attention" &&
      attentionQueryToken === null &&
      axisCountForSite(attentionSiteForSite(sites, selectedSite) ?? selectedSite, "query_token") > 0
    ) {
      setAttentionQueryToken(0);
    }
  };
  const handleSiteChange = (siteName: string) => {
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
  };
  const inspectProbeSite = () => {
    if (!selectedProbeSite) {
      return;
    }
    handleInspectionModeChange("features");
    handleSiteChange(selectedProbeSite.name);
  };
  const jumpToProbeCall = () => {
    if (selectedProbePolicyCall !== null && selectedProbePolicyCall !== undefined) {
      jumpToPolicyCall(selectedProbePolicyCall);
    }
  };
  const jumpToPolicyCall = (policyCallIndex: number) => {
    const call = (policyCalls.data?.calls ?? []).find(
      (item) => Number(item.index) === Number(policyCallIndex),
    );
    if (!call) {
      return;
    }
    setIsPlayingFrames(false);
    setTimestep(Math.max(0, Math.min(maxTimestep, call.env_timestep ?? call.segment_start)));
  };

  useEffect(() => {
    if (!initialProbeArtifactId && typeof initialPolicyCall !== "number" && !initialSiteName) {
      return;
    }
    const routeKey = [
      activeTraceId,
      initialProbeArtifactId,
      initialPolicyCall ?? "",
      initialSiteName,
    ].join("|");
    if (appliedRouteContextRef.current === routeKey) {
      return;
    }
    if (initialProbeArtifactId && activeSelectedProbeArtifactId !== initialProbeArtifactId) {
      return;
    }
    if (initialProbeArtifactId && !selectedProbe) {
      return;
    }
    const targetPolicyCall =
      typeof initialPolicyCall === "number" ? initialPolicyCall : selectedProbePolicyCall;
    const targetCallReady =
      typeof targetPolicyCall !== "number" ||
      (policyCalls.data?.calls ?? []).some((call) => Number(call.index) === targetPolicyCall);
    if (!targetCallReady) {
      return;
    }
    if ((initialSiteName || initialProbeArtifactId) && activationSites.isLoading) {
      return;
    }
    const targetSite = initialSiteName
      ? sites.find((site) => site.name === initialSiteName)
      : selectedProbeSite;
    const targetSiteName = targetSite?.name;
    const targetCallIndex = typeof targetPolicyCall === "number" ? targetPolicyCall : undefined;
    appliedRouteContextRef.current = routeKey;
    const frame = window.requestAnimationFrame(() => {
      if (targetSiteName) {
        const nextSite = sites.find((site) => site.name === targetSiteName);
        const nextAttentionSite = nextSite ? attentionSiteForSite(sites, nextSite) ?? nextSite : undefined;
        setInspectionMode("features");
        setSelectedSiteName(targetSiteName);
        setAttentionHead(null);
        setAttentionQueryToken(
          nextAttentionSite &&
            isAttentionSite(nextAttentionSite) &&
            axisCountForSite(nextAttentionSite, "query_token") > 0
            ? 0
            : null,
        );
        setFeature(0);
        setSelectedPatch(null);
        setSelectedExpertToken(null);
        setSelectedPromptTokenIndex(null);
        setGenerationStep(0);
      }
      if (typeof targetCallIndex === "number") {
        const call = (policyCalls.data?.calls ?? []).find(
          (item) => Number(item.index) === Number(targetCallIndex),
        );
        if (call) {
          setIsPlayingFrames(false);
          setTimestep(Math.max(0, Math.min(maxTimestep, call.env_timestep ?? call.segment_start)));
        }
      }
    });
    return () => window.cancelAnimationFrame(frame);
  }, [
    activeSelectedProbeArtifactId,
    activeTraceId,
    activationSites.isLoading,
    initialPolicyCall,
    initialProbeArtifactId,
    initialSiteName,
    maxTimestep,
    policyCalls.data?.calls,
    selectedProbe,
    selectedProbePolicyCall,
    selectedProbeSite,
    sites,
  ]);

  useEffect(() => {
    if (!isPlayingFrames || maxTimestep <= 0) {
      return;
    }
    const interval = window.setInterval(() => {
      setTimestep((current) => (current >= maxTimestep ? 0 : current + 1));
    }, 1000 / Math.max(1, playbackFps));
    return () => window.clearInterval(interval);
  }, [isPlayingFrames, maxTimestep, playbackFps]);

  useEffect(() => {
    if (
      !isPlayingFrames ||
      !showAttentionOverlay ||
      !activeTraceId ||
      !nextPolicyCall
    ) {
      return;
    }
    if (inspectorContext === "vlm" && activeSelectedSiteName && selectedSiteHasFeatures) {
      void queryClient.prefetchQuery({
        queryKey: imageTokenMapQueryKey(
          activeTraceId,
          nextPolicyCall.index,
          activeSelectedSiteName,
          clampedFeature,
        ),
        queryFn: () =>
          fetchImageTokenMap(
            activeTraceId,
            nextPolicyCall.index,
            activeSelectedSiteName,
            clampedFeature,
          ),
        staleTime: 60_000,
      });
    }
    if (inspectorContext === "attention" && attentionSiteName) {
      const kind = attentionSite?.name.includes(".vlm.") ? "vlm" : "expert";
      void queryClient.prefetchQuery({
        queryKey: [
          "attention-map",
          activeTraceId,
          nextPolicyCall.index,
          inspectorContext,
          attentionSiteName,
          activeGenerationStep,
          attentionHead,
          attentionQueryToken,
        ],
        queryFn: () =>
          fetchAttentionMap(
            activeTraceId,
            nextPolicyCall.index,
            kind,
            activeGenerationStep,
            attentionSiteName,
            attentionHead,
            attentionQueryToken,
          ),
        staleTime: 60_000,
      });
    }
  }, [
    activeSelectedSiteName,
    activeTraceId,
    activeGenerationStep,
    attentionHead,
    attentionQueryToken,
    attentionSite?.name,
    attentionSiteName,
    clampedFeature,
    inspectorContext,
    isPlayingFrames,
    nextPolicyCall,
    queryClient,
    selectedSiteHasFeatures,
    showAttentionOverlay,
  ]);

  const workspaceStyle = {
    "--features-width": `${inspectorWidthPct}%`,
  } as CSSProperties;
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
            episodeCount={episodes.length}
            counterfactualPair={activeCounterfactualPair}
            hasNext={Boolean(nextEpisode)}
            hasPrevious={Boolean(previousEpisode)}
            isSavingAnnotation={saveAnnotation.isPending}
            onNext={() => navigateEpisode(nextEpisode?.trace_id)}
            onPrevious={() => navigateEpisode(previousEpisode?.trace_id)}
            onNavigateTrace={navigateEpisode}
            onSaveAnnotation={(annotation) => saveAnnotation.mutate(annotation)}
          />
          <section className="stage">
            <div className="stage-body stage-view">
              <div className="episode-workspace">
                <div className="workspace-main">
                  <div className="viewer-layout">
                    <div className="viewer-media">
                      <PromptAttentionStrip
                        expertTokenDetails={expertTokenDetails.data}
                        context={inspectorContext}
                        prompt={(episodeDetail.data ?? selectedEpisode)?.prompt}
                        promptAttention={promptAttention.data}
                        promptFeatureMap={promptFeatureMap.data}
                        selectedPromptToken={selectedPromptTokenIndex}
                        onPromptTokenSelect={handlePromptTokenSelect}
                      />
                      <CameraGrid
                        cacheKey={frameCacheKey}
                        cameras={cameras}
                        imageTokenMap={cameraOverlay}
                        overlayStatus={cameraOverlayStatus}
                        isPlaying={isPlayingFrames}
                        maxTimestep={maxTimestep}
                        showAttentionOverlay={showAttentionOverlay}
                        showObjectOverlay={showObjectOverlay}
                        onPatchSelect={handlePatchSelect}
                        selectedPatch={selectedPatch}
                        traceId={activeTraceId}
                        timestep={currentTimestep}
                      />
                      <FramePlaybackControls
                        fps={playbackFps}
                        cacheKey={frameCacheKey}
                        isPlaying={isPlayingFrames}
                        maxTimestep={maxTimestep}
                        policyCalls={policyCalls.data?.calls ?? []}
                        showAttentionOverlay={showAttentionOverlay}
                        showObjectOverlay={showObjectOverlay}
                        timestep={currentTimestep}
                        traceId={activeTraceId}
                        onFpsChange={setPlaybackFps}
                        onAttentionOverlayToggle={() => setShowAttentionOverlay((value) => !value)}
                        onObjectOverlayToggle={() => setShowObjectOverlay((value) => !value)}
                        onReset={() => {
                          setIsPlayingFrames(false);
                          setTimestep(0);
                        }}
                        onToggle={() => setIsPlayingFrames((value) => !value)}
                        onTimestepChange={setTimestep}
                      />
                    </div>
                    <aside className="viewer-plot-panel">
                      <div className="episode-plot-tabs" aria-label="Episode plot tabs">
                        <button
                          className={episodePlotTab === "probes" ? "active" : ""}
                          type="button"
                          onClick={() => setEpisodePlotTab("probes")}
                        >
                          Probes
                        </button>
                        <button
                          className={episodePlotTab === "episode" ? "active" : ""}
                          type="button"
                          onClick={() => setEpisodePlotTab("episode")}
                        >
                          Episode
                        </button>
                      </div>
                      {episodePlotTab === "episode" ? (
                        <>
                          <MetricPlotPanel
                            metrics={metrics}
                            timestep={currentTimestep}
                            onTimestepChange={setTimestep}
                          />
                          <InteractionSummaryPanel
                            interactions={episodeInteractions.data}
                            isError={episodeInteractions.isError}
                            isLoading={episodeInteractions.isLoading}
                            onTimestepChange={(nextTimestep) => {
                              setIsPlayingFrames(false);
                              setTimestep(Math.max(0, Math.min(maxTimestep, nextTimestep)));
                            }}
                          />
                        </>
                      ) : (
                        <EpisodeProbePanel
                          comparisons={observationalComparisons.data}
                          probes={episodeProbes.data}
                          selectedProbe={selectedProbe}
                          selectedProbeRef={selectedProbeRef}
                          isError={episodeProbes.isError}
                          isLoading={episodeProbes.isLoading}
                          isComparisonError={observationalComparisons.isError}
                          isComparisonLoading={observationalComparisons.isFetching}
                          canInspectProbe={Boolean(selectedProbeSite)}
                          canJumpToProbeCall={Boolean(selectedProbeCall)}
                          onOpenComparison={openComparisonCandidate}
                          onInspectProbe={inspectProbeSite}
                          onJumpToProbeCall={jumpToProbeCall}
                          onJumpToPolicyCall={jumpToPolicyCall}
                          onProbeChange={setSelectedProbeArtifactId}
                        />
                      )}
                    </aside>
                  </div>
                </div>
              </div>
            </div>
          </section>

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
              <ActivationSitePanel
                architecture={architecture}
                sites={sites}
                activationSlice={activationSlice.data}
                feature={clampedFeature}
                cameraOverlay={cameraOverlay}
                episodeProbes={episodeProbes.data}
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
                onFeatureChange={(nextFeature) => {
                  setFeature(nextFeature);
                  setSelectedPatch(null);
                  setSelectedExpertToken(null);
                  setSelectedPromptTokenIndex(null);
                }}
                generationStep={activeGenerationStep}
                generationStepCount={generationStepCount}
                expertTokenSiteName={expertTokenSiteName}
                inspectorContext={inspectorContext}
                inspectionMode={inspectionMode}
                onGenerationStepChange={(nextStep) => {
                  setGenerationStep(nextStep);
                  setSelectedExpertToken(null);
                  setSelectedPromptTokenIndex(null);
                }}
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
              <InspectorDebugSections
                artifacts={episodeDetail.data?.artifacts ?? []}
                calls={policyCalls.data?.calls ?? []}
                generationValues={generation.data?.values ?? []}
                inspectorContext={inspectorContext}
                onTimestepChange={setTimestep}
                timestep={currentTimestep}
              />
            </div>
          </section>
        </>
      ) : null}
    </main>
  );

}
