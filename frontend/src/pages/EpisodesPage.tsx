import {
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  BarChart3,
  ChevronLeft,
  ChevronRight,
  Eye,
  EyeOff,
  GripVertical,
  Layers3,
  Maximize2,
  Pause,
  Play,
  Plus,
  RotateCcw,
  Star,
  X,
} from "lucide-react";
import {
  episodeVideoUrl,
  cachedDatasetSnapshot,
  fetchActivationSites,
  fetchActivationSlice,
  fetchAttentionMap,
  fetchDataset,
  fetchDatasetDiagnostics,
  fetchGenerationCommitment,
  fetchEpisode,
  fetchEpisodeAnnotation,
  fetchEpisodeMetrics,
  fetchExpertTokenActivations,
  fetchExpertTokenDetails,
  fetchImageTokenMap,
  fetchObjectCameraOverlay,
  fetchPatchFeatures,
  fetchPolicyCalls,
  fetchPromptAttention,
  fetchPromptFeatureMap,
  frameUrl,
  saveEpisodeAnnotation,
} from "../api/dataset";
import type {
  ActivationSite,
  ActivationSliceResponse,
  ArchitectureMetadata,
  AttentionMapResponse,
  CounterfactualPair,
  DatasetEpisode,
  EpisodeAnnotation,
  EpisodeMetric,
  ExpertTokenActivationsResponse,
  ExpertTokenDetailsResponse,
  ImagePatchAttention,
  ImageTokenMapResponse,
  ObjectCameraOverlayObject,
  PatchFeaturesResponse,
  PolicyCall,
  PromptAttentionResponse,
  PromptTokenAttention,
  SelectedPatch,
} from "../types/dataset";
import type { WorkbenchManifest } from "../types/workbench";

type EpisodesPageProps = {
  manifest?: WorkbenchManifest;
  initialTraceId?: string;
  onTraceChange?: (traceId: string) => void;
};

type InspectorContext = "vlm" | "expert" | "attention" | "other";
const ACTIVATION_CLIP_OPTIONS = [0, 1, 5, 10, 20] as const;
const TOP_CHANNEL_COUNT_OPTIONS = [8, 12, 24, 48, 96] as const;
const DEFAULT_METRIC_X_KEY = "__metric_x__";
const DEFAULT_METRIC_ORDER = [
  "action_norm",
  "eef_speed",
  "rewards",
  "gripper_open_signal",
  "generation_delta",
  "generation_start",
  "generation_end",
  "eef_x",
  "eef_y",
  "eef_z",
] as const;

type MetricPlotConfig = {
  id: string;
  xKey: string;
  yKey: string;
};
type PipelineFamily = "input" | "vlm" | "handoff" | "expert" | "action" | "other";
type InspectionMode = "features" | "attention" | "computation" | "saved_state" | "advanced";
type PipelineSiteChoice = {
  group: CaptureGroupId;
  id: string;
  label: string;
  mode: InspectionMode;
  site: ActivationSite;
};
type CaptureGroupId = "features" | "attention" | "mlp" | "saved_state" | "action" | "other";
type PipelineNode = {
  id: string;
  label: string;
  sublabel: string;
  family: PipelineFamily;
  captured: boolean;
  sites: ActivationSite[];
  allSites: ActivationSite[];
  choices: PipelineSiteChoice[];
  rawChoices: PipelineSiteChoice[];
};
type PipelineStage = {
  id: string;
  label: string;
  family: PipelineFamily;
  nodes: PipelineNode[];
};
type PipelineDiagramNode = {
  node: PipelineNode;
  stageId: string;
  x: number;
  y: number;
  width: number;
  height: number;
};
type PipelineDiagramBand = {
  className: string;
  id: string;
  label: string;
  x: number;
  y: number;
  width: number;
  height: number;
};
type PipelineDiagramArrow = {
  className: string;
  id: string;
  label?: string;
  labelAnchor?: "start" | "middle" | "end";
  labelX?: number;
  labelY?: number;
  path: string;
};
type PipelineDiagramPort = {
  className: string;
  id: string;
  label?: string;
  radius?: number;
  textAnchor?: "start" | "middle" | "end";
  x: number;
  y: number;
};
type PipelineDiagramLayout = {
  arrows: PipelineDiagramArrow[];
  bands: PipelineDiagramBand[];
  height: number;
  nodes: PipelineDiagramNode[];
  ports: PipelineDiagramPort[];
  width: number;
};
type CameraOverlayPayload = Pick<
  ImageTokenMapResponse | ExpertTokenDetailsResponse | AttentionMapResponse,
  "available" | "maps" | "note" | "reason"
>;

export function EpisodesPage({
  manifest,
  initialTraceId = "",
  onTraceChange,
}: EpisodesPageProps) {
  const queryClient = useQueryClient();
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

  const episodes = dataset.data?.episodes ?? episodesFromManifest(manifest);
  const [timestep, setTimestep] = useState(0);
  const [isPlayingFrames, setIsPlayingFrames] = useState(false);
  const [playbackFps, setPlaybackFps] = useState(10);
  const [showObjectOverlay, setShowObjectOverlay] = useState(true);
  const [showAttentionOverlay, setShowAttentionOverlay] = useState(true);
  const [selectedSiteName, setSelectedSiteName] = useState("");
  const [inspectionMode, setInspectionMode] = useState<InspectionMode>("features");
  const [attentionHead, setAttentionHead] = useState<number | null>(null);
  const [attentionQueryToken, setAttentionQueryToken] = useState<number | null>(null);
  const [feature, setFeature] = useState(0);
  const [activationClipPercent, setActivationClipPercent] = useState(0);
  const [topChannelCount, setTopChannelCount] = useState(12);
  const [inspectorWidthPct, setInspectorWidthPct] = useState(38);
  const [selectedPatch, setSelectedPatch] = useState<SelectedPatch | null>(null);
  const [selectedExpertToken, setSelectedExpertToken] = useState<number | null>(null);
  const [generationStep, setGenerationStep] = useState(0);

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
    onTraceChange?.(traceId);
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
  const sites = activationSites.data?.sites ?? [];
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
      selectedExpertToken,
      activeGenerationStep,
    ],
    queryFn: () =>
      fetchExpertTokenDetails(
        activeTraceId,
        activeCall?.index ?? 0,
        expertTokenSiteName,
        clampedFeature,
        selectedExpertToken ?? 0,
        activeGenerationStep,
      ),
    enabled: Boolean(
      inspectorContext === "expert" &&
        activeTraceId &&
        activeCall &&
        expertTokenSiteName &&
      selectedExpertToken !== null,
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
    inspectorContext === "expert" && expertTokenDetails.data?.available
      ? expertTokenDetails.data
      : inspectorContext === "attention" && attentionMap.data?.available
        ? attentionMap.data
      : imageTokenMap.data;

  const currentTimestep = Math.min(timestep, maxTimestep);

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
                      />
                      <CameraGrid
                        cacheKey={frameCacheKey}
                        cameras={cameras}
                        imageTokenMap={cameraOverlay}
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
                      <MetricPlotPanel
                        metrics={metrics}
                        timestep={currentTimestep}
                        onTimestepChange={setTimestep}
                      />
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
                patchFeatures={patchFeatures.data}
                expertTokenActivations={expertTokenActivations.data}
                expertTokenDetails={expertTokenDetails.data}
                activationSliceFetching={activationSlice.isFetching}
                activationSlicePlaceholder={activationSlice.isPlaceholderData}
                activationClipPercent={activationClipPercent}
                attentionHead={attentionHead}
                attentionQueryToken={attentionQueryToken}
                selectedPatch={selectedPatch}
                selectedExpertToken={selectedExpertToken}
                onFeatureChange={(nextFeature) => {
                  setFeature(nextFeature);
                  setSelectedPatch(null);
                  setSelectedExpertToken(null);
                }}
                generationStep={activeGenerationStep}
                generationStepCount={generationStepCount}
                expertTokenSiteName={expertTokenSiteName}
                inspectorContext={inspectorContext}
                inspectionMode={inspectionMode}
                onGenerationStepChange={(nextStep) => {
                  setGenerationStep(nextStep);
                  setSelectedExpertToken(null);
                }}
                onAttentionHeadChange={setAttentionHead}
                onAttentionQueryTokenChange={setAttentionQueryToken}
                onExpertTokenChange={setSelectedExpertToken}
                onActivationClipPercentChange={setActivationClipPercent}
                onInspectionModeChange={setInspectionMode}
                onTopChannelCountChange={setTopChannelCount}
                selectedSite={selectedSite}
                selectedSiteName={activeSelectedSiteName}
                topChannelCount={topChannelCount}
                onSiteChange={(siteName) => {
                  setSelectedSiteName(siteName);
                  setAttentionHead(null);
                  setAttentionQueryToken(null);
                  setFeature(0);
                  setSelectedPatch(null);
                  setSelectedExpertToken(null);
                  setGenerationStep(0);
                }}
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

function CameraGrid({
  cacheKey,
  cameras,
  imageTokenMap,
  isPlaying,
  maxTimestep,
  showAttentionOverlay,
  showObjectOverlay,
  onPatchSelect,
  selectedPatch,
  traceId,
  timestep,
}: {
  cacheKey: string;
  cameras: string[];
  imageTokenMap?: CameraOverlayPayload;
  isPlaying: boolean;
  maxTimestep: number;
  showAttentionOverlay: boolean;
  showObjectOverlay: boolean;
  onPatchSelect: (patch: SelectedPatch | null) => void;
  selectedPatch: SelectedPatch | null;
  traceId: string;
  timestep: number;
}) {
  const preloadedFrames = useRef<Set<string>>(new Set());
  const preloadImages = useRef<HTMLImageElement[]>([]);

  useEffect(() => {
    preloadedFrames.current.clear();
    preloadImages.current = [];
  }, [cacheKey, traceId]);

  useEffect(() => {
    if (!cameras.length || maxTimestep <= timestep) {
      return;
    }
    if (!isPlaying) {
      return;
    }
    for (let next = timestep + 1; next <= Math.min(maxTimestep, timestep + 2); next += 1) {
      for (const camera of cameras) {
        const key = `${cacheKey}:${traceId}:${camera}:${next}`;
        if (preloadedFrames.current.has(key)) {
          continue;
        }
        preloadedFrames.current.add(key);
        const image = new Image();
        image.decoding = "async";
        image.src = frameUrl(traceId, camera, next, cacheKey);
        preloadImages.current.push(image);
      }
    }
    if (preloadImages.current.length > 80) {
      preloadImages.current = preloadImages.current.slice(-80);
    }
  }, [cacheKey, cameras, isPlaying, maxTimestep, timestep, traceId]);

  if (!cameras.length) {
    return <div className="empty-state">No camera streams for this episode.</div>;
  }
  return (
    <div className="camera-grid">
      {cameras.map((camera) => (
        <CameraFrame
          cacheKey={cacheKey}
          camera={camera}
          imageTokenMap={imageTokenMap}
          isPlaying={isPlaying}
          showAttentionOverlay={showAttentionOverlay}
          showObjectOverlay={showObjectOverlay}
          key={camera}
          onPatchSelect={onPatchSelect}
          selectedPatch={selectedPatch}
          timestep={timestep}
          traceId={traceId}
        />
      ))}
    </div>
  );
}

function CameraFrame({
  cacheKey,
  camera,
  imageTokenMap,
  isPlaying,
  showAttentionOverlay,
  showObjectOverlay,
  onPatchSelect,
  selectedPatch,
  timestep,
  traceId,
}: {
  cacheKey: string;
  camera: string;
  imageTokenMap?: CameraOverlayPayload;
  isPlaying: boolean;
  showAttentionOverlay: boolean;
  showObjectOverlay: boolean;
  onPatchSelect: (patch: SelectedPatch | null) => void;
  selectedPatch: SelectedPatch | null;
  timestep: number;
  traceId: string;
}) {
  const frameSrc = frameUrl(traceId, camera, timestep, cacheKey);
  const [loadedFrameSrc, setLoadedFrameSrc] = useState("");
  const frameReady = loadedFrameSrc === frameSrc;
  const objectOverlay = useQuery({
    queryKey: ["object-camera-overlay", traceId, camera, timestep],
    queryFn: () => fetchObjectCameraOverlay(traceId, camera, timestep),
    enabled: !isPlaying,
    staleTime: 60_000,
  });
  const visibleObjects = useMemo(
    () => {
      if (
        isPlaying ||
        !frameReady ||
        objectOverlay.data?.timestep !== timestep
      ) {
        return [];
      }
      return (objectOverlay.data?.objects ?? []).filter(
        (object) =>
          object.in_frame &&
          Number.isFinite(object.x ?? NaN) &&
          Number.isFinite(object.y ?? NaN),
      );
    },
    [frameReady, isPlaying, objectOverlay.data?.objects, objectOverlay.data?.timestep, timestep],
  );
  const [hoveredObject, setHoveredObject] = useState<ObjectCameraOverlayObject | null>(null);

  const activeHoveredObject = isPlaying || !frameReady ? null : hoveredObject;

  return (
    <figure className="camera-frame">
      <div
        className="camera-image-wrap"
        onPointerMove={(event) => {
          if (isPlaying) {
            setHoveredObject(null);
            return;
          }
          if (!visibleObjects.length) {
            setHoveredObject(null);
            return;
          }
          const rect = event.currentTarget.getBoundingClientRect();
          const x = event.clientX - rect.left;
          const y = event.clientY - rect.top;
          const normX = x / Math.max(1, rect.width);
          const normY = y / Math.max(1, rect.height);
          let containing: ObjectCameraOverlayObject | null = null;
          let containingArea = Number.POSITIVE_INFINITY;
          for (const object of visibleObjects) {
            const bbox = normalizedObjectBbox(object);
            if (!bbox) {
              continue;
            }
            if (normX < bbox.x0 || normX > bbox.x1 || normY < bbox.y0 || normY > bbox.y1) {
              continue;
            }
            const area = Math.max(0, bbox.x1 - bbox.x0) * Math.max(0, bbox.y1 - bbox.y0);
            if (area < containingArea) {
              containing = object;
              containingArea = area;
            }
          }
          if (containing) {
            setHoveredObject(containing);
            return;
          }
          let nearest: ObjectCameraOverlayObject | null = null;
          let nearestDistance = Number.POSITIVE_INFINITY;
          for (const object of visibleObjects) {
            const objectX = Number(object.x) * rect.width;
            const objectY = Number(object.y) * rect.height;
            const distance = Math.hypot(objectX - x, objectY - y);
            if (distance < nearestDistance) {
              nearest = object;
              nearestDistance = distance;
            }
          }
          setHoveredObject(nearestDistance <= 44 ? nearest : null);
        }}
        onPointerLeave={() => setHoveredObject(null)}
      >
        <img
          alt={`${traceId} ${camera} timestep ${timestep}`}
          decoding="async"
          src={frameSrc}
          onLoad={() => setLoadedFrameSrc(frameSrc)}
        />
        <ObjectCameraOverlay
          objects={visibleObjects}
          hoveredObject={activeHoveredObject}
          showMarks={showObjectOverlay}
        />
        {showAttentionOverlay && (isPlaying || frameReady) ? (
          <ActivationGridOverlay
            camera={camera}
            imageTokenMap={imageTokenMap}
            selectedPatch={selectedPatch}
            onPatchSelect={onPatchSelect}
          />
        ) : null}
      </div>
      <figcaption>
        <span>{camera}</span>
        <span>t={timestep}</span>
      </figcaption>
    </figure>
  );
}

function ObjectCameraOverlay({
  hoveredObject,
  objects,
  showMarks,
}: {
  hoveredObject: ObjectCameraOverlayObject | null;
  objects: ObjectCameraOverlayObject[];
  showMarks: boolean;
}) {
  if (!objects.length) {
    return null;
  }
  return (
    <div className="object-camera-overlay" aria-hidden="true">
      {showMarks ? objects.map((object) => {
        const bbox = normalizedObjectBbox(object);
        const active = hoveredObject?.object_index === object.object_index;
        if (!bbox) {
          return null;
        }
        return (
          <span
            className={`object-camera-bbox${active ? " active" : ""}`}
            key={`bbox:${object.object_index}:${object.object_name}`}
            style={{
              left: `${bbox.x0 * 100}%`,
              top: `${bbox.y0 * 100}%`,
              width: `${Math.max(0, bbox.x1 - bbox.x0) * 100}%`,
              height: `${Math.max(0, bbox.y1 - bbox.y0) * 100}%`,
            }}
          />
        );
      }) : null}
      {objects.map((object) => {
        const x = Number(object.x);
        const y = Number(object.y);
        const left = `${Math.min(98, Math.max(2, x * 100))}%`;
        const top = `${Math.min(98, Math.max(2, y * 100))}%`;
        const active = hoveredObject?.object_index === object.object_index;
        if (!showMarks && !active) {
          return null;
        }
        const edgeClass = objectOverlayEdgeClass(x, y);
        return (
          <div
            className={["object-camera-marker", active ? "active" : "", edgeClass]
              .filter(Boolean)
              .join(" ")}
            key={`${object.object_index}:${object.object_name}`}
            style={{ left, top }}
          >
            {showMarks ? <span className="object-camera-dot" /> : null}
            {active ? <span className="object-camera-label">{object.object_name}</span> : null}
          </div>
        );
      })}
      {hoveredObject ? (
        <div
          className={["object-camera-tooltip", objectOverlayEdgeClass(
            Number(hoveredObject.x),
            Number(hoveredObject.y),
          )]
            .filter(Boolean)
            .join(" ")}
          style={{
            left: `${Math.min(96, Math.max(4, Number(hoveredObject.x) * 100))}%`,
            top: `${Math.min(94, Math.max(4, Number(hoveredObject.y) * 100))}%`,
          }}
        >
          <strong>{hoveredObject.object_name}</strong>
          <span>{hoveredObject.object_kind ?? "object"}</span>
          {hoveredObject.position_world ? (
            <span>xyz {formatVector(hoveredObject.position_world, 3)}</span>
          ) : null}
          {hoveredObject.geometry_center_world ? (
            <span>geom {formatVector(hoveredObject.geometry_center_world, 3)}</span>
          ) : null}
          {hoveredObject.quaternion_xyzw ? (
            <span>quat {formatVector(hoveredObject.quaternion_xyzw, 3)}</span>
          ) : null}
          {hoveredObject.projection_kind ? <span>{hoveredObject.projection_kind}</span> : null}
        </div>
      ) : null}
    </div>
  );
}

function objectOverlayEdgeClass(x: number, y: number) {
  return [
    x > 0.72 ? "edge-right" : "",
    x < 0.18 ? "edge-left" : "",
    y > 0.72 ? "edge-bottom" : "",
  ]
    .filter(Boolean)
    .join(" ");
}

function normalizedObjectBbox(object: ObjectCameraOverlayObject) {
  const bbox = object.bbox;
  if (!bbox) {
    return null;
  }
  const x0 = Number(bbox.x0);
  const y0 = Number(bbox.y0);
  const x1 = Number(bbox.x1);
  const y1 = Number(bbox.y1);
  if (![x0, y0, x1, y1].every(Number.isFinite)) {
    return null;
  }
  return {
    x0: Math.max(0, Math.min(1, Math.min(x0, x1))),
    y0: Math.max(0, Math.min(1, Math.min(y0, y1))),
    x1: Math.max(0, Math.min(1, Math.max(x0, x1))),
    y1: Math.max(0, Math.min(1, Math.max(y0, y1))),
  };
}

function ActivationGridOverlay({
  camera,
  imageTokenMap,
  onPatchSelect,
  selectedPatch,
}: {
  camera: string;
  imageTokenMap?: CameraOverlayPayload;
  onPatchSelect: (patch: SelectedPatch | null) => void;
  selectedPatch: SelectedPatch | null;
}) {
  const cameraMapValues = imageTokenMap?.maps?.[camera]?.values;
  const values = useMemo(() => cameraMapValues ?? [], [cameraMapValues]);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [layoutVersion, setLayoutVersion] = useState(0);

  useEffect(() => {
    const target = canvasRef.current?.parentElement;
    if (!target) {
      return;
    }
    const observer = new ResizeObserver(() => {
      setLayoutVersion((version) => version + 1);
    });
    observer.observe(target);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageTokenMap?.available || !values.length) {
      return;
    }

    const rect = canvas.getBoundingClientRect();
    const scale = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(rect.width * scale));
    canvas.height = Math.max(1, Math.round(rect.height * scale));

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);

    const flat = values.flat().filter((value) => Number.isFinite(value));
    const maxAbs = Math.max(...flat.map((value) => Math.abs(value)), 1e-6);
    const rows = values.length;
    const cols = Math.max(1, values[0]?.length ?? 1);
    const cellWidth = rect.width / cols;
    const cellHeight = rect.height / rows;

    values.forEach((row, rowIndex) => {
      row.forEach((value, colIndex) => {
        if (!Number.isFinite(value)) {
          return;
        }
        ctx.fillStyle = signedActivationColor(value, maxAbs);
        ctx.fillRect(colIndex * cellWidth, rowIndex * cellHeight, cellWidth, cellHeight);
      });
    });

    ctx.strokeStyle = "rgba(255, 255, 255, 0.28)";
    ctx.lineWidth = 1;
    for (let row = 1; row < rows; row += 1) {
      const y = row * cellHeight;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(rect.width, y);
      ctx.stroke();
    }
    for (let col = 1; col < cols; col += 1) {
      const x = col * cellWidth;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, rect.height);
      ctx.stroke();
    }

    if (selectedPatch?.camera === camera) {
      ctx.strokeStyle = "rgba(255, 255, 255, 0.98)";
      ctx.lineWidth = 3;
      ctx.strokeRect(
        selectedPatch.col * cellWidth + 1.5,
        selectedPatch.row * cellHeight + 1.5,
        Math.max(0, cellWidth - 3),
        Math.max(0, cellHeight - 3),
      );
      ctx.strokeStyle = "rgba(15, 23, 42, 0.88)";
      ctx.lineWidth = 1;
      ctx.strokeRect(
        selectedPatch.col * cellWidth + 4,
        selectedPatch.row * cellHeight + 4,
        Math.max(0, cellWidth - 8),
        Math.max(0, cellHeight - 8),
      );
    }
  }, [camera, imageTokenMap?.available, layoutVersion, selectedPatch, values]);

  if (!imageTokenMap?.available || !values.length) {
    return null;
  }
  return (
    <canvas
      ref={canvasRef}
      className="activation-grid-overlay"
      aria-label={`${camera} activation heatmap overlay`}
      role="button"
      tabIndex={0}
      onClick={(event) => {
        const rect = event.currentTarget.getBoundingClientRect();
        const cols = Math.max(1, values[0]?.length ?? 1);
        const rows = values.length;
        const col = Math.max(
          0,
          Math.min(cols - 1, Math.floor(((event.clientX - rect.left) / rect.width) * cols)),
        );
        const row = Math.max(
          0,
          Math.min(rows - 1, Math.floor(((event.clientY - rect.top) / rect.height) * rows)),
        );
        onPatchSelect({ camera, row, col });
      }}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onPatchSelect({ camera, row: 0, col: 0 });
        }
      }}
    />
  );
}

function FramePlaybackControls({
  cacheKey,
  fps,
  isPlaying,
  maxTimestep,
  policyCalls,
  onAttentionOverlayToggle,
  onFpsChange,
  onObjectOverlayToggle,
  onReset,
  onTimestepChange,
  onToggle,
  showAttentionOverlay,
  showObjectOverlay,
  timestep,
  traceId,
}: {
  cacheKey: string;
  fps: number;
  isPlaying: boolean;
  maxTimestep: number;
  policyCalls: PolicyCall[];
  onAttentionOverlayToggle: () => void;
  onFpsChange: (fps: number) => void;
  onObjectOverlayToggle: () => void;
  onReset: () => void;
  onTimestepChange: (timestep: number) => void;
  onToggle: () => void;
  showAttentionOverlay: boolean;
  showObjectOverlay: boolean;
  timestep: number;
  traceId: string;
}) {
  const activeCall = policyCalls.find(
    (call) => timestep >= call.segment_start && timestep <= call.segment_end,
  );
  return (
    <div className="frame-playback">
      <div className="playback-command-row">
        <button
          aria-label={isPlaying ? "Pause episode" : "Play episode"}
          className="playback-icon-button playback-primary"
          title={isPlaying ? "Pause" : "Play"}
          type="button"
          onClick={onToggle}
        >
          {isPlaying ? <Pause size={15} /> : <Play size={15} />}
        </button>
        <button
          aria-label="Reset episode playback"
          className="playback-icon-button frame-reset"
          title="Reset"
          type="button"
          onClick={onReset}
        >
          <RotateCcw size={15} />
        </button>
        <select
          aria-label="Playback speed"
          className="fps-select"
          value={fps}
          onChange={(event) => onFpsChange(Number(event.target.value))}
        >
          {[2, 5, 10, 15].map((option) => (
            <option key={option} value={option}>
              {option} fps
            </option>
          ))}
        </select>
        <button
          aria-pressed={showObjectOverlay}
          className={`object-overlay-toggle${showObjectOverlay ? " active" : ""}`}
          title="Toggle object boxes and hover labels"
          type="button"
          onClick={onObjectOverlayToggle}
        >
          {showObjectOverlay ? <Eye size={15} /> : <EyeOff size={15} />}
          <span>Objects</span>
        </button>
        <button
          aria-pressed={showAttentionOverlay}
          className={`attention-overlay-toggle${showAttentionOverlay ? " active" : ""}`}
          title="Toggle attention and activation patch overlay"
          type="button"
          onClick={onAttentionOverlayToggle}
        >
          {showAttentionOverlay ? <Layers3 size={15} /> : <EyeOff size={15} />}
          <span>Model overlay</span>
        </button>
        <span className="playback-policy-readout">
          {activeCall
            ? `Policy call ${activeCall.index} / t=${activeCall.segment_start}-${activeCall.segment_end}`
            : `${policyCalls.length} policy calls`}
        </span>
        <a
          className="mp4-link"
          href={episodeVideoUrl(traceId, "all", cacheKey)}
          target="_blank"
          rel="noreferrer"
        >
          MP4
        </a>
      </div>
      <TimelineControl
        timestep={timestep}
        maxTimestep={maxTimestep}
        onChange={onTimestepChange}
        policyCalls={policyCalls}
      />
    </div>
  );
}

function TimelineControl({
  timestep,
  maxTimestep,
  onChange,
  policyCalls = [],
}: {
  timestep: number;
  maxTimestep: number;
  onChange: (timestep: number) => void;
  policyCalls?: PolicyCall[];
}) {
  const [draftTimestep, setDraftTimestep] = useState(timestep);
  const clampTimestep = (value: number) => Math.max(0, Math.min(maxTimestep, value));
  const commitImmediate = (value: number) => {
    const next = clampTimestep(value);
    setDraftTimestep(next);
    onChange(next);
  };

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => setDraftTimestep(timestep));
    return () => window.cancelAnimationFrame(frame);
  }, [timestep]);

  const boundedMax = Math.max(1, maxTimestep);
  return (
    <div className="timeline-control">
      <div className="timeline-main">
        <div className="timeline-readout">
          <span>Timestep {draftTimestep} / {maxTimestep}</span>
        </div>
        <div className="timeline-track-shell">
          <input
            aria-label="Episode timestep"
            max={maxTimestep}
            min={0}
            type="range"
            value={draftTimestep}
            onBlur={() => commitImmediate(draftTimestep)}
            onChange={(event) => setDraftTimestep(clampTimestep(Number(event.target.value)))}
            onKeyUp={() => commitImmediate(draftTimestep)}
            onPointerCancel={() => commitImmediate(draftTimestep)}
            onPointerUp={() => commitImmediate(draftTimestep)}
          />
          <div className="timeline-call-markers" aria-label="Policy call checkpoints">
            {policyCalls.map((call) => {
              const markerTimestep = call.env_timestep ?? call.segment_start;
              const left = Math.max(0, Math.min(100, (markerTimestep / boundedMax) * 100));
              const active =
                draftTimestep >= call.segment_start && draftTimestep <= call.segment_end;
              return (
                <button
                  aria-label={`Jump to policy call ${call.index}, timesteps ${call.segment_start} to ${call.segment_end}`}
                  className={active ? "timeline-call-marker active" : "timeline-call-marker"}
                  key={call.index}
                  style={{ left: `${left}%` }}
                  title={`Policy call ${call.index}: t=${call.segment_start}-${call.segment_end}`}
                  type="button"
                  onClick={() => commitImmediate(markerTimestep)}
                />
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricPlotPanel({
  metrics,
  timestep,
  onTimestepChange,
}: {
  metrics: EpisodeMetric[];
  timestep: number;
  onTimestepChange: (timestep: number) => void;
}) {
  const [plots, setPlots] = useState<MetricPlotConfig[] | null>(null);
  const [addOpen, setAddOpen] = useState(false);
  const [draggedPlotId, setDraggedPlotId] = useState<string | null>(null);

  const metricByKey = useMemo(
    () => new Map(metrics.map((metric) => [metric.key, metric])),
    [metrics],
  );
  const resolvedPlots = useMemo(
    () => reconcileMetricPlots(plots ?? [], metrics),
    [metrics, plots],
  );
  const visiblePlots = resolvedPlots.filter((plot) => metricByKey.has(plot.yKey));

  const movePlot = (plotId: string, direction: -1 | 1) => {
    setPlots((current) => moveMetricPlot(reconcileMetricPlots(current ?? [], metrics), plotId, direction));
  };
  const removePlot = (plotId: string) => {
    setPlots((current) =>
      reconcileMetricPlots(current ?? [], metrics).filter((plot) => plot.id !== plotId),
    );
  };
  const addPlot = (plot: Omit<MetricPlotConfig, "id">) => {
    setPlots((current) => [
      ...reconcileMetricPlots(current ?? [], metrics),
      {
        ...plot,
        id: `custom:${plot.xKey}:${plot.yKey}:${Date.now()}`,
      },
    ]);
    setAddOpen(false);
  };
  const dropPlot = (targetPlotId: string) => {
    if (!draggedPlotId || draggedPlotId === targetPlotId) {
      setDraggedPlotId(null);
      return;
    }
    setPlots((current) =>
      reorderMetricPlots(reconcileMetricPlots(current ?? [], metrics), draggedPlotId, targetPlotId),
    );
    setDraggedPlotId(null);
  };

  return (
    <section className="episode-tool-panel metric-dashboard">
      <header>
        <div className="icon-label">
          <BarChart3 size={16} />
          <strong>Episode Metrics</strong>
        </div>
        <button
          className="metric-add-button"
          disabled={!metrics.length}
          type="button"
          onClick={() => setAddOpen(true)}
        >
          <Plus size={15} />
          <span>Add plot</span>
        </button>
      </header>
      {visiblePlots.length ? (
        <div className="metric-tile-grid">
          {visiblePlots.map((plot, index) => {
            const yMetric = metricByKey.get(plot.yKey);
            const xMetric =
              plot.xKey === DEFAULT_METRIC_X_KEY ? undefined : metricByKey.get(plot.xKey);
            if (!yMetric) {
              return null;
            }
            return (
              <article
                className="metric-tile"
                draggable
                key={plot.id}
                onDragEnd={() => setDraggedPlotId(null)}
                onDragOver={(event) => event.preventDefault()}
                onDragStart={() => setDraggedPlotId(plot.id)}
                onDrop={() => dropPlot(plot.id)}
              >
                <div className="metric-tile-head">
                  <span className="metric-drag-handle" title="Move plot">
                    <GripVertical size={15} />
                  </span>
                  <div className="metric-tile-title">
                    <strong>{yMetric.label}</strong>
                    <span>{metricAxisSummary(yMetric, xMetric)}</span>
                  </div>
                  <div className="metric-tile-actions">
                    <button
                      aria-label={`Move ${yMetric.label} left`}
                      disabled={index === 0}
                      title="Move left"
                      type="button"
                      onClick={() => movePlot(plot.id, -1)}
                    >
                      <ChevronLeft size={14} />
                    </button>
                    <button
                      aria-label={`Move ${yMetric.label} right`}
                      disabled={index === visiblePlots.length - 1}
                      title="Move right"
                      type="button"
                      onClick={() => movePlot(plot.id, 1)}
                    >
                      <ChevronRight size={14} />
                    </button>
                    <button
                      aria-label={`Remove ${yMetric.label}`}
                      title="Remove"
                      type="button"
                      onClick={() => removePlot(plot.id)}
                    >
                      <X size={14} />
                    </button>
                  </div>
                </div>
                <SeriesPlot
                  metric={yMetric}
                  xMetric={xMetric}
                  timestep={timestep}
                  onSelectIndex={onTimestepChange}
                />
                {yMetric.description ? <p>{yMetric.description}</p> : null}
              </article>
            );
          })}
        </div>
      ) : (
        <div className="empty-state">No metrics available for this episode.</div>
      )}
      {addOpen ? (
        <MetricPlotDialog
          metrics={metrics}
          onAdd={addPlot}
          onClose={() => setAddOpen(false)}
        />
      ) : null}
    </section>
  );
}

function MetricPlotDialog({
  metrics,
  onAdd,
  onClose,
}: {
  metrics: EpisodeMetric[];
  onAdd: (plot: Omit<MetricPlotConfig, "id">) => void;
  onClose: () => void;
}) {
  const [xKey, setXKey] = useState(DEFAULT_METRIC_X_KEY);
  const [yKey, setYKey] = useState(metrics[0]?.key ?? "");
  const yMetric = metrics.find((metric) => metric.key === yKey) ?? metrics[0];
  const resolvedYKey = yMetric?.key ?? "";
  const compatibleXMetrics = metrics.filter(
    (metric) => !yMetric || metric.values.length === yMetric.values.length,
  );
  const resolvedXKey =
    xKey === DEFAULT_METRIC_X_KEY ||
    compatibleXMetrics.some((metric) => metric.key === xKey)
      ? xKey
      : DEFAULT_METRIC_X_KEY;
  const canAdd = Boolean(yMetric);

  return (
    <div className="metric-dialog-backdrop" role="presentation" onMouseDown={onClose}>
      <div
        aria-label="Add metric plot"
        aria-modal="true"
        className="metric-dialog"
        role="dialog"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header>
          <div>
            <strong>Add plot</strong>
            <span>Select the series to use for each axis.</span>
          </div>
          <button aria-label="Close add plot" type="button" onClick={onClose}>
            <X size={16} />
          </button>
        </header>
        <div className="metric-dialog-grid">
          <label>
            X axis
            <select value={resolvedXKey} onChange={(event) => setXKey(event.target.value)}>
              <option value={DEFAULT_METRIC_X_KEY}>Metric timeline</option>
              {compatibleXMetrics.map((metric) => (
                <option key={metric.key} value={metric.key}>
                  {metric.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Y axis
            <select value={resolvedYKey} onChange={(event) => setYKey(event.target.value)}>
              {metrics.map((metric) => (
                <option key={metric.key} value={metric.key}>
                  {metric.label}
                </option>
              ))}
            </select>
          </label>
        </div>
        <footer>
          <button type="button" onClick={onClose}>
            Cancel
          </button>
          <button
            className="metric-dialog-primary"
            disabled={!canAdd}
            type="button"
            onClick={() => onAdd({ xKey: resolvedXKey, yKey: resolvedYKey })}
          >
            Add plot
          </button>
        </footer>
      </div>
    </div>
  );
}

function SeriesPlot({
  metric,
  xMetric,
  timestep,
  onSelectIndex,
}: {
  metric: EpisodeMetric;
  xMetric?: EpisodeMetric;
  timestep: number;
  onSelectIndex: (index: number) => void;
}) {
  const yValues = metric.values ?? [];
  const count = xMetric ? Math.min(yValues.length, xMetric.values.length) : yValues.length;
  const values = yValues.slice(0, count);
  const width = 640;
  const height = 168;
  const padLeft = 50;
  const padRight = 18;
  const padTop = 18;
  const padBottom = 34;
  const xValues = xMetric
    ? xMetric.values.slice(0, count)
    : metric.x_values?.length === values.length
      ? metric.x_values
      : values.map((_, index) => index);
  const plotPoints = values
    .map((value, index) => ({ index, value, x: xValues[index] ?? index }))
    .filter((point) => Number.isFinite(point.value) && Number.isFinite(point.x));
  const finiteValues = plotPoints.map((point) => point.value);
  const finiteX = plotPoints.map((point) => point.x);
  const minX = Math.min(...finiteX, 0);
  const maxX = Math.max(...finiteX, Math.max(1, values.length - 1));
  const min = Math.min(...finiteValues, 0);
  const max = Math.max(...finiteValues, 1);
  const span = max - min || 1;
  const xSpan = maxX - minX || 1;
  const xForValue = (value: number) =>
    padLeft + ((value - minX) / xSpan) * (width - padLeft - padRight);
  const xFor = (index: number) => xForValue(xValues[index] ?? index);
  const yFor = (value: number) =>
    height - padBottom - ((value - min) / span) * (height - padTop - padBottom);
  const points = plotPoints.map((point) => `${xForValue(point.x)},${yFor(point.value)}`).join(" ");
  const activeIndex = nearestMetricIndex(metric, timestep);
  const activeX = xFor(Math.min(activeIndex, Math.max(0, values.length - 1)));
  const xLabel = xMetric?.label || metric.x_label || "Environment timestep";
  const yLabel = metric.y_label || metric.label;
  const yUnit = metric.y_unit ? ` (${metric.y_unit})` : "";

  return (
    <svg
      className="episode-line-plot"
      role="img"
      viewBox={`0 0 ${width} ${height}`}
      onClick={(event) => {
        const rect = event.currentTarget.getBoundingClientRect();
        const svgX = ((event.clientX - rect.left) / rect.width) * width;
        const plotRatio = Math.max(
          0,
          Math.min(1, (svgX - padLeft) / (width - padLeft - padRight)),
        );
        const clickedX = minX + plotRatio * xSpan;
        const clickedIndex = nearestXIndex(xValues, clickedX);
        const timelineX = metric.x_values?.[clickedIndex] ?? clickedIndex;
        onSelectIndex(Math.round(timelineX));
      }}
    >
      <line className="plot-axis" x1={padLeft} x2={width - padRight} y1={height - padBottom} y2={height - padBottom} />
      <line className="plot-axis" x1={padLeft} x2={padLeft} y1={padTop} y2={height - padBottom} />
      <polyline className="plot-series" points={points} />
      <line className="plot-cursor" x1={activeX} x2={activeX} y1={padTop} y2={height - padBottom} />
      {plotPoints.map((point) => (
        <circle
          className={point.index === activeIndex ? "plot-point active" : "plot-point"}
          cx={xForValue(point.x)}
          cy={yFor(point.value)}
          key={`${metric.key}-${point.index}`}
          r={point.index === activeIndex ? 3.8 : 2.4}
        />
      ))}
      <text className="plot-label" x={padLeft} y={13}>
        {max.toFixed(3)}
      </text>
      <text className="plot-label" x={padLeft} y={height - padBottom + 15}>
        {min.toFixed(3)}
      </text>
      <text className="plot-label plot-x-label" x={width / 2} y={height - 8}>
        {xLabel}
      </text>
      <text
        className="plot-label plot-y-label"
        transform={`translate(14 ${height / 2}) rotate(-90)`}
      >
        {yLabel}
        {yUnit}
      </text>
    </svg>
  );
}

function reconcileMetricPlots(
  current: MetricPlotConfig[],
  metrics: EpisodeMetric[],
): MetricPlotConfig[] {
  const metricKeys = new Set(metrics.map((metric) => metric.key));
  const validCurrent = current.filter(
    (plot) =>
      metricKeys.has(plot.yKey) &&
      (plot.xKey === DEFAULT_METRIC_X_KEY || metricKeys.has(plot.xKey)),
  );
  if (validCurrent.length) {
    return validCurrent;
  }
  return defaultMetricPlots(metrics);
}

function defaultMetricPlots(metrics: EpisodeMetric[]): MetricPlotConfig[] {
  const metricByKey = new Map(metrics.map((metric) => [metric.key, metric]));
  const defaultMetricKeys = new Set<string>(DEFAULT_METRIC_ORDER);
  const ordered = [
    ...DEFAULT_METRIC_ORDER.flatMap((key) => {
      const metric = metricByKey.get(key);
      return metric ? [metric] : [];
    }),
    ...metrics.filter((metric) => !defaultMetricKeys.has(metric.key)),
  ];
  return ordered.slice(0, 6).map((metric) => ({
    id: `default:${metric.key}`,
    xKey: DEFAULT_METRIC_X_KEY,
    yKey: metric.key,
  }));
}

function moveMetricPlot(
  plots: MetricPlotConfig[],
  plotId: string,
  direction: -1 | 1,
): MetricPlotConfig[] {
  const index = plots.findIndex((plot) => plot.id === plotId);
  const nextIndex = index + direction;
  if (index < 0 || nextIndex < 0 || nextIndex >= plots.length) {
    return plots;
  }
  const next = [...plots];
  const [plot] = next.splice(index, 1);
  next.splice(nextIndex, 0, plot);
  return next;
}

function reorderMetricPlots(
  plots: MetricPlotConfig[],
  draggedPlotId: string,
  targetPlotId: string,
): MetricPlotConfig[] {
  const from = plots.findIndex((plot) => plot.id === draggedPlotId);
  const to = plots.findIndex((plot) => plot.id === targetPlotId);
  if (from < 0 || to < 0 || from === to) {
    return plots;
  }
  const next = [...plots];
  const [plot] = next.splice(from, 1);
  next.splice(to, 0, plot);
  return next;
}

function nearestXIndex(values: number[], target: number): number {
  if (!values.length) {
    return 0;
  }
  let best = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  values.forEach((value, index) => {
    const distance = Math.abs(value - target);
    if (distance < bestDistance) {
      best = index;
      bestDistance = distance;
    }
  });
  return best;
}

function metricAxisSummary(yMetric: EpisodeMetric, xMetric?: EpisodeMetric): string {
  const xLabel = xMetric?.label ?? yMetric.x_label ?? "Timeline";
  const yLabel = yMetric.y_label ?? yMetric.label;
  return `${xLabel} -> ${yLabel}`;
}

function InspectorDebugSections({
  artifacts,
  calls,
  generationValues,
  inspectorContext,
  onTimestepChange,
  timestep,
}: {
  artifacts: Record<string, unknown>[];
  calls: PolicyCall[];
  generationValues: (number | null)[][];
  inspectorContext: InspectorContext;
  onTimestepChange: (timestep: number) => void;
  timestep: number;
}) {
  const generationColumns = generationValues[0]?.length ?? 0;
  return (
    <div className="inspector-debug-stack" aria-label="Episode reference panels">
      {inspectorContext === "expert" ? (
        <details className="inspector-disclosure">
          <summary>
            <span>Action Generation</span>
            <small>
              {generationValues.length} x {generationColumns}
            </small>
          </summary>
          <GenerationMatrixPanel
            calls={calls}
            values={generationValues}
            timestep={timestep}
            onTimestepChange={onTimestepChange}
          />
        </details>
      ) : null}

      <details className="inspector-disclosure">
        <summary>
          <span>Episode Artifacts</span>
          <small>{artifacts.length}</small>
        </summary>
        <EpisodeArtifactPanel artifacts={artifacts} />
      </details>
    </div>
  );
}

function nearestMetricIndex(metric: EpisodeMetric, target: number): number {
  const values = metric.values ?? [];
  const xValues =
    metric.x_values?.length === values.length
      ? metric.x_values
      : values.map((_, index) => index);
  if (!xValues.length) {
    return 0;
  }
  let best = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  xValues.forEach((value, index) => {
    const distance = Math.abs(value - target);
    if (distance < bestDistance) {
      best = index;
      bestDistance = distance;
    }
  });
  return best;
}

function ActivationSitePanel({
  activationSlice,
  activationSliceFetching,
  activationSlicePlaceholder,
  activationClipPercent,
  architecture,
  attentionHead,
  attentionQueryToken,
  cameraOverlay,
  generationStep,
  generationStepCount,
  expertTokenActivations,
  expertTokenDetails,
  expertTokenSiteName,
  feature,
  inspectorContext,
  inspectionMode,
  onAttentionHeadChange,
  onAttentionQueryTokenChange,
  onGenerationStepChange,
  onExpertTokenChange,
  onActivationClipPercentChange,
  onFeatureChange,
  onInspectionModeChange,
  onTopChannelCountChange,
  onSiteChange,
  patchFeatures,
  selectedExpertToken,
  selectedPatch,
  sites,
  selectedSite,
  selectedSiteName,
  topChannelCount,
}: {
  activationSlice?: ActivationSliceResponse;
  activationSliceFetching: boolean;
  activationSlicePlaceholder: boolean;
  activationClipPercent: number;
  architecture?: ArchitectureMetadata;
  attentionHead: number | null;
  attentionQueryToken: number | null;
  cameraOverlay?: CameraOverlayPayload;
  generationStep: number;
  generationStepCount: number;
  expertTokenActivations?: ExpertTokenActivationsResponse;
  expertTokenDetails?: ExpertTokenDetailsResponse;
  expertTokenSiteName: string;
  feature: number;
  inspectorContext: InspectorContext;
  inspectionMode: InspectionMode;
  onAttentionHeadChange: (head: number | null) => void;
  onAttentionQueryTokenChange: (token: number | null) => void;
  onGenerationStepChange: (step: number) => void;
  onExpertTokenChange: (tokenIndex: number | null) => void;
  onActivationClipPercentChange: (clipPercent: number) => void;
  onFeatureChange: (feature: number) => void;
  onInspectionModeChange: (mode: InspectionMode) => void;
  onTopChannelCountChange: (count: number) => void;
  onSiteChange: (siteName: string) => void;
  patchFeatures?: PatchFeaturesResponse;
  selectedExpertToken: number | null;
  selectedPatch: SelectedPatch | null;
  sites: ActivationSite[];
  selectedSite?: ActivationSite;
  selectedSiteName: string;
  topChannelCount: number;
}) {
  const selectedSiteHasFeatures = isFeatureActivationSite(selectedSite);
  const siteFeatureCount = channelCountForSite(selectedSite);
  const featureCount = selectedSiteHasFeatures
    ? Math.max(0, siteFeatureCount || activationSlice?.feature_count || 0)
    : 0;
  const topRows = selectedSiteHasFeatures ? activationSlice?.top_abs ?? [] : [];
  const channelFeatureControl =
    inspectionMode === "features" && selectedSiteHasFeatures ? (
      <ChannelFeatureControl
        feature={feature}
        featureCount={featureCount}
        onFeatureChange={onFeatureChange}
        selectedSiteHasFeatures={selectedSiteHasFeatures}
      />
    ) : null;
  const attentionAxisControls =
    inspectionMode === "attention" && selectedSite ? (
      <AttentionAxisControls
        head={attentionHead}
        queryToken={attentionQueryToken}
        selectedSite={attentionSiteForSite(sites, selectedSite) ?? selectedSite}
        onHeadChange={onAttentionHeadChange}
        onQueryTokenChange={onAttentionQueryTokenChange}
      />
    ) : null;
  const topChannelPanel =
    inspectionMode === "features" && selectedSiteHasFeatures ? (
      <TopChannelPanel
        activationSlice={activationSlice}
        activationSliceFetching={activationSliceFetching}
        activationSlicePlaceholder={activationSlicePlaceholder}
        activationClipPercent={activationClipPercent}
        feature={feature}
        onActivationClipPercentChange={onActivationClipPercentChange}
        onFeatureChange={onFeatureChange}
        onTopChannelCountChange={onTopChannelCountChange}
        selectedSiteHasFeatures={selectedSiteHasFeatures}
        topChannelCount={topChannelCount}
        topRows={topRows}
      />
    ) : null;
  return (
    <section className="episode-tool-panel">
      <ModelPipelineMap
        architecture={architecture}
        feature={feature}
        axisControls={attentionAxisControls ?? channelFeatureControl}
        inspectionMode={inspectionMode}
        onInspectionModeChange={onInspectionModeChange}
        selectedSiteName={selectedSiteName}
        sites={sites}
        topChannelPanel={topChannelPanel}
        onSiteChange={onSiteChange}
      />
      {selectedSite ? (
        <>
          {selectedPatch ? (
            <CurrentImagePatchPanel
              cameraOverlay={cameraOverlay}
              inspectorContext={inspectorContext}
              patchFeatures={patchFeatures}
              selectedPatch={selectedPatch}
            />
          ) : null}

          {inspectorContext !== "vlm" && generationStepCount > 1 ? (
            <GenerationStepControl
              generationStep={generationStep}
              generationStepCount={generationStepCount}
              onGenerationStepChange={onGenerationStepChange}
            />
          ) : null}

          {inspectionMode === "features" && inspectorContext === "expert" ? (
            <>
              <ExpertTokenFlow
                activeFeature={feature}
                details={expertTokenDetails}
                payload={expertTokenActivations}
                selectedToken={selectedExpertToken}
                tokenSiteName={expertTokenSiteName}
                onFeatureChange={onFeatureChange}
                onTokenChange={onExpertTokenChange}
              />
            </>
          ) : null}
        </>
      ) : (
        <div className="empty-state">No activation sites recorded.</div>
      )}
    </section>
  );
}

function ChannelFeatureControl({
  feature,
  featureCount,
  onFeatureChange,
  selectedSiteHasFeatures,
}: {
  feature: number;
  featureCount: number;
  onFeatureChange: (feature: number) => void;
  selectedSiteHasFeatures: boolean;
}) {
  if (!selectedSiteHasFeatures) {
    return null;
  }
  return (
    <div className="channel-feature-control">
      <div className="feature-control">
        <label>
          Channel {feature}
          <input
            max={Math.max(0, featureCount - 1)}
            min={0}
            type="range"
            value={feature}
            onChange={(event) => onFeatureChange(Number(event.target.value))}
          />
        </label>
        <input
          aria-label="Channel index"
          max={Math.max(0, featureCount - 1)}
          min={0}
          type="number"
          value={feature}
          onChange={(event) => onFeatureChange(Number(event.target.value))}
        />
      </div>
    </div>
  );
}

function AttentionAxisControls({
  head,
  onHeadChange,
  onQueryTokenChange,
  queryToken,
  selectedSite,
}: {
  head: number | null;
  onHeadChange: (head: number | null) => void;
  onQueryTokenChange: (token: number | null) => void;
  queryToken: number | null;
  selectedSite?: ActivationSite;
}) {
  const headCount = axisCountForSite(selectedSite, "head");
  const queryCount = axisCountForSite(selectedSite, "query_token");
  const clampedHead =
    head === null || headCount <= 0 ? null : Math.max(0, Math.min(head, headCount - 1));
  const clampedQuery =
    queryToken === null || queryCount <= 0
      ? null
      : Math.max(0, Math.min(queryToken, queryCount - 1));

  if (headCount <= 0 && queryCount <= 0) {
    return null;
  }

  return (
    <div className="attention-axis-controls">
      {selectedSite?.key_token_space_id === "pi05.expert_context" ? (
        <div className="attention-token-ruler" aria-label="Expert attention key token space">
          <span>Keys</span>
          <i>image patches + prompt</i>
          <i>action tokens</i>
        </div>
      ) : null}
      {headCount > 0 ? (
        <label>
          Head
          <select
            value={clampedHead === null ? "avg" : String(clampedHead)}
            onChange={(event) => {
              const value = event.target.value;
              onHeadChange(value === "avg" ? null : Number(value));
            }}
          >
            <option value="avg">Average heads</option>
            {Array.from({ length: headCount }, (_, index) => (
              <option key={index} value={index}>
                Head {index}
              </option>
            ))}
          </select>
        </label>
      ) : null}
      {queryCount > 0 ? (
        <div className="feature-control attention-query-control">
          <label>
            Looking slot {clampedQuery === null ? "average" : clampedQuery}
            <input
              disabled={clampedQuery === null}
              max={Math.max(0, queryCount - 1)}
              min={0}
              type="range"
              value={clampedQuery ?? 0}
              onChange={(event) => onQueryTokenChange(Number(event.target.value))}
            />
          </label>
          <div className="attention-query-inputs">
            <button
              className={clampedQuery === null ? "active" : ""}
              type="button"
              onClick={() => onQueryTokenChange(null)}
            >
              Avg
            </button>
            <input
              aria-label="Looking slot index"
              max={Math.max(0, queryCount - 1)}
              min={0}
              type="number"
              value={clampedQuery ?? 0}
              onChange={(event) => onQueryTokenChange(Number(event.target.value))}
            />
          </div>
        </div>
      ) : null}
    </div>
  );
}

function TopChannelPanel({
  activationSlice,
  activationSliceFetching,
  activationSlicePlaceholder,
  activationClipPercent,
  feature,
  onActivationClipPercentChange,
  onFeatureChange,
  onTopChannelCountChange,
  selectedSiteHasFeatures,
  topChannelCount,
  topRows,
}: {
  activationSlice?: ActivationSliceResponse;
  activationSliceFetching: boolean;
  activationSlicePlaceholder: boolean;
  activationClipPercent: number;
  feature: number;
  onActivationClipPercentChange: (clipPercent: number) => void;
  onFeatureChange: (feature: number) => void;
  onTopChannelCountChange: (count: number) => void;
  selectedSiteHasFeatures: boolean;
  topChannelCount: number;
  topRows: { index: number; value: number }[];
}) {
  if (!selectedSiteHasFeatures) {
    return null;
  }
  return (
    <div className="top-channel-panel">
      <FeatureTable
        rows={topRows}
        activeFeature={feature}
        title="Top Channels"
        loading={activationSliceFetching && !topRows.length}
        reserveHeight
        updating={activationSliceFetching && Boolean(topRows.length)}
        stale={activationSlicePlaceholder}
        onFeatureChange={onFeatureChange}
        clipPercent={activationClipPercent}
        clip={activationSlice?.clip}
        onClipPercentChange={onActivationClipPercentChange}
        rowLimit={topChannelCount}
        rowLimitOptions={TOP_CHANNEL_COUNT_OPTIONS}
        onRowLimitChange={onTopChannelCountChange}
      />
    </div>
  );
}

function GenerationStepControl({
  generationStep,
  generationStepCount,
  onGenerationStepChange,
}: {
  generationStep: number;
  generationStepCount: number;
  onGenerationStepChange: (step: number) => void;
}) {
  return (
    <div className="generation-control">
      <label>
        Generation step {generationStep}
        <input
          max={Math.max(0, generationStepCount - 1)}
          min={0}
          type="range"
          value={generationStep}
          onChange={(event) => onGenerationStepChange(Number(event.target.value))}
        />
      </label>
    </div>
  );
}

function PromptAttentionStrip({
  expertTokenDetails,
  context,
  prompt,
  promptAttention,
  promptFeatureMap,
}: {
  expertTokenDetails?: ExpertTokenDetailsResponse;
  context: InspectorContext;
  prompt?: string | null;
  promptAttention?: PromptAttentionResponse;
  promptFeatureMap?: PromptAttentionResponse;
}) {
  const modelPromptRows = orderedPromptAttentionRows(
    expertTokenDetails?.available ? expertTokenDetails : undefined,
    context === "vlm" ? promptFeatureMap : promptAttention,
  );
  const candidateRows = taskPromptRows(modelPromptRows);
  const rows = promptRowsMatchPrompt(candidateRows, prompt) ? candidateRows : [];
  const maxAttention = Math.max(
    ...rows.map((row) => Math.abs(Number(row.attention))).filter(Number.isFinite),
    1e-6,
  );
  const promptText =
    prompt || promptFeatureMap?.prompt || promptAttention?.prompt || "No prompt recorded.";

  return (
    <div className="viewer-prompt-strip">
      <div className="viewer-prompt-head">
        <strong>Prompt</strong>
      </div>
      {rows.length ? (
        <PromptAttentionChips rows={rows} maxAttention={maxAttention} />
      ) : (
        <p>{promptText}</p>
      )}
    </div>
  );
}

function CurrentImagePatchPanel({
  cameraOverlay,
  inspectorContext,
  patchFeatures,
  selectedPatch,
}: {
  cameraOverlay?: CameraOverlayPayload;
  inspectorContext: InspectorContext;
  patchFeatures?: PatchFeaturesResponse;
  selectedPatch: SelectedPatch;
}) {
  const patchValue = overlayPatchValue(cameraOverlay, selectedPatch);
  const patchMaxAbs = overlayCameraMaxAbs(cameraOverlay, selectedPatch.camera);
  const patchStyle =
    typeof patchValue === "number"
      ? { background: signedActivationColor(patchValue, patchMaxAbs) }
      : undefined;
  const patchValueLabel = inspectorContext === "vlm" ? "Patch feature value" : "Patch attention mass";

  return (
    <div className="current-patch-panel">
      <div className="section-title">
        <span>Current Selection</span>
        <small>{selectedPatch.camera}</small>
      </div>
      <div className="patch-summary-grid">
        <DetailItem label="camera" value={selectedPatch.camera} />
        <DetailItem label="patch" value={`r${selectedPatch.row} c${selectedPatch.col}`} />
        <DetailItem
          label="token"
          value={
            patchFeatures?.available && patchFeatures.token_index !== undefined
              ? String(patchFeatures.token_index)
              : "-"
          }
        />
        <DetailItem
          label="feature rank"
          value={
            patchFeatures?.available && patchFeatures.feature_rank_by_abs
              ? `#${patchFeatures.feature_rank_by_abs}`
              : "-"
          }
        />
      </div>

      <div className="patch-attention-readout">
        <span className="patch-color-chip" style={patchStyle} />
        <div>
          <strong>{patchValueLabel}</strong>
          <span>
            {typeof patchValue === "number"
              ? inspectorContext === "vlm"
                ? formatMaybeNumber(patchValue)
                : formatPercent(patchValue)
              : "unavailable"}
          </span>
        </div>
      </div>
    </div>
  );
}

function PromptAttentionChips({
  maxAttention,
  rows,
}: {
  maxAttention: number;
  rows: PromptTokenAttention[];
}) {
  if (!rows.length) {
    return <div className="empty-state">Prompt attention is not available for the current selection.</div>;
  }
  return (
    <div className="prompt-attention-chips" aria-label="Prompt attention tokens">
      {rows.map((row) => {
        const attention = Number(row.attention);
        const token = promptTokenDisplay(row);
        return (
          <span key={`${row.prefix_index ?? row.local_index}-${row.token_id ?? row.token_piece ?? ""}`}>
            {token.prefix ? <span className="prompt-token-space">{token.prefix}</span> : null}
            {token.text ? (
              <span
                className="prompt-attention-chip"
                style={{ background: signedActivationColor(attention, maxAttention) }}
                title={promptTokenTitle(row)}
              >
                <strong>{token.text}</strong>
              </span>
            ) : null}
          </span>
        );
      })}
    </div>
  );
}

function ModelPipelineMap({
  axisControls,
  architecture,
  feature,
  inspectionMode,
  onInspectionModeChange,
  onSiteChange,
  selectedSiteName,
  sites,
  topChannelPanel,
}: {
  axisControls?: ReactNode;
  architecture?: ArchitectureMetadata;
  feature: number;
  inspectionMode: InspectionMode;
  onInspectionModeChange: (mode: InspectionMode) => void;
  onSiteChange: (siteName: string) => void;
  selectedSiteName: string;
  sites: ActivationSite[];
  topChannelPanel?: ReactNode;
}) {
  const stages = useMemo(() => modelPipelineStages(sites), [sites]);
  const [showPipelineModal, setShowPipelineModal] = useState(false);
  const selectedNodeId = useMemo(
    () => stages.flatMap((stage) => stage.nodes).find((node) =>
      node.allSites.some((site) => site.name === selectedSiteName),
    )?.id,
    [stages, selectedSiteName],
  );
  const [expandedNodeId, setExpandedNodeId] = useState<string>("");
  const allNodes = useMemo(() => stages.flatMap((stage) => stage.nodes), [stages]);
  const expandedNodeStillVisible = allNodes
    .some((node) => node.id === expandedNodeId);
  const activeExpandedNodeId =
    selectedNodeId || (expandedNodeStillVisible ? expandedNodeId : "") || allNodes.find((node) => node.captured)?.id || "";
  const expandedNode = allNodes.find((node) => node.id === activeExpandedNodeId);

  if (!stages.length) {
    return <div className="empty-state">No activation sites captured in this trace.</div>;
  }
  return (
    <div className="model-pipeline">
      <div className="pipeline-toolbar">
        <PipelineLegend />
      </div>
      <PipelineMap2D
        cornerAction={
          <button
            className="pipeline-view-toggle"
            type="button"
            onClick={() => setShowPipelineModal(true)}
          >
            <Maximize2 size={14} />
            Fullsize
          </button>
        }
        feature={feature}
        inspectionMode={inspectionMode}
        onNodeSelect={(node) => {
          setExpandedNodeId(node.id);
          const preferred = preferredSiteWithinNode(
            inspectionMode === "advanced" ? node.allSites : node.sites,
            inspectionMode,
          );
          if (preferred) {
            if (inspectionMode !== "advanced" && inspectionModeForSite(preferred) !== inspectionMode) {
              onInspectionModeChange(inspectionModeForSite(preferred));
            }
            onSiteChange(preferred.name);
          }
        }}
        selectedSiteName={selectedSiteName}
        architecture={architecture}
        stages={stages}
      />
      {axisControls ? <div className="pipeline-channel-row">{axisControls}</div> : null}
      {showPipelineModal ? (
        <PipelineMapModal
          feature={feature}
          inspectionMode={inspectionMode}
          overlayControls={axisControls}
          onClose={() => setShowPipelineModal(false)}
          onNodeSelect={(node) => {
            setExpandedNodeId(node.id);
            const preferred = preferredSiteWithinNode(
              inspectionMode === "advanced" ? node.allSites : node.sites,
              inspectionMode,
            );
            if (preferred) {
              if (inspectionMode !== "advanced" && inspectionModeForSite(preferred) !== inspectionMode) {
                onInspectionModeChange(inspectionModeForSite(preferred));
              }
              onSiteChange(preferred.name);
            }
          }}
          selectedSiteName={selectedSiteName}
          architecture={architecture}
          stages={stages}
        />
      ) : null}
      <PipelineSelectedNodePanel
        inspectionMode={inspectionMode}
        node={expandedNode}
        onInspectionModeChange={onInspectionModeChange}
        onSiteChange={onSiteChange}
        selectedSiteName={selectedSiteName}
      />
      {topChannelPanel ? <div className="pipeline-top-channels">{topChannelPanel}</div> : null}
    </div>
  );
}

function PipelineSelectedNodePanel({
  inspectionMode,
  node,
  onInspectionModeChange,
  onSiteChange,
  selectedSiteName,
}: {
  inspectionMode: InspectionMode;
  node?: PipelineNode;
  onInspectionModeChange: (mode: InspectionMode) => void;
  onSiteChange: (siteName: string) => void;
  selectedSiteName: string;
}) {
  const [rawQuery, setRawQuery] = useState("");
  if (!node) {
    return null;
  }
  const modeCounts = inspectionModes.map((mode) => ({
    mode,
    count: mode === "advanced"
      ? node.rawChoices.length
      : node.choices.filter((choice) => choice.mode === mode).length,
  })).filter(({ count }) => count > 0);
  const modeChoices =
    inspectionMode === "advanced"
      ? node.rawChoices
      : node.choices.filter((choice) => choice.mode === inspectionMode);
  const selectedChoice =
    modeChoices.find((choice) => choice.site.name === selectedSiteName) ?? modeChoices[0];
  const rawQueryText = rawQuery.trim().toLowerCase();
  const rawChoices = rawQueryText
    ? node.rawChoices.filter((choice) =>
        [choice.label, choice.site.name, choice.site.role, choice.site.tensor_type, choice.site.axes?.join(" ")]
          .filter(Boolean)
          .join(" ")
          .toLowerCase()
          .includes(rawQueryText),
      )
    : node.rawChoices;
  const chooseMode = (mode: InspectionMode) => {
    onInspectionModeChange(mode);
    const next = preferredSiteWithinNode(mode === "advanced" ? node.allSites : node.sites, mode);
    if (next) {
      onSiteChange(next.name);
    }
  };

  return (
    <div className={`pipeline-site-detail ${node.family}`}>
      {node.rawChoices.length ? (
        <>
          <div className="pipeline-capture-heading">
            <span>Inspect</span>
            <small>{node.label}</small>
          </div>
          {modeCounts.length > 1 ? (
            <div className="inspection-mode-tabs" aria-label={`Inspection modes for ${node.label}`}>
              {modeCounts.map(({ mode, count }) => (
                <button
                  className={inspectionMode === mode ? "active" : ""}
                  key={mode}
                  type="button"
                  onClick={() => chooseMode(mode)}
                >
                  <span>{inspectionModeLabel(mode)}</span>
                  <small>{count}</small>
                </button>
              ))}
            </div>
          ) : null}
          {inspectionMode === "advanced" ? (
            <AdvancedRawCaptures
              choices={rawChoices}
              onQueryChange={setRawQuery}
              onSiteChange={onSiteChange}
              query={rawQuery}
              selectedSiteName={selectedSiteName}
            />
          ) : modeChoices.length > 1 ? (
            <>
              <label className="capture-point-select">
                <span>{inspectionModeLabel(inspectionMode)}</span>
                <select
                  value={selectedChoice?.site.name ?? ""}
                  onChange={(event) => onSiteChange(event.target.value)}
                >
                  {modeChoices.map((choice) => (
                    <option key={choice.id} value={choice.site.name}>
                      {choice.label}
                    </option>
                  ))}
                </select>
              </label>
              {selectedChoice ? <CaptureDescription choice={selectedChoice} node={node} /> : null}
            </>
          ) : modeChoices.length === 1 && selectedChoice ? (
            <CaptureDescription choice={selectedChoice} node={node} />
          ) : (
            <div className="pipeline-site-empty">
              {inspectionModeEmptyMessage(inspectionMode, node)}
            </div>
          )}
        </>
      ) : (
        <div className="pipeline-site-empty">No captured inspectable site for this node in the current profile.</div>
      )}
    </div>
  );
}

function AdvancedRawCaptures({
  choices,
  onQueryChange,
  onSiteChange,
  query,
  selectedSiteName,
}: {
  choices: PipelineSiteChoice[];
  onQueryChange: (query: string) => void;
  onSiteChange: (siteName: string) => void;
  query: string;
  selectedSiteName: string;
}) {
  return (
    <div className="advanced-capture-drawer">
      <input
        aria-label="Search raw captures"
        placeholder="Search raw captures"
        type="search"
        value={query}
        onChange={(event) => onQueryChange(event.target.value)}
      />
      <div className="advanced-capture-list">
        {choices.length ? (
          choices.map((choice) => (
            <button
              className={choice.site.name === selectedSiteName ? "active" : ""}
              key={choice.id}
              title={choice.site.name}
              type="button"
              onClick={() => onSiteChange(choice.site.name)}
            >
              <strong>{choice.label}</strong>
              <span>{choice.site.name}</span>
              <small>
                {[formatCaptureShape(choice.site), captureStorageLabel(choice.site)]
                  .filter(Boolean)
                  .join(" · ")}
              </small>
            </button>
          ))
        ) : (
          <div className="pipeline-site-empty">No raw captures match this search.</div>
        )}
      </div>
    </div>
  );
}

function CaptureDescription({ choice, node }: { choice: PipelineSiteChoice; node: PipelineNode }) {
  const site = choice.site;
  const description = captureDescription(site, node);
  const shape = site.shape?.length ? formatCaptureShape(site) : "";
  const size = estimateCaptureSize(site);
  const stored = captureStorageLabel(site);
  const controls = captureControlsLabel(site);
  return (
    <div className="pipeline-capture-description">
      <div>
        <span>{choice.label}</span>
        <strong>{description}</strong>
      </div>
      <dl>
        {shape ? (
          <div>
            <dt>Size</dt>
            <dd>{shape}{size ? ` · ${size}` : ""}</dd>
          </div>
        ) : null}
        {stored ? (
          <div>
            <dt>Stored</dt>
            <dd>{stored}</dd>
          </div>
        ) : null}
        {controls ? (
          <div>
            <dt>Controls</dt>
            <dd>{controls}</dd>
          </div>
        ) : null}
      </dl>
    </div>
  );
}

function PipelineLegend() {
  return (
    <div className="pipeline-legend" aria-label="Pipeline legend">
      <span><i className="legend-swatch vlm" />VLM</span>
      <span><i className="legend-swatch expert" />Expert</span>
      <span><i className="legend-swatch action" />State</span>
      <span><i className="legend-swatch missing" />Missing</span>
      <span><i className="legend-swatch feature" />Channel</span>
      <span><i className="legend-conditioning" />K/V bus</span>
      <span><i className="legend-loop" />Denoise</span>
    </div>
  );
}

function PipelineMapModal({
  architecture,
  feature,
  inspectionMode,
  overlayControls,
  onClose,
  onNodeSelect,
  selectedSiteName,
  stages,
}: {
  architecture?: ArchitectureMetadata;
  feature: number;
  inspectionMode: InspectionMode;
  overlayControls?: ReactNode;
  onClose: () => void;
  onNodeSelect: (node: PipelineNode) => void;
  selectedSiteName: string;
  stages: PipelineStage[];
}) {
  useEffect(() => {
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [onClose]);

  return (
    <div className="pipeline-map-modal-backdrop" role="presentation" onMouseDown={onClose}>
      <div
        aria-label="Expanded PI0.5 model pipeline"
        aria-modal="true"
        className="pipeline-map-modal"
        role="dialog"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header>
          <div className="pipeline-map-modal-title">
            <Layers3 size={17} />
            <strong>PI0.5 Pipeline</strong>
          </div>
          <PipelineLegend />
          <button aria-label="Close expanded model pipeline" type="button" onClick={onClose}>
            <X size={17} />
          </button>
        </header>
        <PipelineMap2D
          architecture={architecture}
          className="large"
          feature={feature}
          inspectionMode={inspectionMode}
          onNodeSelect={onNodeSelect}
          selectedSiteName={selectedSiteName}
          stages={stages}
        />
        {overlayControls ? <div className="pipeline-channel-row modal">{overlayControls}</div> : null}
      </div>
    </div>
  );
}

function PipelineMap2D({
  architecture,
  className = "",
  cornerAction,
  feature,
  inspectionMode,
  onNodeSelect,
  selectedSiteName,
  stages,
}: {
  architecture?: ArchitectureMetadata;
  className?: string;
  cornerAction?: ReactNode;
  feature: number;
  inspectionMode: InspectionMode;
  onNodeSelect: (node: PipelineNode) => void;
  selectedSiteName: string;
  stages: PipelineStage[];
}) {
  const layout = useMemo(() => pipelineDiagramLayout(stages, architecture), [architecture, stages]);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const isLargeMap = className.includes("large");
  const maxFitScale = isLargeMap ? 1.35 : 1;
  const minFitScale = isLargeMap ? 0.68 : 0.48;
  const [view, setView] = useState({ scale: isLargeMap ? 1 : 0.58, x: 0, y: 0 });
  const allSites = stages.flatMap((stage) => stage.nodes).flatMap((node) => node.allSites);
  const selectedSite = allSites.find((site) => site.name === selectedSiteName);
  const selectedNode = layout.nodes.find((entry) =>
    entry.node.allSites.some((site) => site.name === selectedSiteName),
  );
  const featureCount = channelCountForSite(selectedSite);
  const showFeatureSlice = Boolean(selectedNode && selectedSite && featureCount > 1 && isFeatureActivationSite(selectedSite));
  const featureFraction = Math.max(0, Math.min(1, feature / Math.max(1, featureCount - 1)));

  useEffect(() => {
    const target = viewportRef.current;
    if (!target) {
      return;
    }
    const observer = new ResizeObserver(([entry]) => {
      const padding = isLargeMap ? 56 : 24;
      const widthScale = (entry.contentRect.width - padding) / layout.width;
      const heightScale = (entry.contentRect.height - padding) / layout.height;
      const nextScale = Math.max(minFitScale, Math.min(maxFitScale, widthScale, heightScale));
      const nextView = {
        scale: nextScale,
        x: Math.max(12, (entry.contentRect.width - layout.width * nextScale) / 2),
        y: isLargeMap ? 24 : Math.max(12, (entry.contentRect.height - layout.height * nextScale) / 2),
      };
      setView(nextView);
    });
    observer.observe(target);
    return () => observer.disconnect();
  }, [isLargeMap, layout.height, layout.width, maxFitScale, minFitScale]);

  return (
    <div className={["pipeline-map2d", className].filter(Boolean).join(" ")}>
      {cornerAction ? <div className="pipeline-map2d-action">{cornerAction}</div> : null}
      <div className="pipeline-map2d-viewport" ref={viewportRef}>
        <div
          className="pipeline-map2d-canvas"
          style={{
            height: layout.height,
            transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})`,
            width: layout.width,
          }}
        >
          <svg
            aria-hidden="true"
            className="pipeline-map2d-svg"
            viewBox={`0 0 ${layout.width} ${layout.height}`}
          >
            <defs>
              <marker id="pipeline-forward-arrow" markerHeight="6" markerWidth="7" orient="auto" refX="6" refY="3">
                <path d="M0,0 L6,3 L0,6 Z" />
              </marker>
              <marker id="pipeline-conditioning-arrow" markerHeight="6" markerWidth="7" orient="auto" refX="6" refY="3">
                <path d="M0,0 L6,3 L0,6 Z" />
              </marker>
              <marker id="pipeline-loop-arrow" markerHeight="6" markerWidth="7" orient="auto" refX="6" refY="3">
                <path d="M0,0 L6,3 L0,6 Z" />
              </marker>
            </defs>
            {layout.bands.map((band) => (
              <g className={`pipeline-map2d-band ${band.className}`} key={band.id}>
                <rect height={band.height} width={band.width} x={band.x} y={band.y} />
                <text x={band.x + 10} y={band.y + 17}>{band.label}</text>
              </g>
            ))}
            {layout.arrows.map((arrow) => (
              <g className={`pipeline-map2d-arrow ${arrow.className}`} key={arrow.id}>
                <path d={arrow.path} />
                {arrow.label ? (
                  <text textAnchor={arrow.labelAnchor ?? "start"} x={arrow.labelX} y={arrow.labelY}>
                    {arrow.label}
                  </text>
                ) : null}
              </g>
            ))}
            {layout.ports.map((port) => (
              <g className={`pipeline-map2d-port ${port.className}`} key={port.id}>
                <circle cx={port.x} cy={port.y} r={port.radius ?? 4} />
                {port.label ? (
                  <text
                    textAnchor={port.textAnchor ?? "start"}
                    x={port.textAnchor === "middle" ? port.x : port.x + 9}
                    y={port.y + 4}
                  >
                    {port.label}
                  </text>
                ) : null}
              </g>
            ))}
          </svg>
          {layout.nodes.map((entry) => {
            const active = entry.node.allSites.some((site) => site.name === selectedSiteName);
            const preferred = preferredSiteWithinNode(
              inspectionMode === "advanced" ? entry.node.allSites : entry.node.sites,
              inspectionMode,
            );
            return (
              <button
                className={[
                  "pipeline-map2d-node",
                  entry.node.family,
                  entry.node.captured ? "captured" : "missing",
                  active ? "active" : "",
                  entry.width < 42 ? "compact" : "",
                ].filter(Boolean).join(" ")}
                disabled={!entry.node.captured}
                key={entry.node.id}
                style={{
                  height: entry.height,
                  left: entry.x,
                  top: entry.y,
                  width: entry.width,
                }}
                title={entry.node.allSites.map((site) => site.name).join("\n") || entry.node.sublabel}
                type="button"
                onClick={() => {
                  if (preferred) {
                    onNodeSelect(entry.node);
                  }
                }}
              >
                <strong>{entry.node.label}</strong>
                {entry.width >= 74 ? <small>{entry.node.sublabel}</small> : null}
                {showFeatureSlice && active ? (
                  <i
                    aria-hidden="true"
                    className={
                      entry.width >= 64
                        ? "pipeline-map2d-feature-slice"
                        : "pipeline-map2d-channel-marker"
                    }
                    style={{ left: `${6 + featureFraction * 88}%` }}
                    title={`Channel ${feature} / ${Math.max(0, featureCount - 1)}`}
                  />
                ) : null}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function ExpertTokenFlow({
  activeFeature,
  details,
  onFeatureChange,
  onTokenChange,
  payload,
  selectedToken,
  tokenSiteName,
}: {
  activeFeature: number;
  details?: ExpertTokenDetailsResponse;
  onFeatureChange: (feature: number) => void;
  onTokenChange: (tokenIndex: number | null) => void;
  payload?: ExpertTokenActivationsResponse;
  selectedToken: number | null;
  tokenSiteName: string;
}) {
  return (
    <>
      <div className="expert-token-panel">
        <ExpertTokenStrip
          payload={payload}
          selectedToken={selectedToken}
          onTokenChange={onTokenChange}
        />
        {!payload?.available ? (
          <div className="empty-state compact">
            {payload?.reason || tokenSiteName || "Select an expert token site to view action-token activations."}
          </div>
        ) : null}
        <ExpertTokenDetails
          activeFeature={activeFeature}
          details={details}
          selectedToken={selectedToken}
          onFeatureChange={onFeatureChange}
        />
      </div>
      <ExpertAttentionSummary details={details} selectedToken={selectedToken} />
    </>
  );
}

function ExpertAttentionSummary({
  details,
  selectedToken,
}: {
  details?: ExpertTokenDetailsResponse;
  selectedToken: number | null;
}) {
  const coarse = details?.attention_coarse;
  if (selectedToken === null) {
    return null;
  }
  return (
    <div className="expert-attention-summary">
      {details?.available && coarse ? (
        <div className="attention-split">
          <AttentionBar label="image" value={coarse.image} />
          <AttentionBar label="prompt" value={coarse.prompt} />
          <AttentionBar label="action" value={coarse.action_suffix} />
        </div>
      ) : (
        <div className="empty-state">
          {details?.reason || "No token-specific attention was found for this site."}
        </div>
      )}
    </div>
  );
}

function AttentionBar({ label, value }: { label: string; value: number | null | undefined }) {
  const numeric = typeof value === "number" && Number.isFinite(value) ? value : 0;
  const width = `${Math.max(0, Math.min(1, numeric)) * 100}%`;
  return (
    <div className="bar-row">
      <span>{label}</span>
      <div className="bar-track">
        <span style={{ width }} />
      </div>
      <span>{formatPercent(value)}</span>
    </div>
  );
}

function ExpertTokenStrip({
  onTokenChange,
  payload,
  selectedToken,
}: {
  onTokenChange: (tokenIndex: number | null) => void;
  payload?: ExpertTokenActivationsResponse;
  selectedToken: number | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const values = useMemo(() => payload?.values ?? [], [payload?.values]);
  const [layoutVersion, setLayoutVersion] = useState(0);

  useEffect(() => {
    const target = canvasRef.current;
    if (!target) {
      return;
    }
    const observer = new ResizeObserver(() => {
      setLayoutVersion((version) => version + 1);
    });
    observer.observe(target);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const scale = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(rect.width * scale));
    canvas.height = Math.max(1, Math.round(rect.height * scale));
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.fillStyle = "#fffefa";
    ctx.fillRect(0, 0, rect.width, rect.height);

    if (!payload?.available || !values.length) {
      ctx.fillStyle = "#61707b";
      ctx.font = "12px Inter, sans-serif";
      ctx.fillText("Select an expert layer to view action-token activations.", 10, 24);
      return;
    }

    const maxAbs = Math.max(...values.map((value) => Math.abs(value)).filter(Number.isFinite), 1e-6);
    const cellWidth = rect.width / values.length;
    values.forEach((value, index) => {
      ctx.fillStyle = signedActivationColor(value, maxAbs);
      ctx.fillRect(index * cellWidth, 0, Math.ceil(cellWidth), rect.height);
    });

    if (selectedToken !== null) {
      const index = Math.max(0, Math.min(values.length - 1, selectedToken));
      ctx.strokeStyle = "rgba(31, 33, 31, 0.82)";
      ctx.lineWidth = 2;
      ctx.strokeRect(index * cellWidth + 1, 1, Math.max(2, cellWidth - 2), rect.height - 2);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.95)";
      ctx.lineWidth = 1;
      ctx.strokeRect(index * cellWidth + 3, 3, Math.max(2, cellWidth - 6), rect.height - 6);
    }
  }, [layoutVersion, payload?.available, selectedToken, values]);

  return (
    <canvas
      ref={canvasRef}
      className="strip-canvas"
      aria-label="Expert action token activation strip"
      role="button"
      tabIndex={0}
      onClick={(event) => {
        if (!values.length) {
          return;
        }
        const rect = event.currentTarget.getBoundingClientRect();
        const index = Math.max(
          0,
          Math.min(values.length - 1, Math.floor(((event.clientX - rect.left) / rect.width) * values.length)),
        );
        onTokenChange(index);
      }}
      onKeyDown={(event) => {
        if ((event.key === "Enter" || event.key === " ") && values.length) {
          event.preventDefault();
          onTokenChange(selectedToken === null ? 0 : selectedToken);
        }
      }}
    />
  );
}

function ExpertTokenDetails({
  activeFeature,
  details,
  onFeatureChange,
  selectedToken,
}: {
  activeFeature: number;
  details?: ExpertTokenDetailsResponse;
  onFeatureChange: (feature: number) => void;
  selectedToken: number | null;
}) {
  if (selectedToken === null) {
    return (
      <div className="token-detail">
        <div className="empty-state">
          Click an action token to inspect its hidden vector and aligned action vector.
        </div>
      </div>
    );
  }
  if (!details?.available) {
    return (
      <div className="token-detail">
        <div className="empty-state">{details?.reason || "Loading token detail."}</div>
      </div>
    );
  }
  const action = details.action;
  const coarse = details.attention_coarse;
  return (
    <div className="token-detail">
      <div className="detail-grid">
        <DetailItem label="channel value" value={formatMaybeNumber(details.feature_value)} />
        <DetailItem
          label="channel rank"
          value={details.feature_rank_by_abs ? `#${details.feature_rank_by_abs}` : "-"}
        />
        <DetailItem label="action norm" value={formatMaybeNumber(action?.norm)} />
        <DetailItem
          label="token"
          value={`${details.token_index ?? selectedToken} / ${Math.max(0, (details.token_count ?? 1) - 1)}`}
        />
        <DetailItem
          label="attention split"
          value={
            coarse
              ? `img ${formatPercent(coarse.image)} / prompt ${formatPercent(coarse.prompt)}`
              : "-"
          }
        />
      </div>
      <FeatureTable
        activeFeature={activeFeature}
        onFeatureChange={onFeatureChange}
        rows={details.top_abs ?? []}
        title="Token Channel Ranking"
      />
      <ImageAttentionTable rows={details.top_image_patches ?? []} />
      <PromptAttentionTable rows={details.top_prompt_tokens ?? []} />
      <FeatureTable
        activeFeature={-1}
        onFeatureChange={() => undefined}
        rows={action?.top_abs ?? []}
        title={`Action Dimensions${action?.source ? ` / ${action.source}` : ""}`}
        indexHeader="Dim"
        indexPrefix="a"
        selectable={false}
      />
      {details.note ? <p className="note">{details.note}</p> : null}
    </div>
  );
}

function ImageAttentionTable({ rows }: { rows: ImagePatchAttention[] }) {
  return (
    <div className="feature-table-wrap">
      <div className="section-title">
        <span>Image Attention Patches</span>
        <small>attention</small>
      </div>
      {rows.length ? (
        <table className="compact-table">
          <thead>
            <tr>
              <th>Patch</th>
              <th>Mass</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 12).map((row) => (
              <tr key={`${row.camera}-${row.row}-${row.col}`}>
                <td>
                  {row.camera} r{row.row} c{row.col}
                </td>
                <td>{formatPercent(row.attention)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <div className="empty-state">No image attention patches found.</div>
      )}
    </div>
  );
}

function PromptAttentionTable({ rows }: { rows: PromptTokenAttention[] }) {
  return (
    <div className="feature-table-wrap">
      <div className="section-title">
        <span>Prompt Attention Tokens</span>
        <small>attention</small>
      </div>
      {rows.length ? (
        <table className="compact-table">
          <thead>
            <tr>
              <th>Token</th>
              <th>ID</th>
              <th>Mass</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 12).map((row) => (
              <tr key={`${row.prefix_index ?? row.local_index}-${row.token_id ?? ""}`}>
                <td>#{row.local_index}</td>
                <td>{displayTokenPiece(row)}</td>
                <td>{formatPercent(row.attention)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <div className="empty-state">No active prompt tokens found.</div>
      )}
    </div>
  );
}

function DetailItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="detail-item">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function FeatureTable({
  activeFeature,
  clip,
  clipPercent,
  indexHeader = "Channel",
  indexPrefix = "c",
  loading = false,
  onClipPercentChange,
  onFeatureChange,
  onRowLimitChange,
  reserveHeight = false,
  rowLimit,
  rowLimitOptions,
  rows,
  selectable = true,
  stale = false,
  title,
  updating = false,
}: {
  activeFeature: number;
  clip?: ActivationSliceResponse["clip"];
  clipPercent?: number;
  indexHeader?: string;
  indexPrefix?: string;
  loading?: boolean;
  onClipPercentChange?: (clipPercent: number) => void;
  onFeatureChange: (feature: number) => void;
  onRowLimitChange?: (count: number) => void;
  reserveHeight?: boolean;
  rowLimit?: number;
  rowLimitOptions?: readonly number[];
  rows: { index: number; value: number }[];
  selectable?: boolean;
  stale?: boolean;
  title: string;
  updating?: boolean;
}) {
  const showClipControl = Boolean(onClipPercentChange);
  const showRowLimitControl = Boolean(onRowLimitChange && rowLimitOptions?.length);
  const rowsAreSelectable = selectable && activeFeature >= 0;
  const visibleRows = rows.slice(0, rowLimit ?? 12);
  return (
    <div
      className={[
        "feature-table-wrap",
        stale ? "updating" : "",
        reserveHeight ? "stable-height" : "",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <div className="section-title">
        <span>{title}</span>
        <div className="section-title-actions">
          {showClipControl ? (
            <div className="activation-clip-control" aria-label="Activation value clip">
              <span>Clip</span>
              {ACTIVATION_CLIP_OPTIONS.map((percent) => (
                <button
                  className={clipPercent === percent ? "active" : ""}
                  key={percent}
                  type="button"
                  onClick={() => onClipPercentChange?.(percent)}
                >
                  {percent === 0 ? "0" : `${percent}%`}
                </button>
              ))}
            </div>
          ) : null}
          {showRowLimitControl ? (
            <label className="feature-row-limit">
              <span>Top</span>
              <select
                aria-label="Number of top channels"
                value={rowLimit}
                onChange={(event) => onRowLimitChange?.(Number(event.target.value))}
              >
                {rowLimitOptions?.map((count) => (
                  <option key={count} value={count}>
                    {count}
                  </option>
                ))}
              </select>
            </label>
          ) : null}
          {updating || loading ? <small>{loading ? "loading" : "updating"}</small> : null}
        </div>
      </div>
      {showClipControl && clip?.enabled ? (
        <div className="activation-clip-note">
          Showing {clip.kept ?? 0} / {clip.total ?? 0} features after percentile trim.
        </div>
      ) : null}
      {rows.length ? (
        <table className="compact-table">
          <thead>
            <tr>
              <th>{indexHeader}</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody>
            {visibleRows.map((row) => (
              <tr
                className={rowsAreSelectable ? "selectable-row" : ""}
                key={row.index}
                onClick={rowsAreSelectable ? () => onFeatureChange(row.index) : undefined}
              >
                <td>
                  {rowsAreSelectable ? (
                    <button
                      className={row.index === activeFeature ? "feature-link active" : "feature-link"}
                      type="button"
                      onClick={(event) => {
                        event.stopPropagation();
                        onFeatureChange(row.index);
                      }}
                    >
                      {indexPrefix}
                      {row.index}
                    </button>
                  ) : (
                    <span className="feature-index-text">
                      {indexPrefix}
                      {row.index}
                    </span>
                  )}
                </td>
                <td className={row.value >= 0 ? "signed-positive" : "signed-negative"}>
                  {row.value.toFixed(4)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : loading ? (
        <div className="feature-table-skeleton" aria-label={`${title} loading`}>
          {Array.from({ length: 8 }, (_, index) => (
            <div className="skeleton-row" key={index}>
              <span />
              <span />
            </div>
          ))}
        </div>
      ) : (
        <div className="empty-state">No channel ranking available.</div>
      )}
    </div>
  );
}

function GenerationMatrixPanel({
  calls,
  values,
  timestep,
  onTimestepChange,
}: {
  calls: PolicyCall[];
  values: (number | null)[][];
  timestep: number;
  onTimestepChange: (timestep: number) => void;
}) {
  const finiteValues = values
    .flat()
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  const max = Math.max(...finiteValues, 1);
  const columns = Math.max(1, values[0]?.length ?? 1);
  return (
    <section className="episode-tool-panel">
      <header>
        <div className="icon-label">
          <BarChart3 size={16} />
          <strong>Generation Commitment</strong>
        </div>
        <span className="panel-note">policy call x generation step</span>
      </header>
      {values.length ? (
        <div
          className="generation-grid"
          style={{ gridTemplateColumns: `72px repeat(${columns}, minmax(26px, 1fr))` }}
        >
          <span />
          {Array.from({ length: columns }, (_, index) => (
            <span className="generation-label" key={`g-${index}`}>
              g{index}
            </span>
          ))}
          {values.map((row, rowIndex) => (
            <GenerationRow
              active={policyCallActive(calls[rowIndex], timestep, rowIndex)}
              call={calls[rowIndex]}
              key={`row-${rowIndex}`}
              max={max}
              onClick={() => onTimestepChange(calls[rowIndex]?.env_timestep ?? rowIndex)}
              row={row}
              rowIndex={rowIndex}
            />
          ))}
        </div>
      ) : (
        <div className="empty-state">No generation actions recorded.</div>
      )}
    </section>
  );
}

function GenerationRow({
  active,
  call,
  max,
  onClick,
  row,
  rowIndex,
}: {
  active: boolean;
  call?: PolicyCall;
  max: number;
  onClick: () => void;
  row: (number | null)[];
  rowIndex: number;
}) {
  const label = call ? `c${call.index}` : `r${rowIndex}`;
  const timestepLabel = call ? `t${call.env_timestep}` : `idx ${rowIndex}`;
  const ariaPrefix = call
    ? `policy call ${call.index} at timestep ${call.env_timestep}`
    : `generation row ${rowIndex}`;
  return (
    <>
      <button className={active ? "generation-row-label active" : "generation-row-label"} type="button" onClick={onClick}>
        <span>{label}</span>
        <small>{timestepLabel}</small>
      </button>
      {row.map((value, columnIndex) => {
        const numericValue = typeof value === "number" && Number.isFinite(value) ? value : null;
        return (
          <button
            aria-label={`${ariaPrefix} generation step ${columnIndex} value ${
              numericValue === null ? "not captured" : numericValue.toFixed(3)
            }`}
            className={active ? "generation-cell active" : "generation-cell"}
            key={`${rowIndex}-${columnIndex}`}
            style={{ background: numericValue === null ? "transparent" : heatColor(numericValue / max) }}
            type="button"
            onClick={onClick}
          >
            {numericValue === null ? "-" : numericValue.toFixed(2)}
          </button>
        );
      })}
    </>
  );
}

function policyCallActive(call: PolicyCall | undefined, timestep: number, fallbackIndex: number): boolean {
  if (!call) {
    return fallbackIndex === timestep;
  }
  return timestep >= call.segment_start && timestep <= call.segment_end;
}

function EpisodeNavigationBar({
  annotation,
  counterfactualPair,
  episode,
  episodeCount,
  episodeIndex,
  hasNext,
  hasPrevious,
  isSavingAnnotation,
  onNext,
  onNavigateTrace,
  onPrevious,
  onSaveAnnotation,
}: {
  annotation?: EpisodeAnnotation;
  counterfactualPair?: CounterfactualPair;
  episode: DatasetEpisode;
  episodeCount: number;
  episodeIndex: number;
  hasNext: boolean;
  hasPrevious: boolean;
  isSavingAnnotation: boolean;
  onNext: () => void;
  onNavigateTrace: (traceId: string | undefined) => void;
  onPrevious: () => void;
  onSaveAnnotation: (annotation: Pick<EpisodeAnnotation, "trace_id" | "starred" | "notes">) => void;
}) {
  const metadataItems = episodeMetadataItems(episode);
  const outcome = String(episode.outcome || "unknown");
  const starred = Boolean(annotation?.starred);
  const savedNotes = annotation?.notes ?? "";
  const [draft, setDraft] = useState({
    notes: savedNotes,
    savedNotes,
    traceId: episode.trace_id,
  });
  const draftNotes =
    draft.traceId === episode.trace_id && draft.savedNotes === savedNotes
      ? draft.notes
      : savedNotes;
  const notesChanged = draftNotes !== savedNotes;
  const saveCurrentAnnotation = (next: { starred?: boolean; notes?: string }) => {
    onSaveAnnotation({
      trace_id: episode.trace_id,
      starred: next.starred ?? starred,
      notes: next.notes ?? draftNotes,
    });
  };
  return (
    <section className="episode-nav-bar" aria-label="Episode metadata and navigation">
      <div className="episode-nav-controls">
        <button
          aria-label="Previous episode"
          disabled={!hasPrevious}
          type="button"
          onClick={onPrevious}
        >
          <ChevronLeft size={16} />
        </button>
        <div className="episode-nav-position">
          <span>Episode</span>
          <strong>
            {episodeIndex >= 0 ? episodeIndex + 1 : "-"} / {episodeCount || "-"}
          </strong>
        </div>
        <button aria-label="Next episode" disabled={!hasNext} type="button" onClick={onNext}>
          <ChevronRight size={16} />
        </button>
      </div>

      <div className="episode-nav-main">
        <div className="episode-nav-meta">
          <span className={`outcome-pill ${outcomeClass(outcome)}`}>{outcome}</span>
          {metadataItems.map((item) => (
            <span className="episode-meta-item" key={item.label}>
              <b>{item.label}</b>
              {item.value}
            </span>
          ))}
        </div>
        {counterfactualPair ? (
          <CounterfactualPairStrip
            activeTraceId={episode.trace_id}
            pair={counterfactualPair}
            onNavigateTrace={onNavigateTrace}
          />
        ) : null}
      </div>
      <div className="episode-annotation-tools">
        <button
          aria-label={starred ? "Unstar episode" : "Star episode"}
          className={starred ? "episode-star-button active" : "episode-star-button"}
          type="button"
          onClick={() => saveCurrentAnnotation({ starred: !starred })}
        >
          <Star size={15} fill={starred ? "currentColor" : "none"} />
        </button>
        <textarea
          aria-label="Episode notes"
          placeholder="Notes"
          value={draftNotes}
          onBlur={() => {
            if (notesChanged) {
              saveCurrentAnnotation({ notes: draftNotes });
            }
          }}
          onChange={(event) =>
            setDraft({
              notes: event.target.value,
              savedNotes,
              traceId: episode.trace_id,
            })
          }
        />
        <button
          disabled={!notesChanged || isSavingAnnotation}
          type="button"
          onClick={() => saveCurrentAnnotation({ notes: draftNotes })}
        >
          {isSavingAnnotation ? "Saving" : "Save"}
        </button>
      </div>
    </section>
  );
}

function CounterfactualPairStrip({
  activeTraceId,
  onNavigateTrace,
  pair,
}: {
  activeTraceId: string;
  onNavigateTrace: (traceId: string | undefined) => void;
  pair: CounterfactualPair;
}) {
  const pairType = pair.type ? pair.type.replaceAll("_", " ") : "paired traces";
  return (
    <div className="counterfactual-pair-strip" aria-label="Counterfactual pair">
      <span className="counterfactual-pair-label">{pairType}</span>
      {pair.members.map((member) => {
        const role = member.role || "trace";
        const isActive = member.trace_id === activeTraceId;
        const target = member.target_object_id || member.counterfactual_target_object_id || "";
        return (
          <button
            className={isActive ? "active" : ""}
            disabled={isActive}
            key={member.trace_id}
            title={member.prompt || member.trace_id}
            type="button"
            onClick={() => onNavigateTrace(member.trace_id)}
          >
            <span>{role}</span>
            {target ? <small>{target}</small> : null}
          </button>
        );
      })}
    </div>
  );
}

function EpisodeArtifactPanel({ artifacts }: { artifacts: Record<string, unknown>[] }) {
  return (
    <section className="episode-tool-panel">
      <header>
        <strong>Episode Artifacts</strong>
        <span className="panel-note">{artifacts.length} linked</span>
      </header>
      {artifacts.length ? (
        <table className="compact-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Type</th>
              <th>Group</th>
            </tr>
          </thead>
          <tbody>
            {artifacts.map((artifact) => (
              <tr key={String(artifact.artifact_id ?? artifact.name)}>
                <td>{String(artifact.name ?? artifact.artifact_id ?? "-")}</td>
                <td>{String(artifact.artifact_type ?? "-")}</td>
                <td>{String(artifact.group_id ?? "-")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <div className="empty-state">No linked artifacts for this episode.</div>
      )}
    </section>
  );
}

function episodesFromManifest(manifest: WorkbenchManifest | undefined): DatasetEpisode[] {
  if (!manifest) {
    return [];
  }
  const byTrace = new Map<string, DatasetEpisode>();
  manifest.image_frames.forEach((frame) => {
    if (!byTrace.has(frame.trace_id)) {
      byTrace.set(frame.trace_id, {
        trace_id: frame.trace_id,
        episode_id: frame.episode_id,
        length: frame.frame_count,
        metadata: frame.provenance,
      });
    }
  });
  return Array.from(byTrace.values());
}

function camerasFromManifest(manifest: WorkbenchManifest | undefined, traceId: string): string[] {
  if (!manifest) {
    return [];
  }
  return manifest.image_frames
    .filter((frame) => frame.trace_id === traceId)
    .map((frame) => frame.camera);
}

function frameVersionKey(episode: DatasetEpisode | undefined, datasetFingerprint: unknown): string {
  const metadata = episode?.metadata ?? {};
  const captureStamp = metadata.timestamp_utc ?? metadata.capture_started_utc ?? metadata.capture_granularity;
  return encodeURIComponent(
    String(datasetFingerprint ?? captureStamp ?? episode?.trace_id ?? "trace_frames_v3"),
  );
}

function episodeMetadataItems(episode: DatasetEpisode): Array<{ label: string; value: string }> {
  return [
    ["Dataset", metadataString(episode, "dataset_id")],
    ["Type", episodeDatasetType(episode)],
    ["Suite", episodeTaskSuite(episode)],
    ["Task", String(episode.task_id ?? metadataString(episode, "task_number") ?? "")],
    ["Seed", metadataString(episode, "seed")],
    ["Length", episode.length === undefined || episode.length === null ? "" : String(episode.length)],
    ["Profile", metadataString(episode, "actual_profile") || metadataString(episode, "capture_profile")],
    ["Batch", metadataString(episode, "batch_id")],
  ]
    .filter(([, value]) => Boolean(value))
    .map(([label, value]) => ({ label, value }));
}

function episodeDatasetType(episode: DatasetEpisode): string {
  const explicit =
    metadataString(episode, "dataset_type") ||
    metadataString(episode, "benchmark") ||
    metadataString(episode, "environment");
  if (explicit) {
    return explicit;
  }
  const env = String(episode.env_id || "");
  if (env.toLowerCase().includes("libero")) {
    return "LIBERO";
  }
  return env;
}

function episodeTaskSuite(episode: DatasetEpisode): string {
  const explicit =
    metadataString(episode, "task_suite") ||
    metadataString(episode, "suite") ||
    metadataString(episode, "benchmark");
  if (explicit) {
    return explicit;
  }
  return String(episode.env_id || "");
}

function metadataString(episode: DatasetEpisode, key: string): string {
  const value = episode.metadata?.[key];
  if (value === undefined || value === null || value === "") {
    return "";
  }
  if (typeof value === "object") {
    return "";
  }
  return String(value);
}

function outcomeClass(outcome: string): string {
  const normalized = outcome.toLowerCase();
  if (["success", "passed", "pass", "true"].includes(normalized)) {
    return "success";
  }
  if (["failure", "failed", "fail", "false", "timeout"].includes(normalized)) {
    return "failure";
  }
  return "";
}

function imageTokenMapQueryKey(
  traceId: string,
  callIndex: number | undefined,
  siteName: string,
  feature: number,
) {
  return ["image-token-map", traceId, callIndex, siteName, feature] as const;
}

function preferredPipelineSite(sites: ActivationSite[]): ActivationSite | undefined {
  return (
    sites.find(
      (site) =>
        site.name === "pi05.vlm.prefix.image_hidden_tokens" &&
        site.axes?.includes("token"),
    ) ??
    sites.find(
      (site) =>
        site.name.includes(".vlm.") &&
        site.token_kind === "image" &&
        site.tensor_type === "hidden_tokens",
    ) ??
    sites.find(
      (site) =>
        site.name.includes(".vlm.") &&
        site.tensor_type === "hidden_tokens" &&
        site.axes?.includes("token"),
    ) ??
    sites.find(
      (site) =>
        site.name.includes(".expert.") &&
        site.tensor_type === "hidden_tokens" &&
        Number(site.layer) === 17,
    ) ??
    sites.find((site) => site.name.includes(".expert.") && site.tensor_type === "hidden_tokens") ??
    sites[0]
  );
}

function inspectorContextForSite(site?: ActivationSite): InspectorContext {
  if (!site) {
    return "other";
  }
  if (isAttentionSite(site)) {
    return "attention";
  }
  if (site.name.includes(".expert.") || site.segment === "action_expert") {
    return "expert";
  }
  if (site.name.includes(".vlm.") || site.segment === "vlm_prefix" || site.token_space_id === "pi05.prefix") {
    return "vlm";
  }
  return "other";
}

function isFeatureActivationSite(site?: ActivationSite): boolean {
  return Boolean(
    site?.axes?.includes("channel") &&
      !isAttentionSite(site),
  );
}

function isAttentionSite(site?: ActivationSite): boolean {
  return Boolean(
    site &&
      (site.tensor_type === "attention" ||
        site.tensor_type === "attention_probs" ||
        site.family === "attention" ||
        site.role === "attention_probs" ||
        site.name.includes("attention")),
  );
}

function isAttentionMapSite(site?: ActivationSite): boolean {
  return Boolean(
    site &&
      (site.tensor_type === "attention" ||
        site.tensor_type === "attention_probs" ||
        site.role === "attention_probs" ||
        site.name.endsWith(".by_step.attention") ||
        site.name.endsWith(".prefix.attention") ||
        site.name.endsWith(".attention.attention_probs")),
  );
}

function siteUsesGenerationStep(site?: ActivationSite): boolean {
  return Boolean(site?.axes?.includes("generation_step"));
}

function expertTokenSiteForSite(
  sites: ActivationSite[],
  selectedSite?: ActivationSite,
): ActivationSite | undefined {
  if (selectedSite?.name.includes(".expert.") && selectedSite.tensor_type === "hidden_tokens") {
    return selectedSite;
  }
  const selectedLayer = selectedSite?.layer;
  if (selectedLayer !== null && selectedLayer !== undefined) {
    const layer = Number(selectedLayer);
    const layerTokenSite = sites.find(
      (site) =>
        site.name.includes(".expert.") &&
        site.tensor_type === "hidden_tokens" &&
        Number(site.layer) === layer,
    );
    if (layerTokenSite) {
      return layerTokenSite;
    }
  }
  return sites.find((site) => site.name.includes(".expert.") && site.tensor_type === "hidden_tokens");
}

function attentionSiteForSite(
  sites: ActivationSite[],
  selectedSite: ActivationSite | undefined,
): ActivationSite | undefined {
  if (isAttentionMapSite(selectedSite)) {
    return selectedSite;
  }
  const context = selectedSite?.name.includes(".vlm.")
    ? "vlm"
    : selectedSite?.name.includes(".expert.")
      ? "expert"
      : inspectorContextForSite(selectedSite);
  const selectedLayer = selectedSite?.layer;
  const sameLayer = (site: ActivationSite) =>
    selectedLayer !== null &&
    selectedLayer !== undefined &&
    Number(site.layer) === Number(selectedLayer);
  const attentionSites = sites.filter(
    (site) =>
      isAttentionSite(site) &&
      (context === "vlm"
        ? site.name.includes(".vlm.") || site.segment === "vlm_prefix"
        : context === "expert"
          ? site.name.includes(".expert.") || site.segment === "action_expert"
          : true),
  );
  return (
    attentionSites.find(
      (site) => sameLayer(site) && site.name.endsWith(".attention.attention_probs"),
    ) ??
    attentionSites.find((site) => sameLayer(site) && site.name.endsWith(".prefix.attention")) ??
    attentionSites.find((site) => sameLayer(site) && site.name.endsWith(".by_step.attention")) ??
    attentionSites.find(sameLayer) ??
    attentionSites[0]
  );
}

function generationStepCountForSite(site?: ActivationSite): number {
  if (!site?.axes?.length || !site.shape?.length) {
    return 1;
  }
  const index = site.axes.indexOf("generation_step");
  if (index < 0) {
    return 1;
  }
  return Math.max(1, Number(site.shape[index]) || 1);
}

function channelCountForSite(site?: ActivationSite): number {
  return axisCountForSite(site, "channel");
}

function axisCountForSite(site: ActivationSite | undefined, axis: string): number {
  if (!site?.axes?.length || !site.shape?.length) {
    return 0;
  }
  const index = site.axes.indexOf(axis);
  if (index < 0) {
    return 0;
  }
  return Math.max(0, Number(site.shape[index]) || 0);
}

function modelPipelineStages(sites: ActivationSite[]): PipelineStage[] {
  if (!sites.length) {
    return [];
  }
  if (!sites.some((site) => site.name.startsWith("pi05."))) {
    return fallbackPipelineStages(sites);
  }

  const prefixSites = sites.filter(
    (site) =>
      site.name === "pi05.vlm.prefix.image_hidden_tokens" ||
      site.name === "pi05.vlm.prefix.input_embeddings",
  );
  const vlmLayerNodes = Array.from({ length: 18 }, (_, layer) =>
    pipelineLayerNode("vlm", layer, sites.filter((site) => site.name.includes(`.vlm.layers.${layer}.`))),
  );
  const generationSites = sites.filter(
    (site) =>
      site.name === "pi05.expert.by_step.input_embeddings" ||
      site.name === "pi05.expert.by_step.position_ids" ||
      site.name === "pi05.expert.by_step.attention_mask" ||
      site.name === "pi05.expert.by_step.causal_mask" ||
      site.name === "pi05.expert.by_step.rope.cos" ||
      site.name === "pi05.expert.by_step.rope.sin",
  );
  const expertLayerNodes = Array.from({ length: 18 }, (_, layer) =>
    pipelineLayerNode("expert", layer, sites.filter((site) => site.name.includes(`.expert.layers.${layer}.`))),
  );
  const actionHeadSites = sites.filter(
    (site) => site.name.includes("pi05.action_head") && site.role !== "action_head_output",
  );
  const actionOutputSites = sites.filter(
    (site) =>
      site.name.includes("action_chunk") ||
      site.name.includes("action_output") ||
      site.role === "action_head_output",
  );

  const stages: PipelineStage[] = [];
  stages.push({
    id: "prefix",
    label: "Input Embed",
    family: "input",
    nodes: [pipelineNode("prefix-embed", "Inputs", "prompt + images", "input", prefixSites)],
  });
  stages.push({
    id: "vlm",
    label: "VLM",
    family: "vlm",
    nodes: vlmLayerNodes,
  });
  stages.push({
    id: "generation",
    label: "Denoising State",
    family: "action",
    nodes: [pipelineNode("generation-step", "x_t", "current action state", "action", generationSites)],
  });
  stages.push({
    id: "expert",
    label: "Action Expert",
    family: "expert",
    nodes: expertLayerNodes,
  });
  stages.push({
    id: "action-head",
    label: "Action Head",
    family: "action",
    nodes: [
      pipelineNode("action-head", "Head", "projection", "action", actionHeadSites),
      pipelineNode("action-output", "Action", "final chunk", "action", actionOutputSites),
    ],
  });
  return stages;
}

function pipelineDiagramLayout(
  stages: PipelineStage[],
  architecture?: ArchitectureMetadata,
): PipelineDiagramLayout {
  const stage = (id: string) => stages.find((entry) => entry.id === id)?.nodes ?? [];
  const prefixNodes = stage("prefix");
  const vlmNodes = stage("vlm");
  const generationNodes = stage("generation");
  const expertNodes = stage("expert");
  const actionHeadNodes = stage("action-head");
  const isPi05Layout = prefixNodes.length > 0 || expertNodes.length > 0;
  const nodes: PipelineDiagramNode[] = [];
  const bands: PipelineDiagramBand[] = [];

  if (!isPi05Layout) {
    let cursor = 0;
    for (const entry of stages) {
      for (const node of entry.nodes) {
        nodes.push({ height: 62, node, stageId: entry.id, width: 92, x: cursor, y: 84 });
        cursor += 104;
      }
      cursor += 36;
    }
    return {
      arrows: sequentialArrows(nodes, "fallback"),
      bands: nodes.length ? [{ className: "other", height: 108, id: "captured", label: "Captured Sites", width: cursor, x: 0, y: 58 }] : [],
      height: 230,
      nodes,
      ports: [],
      width: Math.max(860, cursor + 40),
    };
  }

  const layerWidth = 24;
  const layerHeight = 82;
  const layerStep = 34;
  const yPrefix = 78;
  const yExpert = 254;
  const prefix = prefixNodes[0];
  const currentAction = generationNodes[0];
  const actionHead = actionHeadNodes[0];
  const actionOutput = actionHeadNodes[1];
  const vlmStart = 166;
  const expertStart = vlmStart;
  const vlmEnd = vlmStart + Math.max(0, vlmNodes.length - 1) * layerStep + layerWidth;
  const expertEnd = expertStart + Math.max(0, expertNodes.length - 1) * layerStep + layerWidth;
  const kvBusY = 196;
  const headX = expertEnd + 48;
  const outputX = headX + 118;

  const add = (
    node: PipelineNode | undefined,
    stageId: string,
    x: number,
    y: number,
    width: number,
    height: number,
  ) => {
    if (!node) {
      return;
    }
    nodes.push({ height, node, stageId, width, x, y });
  };

  add(prefix, "prefix", 42, yPrefix + 9, 92, 64);
  vlmNodes.forEach((node, index) => add(node, "vlm", vlmStart + index * layerStep, yPrefix, layerWidth, layerHeight));
  add(currentAction, "generation", 42, yExpert + 9, 92, 64);
  expertNodes.forEach((node, index) => add(node, "expert", expertStart + index * layerStep, yExpert, layerWidth, layerHeight));
  add(actionHead, "action-head", headX, yExpert + 9, 96, 64);
  add(actionOutput, "action-head", outputX, yExpert + 9, 104, 64);

  bands.push(
    { className: "vlm", height: 116, id: "vlm-band", label: "Vision-language input pass", width: vlmEnd + 36, x: 26, y: 48 },
    { className: "expert", height: 134, id: "expert-band", label: "Action denoiser", width: outputX + 132 - 26, x: 26, y: 224 },
  );

  const byId = (id: string) => nodes.find((entry) => entry.node.id === id);
  const firstVlm = vlmNodes[0] ? byId(vlmNodes[0].id) : undefined;
  const lastVlm = vlmNodes.at(-1) ? byId(vlmNodes.at(-1)!.id) : undefined;
  const firstExpert = expertNodes[0] ? byId(expertNodes[0].id) : undefined;
  const lastExpert = expertNodes.at(-1) ? byId(expertNodes.at(-1)!.id) : undefined;
  const prefixBox = prefix ? byId(prefix.id) : undefined;
  const currentBox = currentAction ? byId(currentAction.id) : undefined;
  const headBox = actionHead ? byId(actionHead.id) : undefined;
  const outputBox = actionOutput ? byId(actionOutput.id) : undefined;
  const arrows: PipelineDiagramArrow[] = [];
  const ports: PipelineDiagramPort[] = [];
  if (prefixBox && firstVlm) {
    const y = centerY(prefixBox);
    arrows.push({
      className: "forward",
      id: "prefix-to-vlm",
      path: `M ${right(prefixBox) + 4} ${y} L ${left(firstVlm) - 8} ${y}`,
    });
  }
  if (vlmNodes.length && expertNodes.length) {
    const kvLayers = perLayerKvEdgeLayers(architecture);
    const kvEndpointLayers = kvLayers.length
      ? uniqueNumbers([kvLayers[0], kvLayers[kvLayers.length - 1]])
      : [];
    const pairEndpoints = kvEndpointLayers.length
      ? kvEndpointLayers.map((layer) => byId(`vlm-${layer}`)).filter(isDiagramNode)
      : uniqueDiagramNodes([firstVlm, lastVlm]);
    pairEndpoints.forEach((vlm) => {
      const expert = byId(vlm.node.id.replace("vlm", "expert"));
      if (!expert) {
        return;
      }
      const pairX = centerX(vlm);
      arrows.push({
        className: "conditioning kv-pair",
        id: `kv-pair-${vlm.node.id}`,
        path: `M ${pairX} ${bottom(vlm) + 4} V ${top(expert) - 8}`,
      });
    });
    if (firstVlm && lastVlm) {
      ports.push({
        className: "kv-label",
        id: "kv-same-index-label",
        label: "same-index prefix memory",
        textAnchor: "middle",
        x: (centerX(firstVlm) + centerX(lastVlm)) / 2,
        y: kvBusY - 16,
      });
    }
    const dotLayers = kvLayers.length
      ? kvLayers.filter((layer) => !kvEndpointLayers.includes(layer))
      : [4, 8, 12];
    const dotLabelIndex = Math.floor(dotLayers.length / 2);
    dotLayers.forEach((layer, index) => {
      const vlm = byId(`vlm-${layer}`);
      if (!vlm) {
        return;
      }
      ports.push({
        className: "kv-dot",
        id: `kv-dot-${layer}`,
        label: index === dotLabelIndex ? "..." : undefined,
        radius: 2.6,
        textAnchor: "middle",
        x: centerX(vlm),
        y: kvBusY,
      });
    });
  }
  if (currentBox && firstExpert) {
    arrows.push({
      className: "forward",
      id: "current-action-to-expert",
      path: `M ${right(currentBox) + 4} ${centerY(currentBox)} L ${left(firstExpert) - 8} ${centerY(firstExpert)}`,
    });
  }
  if (lastExpert && headBox) {
    arrows.push({
      className: "forward",
      id: "expert-to-head",
      path: `M ${right(lastExpert) + 8} ${centerY(lastExpert)} L ${left(headBox) - 8} ${centerY(headBox)}`,
    });
  }
  if (headBox && outputBox) {
    arrows.push({
      className: "forward",
      id: "head-to-output",
      path: `M ${right(headBox) + 6} ${centerY(headBox)} L ${left(outputBox) - 8} ${centerY(outputBox)}`,
    });
  }
  if (currentBox && outputBox) {
    const loopY = bottom(outputBox) + 44;
    const returnX = centerX(currentBox);
    const outputXCenter = centerX(outputBox);
    arrows.push({
      className: "loop",
      id: "denoise-update-loop",
      label: "Euler update: x_t + dt * v_t",
      labelAnchor: "middle",
      labelX: (returnX + outputXCenter) / 2,
      labelY: loopY + 19,
      path: `M ${outputXCenter} ${bottom(outputBox) + 4} C ${outputXCenter} ${loopY - 16}, ${outputXCenter - 18} ${loopY}, ${outputXCenter - 42} ${loopY} H ${returnX + 42} C ${returnX + 18} ${loopY}, ${returnX} ${loopY - 16}, ${returnX} ${bottom(currentBox) + 6}`,
    });
  }
  if (outputBox) {
    arrows.push({
      className: "final",
      id: "final-action",
      path: `M ${right(outputBox) + 6} ${centerY(outputBox)} L ${right(outputBox) + 74} ${centerY(outputBox)}`,
    });
  }

  return {
    arrows,
    bands,
    height: 408,
    nodes,
    ports,
    width: Math.max(1080, outputX + 190),
  };
}

function perLayerKvEdgeLayers(architecture?: ArchitectureMetadata): number[] {
  const layers = architecture?.edges
    ?.filter((edge) => edge.kind === "per_layer_kv_conditioning")
    .map((edge) => Number(edge.layer))
    .filter((layer) => Number.isInteger(layer)) ?? [];
  return uniqueNumbers(layers).sort((left, right) => left - right);
}

function uniqueNumbers(values: number[]): number[] {
  return Array.from(new Set(values));
}

function isDiagramNode(value: PipelineDiagramNode | undefined): value is PipelineDiagramNode {
  return Boolean(value);
}

function uniqueDiagramNodes(
  entries: Array<PipelineDiagramNode | undefined>,
): PipelineDiagramNode[] {
  const seen = new Set<string>();
  const nodes: PipelineDiagramNode[] = [];
  for (const entry of entries) {
    if (!entry || seen.has(entry.node.id)) {
      continue;
    }
    seen.add(entry.node.id);
    nodes.push(entry);
  }
  return nodes;
}

function sequentialArrows(nodes: PipelineDiagramNode[], prefix: string): PipelineDiagramArrow[] {
  return nodes.slice(0, -1).map((node, index) => {
    const next = nodes[index + 1];
    return {
      className: "forward",
      id: `${prefix}-${node.node.id}-${next.node.id}`,
      path: `M ${right(node)} ${centerY(node)} L ${left(next)} ${centerY(next)}`,
    };
  });
}

function left(box: PipelineDiagramNode): number {
  return box.x;
}

function right(box: PipelineDiagramNode): number {
  return box.x + box.width;
}

function top(box: PipelineDiagramNode): number {
  return box.y;
}

function bottom(box: PipelineDiagramNode): number {
  return box.y + box.height;
}

function centerX(box: PipelineDiagramNode): number {
  return box.x + box.width / 2;
}

function centerY(box: PipelineDiagramNode): number {
  return box.y + box.height / 2;
}

function fallbackPipelineStages(sites: ActivationSite[]): PipelineStage[] {
  const byModule = new Map<string, ActivationSite[]>();
  for (const site of sites) {
    const key = site.module || site.segment || site.family || "model";
    byModule.set(key, [...(byModule.get(key) ?? []), site]);
  }
  return [
    {
      id: "captured-sites",
      label: "Captured Sites",
      family: "other",
      nodes: [...byModule.entries()].map(([module, moduleSites]) =>
        pipelineNode(
          `site-${module}`,
          module.split(".").at(-1) || "Site",
          summarizeSiteKinds(moduleSites),
          "other",
          moduleSites,
        ),
      ),
    },
  ];
}

function pipelineLayerNode(
  family: Extract<PipelineFamily, "vlm" | "expert">,
  layer: number,
  sites: ActivationSite[],
): PipelineNode {
  return pipelineNode(`${family}-${layer}`, `L${layer}`, summarizeSiteKinds(sites), family, sites);
}

function pipelineNode(
  id: string,
  label: string,
  fallbackSublabel: string,
  family: PipelineFamily,
  sites: ActivationSite[],
): PipelineNode {
  const inspectableSites = sortSites(sites.filter(isInspectableSite));
  const sortedSites = sortSites(filterVisibleCaptureSites(inspectableSites));
  const toChoice = (site: ActivationSite): PipelineSiteChoice => ({
    group: captureGroupForSite(site),
    id: site.site_id || site.name,
    label: siteOptionLabel(site),
    mode: inspectionModeForSite(site),
    site,
  });
  return {
    id,
    label,
    sublabel: sortedSites.length ? summarizeSiteKinds(sortedSites) : fallbackSublabel,
    family,
    captured: sortedSites.length > 0,
    sites: sortedSites,
    allSites: inspectableSites,
    choices: sortedSites.map(toChoice),
    rawChoices: inspectableSites.map(toChoice),
  };
}

function isInspectableSite(site: ActivationSite): boolean {
  const role = String(site.role || "");
  return Boolean(
    isAttentionSite(site) ||
      site.family === "attention" ||
      (site.tensor_type === "hidden_tokens" && site.axes?.includes("channel")) ||
      (site.tensor_type === "hidden_mean" && site.axes?.includes("channel")) ||
      site.role === "input_embeddings" ||
      site.tensor_type === "embedding" ||
      site.family === "residual" ||
      site.family === "normalization" ||
      site.family === "mlp" ||
      site.family === "cache" ||
      site.tensor_type === "cache" ||
      site.name.includes(".kv_cache.") ||
      site.name.includes("mask") ||
      site.name.includes("position_ids") ||
      site.name.includes(".rope.") ||
      role.includes("mask") ||
      role.includes("position") ||
      role.includes("rope") ||
      site.family === "action_head" ||
      site.tensor_type === "action_head" ||
      site.name.includes("action_head"),
  );
}

function filterVisibleCaptureSites(sites: ActivationSite[]): ActivationSite[] {
  const hasTokenFeatures = sites.some(
    (site) => site.tensor_type === "hidden_tokens" && site.axes?.includes("channel"),
  );
  if (!hasTokenFeatures) {
    return sites;
  }
  return sites.filter((site) => site.tensor_type !== "hidden_mean");
}

function preferredSiteWithinNode(sites: ActivationSite[], mode: InspectionMode = "features"): ActivationSite | undefined {
  const modeSites = mode === "advanced" ? sites : sites.filter((site) => inspectionModeForSite(site) === mode);
  const candidates = modeSites.length ? modeSites : sites;
  return (
    candidates.find((site) => site.token_kind === "image" && site.tensor_type === "hidden_tokens") ??
    candidates.find((site) => site.tensor_type === "hidden_tokens" && !isAttentionSite(site)) ??
    candidates.find(isAttentionMapSite) ??
    candidates.find((site) => String(site.role || "") === "q") ??
    candidates.find((site) => String(site.role || "") === "k") ??
    candidates.find((site) => String(site.role || "") === "v") ??
    candidates.find((site) => site.role === "input_embeddings" || site.tensor_type === "embedding") ??
    candidates[0]
  );
}

function sortSites(sites: ActivationSite[]): ActivationSite[] {
  const rank = (site: ActivationSite) => {
    if (site.tensor_type === "hidden_tokens") return 0;
    if (site.role === "input_embeddings" || site.tensor_type === "embedding") return 1;
    if (site.family === "residual") return 2;
    if (site.family === "normalization") return 3;
    if (isAttentionMapSite(site)) return 4;
    if (String(site.role || "").startsWith("q") || site.tensor_type === "attention_q") return 5;
    if (String(site.role || "").startsWith("k") || site.tensor_type === "attention_k") return 6;
    if (String(site.role || "").startsWith("v") || site.tensor_type === "attention_v") return 7;
    if (isAttentionSite(site)) return 8;
    if (site.family === "mlp") return 9;
    if (site.family === "cache") return 10;
    if (site.tensor_type === "hidden_mean") return 11;
    return 12;
  };
  return [...sites].sort((left, right) => rank(left) - rank(right) || left.name.localeCompare(right.name));
}

function siteOptionLabel(site: ActivationSite): string {
  const role = String(site.role || "");
  if (role === "input_embeddings" || site.tensor_type === "embedding") {
    return "Input features";
  }
  if (site.tensor_type === "hidden_mean") {
    return "Average features";
  }
  if (site.tensor_type === "hidden_tokens") {
    if (site.token_kind === "image") {
      return "Image features";
    }
    if (site.token_kind === "prefix") {
      return "Text + image features";
    }
    return "Action features";
  }
  if (["q", "k", "v"].includes(role)) {
    if (role === "q") return "Query vectors";
    if (role === "k") return "Key vectors";
    return "Value vectors";
  }
  if (role.includes("logits") || role.includes("scores")) {
    return "Attention scores";
  }
  if (role.includes("o_proj")) {
    return "Attention output";
  }
  if (isAttentionMapSite(site)) {
    return site.name.includes("key_mass") ? "Attention summary" : "Attention map";
  }
  if (isAttentionSite(site)) {
    return labelFromSnake(role || site.tensor_type || "Attention capture");
  }
  if (site.family === "residual") {
    return residualCaptureLabel(role);
  }
  if (site.family === "normalization") {
    return role.includes("adarms") ? adarmsCaptureLabel(role) : "Normalized features";
  }
  if (site.family === "mlp") {
    return mlpCaptureLabel(role);
  }
  if (site.family === "cache") {
    return role.includes("key") ? "KV cache key" : role.includes("value") ? "KV cache value" : "Cache";
  }
  if (site.name.includes("action_head")) {
    return role === "action_head_output" ? "Action output" : "Head input";
  }
  return labelFromSnake(site.tensor_type || site.token_kind || site.module?.split(".").at(-1) || "Site");
}

function residualCaptureLabel(role: string): string {
  if (role.includes("pre_attention")) return "Layer input";
  if (role.includes("post_attention")) return "After attention";
  if (role.includes("pre_mlp")) return "Before MLP";
  if (role.includes("post_mlp")) return "Layer output";
  return "Layer state";
}

function adarmsCaptureLabel(role: string): string {
  if (role.includes("scale")) return "Conditioning scale";
  if (role.includes("shift")) return "Conditioning shift";
  if (role.includes("gate")) return "Conditioning gate";
  return "Conditioning";
}

function mlpCaptureLabel(role: string): string {
  if (role.includes("gate")) return "MLP gate";
  if (role.includes("up")) return "MLP up";
  if (role.includes("intermediate")) return "MLP hidden";
  if (role.includes("down")) return "MLP down";
  if (role.includes("output")) return "MLP output";
  return "MLP";
}

function captureGroupForSite(site: ActivationSite): CaptureGroupId {
  const role = String(site.role || "");
  if (site.family === "mlp") return "mlp";
  if (site.family === "normalization" && role.includes("adarms")) return "mlp";
  if (site.family === "cache" || site.tensor_type === "cache" || site.name.includes(".kv_cache.")) return "saved_state";
  if (
    isAttentionSite(site) ||
    site.family === "attention" ||
    role.startsWith("q") ||
    role.startsWith("k") ||
    role.startsWith("v")
  ) {
    return "attention";
  }
  if (site.family === "action_head" || site.tensor_type === "action_head" || site.name.includes("action_head")) {
    return "action";
  }
  if (
    site.tensor_type === "hidden_tokens" ||
    site.tensor_type === "hidden_mean" ||
    site.tensor_type === "embedding" ||
    site.family === "residual" ||
    site.family === "normalization"
  ) {
    return "features";
  }
  return "other";
}

const inspectionModes: InspectionMode[] = [
  "features",
  "attention",
  "computation",
  "saved_state",
  "advanced",
];

function inspectionModeForSite(site?: ActivationSite): InspectionMode {
  if (!site) {
    return "features";
  }
  const group = captureGroupForSite(site);
  if (group === "attention") {
    return "attention";
  }
  if (group === "mlp") {
    return "computation";
  }
  if (group === "saved_state") {
    return "saved_state";
  }
  if (group === "features" || group === "action") {
    return "features";
  }
  return "advanced";
}

function inspectionModeLabel(mode: InspectionMode): string {
  const labels: Record<InspectionMode, string> = {
    advanced: "Raw details",
    attention: "Attention",
    computation: "Computation",
    features: "Features",
    saved_state: "Cache",
  };
  return labels[mode];
}

function inspectionModeEmptyMessage(mode: InspectionMode, node: PipelineNode): string {
  if (!node.captured) {
    return "This layer is not captured in the current profile.";
  }
  if (mode === "attention") {
    return "No attention captures are available for this layer in the current profile.";
  }
  if (mode === "computation") {
    return "No MLP, normalization, or conditioning captures are available for this layer in the current profile.";
  }
  if (mode === "saved_state") {
    return "No saved cache, mask, position, or RoPE captures are available for this layer.";
  }
  if (mode === "advanced") {
    return "No raw captures are available for this node.";
  }
  return "No feature captures are available for this layer in the current profile.";
}

function summarizeSiteKinds(sites: ActivationSite[]): string {
  const labels = Array.from(new Set(sites.filter(isInspectableSite).map(siteOptionLabel)));
  return labels.slice(0, 2).join(" + ") || "site";
}

function captureDescription(site: ActivationSite, node: PipelineNode): string {
  const role = String(site.role || "");
  const stack = node.family === "expert" ? "expert" : node.family === "vlm" ? "vlm" : "other";
  if (site.tensor_type === "hidden_tokens") {
    if (site.token_kind === "image") {
      return "One feature vector for each image patch. Pick a channel to see where that feature is strong in the camera view.";
    }
    if (site.token_kind === "action") {
      return "One feature vector for each action slot while the model is refining its action plan.";
    }
    return "Feature vectors for the text and image slots after this layer. Pick a channel to inspect one coordinate.";
  }
  if (site.tensor_type === "hidden_mean") {
    return "A smaller average of the layer features. Useful as a quick summary, but it loses token-level detail.";
  }
  if (role === "input_embeddings" || site.tensor_type === "embedding") {
    return stack === "expert"
      ? "The current action state after it has been converted into model features."
      : "The prompt and camera inputs after they have been converted into model features.";
  }
  if (role === "q") {
    return "The vectors each slot uses to ask for information before attention is computed.";
  }
  if (role === "k") {
    return "The vectors each slot exposes so other slots can match against it. Same shape as values, different contents.";
  }
  if (role === "v") {
    return "The content vectors that get mixed together after attention decides what to read. Same shape as keys, different contents.";
  }
  if (role.includes("logits") || role.includes("scores")) {
    return "Attention scores before they are turned into final attention weights.";
  }
  if (role.includes("o_proj")) {
    return "The attention result after the output projection, ready to be added back into the layer state.";
  }
  if (role.includes("attn_output_pre_o_proj")) {
    return "The attention result before the output projection.";
  }
  if (isAttentionMapSite(site)) {
    if (site.name.includes("key_mass")) {
      return "A compact attention summary. It is cheaper to view, but it is not the full attention map.";
    }
    return stack === "expert"
      ? "Pick a looking action slot, then see which saved scene/text slots and action slots it reads from."
      : "Pick a looking prompt/image slot, then see which prompt and image slots it reads from.";
  }
  if (isAttentionSite(site)) {
    return "An attention-family capture from this layer.";
  }
  if (site.family === "cache" || site.tensor_type === "cache" || site.name.includes(".kv_cache.")) {
    return "One layer's saved prefix keys or values. The action denoiser receives the full list of these per-layer tensors as past_key_values.";
  }
  if (site.family === "residual") {
    return "The layer's running state at this boundary. These are useful points for probes because they show what the layer has added.";
  }
  if (site.family === "normalization") {
    return role.includes("adarms")
      ? "A conditioning signal that scales, shifts, or gates the expert layer."
      : "The layer state after normalization, before the next major computation.";
  }
  if (site.family === "mlp") {
    return "The feed-forward part of the layer. These captures show the non-attention computation inside the block.";
  }
  if (site.family === "action_head" || site.tensor_type === "action_head" || site.name.includes("action_head")) {
    return role === "action_head_output"
      ? "The action chunk predicted from the expert state."
      : "The expert features just before they are projected into action values.";
  }
  return "A captured tensor from this part of the model.";
}

function formatCaptureShape(site: ActivationSite): string {
  const axes = site.axes ?? [];
  const shape = site.shape ?? [];
  return shape.map((value, index) => {
    const axis = axes[index] ? plainAxisLabel(axes[index]) : "dim";
    return `${Number(value).toLocaleString()} ${axis}`;
  }).join(" x ");
}

function plainAxisLabel(axis: string): string {
  const labels: Record<string, string> = {
    action_dim: "action dims",
    cached_token: "saved slots",
    channel: "channels",
    generation_step: "denoise steps",
    head: "heads",
    head_channel: "head channels",
    horizon: "action slots",
    key_token: "looked-at slots",
    policy_call: "model calls",
    query_token: "looking slots",
    timestep: "timesteps",
    token: "slots",
  };
  return labels[axis] ?? labelFromSnake(axis).toLowerCase();
}

function captureStorageLabel(site: ActivationSite): string {
  const parts = [site.materialization, site.exactness, site.dtype].filter(Boolean);
  if (!parts.length) {
    return "";
  }
  return parts
    .join(" / ")
    .replace("raw", "raw")
    .replace("exact", "exact")
    .replace("summary", "summary")
    .replace("lossy_summary", "lossy summary");
}

function captureControlsLabel(site: ActivationSite): string {
  const axes = new Set(site.axes ?? []);
  if (axes.has("query_token") && axes.has("key_token")) {
    return "head selector + looking-slot selector; the overlay shows looked-at slots";
  }
  if (axes.has("head_channel") || axes.has("head")) {
    return axes.has("head_channel")
      ? "head and head-channel controls in raw inspection"
      : "head selector";
  }
  const controls = [];
  if (axes.has("channel")) controls.push("channel slider");
  if (axes.has("generation_step")) controls.push("denoise step");
  if (axes.has("token")) controls.push(site.token_kind === "action" ? "action slot" : "token/patch");
  return controls.length ? controls.join(" + ") : "No special control for this capture yet.";
}

function estimateCaptureSize(site: ActivationSite): string {
  const shape = site.shape ?? [];
  if (!shape.length) {
    return "";
  }
  const dtype = String(site.dtype ?? "").toLowerCase();
  const bytesPerElement =
    dtype.includes("float16") || dtype.includes("bfloat16") || dtype.includes("int16")
      ? 2
      : dtype.includes("float64") || dtype.includes("int64")
        ? 8
        : dtype.includes("int8") || dtype.includes("uint8") || dtype.includes("bool")
          ? 1
          : 4;
  const elements = shape.reduce((product, value) => product * Math.max(1, Number(value) || 1), 1);
  return formatBytes(elements * bytesPerElement);
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "";
  }
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  const digits = value >= 100 || unitIndex === 0 ? 0 : value >= 10 ? 1 : 2;
  return `${value.toFixed(digits)} ${units[unitIndex]}`;
}

function labelFromSnake(value: string): string {
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function heatColor(ratio: number): string {
  const clamped = Math.max(0, Math.min(1, ratio));
  const hue = 205 - clamped * 175;
  const light = 92 - clamped * 34;
  return `hsl(${hue} 78% ${light}%)`;
}

function overlayPatchValue(
  overlay: CameraOverlayPayload | undefined,
  patch: SelectedPatch,
): number | null {
  const value = overlay?.maps?.[patch.camera]?.values?.[patch.row]?.[patch.col];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function overlayCameraMaxAbs(
  overlay: CameraOverlayPayload | undefined,
  camera: string,
): number {
  const values = overlay?.maps?.[camera]?.values ?? [];
  const finite = values.flat().filter((value) => Number.isFinite(value));
  return Math.max(...finite.map((value) => Math.abs(value)), 1e-6);
}

function orderedPromptAttentionRows(
  expertTokenDetails?: ExpertTokenDetailsResponse,
  promptAttention?: PromptAttentionResponse,
): PromptTokenAttention[] {
  if (expertTokenDetails?.available && expertTokenDetails.prompt_tokens?.length) {
    return expertTokenDetails.prompt_tokens;
  }
  if (promptAttention?.prompt_tokens?.length) {
    return promptAttention.prompt_tokens;
  }
  if (expertTokenDetails?.available && expertTokenDetails.top_prompt_tokens?.length) {
    return [...expertTokenDetails.top_prompt_tokens].sort(
      (left, right) => left.local_index - right.local_index,
    );
  }
  return [...(promptAttention?.top_text_tokens ?? [])].sort(
    (left, right) => left.local_index - right.local_index,
  );
}

function taskPromptRows(rows: PromptTokenAttention[]): PromptTokenAttention[] {
  if (!rows.length) {
    return rows;
  }
  const labels = rows.map((row) => displayTokenPiece(row).trim());
  let start = 0;
  const taskIndex = labels.findIndex((label) => label === "Task" || label === "Task:");
  if (taskIndex >= 0) {
    const colonIndex = labels.findIndex((label, index) => index > taskIndex && label === ":");
    start = colonIndex >= 0 ? colonIndex + 1 : taskIndex + 1;
  } else if (labels[0] === "<bos>") {
    start = 1;
  }

  const stateIndex = labels.findIndex(
    (label, index) => index > start && (label === "State" || label === "State:"),
  );
  const actionIndex = labels.findIndex(
    (label, index) => index > start && (label === "Action" || label === "Action:"),
  );
  const stopCandidates = [stateIndex, actionIndex].filter((index) => index >= 0);
  let end = stopCandidates.length ? Math.min(...stopCandidates) : rows.length;
  while (end > start && labels[end - 1] === ",") {
    end -= 1;
  }
  return rows.slice(start, end).filter((row) => {
    const label = displayTokenPiece(row).trim();
    return label && label !== "<bos>" && label !== "<eos>";
  });
}

function promptRowsMatchPrompt(rows: PromptTokenAttention[], prompt?: string | null): boolean {
  if (!rows.length || !prompt) {
    return true;
  }
  const reconstructed = normalizePromptText(rows.map(displayTokenPiece).join(""));
  const expected = normalizePromptText(prompt);
  return reconstructed === expected || reconstructed.includes(expected) || expected.includes(reconstructed);
}

function normalizePromptText(value: string): string {
  return value.replace(/\s+/gu, " ").trim().toLowerCase();
}

function displayTokenPiece(row: PromptTokenAttention): string {
  const token = promptTokenDisplay(row);
  const text = `${token.prefix}${token.text}`;
  return text || (row.token_id === null || row.token_id === undefined ? "?" : "token");
}

function promptTokenTitle(row: PromptTokenAttention): string {
  const parts = [`token ${row.local_index}`, displayTokenPiece(row)];
  if (row.token_id !== null && row.token_id !== undefined) {
    parts.push(`id ${row.token_id}`);
  }
  return parts.join(" - ");
}

function promptTokenDisplay(row: PromptTokenAttention): { prefix: string; text: string } {
  const rawPiece = row.token_piece;
  if (rawPiece === null || rawPiece === undefined || rawPiece === "") {
    return {
      prefix: "",
      text: row.token_id === null || row.token_id === undefined ? "?" : "",
    };
  }
  return cleanPromptTokenPiece(String(rawPiece));
}

function cleanPromptTokenPiece(piece: string): { prefix: string; text: string } {
  let text = piece.replaceAll("<0x0A>", "\n").replaceAll("Ċ", "\n");
  let prefix = "";
  const boundary = text.match(/^[▁_]+/u);
  if (boundary) {
    prefix += " ";
    text = text.slice(boundary[0].length);
  }
  const leadingWhitespace = text.match(/^\s+/u);
  if (leadingWhitespace) {
    prefix += leadingWhitespace[0];
    text = text.slice(leadingWhitespace[0].length);
  }
  return { prefix, text };
}

function signedActivationColor(value: number, maxAbs: number): string {
  const ratio = Math.min(1, Math.abs(value) / Math.max(maxAbs, 1e-6));
  if (value >= 0) {
    return `rgba(156, 74, 88, ${0.07 + ratio * 0.48})`;
  }
  return `rgba(47, 111, 127, ${0.07 + ratio * 0.44})`;
}

function formatMaybeNumber(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(4) : "-";
}

function formatVector(values: number[], precision: number): string {
  return `[${values
    .map((value) =>
      typeof value === "number" && Number.isFinite(value) ? value.toFixed(precision) : "-",
    )
    .join(", ")}]`;
}

function formatPercent(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "-";
}
