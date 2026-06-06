import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useMemo } from "react";

import {
  fetchActivationSlice,
  fetchAttentionMap,
  fetchExpertTokenActivations,
  fetchExpertTokenDetails,
  fetchGenerationCommitment,
  fetchImageTokenMap,
  fetchObservationalComparisons,
  fetchPatchFeatures,
  fetchPromptAttention,
  fetchPromptFeatureMap,
} from "../../api/dataset";
import type {
  ActivationSitesResponse,
  PolicyCall,
  PolicyCallsResponse,
  EpisodeProbesResponse,
  SelectedPatch,
} from "../../types/dataset";
import { episodeCapabilityGates, episodeQueryGates } from "../capabilityGating";
import { imageTokenMapQueryKey, overlayStatusForSelection } from "./episodeData";
import { probeLayerReferences, selectedEpisodeProbe } from "./episodeProbeModel";
import {
  attentionSiteForSite,
  expertTokenSiteForSite,
  generationStepCountForSite,
  inspectorContextForSite,
  isFeatureActivationSite,
  preferredPipelineSite,
  siteForProbeRef,
  siteUsesGenerationStep,
} from "./siteModel";
import {
  EMPTY_ACTIVATION_SITES,
  type CameraOverlayPayload,
  type InspectionMode,
  type InspectorContext,
} from "./shared";

const EMPTY_POLICY_CALLS: PolicyCall[] = [];

type UseEpisodeInspectorModelParams = {
  activationClipPercent: number;
  activationSitesData?: ActivationSitesResponse;
  activeTraceId: string;
  attentionHead: number | null;
  attentionQueryToken: number | null;
  capabilities: ReturnType<typeof episodeCapabilityGates>;
  feature: number;
  generationStep: number;
  inspectionMode: InspectionMode;
  policyCallsData?: PolicyCallsResponse;
  selectedExpertToken: number | null;
  selectedPatch: SelectedPatch | null;
  selectedProbeArtifactId: string;
  selectedSiteName: string;
  timestep: number;
  topChannelCount: number;
  episodeProbesData?: EpisodeProbesResponse;
};

export function useEpisodeInspectorModel({
  activationClipPercent,
  activationSitesData,
  activeTraceId,
  attentionHead,
  attentionQueryToken,
  capabilities,
  feature,
  generationStep,
  inspectionMode,
  policyCallsData,
  selectedExpertToken,
  selectedPatch,
  selectedProbeArtifactId,
  selectedSiteName,
  timestep,
  topChannelCount,
  episodeProbesData,
}: UseEpisodeInspectorModelParams) {
  const { hasActionGeneration, hasProbeArtifacts } = capabilities;
  const sites = activationSitesData?.sites ?? EMPTY_ACTIVATION_SITES;
  const selectedProbe = useMemo(
    () => selectedEpisodeProbe(episodeProbesData, selectedProbeArtifactId),
    [episodeProbesData, selectedProbeArtifactId],
  );
  const activeSelectedProbeArtifactId = selectedProbe?.artifact_id ?? "";
  const episodeProbeRefs = useMemo(
    () => probeLayerReferences(episodeProbesData),
    [episodeProbesData],
  );
  const selectedProbeRef = useMemo(
    () => episodeProbeRefs.find((ref) => ref.artifactId === activeSelectedProbeArtifactId),
    [activeSelectedProbeArtifactId, episodeProbeRefs],
  );
  const selectedProbeSite = useMemo(
    () => siteForProbeRef(sites, selectedProbeRef),
    [selectedProbeRef, sites],
  );
  const selectedProbePolicyCall = selectedProbeRef?.policyCall;
  const selectedProbeCall =
    selectedProbePolicyCall === null || selectedProbePolicyCall === undefined
      ? undefined
      : (policyCallsData?.calls ?? []).find(
          (item) => Number(item.index) === Number(selectedProbePolicyCall),
        );
  const observationalComparisons = useQuery({
    queryKey: ["observational-comparisons", activeTraceId, activeSelectedProbeArtifactId],
    queryFn: () => fetchObservationalComparisons(activeTraceId, activeSelectedProbeArtifactId),
    enabled: Boolean(activeTraceId && activeSelectedProbeArtifactId && hasProbeArtifacts),
    placeholderData: keepPreviousData,
  });
  const architecture = activationSitesData?.architecture;
  const defaultSite = preferredPipelineSite(sites);
  const selectedSite = sites.find((site) => site.name === selectedSiteName) ?? defaultSite;
  const selectedSiteContext = inspectorContextForSite(selectedSite);
  const generation = useQuery({
    queryKey: ["generation-commitment", activeTraceId],
    queryFn: () => fetchGenerationCommitment(activeTraceId),
    enabled: Boolean(activeTraceId && hasActionGeneration && selectedSiteContext === "expert"),
  });
  const selectedSiteHasFeatures = isFeatureActivationSite(selectedSite);
  const expertTokenSite = expertTokenSiteForSite(sites, selectedSite);
  const expertTokenSiteName = expertTokenSite?.name ?? "";
  const attentionSite = attentionSiteForSite(sites, selectedSite);
  const attentionSiteName = attentionSite?.name ?? "";
  const inspectorContext: InspectorContext =
    inspectionMode === "attention" && attentionSiteName ? "attention" : selectedSiteContext;
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
    (policyCallsData?.calls ?? []).find(
      (call) => timestep >= call.segment_start && timestep <= call.segment_end,
    ) ?? policyCallsData?.calls[0];
  const policyCallList = policyCallsData?.calls ?? EMPTY_POLICY_CALLS;
  const activeCallPosition = activeCall
    ? policyCallList.findIndex((call) => call.index === activeCall.index)
    : -1;
  const nextPolicyCall = activeCallPosition >= 0 ? policyCallList[activeCallPosition + 1] : undefined;
  const queryGates = episodeQueryGates(capabilities, {
    activeTraceId,
    activeSelectedProbeArtifactId,
    activeSelectedSiteName,
    activeCall,
    attentionSiteName,
    effectiveSelectedExpertToken,
    expertTokenSiteName,
    inspectorContext,
    selectedPatch,
    selectedSiteHasFeatures,
  });
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
    queryFn: ({ signal }) =>
      fetchActivationSlice(
        activeTraceId,
        activeCall?.index ?? 0,
        activeSelectedSiteName,
        feature,
        selectedSiteUsesGenerationStep ? activeGenerationStep : undefined,
        activationClipPercent,
        topChannelCount,
        signal,
      ),
    enabled: queryGates.activationSlice,
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
    queryFn: ({ signal }) =>
      fetchImageTokenMap(
        activeTraceId,
        activeCall?.index ?? 0,
        activeSelectedSiteName,
        clampedFeature,
        signal,
      ),
    enabled: queryGates.imageTokenMap,
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
    queryFn: ({ signal }) =>
      fetchPatchFeatures(
        activeTraceId,
        activeCall?.index ?? 0,
        activeSelectedSiteName,
        clampedFeature,
        selectedPatch as SelectedPatch,
        signal,
      ),
    enabled: queryGates.patchFeatures,
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
    queryFn: ({ signal }) =>
      fetchExpertTokenActivations(
        activeTraceId,
        activeCall?.index ?? 0,
        expertTokenSiteName,
        clampedFeature,
        activeGenerationStep,
        signal,
      ),
    enabled: queryGates.expertTokenActivations,
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
    queryFn: ({ signal }) =>
      fetchExpertTokenDetails(
        activeTraceId,
        activeCall?.index ?? 0,
        expertTokenSiteName,
        clampedFeature,
        effectiveSelectedExpertToken ?? 0,
        activeGenerationStep,
        signal,
      ),
    enabled: queryGates.expertTokenDetails,
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
    queryFn: ({ signal }) =>
      fetchAttentionMap(
        activeTraceId,
        activeCall?.index ?? 0,
        attentionSite?.name.includes(".vlm.") ? "vlm" : "expert",
        activeGenerationStep,
        attentionSiteName,
        attentionHead,
        attentionQueryToken,
        signal,
      ),
    enabled: queryGates.attentionMap,
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
    queryFn: ({ signal }) =>
      fetchPromptAttention(
        activeTraceId,
        activeCall?.index ?? 0,
        activeGenerationStep,
        promptAttentionKind,
        attentionSiteName,
        attentionHead,
        attentionQueryToken,
        signal,
      ),
    enabled: queryGates.promptAttention,
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
    queryFn: ({ signal }) =>
      fetchPromptFeatureMap(
        activeTraceId,
        activeCall?.index ?? 0,
        activeSelectedSiteName,
        clampedFeature,
        signal,
      ),
    enabled: queryGates.promptFeatureMap,
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

  return {
    activationSlice,
    activeCall,
    activeGenerationStep,
    activeSelectedProbeArtifactId,
    activeSelectedSiteName,
    architecture,
    attentionMap,
    attentionSite,
    attentionSiteName,
    cameraOverlay,
    cameraOverlayStatus,
    clampedFeature,
    effectiveSelectedExpertToken,
    episodeProbeRefs,
    expertTokenActivations,
    expertTokenDetails,
    expertTokenSiteName,
    generation,
    generationStepCount,
    imageTokenMap,
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
  };
}
