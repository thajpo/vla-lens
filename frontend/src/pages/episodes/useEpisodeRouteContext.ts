import { type MutableRefObject, type RefObject, useEffect } from "react";

import type {
  ActivationSite,
  EpisodeProbeSummary,
  PolicyCall,
  SelectedPatch,
} from "../../types/dataset";
import {
  attentionSiteForSite,
  axisCountForSite,
  isAttentionSite,
  inspectionModeForSite,
} from "./siteModel";
import type { InspectionMode } from "./shared";

type UseEpisodeRouteContextParams = {
  activeSelectedProbeArtifactId: string;
  activeTraceId: string;
  activationSitesIsLoading: boolean;
  appliedRouteContextRef: MutableRefObject<string> | RefObject<string>;
  initialFeature?: number;
  initialInspectionMode?: InspectionMode;
  initialPolicyCall?: number;
  initialProbeArtifactId: string;
  initialSiteName: string;
  maxTimestep: number;
  policyCalls: PolicyCall[];
  selectedProbe?: EpisodeProbeSummary;
  selectedProbePolicyCall?: number | null;
  selectedProbeSite?: ActivationSite;
  sites: ActivationSite[];
  setAttentionHead: (value: number | null) => void;
  setAttentionQueryToken: (value: number | null) => void;
  setFeature: (value: number) => void;
  setGenerationStep: (value: number) => void;
  setInspectionMode: (value: InspectionMode) => void;
  setIsPlayingFrames: (value: boolean) => void;
  setSelectedExpertToken: (value: number | null) => void;
  setSelectedPatch: (value: SelectedPatch | null) => void;
  setSelectedPromptTokenIndex: (value: number | null) => void;
  setSelectedSiteName: (value: string) => void;
  setTimestep: (value: number) => void;
};

export function useEpisodeRouteContext({
  activeSelectedProbeArtifactId,
  activeTraceId,
  activationSitesIsLoading,
  appliedRouteContextRef,
  initialFeature,
  initialInspectionMode,
  initialPolicyCall,
  initialProbeArtifactId,
  initialSiteName,
  maxTimestep,
  policyCalls,
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
}: UseEpisodeRouteContextParams) {
  useEffect(() => {
    if (
      !initialProbeArtifactId &&
      typeof initialPolicyCall !== "number" &&
      typeof initialFeature !== "number" &&
      !initialInspectionMode &&
      !initialSiteName
    ) {
      return;
    }
    const routeKey = [
      activeTraceId,
      initialFeature ?? "",
      initialInspectionMode ?? "",
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
    const targetCall = typeof targetPolicyCall === "number"
      ? policyCalls.find((call) => Number(call.index) === targetPolicyCall)
      : undefined;
    if (typeof targetPolicyCall === "number" && !targetCall) {
      return;
    }
    const targetCallTimestep = targetCall?.env_timestep ?? targetCall?.segment_start;
    if (
      typeof targetCallTimestep === "number" &&
      targetCallTimestep > maxTimestep &&
      maxTimestep <= 0
    ) {
      return;
    }
    if ((initialSiteName || initialProbeArtifactId) && activationSitesIsLoading) {
      return;
    }
    const targetSite = initialSiteName
      ? sites.find((site) => site.name === initialSiteName)
      : selectedProbeSite;
    const targetSiteName = targetSite?.name;
    const targetCallIndex = typeof targetPolicyCall === "number" ? targetPolicyCall : undefined;
    const frame = window.requestAnimationFrame(() => {
      appliedRouteContextRef.current = routeKey;
      if (targetSiteName) {
        const nextSite = sites.find((site) => site.name === targetSiteName);
        const nextAttentionSite = nextSite
          ? attentionSiteForSite(sites, nextSite) ?? nextSite
          : undefined;
        setInspectionMode(initialInspectionMode ?? inspectionModeForSite(nextSite));
        setSelectedSiteName(targetSiteName);
        setAttentionHead(null);
        setAttentionQueryToken(
          nextAttentionSite &&
            isAttentionSite(nextAttentionSite) &&
            axisCountForSite(nextAttentionSite, "query_token") > 0
            ? 0
            : null,
        );
        setFeature(initialFeature ?? 0);
        setSelectedPatch(null);
        setSelectedExpertToken(null);
        setSelectedPromptTokenIndex(null);
        setGenerationStep(0);
      } else if (typeof initialFeature === "number") {
        setFeature(initialFeature);
      } else if (initialInspectionMode) {
        setInspectionMode(initialInspectionMode);
      }
      if (typeof targetCallIndex === "number") {
        if (targetCall) {
          setIsPlayingFrames(false);
          setTimestep(Math.max(0, Math.min(maxTimestep, targetCall.env_timestep ?? targetCall.segment_start)));
        }
      }
    });
    return () => window.cancelAnimationFrame(frame);
  }, [
    activeSelectedProbeArtifactId,
    activeTraceId,
    activationSitesIsLoading,
    appliedRouteContextRef,
    initialFeature,
    initialInspectionMode,
    initialPolicyCall,
    initialProbeArtifactId,
    initialSiteName,
    maxTimestep,
    policyCalls,
    selectedProbe,
    selectedProbePolicyCall,
    selectedProbeSite,
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
    sites,
  ]);
}
