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
} from "./siteModel";
import type { InspectionMode } from "./shared";

type UseEpisodeRouteContextParams = {
  activeSelectedProbeArtifactId: string;
  activeTraceId: string;
  activationSitesIsLoading: boolean;
  appliedRouteContextRef: MutableRefObject<string> | RefObject<string>;
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
      policyCalls.some((call) => Number(call.index) === targetPolicyCall);
    if (!targetCallReady) {
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
    appliedRouteContextRef.current = routeKey;
    const frame = window.requestAnimationFrame(() => {
      if (targetSiteName) {
        const nextSite = sites.find((site) => site.name === targetSiteName);
        const nextAttentionSite = nextSite
          ? attentionSiteForSite(sites, nextSite) ?? nextSite
          : undefined;
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
        const call = policyCalls.find((item) => Number(item.index) === Number(targetCallIndex));
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
    activationSitesIsLoading,
    appliedRouteContextRef,
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
