import { type QueryClient } from "@tanstack/react-query";
import { useEffect } from "react";

import {
  fetchActivationSites,
  fetchAttentionMap,
  fetchEpisode,
  fetchImageTokenMap,
  fetchPolicyCalls,
} from "../../api/dataset";
import type { ActivationSite, PolicyCall } from "../../types/dataset";
import { imageTokenMapQueryKey } from "./episodeData";
import type { InspectorContext } from "./shared";

type UseEpisodePlaybackParams = {
  isPlayingFrames: boolean;
  maxTimestep: number;
  playbackFps: number;
  setTimestep: (update: (current: number) => number) => void;
};

export function useEpisodePlayback({
  isPlayingFrames,
  maxTimestep,
  playbackFps,
  setTimestep,
}: UseEpisodePlaybackParams) {
  useEffect(() => {
    if (!isPlayingFrames || maxTimestep <= 0) {
      return;
    }
    const interval = window.setInterval(() => {
      setTimestep((current) => (current >= maxTimestep ? 0 : current + 1));
    }, 1000 / Math.max(1, playbackFps));
    return () => window.clearInterval(interval);
  }, [isPlayingFrames, maxTimestep, playbackFps, setTimestep]);
}

type UseAdjacentEpisodePrefetchParams = {
  hasModelSites: boolean;
  hasPolicyCalls: boolean;
  nextTraceId?: string;
  previousTraceId?: string;
  queryClient: QueryClient;
};

export function useAdjacentEpisodePrefetch({
  hasModelSites,
  hasPolicyCalls,
  nextTraceId,
  previousTraceId,
  queryClient,
}: UseAdjacentEpisodePrefetchParams) {
  useEffect(() => {
    const traceIds = [previousTraceId, nextTraceId].filter(
      (traceId): traceId is string => Boolean(traceId),
    );
    const timer = window.setTimeout(() => {
      for (const traceId of traceIds) {
        void queryClient.prefetchQuery({
          queryKey: ["episode", traceId],
          queryFn: () => fetchEpisode(traceId),
          staleTime: 60_000,
        });
        if (hasPolicyCalls) {
          void queryClient.prefetchQuery({
            queryKey: ["policy-calls", traceId],
            queryFn: () => fetchPolicyCalls(traceId),
            staleTime: 60_000,
          });
        }
        if (hasModelSites) {
          void queryClient.prefetchQuery({
            queryKey: ["activation-sites", traceId],
            queryFn: () => fetchActivationSites(traceId),
            staleTime: 60_000,
          });
        }
      }
    }, 250);
    return () => window.clearTimeout(timer);
  }, [
    hasModelSites,
    hasPolicyCalls,
    nextTraceId,
    previousTraceId,
    queryClient,
  ]);
}

type UseOverlayPrefetchParams = {
  activeGenerationStep: number;
  activeSelectedSiteName: string;
  activeTraceId: string;
  attentionHead: number | null;
  attentionQueryToken: number | null;
  attentionSite?: ActivationSite;
  attentionSiteName: string;
  clampedFeature: number;
  hasAttentionMaps: boolean;
  hasImageTokenMaps: boolean;
  hasPolicyCalls: boolean;
  hasTokenSpaces: boolean;
  inspectorContext: InspectorContext;
  isPlayingFrames: boolean;
  nextPolicyCall?: PolicyCall;
  queryClient: QueryClient;
  selectedSiteHasFeatures: boolean;
  showAttentionOverlay: boolean;
};

export function useOverlayPrefetch({
  activeGenerationStep,
  activeSelectedSiteName,
  activeTraceId,
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
}: UseOverlayPrefetchParams) {
  useEffect(() => {
    if (
      !isPlayingFrames ||
      !showAttentionOverlay ||
      !activeTraceId ||
      !nextPolicyCall ||
      !hasPolicyCalls
    ) {
      return;
    }
    if (
      hasImageTokenMaps &&
      inspectorContext === "vlm" &&
      activeSelectedSiteName &&
      selectedSiteHasFeatures
    ) {
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
    if (hasAttentionMaps && hasTokenSpaces && inspectorContext === "attention" && attentionSiteName) {
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
    activeGenerationStep,
    activeSelectedSiteName,
    activeTraceId,
    attentionHead,
    attentionQueryToken,
    attentionSite?.name,
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
  ]);
}
