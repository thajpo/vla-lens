import { useCallback, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";

import { fetchEpisodeNeighbors } from "../../api/dataset";

type CounterfactualPairLike = {
  members: Array<{ trace_id: string }>;
};

type EpisodeNavigationArgs<TPair extends CounterfactualPairLike> = {
  activeTraceId: string;
  counterfactualPairs: TPair[];
  onNavigate: (traceId: string) => void;
};

export function useEpisodeNavigation<TPair extends CounterfactualPairLike>({
  activeTraceId,
  counterfactualPairs,
  onNavigate,
}: EpisodeNavigationArgs<TPair>) {
  const neighbors = useQuery({
    queryKey: ["episode-neighbors", activeTraceId],
    queryFn: () => fetchEpisodeNeighbors(activeTraceId),
    enabled: Boolean(activeTraceId),
    staleTime: 30_000,
  });
  const previousTraceId = neighbors.data?.previous_trace_id ?? undefined;
  const nextTraceId = neighbors.data?.next_trace_id ?? undefined;
  const navigateEpisode = useCallback(
    (traceId: string | undefined) => {
      if (traceId) {
        onNavigate(traceId);
      }
    },
    [onNavigate],
  );
  const navigatePreviousEpisode = useCallback(
    () => navigateEpisode(previousTraceId),
    [navigateEpisode, previousTraceId],
  );
  const navigateNextEpisode = useCallback(
    () => navigateEpisode(nextTraceId),
    [navigateEpisode, nextTraceId],
  );
  const activeCounterfactualPair = useMemo(
    () => counterfactualPairs.find((pair) =>
      pair.members.some((member) => member.trace_id === activeTraceId),
    ),
    [activeTraceId, counterfactualPairs],
  );

  return {
    activeCounterfactualPair,
    navigateEpisode,
    navigateNextEpisode,
    navigatePreviousEpisode,
    nextTraceId,
    previousTraceId,
  };
}
