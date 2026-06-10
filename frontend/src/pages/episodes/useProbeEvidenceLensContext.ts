import { useQuery } from "@tanstack/react-query";

import { fetchProbeEvidenceBundle } from "../../api/dataset";
import type { EpisodeLensView } from "../../types/dataset";
import type { ResearchSelectionState } from "../../types/probeEvidence";
import {
  lensTimelineMarks,
  probeEvidenceSelection,
  probeEvidenceTimelineMarks,
} from "./episodeLensModel";

export type ProbeEvidenceStatus = "idle" | "loading" | "error" | "ready";

type UseProbeEvidenceLensContextArgs = {
  activeEpisodeLensView?: EpisodeLensView | null;
  activeSelectedProbeArtifactId: string;
  activeTraceId: string;
  currentTimestep: number;
  datasetIdentityKey: string;
  hasProbeArtifacts: boolean;
  initialLensRunId?: string;
  initialResearchSelection?: ResearchSelectionState;
  policyCallIndex?: number;
};

export function useProbeEvidenceLensContext({
  activeEpisodeLensView,
  activeSelectedProbeArtifactId,
  activeTraceId,
  currentTimestep,
  datasetIdentityKey,
  hasProbeArtifacts,
  initialLensRunId,
  initialResearchSelection,
  policyCallIndex,
}: UseProbeEvidenceLensContextArgs) {
  const probeEvidenceBundle = useQuery({
    queryKey: [
      "probe-evidence-bundle",
      datasetIdentityKey,
      initialResearchSelection?.dataset_id ?? "all",
      activeSelectedProbeArtifactId,
    ],
    queryFn: ({ signal }) =>
      fetchProbeEvidenceBundle(
        activeSelectedProbeArtifactId,
        { dataset_id: initialResearchSelection?.dataset_id ?? undefined, limit: 100 },
        signal,
      ),
    enabled: Boolean(hasProbeArtifacts && activeTraceId && activeSelectedProbeArtifactId),
    staleTime: 60_000,
  });
  const activeProbeEvidenceSelection = probeEvidenceSelection(probeEvidenceBundle.data, {
    activeTraceId,
    currentTimestep,
    initialLensRunId,
    initialResearchSelection,
    policyCallIndex,
  });
  const activeLensTimelineMarks = probeEvidenceBundle.data && activeProbeEvidenceSelection
    ? probeEvidenceTimelineMarks(probeEvidenceBundle.data, activeProbeEvidenceSelection)
    : lensTimelineMarks(activeEpisodeLensView);
  const activeProbeEvidenceStatus: ProbeEvidenceStatus = !activeSelectedProbeArtifactId
    ? "idle"
    : probeEvidenceBundle.isError
      ? "error"
      : probeEvidenceBundle.isLoading || probeEvidenceBundle.isFetching
        ? "loading"
        : probeEvidenceBundle.data
          ? "ready"
          : "idle";

  return {
    activeLensTimelineMarks,
    activeProbeEvidenceBundle: probeEvidenceBundle.data,
    activeProbeEvidenceSelection,
    activeProbeEvidenceStatus,
  };
}
