import { useEffect } from "react";

import type { ResearchSelectionState } from "../../types/probeEvidence";
import type { LensRankingMode } from "./episodeLensModel";
import type { InspectionMode } from "./shared";

type UseEpisodeHashSyncArgs = {
  activeSelectedProbeArtifactId: string;
  activeSelectedSiteName: string;
  activeTraceId: string;
  clampedFeature: number;
  cohortReturnHref?: string;
  isPlayingFrames: boolean;
  lensRankingMode: LensRankingMode;
  inspectionMode: InspectionMode;
  lensRunId?: string;
  policyCallIndex?: number;
  researchSelection?: ResearchSelectionState | null;
};

export type EpisodeHashModelArgs = Omit<UseEpisodeHashSyncArgs, "isPlayingFrames">;

export function episodeHashFromSelection({
  activeSelectedProbeArtifactId,
  activeSelectedSiteName,
  activeTraceId,
  clampedFeature,
  cohortReturnHref,
  lensRankingMode,
  lensRunId,
  inspectionMode,
  policyCallIndex,
  researchSelection,
}: EpisodeHashModelArgs): string {
  const params = new URLSearchParams();
  if (activeSelectedProbeArtifactId) {
    params.set("probe_id", activeSelectedProbeArtifactId);
  }
  if (researchSelection?.lens_run_id || lensRunId) {
    params.set("lens_run_id", researchSelection?.lens_run_id || lensRunId || "");
  }
  if (researchSelection?.dataset_id) {
    params.set("dataset_id", researchSelection.dataset_id);
  }
  if (researchSelection?.ranking) {
    params.set("rank", researchSelection.ranking);
  }
  if (typeof policyCallIndex === "number") {
    params.set("call", String(policyCallIndex));
  }
  if (typeof researchSelection?.timestep === "number") {
    params.set("timestep", String(researchSelection.timestep));
  }
  if (activeSelectedSiteName) {
    params.set("site", activeSelectedSiteName);
  }
  if (clampedFeature > 0) {
    params.set("feature", String(clampedFeature));
  }
  if (lensRankingMode !== "probe_contribution") {
    params.set("ranking", lensRankingMode);
  }
  if (inspectionMode !== "features") {
    params.set("mode", inspectionMode);
  }
  if (cohortReturnHref) {
    params.set("from", "cohort");
  }
  const query = params.toString();
  return `#episode/${encodeURIComponent(activeTraceId)}${query ? `?${query}` : ""}`;
}

export function useEpisodeHashSync({
  activeSelectedProbeArtifactId,
  activeSelectedSiteName,
  activeTraceId,
  clampedFeature,
  cohortReturnHref,
  isPlayingFrames,
  lensRankingMode,
  lensRunId,
  inspectionMode,
  policyCallIndex,
  researchSelection,
}: UseEpisodeHashSyncArgs) {
  useEffect(() => {
    if (!activeTraceId || isPlayingFrames || !window.location.hash.startsWith("#episode/")) {
      return;
    }
    const nextHash = episodeHashFromSelection({
      activeSelectedProbeArtifactId,
      activeSelectedSiteName,
      activeTraceId,
      clampedFeature,
      cohortReturnHref,
      lensRankingMode,
      lensRunId,
      inspectionMode,
      policyCallIndex,
      researchSelection,
    });
    if (window.location.hash !== nextHash) {
      window.history.replaceState(null, "", nextHash);
    }
  }, [
    activeSelectedProbeArtifactId,
    activeSelectedSiteName,
    activeTraceId,
    clampedFeature,
    cohortReturnHref,
    inspectionMode,
    isPlayingFrames,
    lensRunId,
    lensRankingMode,
    policyCallIndex,
    researchSelection,
  ]);
}
