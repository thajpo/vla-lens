import { useEffect } from "react";

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
  policyCallIndex?: number;
};

export function useEpisodeHashSync({
  activeSelectedProbeArtifactId,
  activeSelectedSiteName,
  activeTraceId,
  clampedFeature,
  cohortReturnHref,
  isPlayingFrames,
  lensRankingMode,
  inspectionMode,
  policyCallIndex,
}: UseEpisodeHashSyncArgs) {
  useEffect(() => {
    if (!activeTraceId || isPlayingFrames || !window.location.hash.startsWith("#episode/")) {
      return;
    }
    const params = new URLSearchParams();
    if (activeSelectedProbeArtifactId) {
      params.set("probe_id", activeSelectedProbeArtifactId);
    }
    if (typeof policyCallIndex === "number") {
      params.set("call", String(policyCallIndex));
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
    const nextHash = `#episode/${encodeURIComponent(activeTraceId)}${query ? `?${query}` : ""}`;
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
    lensRankingMode,
    policyCallIndex,
  ]);
}
