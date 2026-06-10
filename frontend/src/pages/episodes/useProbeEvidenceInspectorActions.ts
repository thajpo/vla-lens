import { useCallback } from "react";

import type { ResearchSelectionState } from "../../types/probeEvidence";
import type { InspectionMode } from "./shared";

type UseProbeEvidenceDefaultSiteActionArgs = {
  fallback: () => void;
  onInspectionModeChange: (mode: InspectionMode) => void;
  onSiteChange: (siteName: string) => void;
  selection?: ResearchSelectionState | null;
};

export function probeEvidenceDefaultSiteName(
  selection?: ResearchSelectionState | null,
): string | null {
  return selection?.model_locus?.model_site_id ?? selection?.model_locus?.module ?? null;
}

export function useProbeEvidenceDefaultSiteAction({
  fallback,
  onInspectionModeChange,
  onSiteChange,
  selection,
}: UseProbeEvidenceDefaultSiteActionArgs) {
  const siteName = probeEvidenceDefaultSiteName(selection);
  return useCallback(() => {
    if (!siteName) {
      fallback();
      return;
    }
    onInspectionModeChange("features");
    onSiteChange(siteName);
  }, [fallback, onInspectionModeChange, onSiteChange, siteName]);
}
