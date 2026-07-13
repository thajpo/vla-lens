import { useCallback, useEffect, useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { fetchDiscoveryArtifactEpisodeLensView } from "../../api/dataset";
import type { InterventionLabSeed } from "../../types/interventions";
import {
  lensDefaultApplicationKey,
  shouldApplyLensDefault,
  type LensRankingMode,
} from "./episodeLensModel";
import { buildEpisodeLensInterventionSeed } from "./interventionSeed";
import type { InspectionMode } from "./shared";

type UseEpisodeLensViewArgs = {
  activeSelectedProbeArtifactId: string;
  activeSelectedSiteName: string;
  activeTraceId: string;
  clampedFeature: number;
  currentTimestep: number;
  handleFeatureChange: (feature: number) => void;
  handleInspectionModeChange: (mode: InspectionMode) => void;
  handleSiteChange: (siteName: string) => void;
  hasProbeArtifacts: boolean;
  initialLensRankingMode?: string;
  jumpToPolicyCall: (policyCallIndex: number) => void;
  maxTimestep: number;
  policyCallIndex?: number;
  onSendToIntervention?: (seed: InterventionLabSeed) => void;
  selectedProbePolicyCall?: number | null;
  sendProbeToIntervention: () => void;
  setIsPlayingFrames: (value: boolean) => void;
  setTimestep: (value: number) => void;
  suppressInitialLensDefault?: boolean;
  topChannelCount: number;
};

export function useEpisodeLensView({
  activeSelectedProbeArtifactId,
  activeSelectedSiteName,
  activeTraceId,
  clampedFeature,
  currentTimestep,
  handleFeatureChange,
  handleInspectionModeChange,
  handleSiteChange,
  hasProbeArtifacts,
  initialLensRankingMode,
  jumpToPolicyCall,
  maxTimestep,
  policyCallIndex,
  onSendToIntervention,
  selectedProbePolicyCall,
  sendProbeToIntervention,
  setIsPlayingFrames,
  setTimestep,
  suppressInitialLensDefault = false,
  topChannelCount,
}: UseEpisodeLensViewArgs) {
  const [lensRankingMode, setLensRankingMode] =
    useState<LensRankingMode>(coerceLensRankingMode(initialLensRankingMode));
  const [appliedLensDefaultKey, setAppliedLensDefaultKey] = useState("");
  const episodeLensView = useQuery({
    queryKey: [
      "episode-lens-view",
      activeSelectedProbeArtifactId,
      activeTraceId,
      currentTimestep,
      policyCallIndex,
      activeSelectedSiteName,
      clampedFeature,
      lensRankingMode,
      topChannelCount,
    ],
    queryFn: () =>
      fetchDiscoveryArtifactEpisodeLensView(activeSelectedProbeArtifactId, {
        feature: clampedFeature,
        model_site_id: activeSelectedSiteName,
        policy_call_index: policyCallIndex ?? selectedProbePolicyCall ?? undefined,
        ranking_mode: lensRankingMode,
        timestep: currentTimestep,
        top_k: topChannelCount,
        trace_id: activeTraceId,
    }),
    enabled: Boolean(hasProbeArtifacts && activeTraceId && activeSelectedProbeArtifactId),
    placeholderData: keepPreviousData,
    staleTime: 15_000,
  });
  const activeEpisodeLensView = episodeLensView.data?.view;
  const applyLensSelection = useCallback(
    (selection = activeEpisodeLensView?.recommended_selection) => {
      if (!selection) {
        return;
      }
      if (selection.mode === "features") {
        handleInspectionModeChange("features");
      }
      if (selection.model_site_id && selection.model_site_id !== activeSelectedSiteName) {
        handleSiteChange(selection.model_site_id);
      }
      if (typeof selection.feature === "number") {
        handleFeatureChange(selection.feature);
      }
      if (typeof selection.policy_call_index === "number") {
        jumpToPolicyCall(selection.policy_call_index);
      }
      if (typeof selection.timestep === "number") {
        setIsPlayingFrames(false);
        setTimestep(Math.max(0, Math.min(maxTimestep, selection.timestep)));
      }
    },
    [
      activeEpisodeLensView?.recommended_selection,
      activeSelectedSiteName,
      handleFeatureChange,
      handleInspectionModeChange,
      handleSiteChange,
      jumpToPolicyCall,
      maxTimestep,
      setIsPlayingFrames,
      setTimestep,
    ],
  );

  useEffect(() => {
    if (!shouldApplyLensDefault(activeEpisodeLensView, appliedLensDefaultKey)) {
      return;
    }
    if (suppressInitialLensDefault) {
      const timer = window.setTimeout(() => {
        setAppliedLensDefaultKey(lensDefaultApplicationKey(activeEpisodeLensView));
      }, 0);
      return () => window.clearTimeout(timer);
    }
    const timer = window.setTimeout(() => {
      applyLensSelection(activeEpisodeLensView?.recommended_selection);
      setAppliedLensDefaultKey(lensDefaultApplicationKey(activeEpisodeLensView));
    }, 0);
    return () => window.clearTimeout(timer);
  }, [
    activeEpisodeLensView,
    appliedLensDefaultKey,
    applyLensSelection,
    setAppliedLensDefaultKey,
    suppressInitialLensDefault,
  ]);

  const jumpToLensDefault = useCallback(() => {
    applyLensSelection(activeEpisodeLensView?.recommended_selection);
  }, [activeEpisodeLensView?.recommended_selection, applyLensSelection]);
  const sendLensToIntervention = useCallback(async () => {
    if (!onSendToIntervention) {
      return;
    }
    const action = activeEpisodeLensView?.actions.find(
      (item) => item.kind === "send_to_intervention",
    );
    if (action?.kind === "send_to_intervention" && "seed" in action && action.seed) {
      const seed = await buildEpisodeLensInterventionSeed({
        displayName: activeEpisodeLensView?.lens.display_name,
        seed: action.seed,
      });
      onSendToIntervention(seed);
      return;
    }
    sendProbeToIntervention();
  }, [activeEpisodeLensView, onSendToIntervention, sendProbeToIntervention]);

  return {
    activeEpisodeLensView,
    jumpToLensDefault,
    lensRankingMode,
    sendLensToIntervention,
    setLensRankingMode,
  };
}

function coerceLensRankingMode(value?: string): LensRankingMode {
  return value === "raw_activation" ? "raw_activation" : "probe_contribution";
}
