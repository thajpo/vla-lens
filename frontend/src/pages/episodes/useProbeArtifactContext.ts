import { useCallback } from "react";
import { useQuery } from "@tanstack/react-query";

import { fetchDiscoveryArtifactReadout } from "../../api/dataset";
import type { InterventionLabSeed } from "../../types/interventions";
import { artifactReadoutParams as buildArtifactReadoutParams } from "./artifactReadoutParams";
import { buildProbeInterventionSeed } from "./interventionSeed";

type ArtifactReadoutArgs = Parameters<typeof buildArtifactReadoutParams>[0] & {
  activeSelectedProbeArtifactId: string;
  hasProbeArtifacts: boolean;
};

type ProbeInterventionSenderArgs = Parameters<typeof buildProbeInterventionSeed>[0] & {
  onSendToIntervention?: (seed: InterventionLabSeed) => void;
};

export function useDiscoveryArtifactReadout({
  activeSelectedProbeArtifactId,
  hasProbeArtifacts,
  ...paramsArgs
}: ArtifactReadoutArgs) {
  const artifactReadoutParams = buildArtifactReadoutParams(paramsArgs);
  return useQuery({
    queryKey: ["discovery-artifact-readout", activeSelectedProbeArtifactId, artifactReadoutParams],
    queryFn: () => fetchDiscoveryArtifactReadout(activeSelectedProbeArtifactId, artifactReadoutParams),
    enabled: Boolean(hasProbeArtifacts && paramsArgs.activeTraceId && activeSelectedProbeArtifactId),
    staleTime: 15_000,
  });
}

export function useProbeInterventionSender({
  activeSelectedProbeArtifactId,
  activeSelectedSiteName,
  activeTraceId,
  onSendToIntervention,
  selectedProbe,
  selectedProbePolicyCall,
  selectedProbeRef,
  selectedProbeSite,
}: ProbeInterventionSenderArgs) {
  return useCallback(async () => {
    if (!onSendToIntervention || !activeSelectedProbeArtifactId) {
      return;
    }
    const seed = await buildProbeInterventionSeed({
      activeSelectedProbeArtifactId,
      activeSelectedSiteName,
      activeTraceId,
      selectedProbe,
      selectedProbePolicyCall,
      selectedProbeRef,
      selectedProbeSite,
    });
    onSendToIntervention(seed);
  }, [
    activeSelectedProbeArtifactId,
    activeSelectedSiteName,
    activeTraceId,
    onSendToIntervention,
    selectedProbe,
    selectedProbePolicyCall,
    selectedProbeRef,
    selectedProbeSite,
  ]);
}
