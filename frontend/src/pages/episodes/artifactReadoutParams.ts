import type { DiscoveryArtifactReadoutParams } from "../../api/dataset";

type ReadoutProbeRef = {
  modelSiteId?: string | null;
};

type ReadoutSite = {
  name?: string | null;
  token_space_id?: string | null;
};

export function artifactReadoutParams({
  activeSelectedSiteName,
  activeTraceId,
  selectedProbePolicyCall,
  selectedProbeRef,
  selectedProbeSite,
}: {
  activeSelectedSiteName: string;
  activeTraceId: string;
  selectedProbePolicyCall?: number | null;
  selectedProbeRef?: ReadoutProbeRef;
  selectedProbeSite?: ReadoutSite;
}): DiscoveryArtifactReadoutParams {
  return {
    model_site: selectedProbeSite?.name ?? selectedProbeRef?.modelSiteId ?? activeSelectedSiteName,
    policy_call: selectedProbePolicyCall,
    token_space: selectedProbeSite?.token_space_id,
    trace_id: activeTraceId,
  };
}
