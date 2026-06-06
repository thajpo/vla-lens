import { fetchDiscoveryArtifactTarget } from "../../api/dataset";
import type { ActivationSite, EpisodeProbeSummary } from "../../types/dataset";
import type { InterventionLabSeed } from "../../types/interventions";
import type { ProbeLayerRef } from "./shared";

export async function buildProbeInterventionSeed({
  activeSelectedProbeArtifactId,
  activeSelectedSiteName,
  activeTraceId,
  selectedProbe,
  selectedProbePolicyCall,
  selectedProbeRef,
  selectedProbeSite,
}: {
  activeSelectedProbeArtifactId: string;
  activeSelectedSiteName: string;
  activeTraceId: string;
  selectedProbe?: EpisodeProbeSummary;
  selectedProbePolicyCall?: number | null;
  selectedProbeRef?: ProbeLayerRef;
  selectedProbeSite?: ActivationSite;
}): Promise<InterventionLabSeed> {
  const policyCallIndex = selectedProbePolicyCall ?? selectedProbeRef?.policyCall ?? 0;
  const modelSiteName =
    selectedProbeSite?.name ?? selectedProbeRef?.modelSiteId ?? activeSelectedSiteName;
  const tokenSpace = selectedProbeSite?.token_space_id ?? "synthetic.action_suffix";
  let target: Record<string, unknown> | undefined;
  try {
    const response = await fetchDiscoveryArtifactTarget(activeSelectedProbeArtifactId, {
      model_site: modelSiteName,
      policy_call: policyCallIndex,
      token_space: tokenSpace,
      trace_id: activeTraceId,
    });
    target = response.target ?? undefined;
  } catch {
    target = undefined;
  }
  return {
    artifactId: activeSelectedProbeArtifactId,
    artifactType: "probe_suite",
    modelFamily: selectedProbeSite?.family ?? "pi05",
    modelSite: modelSiteName,
    policyCallIndex,
    target,
    title: selectedProbe?.name ? `Intervene with ${selectedProbe.name}` : undefined,
    tokenSpace,
    traceId: activeTraceId,
  };
}
