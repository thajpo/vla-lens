import { fetchDiscoveryArtifactTarget } from "../../api/dataset";
import type {
  ActivationSite,
  EpisodeProbeSummary,
  InterventionSeedActionPayload,
} from "../../types/dataset";
import type { InterventionLabSeed } from "../../types/interventions";
import { buildBackendTargetInterventionSeed } from "../../components/interventions/interventionLabModel";
import type { ProbeLayerRef } from "./shared";

type TargetFetcher = typeof fetchDiscoveryArtifactTarget;

/**
 * Seeds the Intervention Lab from a probe selection, preferring the backend
 * TargetSpec so probe and Episode Lens paths share target normalization.
 */
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
  const target = await fetchBackendTarget({
    artifactId: activeSelectedProbeArtifactId,
    fetchTarget: fetchDiscoveryArtifactTarget,
    modelSite: modelSiteName,
    policyCallIndex,
    tokenSpace,
    traceId: activeTraceId,
  });
  return buildBackendTargetInterventionSeed({
    artifactId: activeSelectedProbeArtifactId,
    artifactType: "probe_suite",
    modelFamily: selectedProbeSite?.family ?? "pi05",
    modelSite: modelSiteName,
    policyCallIndex,
    target,
    title: selectedProbe?.name ? `Intervene with ${selectedProbe.name}` : undefined,
    tokenSpace,
    traceId: activeTraceId,
  });
}

/**
 * Seeds the Intervention Lab from an Episode Lens action after validating and
 * normalizing the resolved trace/site/call/token address through the backend.
 */
export async function buildEpisodeLensInterventionSeed({
  displayName,
  fetchTarget = fetchDiscoveryArtifactTarget,
  seed,
}: {
  displayName?: string | null;
  fetchTarget?: TargetFetcher;
  seed: InterventionSeedActionPayload;
}): Promise<InterventionLabSeed> {
  const target = await fetchBackendTarget({
    artifactId: seed.artifact_id,
    fetchTarget,
    modelSite: seed.model_site_id,
    policyCallIndex: seed.policy_call_index,
    tokenSpace: seed.token_space,
    traceId: seed.trace_id,
  });
  return buildBackendTargetInterventionSeed({
    artifactId: seed.artifact_id,
    artifactType: seed.family,
    basis: ["episode_lens_view", "probe_contributors"],
    modelSite: seed.model_site_id,
    operator: seed.suggested_operator ?? "ablate",
    policyCallIndex: seed.policy_call_index,
    target,
    title: displayName ? `Intervene with ${displayName}` : undefined,
    tokenSpace: seed.token_space ?? undefined,
    traceId: seed.trace_id,
  });
}

async function fetchBackendTarget({
  artifactId,
  fetchTarget,
  modelSite,
  policyCallIndex,
  tokenSpace,
  traceId,
}: {
  artifactId: string;
  fetchTarget: TargetFetcher;
  modelSite: string;
  policyCallIndex: number;
  tokenSpace?: string | null;
  traceId: string;
}): Promise<Record<string, unknown> | undefined> {
  try {
    const response = await fetchTarget(artifactId, {
      model_site: modelSite,
      policy_call: policyCallIndex,
      token_space: tokenSpace,
      trace_id: traceId,
    });
    return response.target ?? undefined;
  } catch {
    return undefined;
  }
}
