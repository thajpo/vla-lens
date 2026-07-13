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
    layer: selectedProbeSite?.layer ?? selectedProbeRef?.layer ?? null,
    modelFamily: selectedProbeSite?.family ?? "pi05",
    modelSite: modelSiteName,
    policyCallIndex,
    selectionSource: "probe_model_locus",
    sourceObjectRef: {
      artifactId: activeSelectedProbeArtifactId,
      artifactType: "probe_suite",
      kind: "probe_suite",
      label: selectedProbe?.name,
      layer: selectedProbeSite?.layer ?? selectedProbeRef?.layer ?? null,
      modelSite: modelSiteName,
      policyCallIndex,
      probeId: activeSelectedProbeArtifactId,
      traceId: activeTraceId,
    },
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
  lensId,
  rankingMode,
  seed,
}: {
  displayName?: string | null;
  fetchTarget?: TargetFetcher;
  lensId?: string;
  rankingMode?: string;
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
    feature: seed.feature ?? null,
    layer: seed.layer ?? null,
    modelSite: seed.model_site_id,
    operator: seed.suggested_operator ?? "ablate",
    policyCallIndex: seed.policy_call_index,
    rankingMode,
    selectionSource: rankingMode === "raw_activation" ? "raw_activation" : "probe_contributor",
    sourceObjectRef: {
      artifactId: seed.artifact_id,
      artifactType: seed.family,
      feature: seed.feature ?? null,
      kind: seed.family,
      label: displayName ?? undefined,
      layer: seed.layer ?? null,
      lensId,
      modelSite: seed.model_site_id,
      policyCallIndex: seed.policy_call_index,
      probeId: seed.probe_id ?? seed.artifact_id,
      rankingMode,
      timestep: seed.timestep ?? null,
      traceId: seed.trace_id,
    },
    target,
    title: displayName ? `Intervene with ${displayName}` : undefined,
    tokenSpace: seed.token_space ?? undefined,
    traceId: seed.trace_id,
    timestep: seed.timestep ?? null,
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
