export type DatasetCapabilityFlags = Record<string, boolean> | undefined;

export type EpisodeCapabilityGates = {
  hasPolicyCalls: boolean;
  hasModelSites: boolean;
  hasTokenSpaces: boolean;
  hasImageTokenMaps: boolean;
  hasAttentionMaps: boolean;
  hasActionGeneration: boolean;
  hasProbeArtifacts: boolean;
};

export type EpisodeQueryGateInputs = {
  activeTraceId: string;
  activeSelectedProbeArtifactId: string;
  activeSelectedSiteName: string;
  activeCall?: unknown;
  attentionSiteName: string;
  effectiveSelectedExpertToken: number | null;
  expertTokenSiteName: string;
  inspectorContext: string;
  selectedPatch?: unknown;
  selectedSiteHasFeatures: boolean;
};

export type EpisodeQueryGates = {
  policyCalls: boolean;
  episodeProbes: boolean;
  activationSites: boolean;
  observationalComparisons: boolean;
  generation: boolean;
  activationSlice: boolean;
  imageTokenMap: boolean;
  patchFeatures: boolean;
  expertTokenActivations: boolean;
  expertTokenDetails: boolean;
  attentionMap: boolean;
  promptAttention: boolean;
  promptFeatureMap: boolean;
};

export type DatasetBrowserCapabilityGates = {
  hasProbeArtifacts: boolean;
};

function enabledByDefault(flags: DatasetCapabilityFlags, key: string): boolean {
  return flags?.[key] ?? true;
}

export function episodeCapabilityGates(
  flags: DatasetCapabilityFlags,
): EpisodeCapabilityGates {
  return {
    hasPolicyCalls: enabledByDefault(flags, "policy_calls"),
    hasModelSites: enabledByDefault(flags, "model_sites"),
    hasTokenSpaces: enabledByDefault(flags, "token_spaces"),
    hasImageTokenMaps: enabledByDefault(flags, "image_token_maps"),
    hasAttentionMaps: enabledByDefault(flags, "attention_maps"),
    hasActionGeneration: enabledByDefault(flags, "action_generation"),
    hasProbeArtifacts: enabledByDefault(flags, "probe_artifacts"),
  };
}

export function datasetBrowserCapabilityGates(
  flags: DatasetCapabilityFlags,
): DatasetBrowserCapabilityGates {
  return {
    hasProbeArtifacts: enabledByDefault(flags, "probe_artifacts"),
  };
}

export function episodeQueryGates(
  capabilities: EpisodeCapabilityGates,
  inputs: EpisodeQueryGateInputs,
): EpisodeQueryGates {
  const hasTrace = Boolean(inputs.activeTraceId);
  const hasPolicyContext = hasTrace && Boolean(inputs.activeCall);
  const hasSelectedSite = Boolean(inputs.activeSelectedSiteName);
  const hasFeatureSite = hasSelectedSite && inputs.selectedSiteHasFeatures;
  const hasAttentionSite = Boolean(inputs.attentionSiteName);
  const isExpert = inputs.inspectorContext === "expert";
  const isAttention = inputs.inspectorContext === "attention";
  const isVlm = inputs.inspectorContext === "vlm";

  return {
    policyCalls: hasTrace && capabilities.hasPolicyCalls,
    episodeProbes: hasTrace && capabilities.hasProbeArtifacts,
    activationSites: hasTrace && capabilities.hasModelSites,
    observationalComparisons:
      hasTrace && Boolean(inputs.activeSelectedProbeArtifactId) && capabilities.hasProbeArtifacts,
    generation: hasTrace && capabilities.hasActionGeneration && isExpert,
    activationSlice:
      capabilities.hasModelSites &&
      capabilities.hasPolicyCalls &&
      hasPolicyContext &&
      hasFeatureSite,
    imageTokenMap:
      capabilities.hasImageTokenMaps &&
      capabilities.hasPolicyCalls &&
      isVlm &&
      hasPolicyContext &&
      hasFeatureSite,
    patchFeatures:
      capabilities.hasImageTokenMaps &&
      capabilities.hasPolicyCalls &&
      hasPolicyContext &&
      hasFeatureSite &&
      Boolean(inputs.selectedPatch),
    expertTokenActivations:
      capabilities.hasModelSites &&
      capabilities.hasPolicyCalls &&
      isExpert &&
      hasPolicyContext &&
      Boolean(inputs.expertTokenSiteName),
    expertTokenDetails:
      capabilities.hasModelSites &&
      capabilities.hasPolicyCalls &&
      isExpert &&
      hasPolicyContext &&
      Boolean(inputs.expertTokenSiteName) &&
      inputs.effectiveSelectedExpertToken !== null,
    attentionMap:
      capabilities.hasAttentionMaps &&
      capabilities.hasTokenSpaces &&
      capabilities.hasPolicyCalls &&
      isAttention &&
      hasPolicyContext &&
      hasAttentionSite,
    promptAttention:
      capabilities.hasAttentionMaps &&
      capabilities.hasTokenSpaces &&
      capabilities.hasPolicyCalls &&
      hasPolicyContext &&
      hasAttentionSite,
    promptFeatureMap:
      capabilities.hasTokenSpaces &&
      capabilities.hasPolicyCalls &&
      isVlm &&
      hasPolicyContext &&
      hasFeatureSite,
  };
}
