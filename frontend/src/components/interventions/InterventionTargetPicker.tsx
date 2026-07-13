import { useCallback, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  fetchActivationSites,
  fetchDataset,
  fetchDiscoveryArtifactTarget,
  fetchEpisode,
  fetchEpisodeProbes,
  fetchEpisodesPage,
  fetchPolicyCalls,
} from "../../api/dataset";
import type { InterventionLabSeed } from "../../types/interventions";
import { episodeCapabilityGates } from "../../pages/capabilityGating";
import { ModelPipelineMap } from "../../pages/episodes/PipelineMap";
import { LensCompactReadout, TopChannelPanel } from "../../pages/episodes/LensInspectorPanels";
import { useEpisodeInspectorModel } from "../../pages/episodes/useEpisodeInspectorModel";
import { useEpisodeLensView } from "../../pages/episodes/useEpisodeLensView";
import {
  lensFeatureRows,
  probeLayerReferencesFromLensView,
  probeSiteReadoutFromLensView,
  type LensRankingMode,
} from "../../pages/episodes/episodeLensModel";
import { probeLayerReferences } from "../../pages/episodes/episodeProbeModel";
import {
  attentionSiteForSite,
  axisCountForSite,
  inspectionModeForSite,
  isAttentionSite,
  isFeatureActivationSite,
} from "../../pages/episodes/siteModel";
import type { InspectionMode } from "../../pages/episodes/shared";

type InterventionTargetPickerProps = {
  initialDraft?: InterventionLabSeed;
  onDraftChange: (draft: InterventionLabSeed) => void;
};

export function InterventionTargetPicker({
  initialDraft,
  onDraftChange,
}: InterventionTargetPickerProps) {
  const [traceId, setTraceId] = useState(initialDraft?.traceId ?? "");
  const [selectedProbeArtifactId, setSelectedProbeArtifactId] = useState(initialDraft?.artifactId ?? "");
  const [selectedSiteName, setSelectedSiteName] = useState(initialDraft?.modelSite ?? "");
  const [inspectionMode, setInspectionMode] = useState<InspectionMode>("features");
  const [feature, setFeature] = useState(initialDraft?.feature ?? 0);
  const [policyCallIndex, setPolicyCallIndex] = useState(initialDraft?.policyCallIndex ?? 0);
  const [timestep, setTimestep] = useState(initialDraft?.timestep ?? 0);
  const [activationClipPercent, setActivationClipPercent] = useState(0);
  const [topChannelCount, setTopChannelCount] = useState(12);
  const [attentionHead, setAttentionHead] = useState<number | null>(null);
  const [attentionQueryToken, setAttentionQueryToken] = useState<number | null>(null);
  const [generationStep, setGenerationStep] = useState(0);
  const [selectedExpertToken, setSelectedExpertToken] = useState<number | null>(null);
  const [targetMessage, setTargetMessage] = useState("");

  const dataset = useQuery({
    queryKey: ["dataset"],
    queryFn: fetchDataset,
    staleTime: 30_000,
  });
  const capabilities = episodeCapabilityGates(dataset.data?.capabilities?.flags);
  const episodes = useQuery({
    queryKey: ["episodes", "intervention-target-picker"],
    queryFn: ({ signal }) => fetchEpisodesPage({ limit: 100 }, signal),
    staleTime: 30_000,
  });
  const firstEpisode = episodes.data?.episodes[0];
  const activeTraceId = traceId || firstEpisode?.trace_id || "";
  const selectedEpisodeSummary = episodes.data?.episodes.find((episode) => episode.trace_id === activeTraceId);
  const episode = useQuery({
    queryKey: ["episode", activeTraceId, "intervention-target-picker"],
    queryFn: () => fetchEpisode(activeTraceId),
    enabled: Boolean(activeTraceId),
    staleTime: 30_000,
  });
  const policyCalls = useQuery({
    queryKey: ["policy-calls", activeTraceId],
    queryFn: () => fetchPolicyCalls(activeTraceId),
    enabled: Boolean(activeTraceId && capabilities.hasPolicyCalls),
    staleTime: 30_000,
  });
  const episodeProbes = useQuery({
    queryKey: ["episode-probes", activeTraceId],
    queryFn: () => fetchEpisodeProbes(activeTraceId),
    enabled: Boolean(activeTraceId && capabilities.hasProbeArtifacts),
    staleTime: 30_000,
  });
  const activationSites = useQuery({
    queryKey: ["activation-sites", activeTraceId],
    queryFn: () => fetchActivationSites(activeTraceId),
    enabled: Boolean(activeTraceId && capabilities.hasModelSites),
    staleTime: 30_000,
  });

  const selectedEpisode = episode.data ?? selectedEpisodeSummary;
  const maxTimestep = Math.max(0, Number(selectedEpisode?.length ?? 1) - 1);
  const currentTimestep = Math.max(0, Math.min(Number(timestep) || 0, maxTimestep));

  const inspector = useEpisodeInspectorModel({
    activationClipPercent,
    activationSitesData: activationSites.data,
    activeTraceId,
    attentionHead,
    attentionQueryToken,
    capabilities,
    episodeProbesData: episodeProbes.data,
    feature,
    generationStep,
    inspectionMode,
    policyCallsData: policyCalls.data,
    selectedExpertToken,
    selectedPatch: null,
    selectedProbeArtifactId,
    selectedSiteName,
    timestep: currentTimestep,
    topChannelCount,
  });

  const handleFeatureChange = useCallback((nextFeature: number) => {
    setFeature(nextFeature);
    setSelectedExpertToken(null);
  }, []);
  const handleInspectionModeChange = useCallback((mode: InspectionMode) => {
    setInspectionMode(mode);
    if (
      mode === "attention" &&
      attentionQueryToken === null &&
      axisCountForSite(
        attentionSiteForSite(inspector.sites, inspector.selectedSite) ?? inspector.selectedSite,
        "query_token",
      ) > 0
    ) {
      setAttentionQueryToken(0);
    }
  }, [attentionQueryToken, inspector.selectedSite, inspector.sites]);
  const handleSiteChange = useCallback((siteName: string) => {
    const nextSite = inspector.sites.find((site) => site.name === siteName);
    const nextAttentionSite = nextSite
      ? attentionSiteForSite(inspector.sites, nextSite) ?? nextSite
      : undefined;
    setSelectedSiteName(siteName);
    setInspectionMode(inspectionModeForSite(nextSite));
    setAttentionHead(null);
    setAttentionQueryToken(
      nextAttentionSite &&
        isAttentionSite(nextAttentionSite) &&
        axisCountForSite(nextAttentionSite, "query_token") > 0
        ? 0
        : null,
    );
    setFeature(0);
    setSelectedExpertToken(null);
    setGenerationStep(0);
  }, [inspector.sites]);
  const jumpToPolicyCall = useCallback((nextPolicyCallIndex: number) => {
    const call = (policyCalls.data?.calls ?? []).find(
      (item) => Number(item.index) === Number(nextPolicyCallIndex),
    );
    setPolicyCallIndex(nextPolicyCallIndex);
    if (call) {
      setTimestep(call.env_timestep ?? call.segment_start);
    }
  }, [policyCalls.data?.calls]);

  const {
    activeEpisodeLensView,
    jumpToLensDefault,
    lensRankingMode,
    setLensRankingMode,
  } = useEpisodeLensView({
    activeSelectedProbeArtifactId: inspector.activeSelectedProbeArtifactId,
    activeSelectedSiteName: inspector.activeSelectedSiteName,
    activeTraceId,
    clampedFeature: inspector.clampedFeature,
    currentTimestep,
    handleFeatureChange,
    handleInspectionModeChange,
    handleSiteChange,
    hasProbeArtifacts: capabilities.hasProbeArtifacts,
    initialLensRankingMode: initialDraft?.rankingMode,
    jumpToPolicyCall,
    maxTimestep,
    policyCallIndex,
    selectedProbePolicyCall: inspector.selectedProbePolicyCall,
    sendProbeToIntervention: () => undefined,
    setIsPlayingFrames: () => undefined,
    setTimestep,
    suppressInitialLensDefault: Boolean(initialDraft?.modelSite),
    topChannelCount,
  });

  const lensSiteReadout = probeSiteReadoutFromLensView(activeEpisodeLensView);
  const lensRows = lensFeatureRows(activeEpisodeLensView, lensRankingMode);
  const probeRefs = useMemo(() => {
    const lensRefs = probeLayerReferencesFromLensView(activeEpisodeLensView);
    return lensRefs.length ? lensRefs : probeLayerReferences(episodeProbes.data);
  }, [activeEpisodeLensView, episodeProbes.data]);
  const selectedSiteHasFeatures = Boolean(inspector.selectedSiteHasFeatures);
  const topRows = selectedSiteHasFeatures ? inspector.activationSlice.data?.top_abs ?? [] : [];
  const channelFeatureControl =
    inspectionMode === "features" && selectedSiteHasFeatures ? (
      <label className="intervention-target-axis-control">
        <span>Feature</span>
        <input
          min={0}
          type="number"
          value={inspector.clampedFeature}
          onChange={(event) => handleFeatureChange(Number(event.target.value) || 0)}
        />
      </label>
    ) : null;
  const topChannelPanel =
    inspectionMode === "features" && selectedSiteHasFeatures ? (
      <TopChannelPanel
        activationSlice={inspector.activationSlice.data}
        activationSliceFetching={inspector.activationSlice.isFetching}
        activationSlicePlaceholder={inspector.activationSlice.isPlaceholderData}
        activationClipPercent={activationClipPercent}
        feature={inspector.clampedFeature}
        lensRankingMode={lensRankingMode}
        lensRows={lensRows}
        lensSiteReadout={lensSiteReadout}
        onActivationClipPercentChange={setActivationClipPercent}
        onFeatureChange={handleFeatureChange}
        onLensRankingModeChange={setLensRankingMode}
        onTopChannelCountChange={setTopChannelCount}
        selectedSiteHasFeatures={selectedSiteHasFeatures}
        selectedSiteName={inspector.activeSelectedSiteName}
        topChannelCount={topChannelCount}
        topRows={topRows}
      />
    ) : null;

  async function handleUseTarget() {
    const recommendedSelection = selectedSiteName
      ? undefined
      : activeEpisodeLensView?.recommended_selection;
    const recommendedSite = recommendedSelection?.model_site_id
      ? inspector.sites.find((site) => site.name === recommendedSelection.model_site_id)
      : undefined;
    const activeSite = selectedSiteName
      ? inspector.selectedSite
      : recommendedSite ?? inspector.selectedProbeSite ?? inspector.selectedSite;
    const activeSiteName =
      activeSite?.name ?? recommendedSelection?.model_site_id ?? inspector.activeSelectedSiteName;
    const activePolicyCall =
      recommendedSelection?.policy_call_index ??
      (!selectedSiteName ? inspector.selectedProbePolicyCall ?? undefined : undefined) ??
      policyCallIndex ??
      inspector.activeCall?.index ??
      0;
    const activeFeature =
      typeof recommendedSelection?.feature === "number"
        ? recommendedSelection.feature
        : inspector.clampedFeature;
    const activeTimestep =
      typeof recommendedSelection?.timestep === "number"
        ? recommendedSelection.timestep
        : !selectedSiteName && inspector.selectedProbeCall
          ? inspector.selectedProbeCall.env_timestep ?? inspector.selectedProbeCall.segment_start
        : currentTimestep;
    const activeSiteHasFeatures = Boolean(activeSite && isFeatureActivationSite(activeSite));
    const selectionSource = targetSelectionSource(
      inspector.activeSelectedProbeArtifactId,
      lensRankingMode,
      activeSiteHasFeatures,
    );
    let target: Record<string, unknown> | undefined;
    if (inspector.activeSelectedProbeArtifactId) {
      try {
        const response = await fetchDiscoveryArtifactTarget(inspector.activeSelectedProbeArtifactId, {
          model_site: activeSiteName,
          policy_call: activePolicyCall,
          token_space: activeSite?.token_space_id,
          trace_id: activeTraceId,
        });
        target = response.target ?? undefined;
      } catch {
        target = undefined;
      }
    }
    const title = inspector.selectedProbe?.name
      ? `Intervene with ${inspector.selectedProbe.name}`
      : `Intervene at ${inspector.activeSelectedSiteName || "selected site"}`;
    const draft: InterventionLabSeed = {
      artifactId: inspector.activeSelectedProbeArtifactId,
      artifactType: inspector.activeSelectedProbeArtifactId ? "probe_suite" : "",
      feature: activeSiteHasFeatures ? activeFeature : null,
      layer: activeSite?.layer ?? null,
      modelFamily: activeSite?.family ?? "pi05",
      modelSite: activeSiteName,
      operator: inspector.activeSelectedProbeArtifactId ? "add_direction" : "ablate",
      policyCallIndex: activePolicyCall,
      rankingMode: lensRankingMode,
      selectionSource,
      sourceObjectRef: {
        artifactId: inspector.activeSelectedProbeArtifactId || undefined,
        artifactType: inspector.activeSelectedProbeArtifactId ? "probe_suite" : undefined,
        feature: activeSiteHasFeatures ? activeFeature : null,
        kind: inspector.activeSelectedProbeArtifactId ? "probe_suite" : "manual_model_site",
        label: inspector.selectedProbe?.name ?? activeEpisodeLensView?.lens.display_name ?? undefined,
        layer: activeSite?.layer ?? null,
        lensId: activeEpisodeLensView?.lens.artifact_id,
        modelSite: activeSiteName,
        policyCallIndex: activePolicyCall,
        probeId: inspector.activeSelectedProbeArtifactId || undefined,
        rankingMode: lensRankingMode,
        timestep: activeTimestep,
        traceId: activeTraceId,
      },
      target,
      title,
      tokenSpace: activeSite?.token_space_id ?? "pi05.action_suffix",
      traceId: activeTraceId,
      timestep: activeTimestep,
    };
    onDraftChange(draft);
    setTargetMessage(`Target set: ${targetLabel(draft)}`);
  }

  const probeOptions = episodeProbes.data?.probes ?? [];
  const policyCallOptions = policyCalls.data?.calls ?? [];

  return (
    <section className="intervention-target-picker">
      <header className="intervention-target-header">
        <div>
          <span>Model target</span>
          <h2>Choose where to intervene</h2>
          <p>
            Select an episode, probe, policy call, activation tensor, layer, and feature before
            saving or preflighting the intervention recipe.
          </p>
        </div>
        <button
          className="primary-button"
          disabled={!activeTraceId || !inspector.activeSelectedSiteName}
          type="button"
          onClick={handleUseTarget}
        >
          Use as target
        </button>
      </header>

      <div className="intervention-target-controls">
        <LabeledSelect
          label="Episode"
          onChange={(value) => {
            setTraceId(value);
            setSelectedSiteName("");
            setSelectedProbeArtifactId("");
            setFeature(0);
            setPolicyCallIndex(0);
            setTimestep(0);
          }}
          options={(episodes.data?.episodes ?? []).map((item) => ({
            label: item.trace_id,
            value: item.trace_id,
          }))}
          value={activeTraceId}
        />
        <LabeledSelect
          label="Probe view"
          onChange={(value) => setSelectedProbeArtifactId(value)}
          options={[
            { label: "Manual model target", value: "" },
            ...probeOptions.map((probe) => ({
              label: probe.name || probe.artifact_id,
              value: probe.artifact_id,
            })),
          ]}
          value={inspector.activeSelectedProbeArtifactId}
        />
        <LabeledSelect
          label="Policy call"
          onChange={(value) => jumpToPolicyCall(Number(value) || 0)}
          options={policyCallOptions.map((call) => ({
            label: `call ${call.index} · t${call.env_timestep}`,
            value: String(call.index),
          }))}
          value={String(inspector.activeCall?.index ?? policyCallIndex ?? 0)}
        />
      </div>

      <div className="intervention-target-context">
        <span>{selectedEpisode?.prompt || selectedEpisode?.task_id || "Episode context unavailable"}</span>
        <small>
          {[
            activeTraceId,
            inspector.activeSelectedSiteName || "no activation target selected",
            inspector.selectedSite?.layer === null || inspector.selectedSite?.layer === undefined
              ? ""
              : `layer ${inspector.selectedSite.layer}`,
            selectedSiteHasFeatures ? `feature ${inspector.clampedFeature}` : "",
          ].filter(Boolean).join(" · ")}
        </small>
      </div>

      {activationSites.isLoading ? <p className="app-message">Loading model sites.</p> : null}
      {activationSites.isError ? <p className="app-message">Unable to load model sites.</p> : null}
      {activationSites.data?.sites.length ? (
        <ModelPipelineMap
          architecture={inspector.architecture}
          axisControls={channelFeatureControl}
          feature={inspector.clampedFeature}
          inspectionMode={inspectionMode}
          lensContextPanel={
            <LensCompactReadout
              feature={inspector.clampedFeature}
              isError={false}
              isLoading={false}
              lensRequested={Boolean(inspector.activeSelectedProbeArtifactId)}
              selectedSiteName={inspector.activeSelectedSiteName}
              view={activeEpisodeLensView}
              onJumpDefault={jumpToLensDefault}
              onSendToIntervention={handleUseTarget}
            />
          }
          onInspectionModeChange={handleInspectionModeChange}
          onProbeSelect={setSelectedProbeArtifactId}
          onSiteChange={handleSiteChange}
          probeLayerRefs={probeRefs}
          selectedProbeArtifactId={inspector.activeSelectedProbeArtifactId}
          selectedSiteName={inspector.activeSelectedSiteName}
          sites={inspector.sites}
          topChannelPanel={topChannelPanel}
        />
      ) : !activationSites.isLoading ? (
        <p className="empty-state compact">No captured model sites are available for this episode.</p>
      ) : null}
      {targetMessage ? <p className="intervention-target-message">{targetMessage}</p> : null}
    </section>
  );
}

function LabeledSelect({
  label,
  onChange,
  options,
  value,
}: {
  label: string;
  onChange: (value: string) => void;
  options: { label: string; value: string }[];
  value: string;
}) {
  return (
    <label className="lab-field">
      <span>{label}</span>
      <select onChange={(event) => onChange(event.target.value)} value={value}>
        {!options.length ? <option value="">None</option> : null}
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function targetSelectionSource(
  artifactId: string,
  rankingMode: LensRankingMode,
  hasFeature: boolean,
): string {
  if (!artifactId) {
    return "manual_model_site";
  }
  if (hasFeature && rankingMode === "raw_activation") {
    return "raw_activation";
  }
  if (hasFeature) {
    return "probe_contributor";
  }
  return "probe_model_locus";
}

function targetLabel(draft: InterventionLabSeed): string {
  return [
    draft.modelSite,
    draft.layer === null || draft.layer === undefined ? "" : `L${draft.layer}`,
    draft.feature === null || draft.feature === undefined ? "" : `feature ${draft.feature}`,
  ].filter(Boolean).join(" · ");
}
