import { memo, useEffect, useRef, useState } from "react";
import type {
  ActivationSite,
  ActivationSliceResponse,
  ArchitectureMetadata,
  EpisodeLensView,
  EpisodeProbesResponse,
  ExpertTokenActivationsResponse,
  ExpertTokenDetailsResponse,
  PatchFeaturesResponse,
  PromptAttentionResponse,
  PromptTokenAttention,
  SelectedPatch,
} from "../../types/dataset";
import type { ProbeEvidenceBundle, ResearchSelectionState } from "../../types/probeEvidence";
import {
  type CameraOverlayPayload,
  type InspectionMode,
  type InspectorContext,
} from "./shared";
import { ExpertTokenFlow } from "./ExpertPanels";
import { LensCompactReadout, TopChannelPanel } from "./LensInspectorPanels";
import { DetailItem } from "./InspectorTables";
import { ModelPipelineMap } from "./PipelineMap";
import {
  lensFeatureRows,
  probeEvidenceFeatureRows,
  probeEvidenceSiteReadoutFromBundle,
  probeLayerReferencesFromEvidenceBundle,
  probeLayerReferencesFromLensView,
  probeSiteReadoutFromLensView,
  type LensRankingMode,
} from "./episodeLensModel";
import { probeLayerReferences } from "./episodeProbeModel";
import {
  attentionSiteForSite,
  axisCountForSite,
  channelCountForSite,
  isFeatureActivationSite,
} from "./siteModel";
import {
  formatMaybeNumber,
  formatPercent,
  orderedPromptAttentionRows,
  overlayCameraMaxAbs,
  overlayPatchValue,
  promptRowsMatchPrompt,
  promptTokenDisplay,
  promptTokenTitle,
  signedActivationColor,
  taskPromptRows,
} from "./formatters";

export { InspectorDebugSections } from "./InspectorDebugPanels";

function ActivationSitePanelImpl({
  activationSlice,
  activationSliceFetching,
  activationSlicePlaceholder,
  activationClipPercent,
  architecture,
  attentionHead,
  attentionQueryToken,
  cameraOverlay,
  episodeLensView,
  episodeProbes,
  probeEvidenceBundle,
  probeEvidenceSelection,
  generationStep,
  generationStepCount,
  expertTokenActivations,
  expertTokenDetails,
  expertTokenSiteName,
  feature,
  inspectorContext,
  inspectionMode,
  onAttentionHeadChange,
  onAttentionQueryTokenChange,
  onGenerationStepChange,
  onExpertTokenChange,
  onActivationClipPercentChange,
  onFeatureChange,
  onInspectionModeChange,
  onPatchSelect,
  onLensDefaultJump,
  onLensRankingModeChange,
  onLensSendToIntervention,
  onProbeSelect,
  onPromptTokenSelect,
  onTopChannelCountChange,
  onSiteChange,
  patchFeatures,
  selectedExpertToken,
  selectedPatch,
  selectedPromptToken,
  selectedProbeArtifactId,
  sites,
  selectedSite,
  selectedSiteName,
  topChannelCount,
  lensRankingMode,
}: {
  activationSlice?: ActivationSliceResponse;
  activationSliceFetching: boolean;
  activationSlicePlaceholder: boolean;
  activationClipPercent: number;
  architecture?: ArchitectureMetadata;
  attentionHead: number | null;
  attentionQueryToken: number | null;
  cameraOverlay?: CameraOverlayPayload;
  episodeLensView?: EpisodeLensView;
  episodeProbes?: EpisodeProbesResponse;
  probeEvidenceBundle?: ProbeEvidenceBundle;
  probeEvidenceSelection?: ResearchSelectionState | null;
  generationStep: number;
  generationStepCount: number;
  expertTokenActivations?: ExpertTokenActivationsResponse;
  expertTokenDetails?: ExpertTokenDetailsResponse;
  expertTokenSiteName: string;
  feature: number;
  inspectorContext: InspectorContext;
  inspectionMode: InspectionMode;
  onAttentionHeadChange: (head: number | null) => void;
  onAttentionQueryTokenChange: (token: number | null) => void;
  onGenerationStepChange: (step: number) => void;
  onExpertTokenChange: (tokenIndex: number | null) => void;
  onActivationClipPercentChange: (clipPercent: number) => void;
  onFeatureChange: (feature: number) => void;
  onInspectionModeChange: (mode: InspectionMode) => void;
  onLensDefaultJump: () => void;
  onLensRankingModeChange: (mode: LensRankingMode) => void;
  onLensSendToIntervention: () => void;
  onPatchSelect: (patch: SelectedPatch | null) => void;
  onProbeSelect: (artifactId: string) => void;
  onPromptTokenSelect: (tokenIndex: number | null) => void;
  onTopChannelCountChange: (count: number) => void;
  onSiteChange: (siteName: string) => void;
  patchFeatures?: PatchFeaturesResponse;
  selectedExpertToken: number | null;
  selectedPatch: SelectedPatch | null;
  selectedPromptToken: number | null;
  selectedProbeArtifactId: string;
  sites: ActivationSite[];
  selectedSite?: ActivationSite;
  selectedSiteName: string;
  topChannelCount: number;
  lensRankingMode: LensRankingMode;
}) {
  const selectedSiteHasFeatures = isFeatureActivationSite(selectedSite);
  const siteFeatureCount = channelCountForSite(selectedSite);
  const featureCount = selectedSiteHasFeatures
    ? Math.max(0, siteFeatureCount || activationSlice?.feature_count || 0)
    : 0;
  const topRows = selectedSiteHasFeatures ? activationSlice?.top_abs ?? [] : [];
  const evidenceProbeLayerRefs = probeLayerReferencesFromEvidenceBundle(
    probeEvidenceBundle,
    probeEvidenceSelection ?? null,
  );
  const lensProbeLayerRefs = evidenceProbeLayerRefs.length
    ? evidenceProbeLayerRefs
    : probeLayerReferencesFromLensView(episodeLensView);
  const probeLayerRefs = lensProbeLayerRefs.length
    ? lensProbeLayerRefs
    : probeLayerReferences(episodeProbes);
  const evidenceSiteReadout = probeEvidenceSiteReadoutFromBundle(
    probeEvidenceBundle,
    probeEvidenceSelection ?? null,
    selectedSiteName,
  );
  const lensSiteReadout = evidenceSiteReadout ?? probeSiteReadoutFromLensView(episodeLensView);
  const evidenceRows = probeEvidenceFeatureRows(
    probeEvidenceBundle,
    probeEvidenceSelection ?? null,
    lensRankingMode,
  );
  const lensRows = evidenceRows.length ? evidenceRows : lensFeatureRows(episodeLensView, lensRankingMode);
  const channelFeatureControl =
    inspectionMode === "features" && selectedSiteHasFeatures ? (
      <ChannelFeatureControl
        feature={feature}
        featureCount={featureCount}
        onFeatureChange={onFeatureChange}
        selectedSiteHasFeatures={selectedSiteHasFeatures}
      />
    ) : null;
  const attentionAxisControls =
    inspectionMode === "attention" && selectedSite ? (
      <AttentionAxisControls
        head={attentionHead}
        queryToken={attentionQueryToken}
        selectedSite={attentionSiteForSite(sites, selectedSite) ?? selectedSite}
        onHeadChange={onAttentionHeadChange}
        onQueryTokenChange={onAttentionQueryTokenChange}
      />
    ) : null;
  const topChannelPanel =
    inspectionMode === "features" && selectedSiteHasFeatures ? (
      <TopChannelPanel
        activationSlice={activationSlice}
        activationSliceFetching={activationSliceFetching}
        activationSlicePlaceholder={activationSlicePlaceholder}
        activationClipPercent={activationClipPercent}
        feature={feature}
        onActivationClipPercentChange={onActivationClipPercentChange}
        onFeatureChange={onFeatureChange}
        lensRankingMode={lensRankingMode}
        lensRows={lensRows}
        lensSiteReadout={lensSiteReadout}
        onLensRankingModeChange={onLensRankingModeChange}
        onTopChannelCountChange={onTopChannelCountChange}
        selectedSiteHasFeatures={selectedSiteHasFeatures}
        selectedSiteName={selectedSiteName}
        topChannelCount={topChannelCount}
        topRows={topRows}
      />
    ) : null;
  return (
    <section className="episode-tool-panel">
      <ModelPipelineMap
        architecture={architecture}
        feature={feature}
        axisControls={attentionAxisControls ?? channelFeatureControl}
        inspectionMode={inspectionMode}
        lensContextPanel={
          <LensCompactReadout
            feature={feature}
            probeEvidenceBundle={probeEvidenceBundle}
            probeEvidenceSelection={probeEvidenceSelection}
            selectedSiteName={selectedSiteName}
            view={episodeLensView}
            onJumpDefault={onLensDefaultJump}
            onSendToIntervention={onLensSendToIntervention}
          />
        }
        onInspectionModeChange={onInspectionModeChange}
        onProbeSelect={onProbeSelect}
        probeLayerRefs={probeLayerRefs}
        selectedProbeArtifactId={selectedProbeArtifactId}
        selectedSiteName={selectedSiteName}
        sites={sites}
        topChannelPanel={topChannelPanel}
        onSiteChange={onSiteChange}
      />
      {selectedSite ? (
        <>
          {selectedPatch ? (
            <CurrentImagePatchPanel
              cameraOverlay={cameraOverlay}
              inspectorContext={inspectorContext}
              patchFeatures={patchFeatures}
              selectedPatch={selectedPatch}
            />
          ) : null}

          {inspectorContext !== "vlm" && generationStepCount > 1 ? (
            <GenerationStepControl
              generationStep={generationStep}
              generationStepCount={generationStepCount}
              onGenerationStepChange={onGenerationStepChange}
            />
          ) : null}

          {inspectionMode === "features" && inspectorContext === "expert" ? (
            <>
              <ExpertTokenFlow
                activeFeature={feature}
                details={expertTokenDetails}
                payload={expertTokenActivations}
                selectedPatch={selectedPatch}
                selectedPromptToken={selectedPromptToken}
                selectedToken={selectedExpertToken}
                tokenSiteName={expertTokenSiteName}
                onFeatureChange={onFeatureChange}
                onPatchSelect={onPatchSelect}
                onPromptTokenSelect={onPromptTokenSelect}
                onTokenChange={onExpertTokenChange}
              />
            </>
          ) : null}
        </>
      ) : (
        <div className="empty-state">No activation sites recorded.</div>
      )}
    </section>
  );
}

export const ActivationSitePanel = memo(ActivationSitePanelImpl);

function ChannelFeatureControl({
  feature,
  featureCount,
  onFeatureChange,
  selectedSiteHasFeatures,
}: {
  feature: number;
  featureCount: number;
  onFeatureChange: (feature: number) => void;
  selectedSiteHasFeatures: boolean;
}) {
  const maxFeature = Math.max(0, featureCount - 1);
  const clampFeature = (value: number) => Math.max(0, Math.min(maxFeature, Number.isFinite(value) ? value : 0));
  const [draftFeature, setDraftFeature] = useState<number | null>(null);
  const displayedFeature = draftFeature ?? clampFeature(feature);
  const commitFeature = (value: number) => {
    const next = clampFeature(value);
    setDraftFeature(null);
    if (next !== feature) {
      onFeatureChange(next);
    }
  };

  if (!selectedSiteHasFeatures) {
    return null;
  }

  return (
    <div className="channel-feature-control">
      <div className="feature-control">
        <label>
          Feature {displayedFeature}
          <input
            title="Choose which feature/channel to project onto the episode image or inspect in the ranking table."
            max={maxFeature}
            min={0}
            type="range"
            value={displayedFeature}
            onChange={(event) => setDraftFeature(clampFeature(Number(event.target.value)))}
            onBlur={() => commitFeature(displayedFeature)}
            onKeyUp={() => commitFeature(displayedFeature)}
            onMouseUp={() => commitFeature(displayedFeature)}
            onTouchEnd={() => commitFeature(displayedFeature)}
          />
        </label>
        <input
          aria-label="Feature index"
          title="Feature/channel index within the selected model site."
          max={maxFeature}
          min={0}
          type="number"
          value={displayedFeature}
          onBlur={() => commitFeature(displayedFeature)}
          onChange={(event) => {
            const next = clampFeature(Number(event.target.value));
            commitFeature(next);
          }}
        />
      </div>
    </div>
  );
}

function AttentionAxisControls({
  head,
  onHeadChange,
  onQueryTokenChange,
  queryToken,
  selectedSite,
}: {
  head: number | null;
  onHeadChange: (head: number | null) => void;
  onQueryTokenChange: (token: number | null) => void;
  queryToken: number | null;
  selectedSite?: ActivationSite;
}) {
  const headCount = axisCountForSite(selectedSite, "head");
  const queryCount = axisCountForSite(selectedSite, "query_token");
  const clampedHead =
    head === null || headCount <= 0 ? null : Math.max(0, Math.min(head, headCount - 1));
  const clampedQuery =
    queryToken === null || queryCount <= 0
      ? null
      : Math.max(0, Math.min(queryToken, queryCount - 1));
  const queryCommitTimer = useRef<number | null>(null);
  const activeQuery = clampedQuery ?? 0;
  const [draftQuery, setDraftQuery] = useState<number | null>(null);
  const displayedQuery = draftQuery ?? activeQuery;
  const queryLabel =
    selectedSite?.query_token_space_id === "pi05.action_horizon" ||
    selectedSite?.token_kind === "action"
      ? "Action slot"
      : "Query slot";
  const keyHint = attentionKeyHint(selectedSite);
  const clampQuery = (value: number) =>
    Math.max(0, Math.min(Math.max(0, queryCount - 1), Number.isFinite(value) ? value : 0));
  const commitQuery = (value: number) => {
    const next = clampQuery(value);
    if (queryCommitTimer.current !== null) {
      window.clearTimeout(queryCommitTimer.current);
      queryCommitTimer.current = null;
    }
    setDraftQuery(null);
    onQueryTokenChange(next);
  };
  const scheduleQueryCommit = (value: number) => {
    const next = clampQuery(value);
    setDraftQuery(next);
    if (queryCommitTimer.current !== null) {
      window.clearTimeout(queryCommitTimer.current);
    }
    queryCommitTimer.current = window.setTimeout(() => {
      queryCommitTimer.current = null;
      setDraftQuery(null);
      onQueryTokenChange(next);
    }, 120);
  };

  useEffect(() => {
    return () => {
      if (queryCommitTimer.current !== null) {
        window.clearTimeout(queryCommitTimer.current);
      }
    };
  }, []);

  if (headCount <= 0 && queryCount <= 0) {
    return null;
  }

  return (
    <div className="attention-axis-controls">
      {keyHint ? (
        <p className="attention-token-hint" title={keyHint.title}>
          {keyHint.label}
        </p>
      ) : null}
      {headCount > 0 ? (
        <div className="attention-head-control">
          <span className="attention-control-label">Head</span>
          <div className="attention-head-blocks" aria-label="Attention heads">
            <button
              className={clampedHead === null ? "active summary" : "summary"}
              title="Average attention across all heads."
              type="button"
              onClick={() => onHeadChange(null)}
            >
              Mean
            </button>
            {Array.from({ length: headCount }, (_, index) => (
              <button
                className={clampedHead === index ? "active" : ""}
                key={index}
                title={`Inspect attention head ${index}.`}
                type="button"
                onClick={() => onHeadChange(index)}
              >
                {index}
              </button>
            ))}
          </div>
        </div>
      ) : null}
      {queryCount > 0 ? (
        <div className="feature-control attention-query-control">
          <label>
            {queryLabel} {clampedQuery === null ? "mean" : clampedQuery}
            <input
              title="Choose which query token or action slot is looking at the keys."
              max={Math.max(0, queryCount - 1)}
              min={0}
              type="range"
              value={displayedQuery}
              onBlur={() => commitQuery(displayedQuery)}
              onChange={(event) => scheduleQueryCommit(Number(event.target.value))}
              onMouseUp={() => commitQuery(displayedQuery)}
              onTouchEnd={() => commitQuery(displayedQuery)}
            />
          </label>
          <div className="attention-query-inputs">
            <span>Slot</span>
            <input
              aria-label={`${queryLabel} index`}
              title="Query token/action slot index."
              max={Math.max(0, queryCount - 1)}
              min={0}
              type="number"
              value={displayedQuery}
              onBlur={() => commitQuery(displayedQuery)}
              onChange={(event) => scheduleQueryCommit(Number(event.target.value))}
            />
          </div>
        </div>
      ) : null}
    </div>
  );
}

function attentionKeyHint(selectedSite?: ActivationSite): { label: string; title: string } | null {
  if (!selectedSite?.key_token_space_id) {
    return null;
  }
  if (selectedSite.key_token_space_id === "pi05.expert_context") {
    return {
      label: "Reads from scene/prompt memory and earlier action slots.",
      title: "Expert attention keys contain VLM prefix memory first, followed by action-context tokens.",
    };
  }
  if (selectedSite.key_token_space_id === "pi05.prefix") {
    return {
      label: "Reads from prompt and image-patch tokens.",
      title: "VLM attention keys are the prefix tokens: camera image patches plus prompt text.",
    };
  }
  return {
    label: `Reads from ${selectedSite.key_token_space_id}.`,
    title: "Attention maps show where the selected query token reads from.",
  };
}



function GenerationStepControl({
  generationStep,
  generationStepCount,
  onGenerationStepChange,
}: {
  generationStep: number;
  generationStepCount: number;
  onGenerationStepChange: (step: number) => void;
}) {
  return (
    <div className="generation-control">
      <label>
        Generation step {generationStep}
        <input
          max={Math.max(0, generationStepCount - 1)}
          min={0}
          type="range"
          value={generationStep}
          onChange={(event) => onGenerationStepChange(Number(event.target.value))}
        />
      </label>
    </div>
  );
}

function PromptAttentionStripImpl({
  expertTokenDetails,
  context,
  onPromptTokenSelect,
  prompt,
  promptAttention,
  promptFeatureMap,
  selectedPromptToken,
}: {
  expertTokenDetails?: ExpertTokenDetailsResponse;
  context: InspectorContext;
  onPromptTokenSelect: (tokenIndex: number | null) => void;
  prompt?: string | null;
  promptAttention?: PromptAttentionResponse;
  promptFeatureMap?: PromptAttentionResponse;
  selectedPromptToken: number | null;
}) {
  const modelPromptRows = orderedPromptAttentionRows(
    expertTokenDetails?.available ? expertTokenDetails : undefined,
    context === "vlm" ? promptFeatureMap : promptAttention,
  );
  const candidateRows = taskPromptRows(modelPromptRows);
  const rows = promptRowsMatchPrompt(candidateRows, prompt) ? candidateRows : [];
  const maxAttention = Math.max(
    ...rows.map((row) => Math.abs(Number(row.attention))).filter(Number.isFinite),
    1e-6,
  );
  const promptText =
    prompt || promptFeatureMap?.prompt || promptAttention?.prompt || "No prompt recorded.";

  return (
    <div className="viewer-prompt-strip">
      <div className="viewer-prompt-head">
        <strong>Prompt</strong>
      </div>
      {rows.length ? (
        <PromptAttentionChips
          rows={rows}
          maxAttention={maxAttention}
          selectedPromptToken={selectedPromptToken}
          onPromptTokenSelect={onPromptTokenSelect}
        />
      ) : (
        <p>{promptText}</p>
      )}
    </div>
  );
}

export const PromptAttentionStrip = memo(PromptAttentionStripImpl);

function CurrentImagePatchPanel({
  cameraOverlay,
  inspectorContext,
  patchFeatures,
  selectedPatch,
}: {
  cameraOverlay?: CameraOverlayPayload;
  inspectorContext: InspectorContext;
  patchFeatures?: PatchFeaturesResponse;
  selectedPatch: SelectedPatch;
}) {
  const patchValue = overlayPatchValue(cameraOverlay, selectedPatch);
  const patchMaxAbs = overlayCameraMaxAbs(cameraOverlay, selectedPatch.camera);
  const patchStyle =
    typeof patchValue === "number"
      ? { background: signedActivationColor(patchValue, patchMaxAbs) }
      : undefined;
  const patchValueLabel = inspectorContext === "vlm" ? "Patch feature value" : "Patch attention mass";

  return (
    <div className="current-patch-panel">
      <div className="section-title">
        <span>Current Selection</span>
        <small>{selectedPatch.camera}</small>
      </div>
      <div className="patch-summary-grid">
        <DetailItem label="camera" value={selectedPatch.camera} />
        <DetailItem label="patch" value={`r${selectedPatch.row} c${selectedPatch.col}`} />
        <DetailItem
          label="token"
          value={
            patchFeatures?.available && patchFeatures.token_index !== undefined
              ? String(patchFeatures.token_index)
              : "-"
          }
        />
        <DetailItem
          label="feature rank"
          value={
            patchFeatures?.available && patchFeatures.feature_rank_by_abs
              ? `#${patchFeatures.feature_rank_by_abs}`
              : "-"
          }
        />
      </div>

      <div className="patch-attention-readout">
        <span className="patch-color-chip" style={patchStyle} />
        <div>
          <strong>{patchValueLabel}</strong>
          <span>
            {typeof patchValue === "number"
              ? inspectorContext === "vlm"
                ? formatMaybeNumber(patchValue)
                : formatPercent(patchValue)
              : "unavailable"}
          </span>
        </div>
      </div>
    </div>
  );
}

function PromptAttentionChips({
  maxAttention,
  onPromptTokenSelect,
  rows,
  selectedPromptToken,
}: {
  maxAttention: number;
  onPromptTokenSelect: (tokenIndex: number | null) => void;
  rows: PromptTokenAttention[];
  selectedPromptToken: number | null;
}) {
  if (!rows.length) {
    return <div className="empty-state">Prompt attention is not available for the current selection.</div>;
  }
  return (
    <div className="prompt-attention-chips" aria-label="Prompt attention tokens">
      {rows.map((row) => {
        const attention = Number(row.attention);
        const token = promptTokenDisplay(row);
        const selected = selectedPromptToken === row.local_index;
        return (
          <span key={`${row.prefix_index ?? row.local_index}-${row.token_id ?? row.token_piece ?? ""}`}>
            {token.prefix ? <span className="prompt-token-space">{token.prefix}</span> : null}
            {token.text ? (
              <button
                aria-pressed={selected}
                className={selected ? "prompt-attention-chip active" : "prompt-attention-chip"}
                style={{ background: signedActivationColor(attention, maxAttention) }}
                title={promptTokenTitle(row)}
                type="button"
                onClick={() => onPromptTokenSelect(row.local_index)}
              >
                <strong>{token.text}</strong>
              </button>
            ) : null}
          </span>
        );
      })}
    </div>
  );
}
