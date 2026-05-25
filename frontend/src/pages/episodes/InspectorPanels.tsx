import { useEffect, useRef, useState } from "react";
import { BarChart3 } from "lucide-react";
import type {
  ActivationSite,
  ActivationSliceResponse,
  ArchitectureMetadata,
  EpisodeProbesResponse,
  ExpertTokenActivationsResponse,
  ExpertTokenDetailsResponse,
  PatchFeaturesResponse,
  PolicyCall,
  PromptAttentionResponse,
  PromptTokenAttention,
  SelectedPatch,
} from "../../types/dataset";
import {
  TOP_CHANNEL_COUNT_OPTIONS,
  type CameraOverlayPayload,
  type InspectionMode,
  type InspectorContext,
} from "./shared";
import { ExpertTokenFlow } from "./ExpertPanels";
import { DetailItem, FeatureTable } from "./InspectorTables";
import { ModelPipelineMap } from "./PipelineMap";
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
  heatColor,
  orderedPromptAttentionRows,
  overlayCameraMaxAbs,
  overlayPatchValue,
  promptRowsMatchPrompt,
  promptTokenDisplay,
  promptTokenTitle,
  signedActivationColor,
  taskPromptRows,
} from "./formatters";

export function InspectorDebugSections({
  artifacts,
  calls,
  generationValues,
  inspectorContext,
  onTimestepChange,
  timestep,
}: {
  artifacts: Record<string, unknown>[];
  calls: PolicyCall[];
  generationValues: (number | null)[][];
  inspectorContext: InspectorContext;
  onTimestepChange: (timestep: number) => void;
  timestep: number;
}) {
  const generationColumns = generationValues[0]?.length ?? 0;
  return (
    <div className="inspector-debug-stack" aria-label="Episode reference panels">
      {inspectorContext === "expert" ? (
        <details className="inspector-disclosure">
          <summary>
            <span>Action Generation</span>
            <small>
              {generationValues.length} x {generationColumns}
            </small>
          </summary>
          <GenerationMatrixPanel
            calls={calls}
            values={generationValues}
            timestep={timestep}
            onTimestepChange={onTimestepChange}
          />
        </details>
      ) : null}

      <details className="inspector-disclosure">
        <summary>
          <span>Episode Artifacts</span>
          <small>{artifacts.length}</small>
        </summary>
        <EpisodeArtifactPanel artifacts={artifacts} />
      </details>
    </div>
  );
}


export function ActivationSitePanel({
  activationSlice,
  activationSliceFetching,
  activationSlicePlaceholder,
  activationClipPercent,
  architecture,
  attentionHead,
  attentionQueryToken,
  cameraOverlay,
  episodeProbes,
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
}: {
  activationSlice?: ActivationSliceResponse;
  activationSliceFetching: boolean;
  activationSlicePlaceholder: boolean;
  activationClipPercent: number;
  architecture?: ArchitectureMetadata;
  attentionHead: number | null;
  attentionQueryToken: number | null;
  cameraOverlay?: CameraOverlayPayload;
  episodeProbes?: EpisodeProbesResponse;
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
}) {
  const selectedSiteHasFeatures = isFeatureActivationSite(selectedSite);
  const siteFeatureCount = channelCountForSite(selectedSite);
  const featureCount = selectedSiteHasFeatures
    ? Math.max(0, siteFeatureCount || activationSlice?.feature_count || 0)
    : 0;
  const topRows = selectedSiteHasFeatures ? activationSlice?.top_abs ?? [] : [];
  const probeLayerRefs = probeLayerReferences(episodeProbes);
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
        onTopChannelCountChange={onTopChannelCountChange}
        selectedSiteHasFeatures={selectedSiteHasFeatures}
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
          Channel {displayedFeature}
          <input
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
          aria-label="Channel index"
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
      {selectedSite?.key_token_space_id === "pi05.expert_context" ? (
        <div className="attention-token-ruler" aria-label="Expert attention key token space">
          <span>Keys</span>
          <i>image patches + prompt</i>
          <i>action tokens</i>
        </div>
      ) : null}
      {headCount > 0 ? (
        <div className="attention-head-control">
          <span className="attention-control-label">Head</span>
          <div className="attention-head-blocks" aria-label="Attention heads">
            <button
              className={clampedHead === null ? "active summary" : "summary"}
              type="button"
              onClick={() => onHeadChange(null)}
            >
              Mean
            </button>
            {Array.from({ length: headCount }, (_, index) => (
              <button
                className={clampedHead === index ? "active" : ""}
                key={index}
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

function TopChannelPanel({
  activationSlice,
  activationSliceFetching,
  activationSlicePlaceholder,
  activationClipPercent,
  feature,
  onActivationClipPercentChange,
  onFeatureChange,
  onTopChannelCountChange,
  selectedSiteHasFeatures,
  topChannelCount,
  topRows,
}: {
  activationSlice?: ActivationSliceResponse;
  activationSliceFetching: boolean;
  activationSlicePlaceholder: boolean;
  activationClipPercent: number;
  feature: number;
  onActivationClipPercentChange: (clipPercent: number) => void;
  onFeatureChange: (feature: number) => void;
  onTopChannelCountChange: (count: number) => void;
  selectedSiteHasFeatures: boolean;
  topChannelCount: number;
  topRows: { index: number; value: number }[];
}) {
  if (!selectedSiteHasFeatures) {
    return null;
  }
  return (
    <div className="top-channel-panel">
      <FeatureTable
        rows={topRows}
        activeFeature={feature}
        title="Top Channels"
        loading={activationSliceFetching && !topRows.length}
        reserveHeight
        updating={activationSliceFetching && Boolean(topRows.length)}
        stale={activationSlicePlaceholder}
        onFeatureChange={onFeatureChange}
        clipPercent={activationClipPercent}
        clip={activationSlice?.clip}
        onClipPercentChange={onActivationClipPercentChange}
        rowLimit={topChannelCount}
        rowLimitOptions={TOP_CHANNEL_COUNT_OPTIONS}
        onRowLimitChange={onTopChannelCountChange}
      />
    </div>
  );
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

export function PromptAttentionStrip({
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


function GenerationMatrixPanel({
  calls,
  values,
  timestep,
  onTimestepChange,
}: {
  calls: PolicyCall[];
  values: (number | null)[][];
  timestep: number;
  onTimestepChange: (timestep: number) => void;
}) {
  const finiteValues = values
    .flat()
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  const max = Math.max(...finiteValues, 1);
  const columns = Math.max(1, values[0]?.length ?? 1);
  return (
    <section className="episode-tool-panel">
      <header>
        <div className="icon-label">
          <BarChart3 size={16} />
          <strong>Generation Commitment</strong>
        </div>
        <span className="panel-note">policy call x generation step</span>
      </header>
      {values.length ? (
        <div
          className="generation-grid"
          style={{ gridTemplateColumns: `72px repeat(${columns}, minmax(26px, 1fr))` }}
        >
          <span />
          {Array.from({ length: columns }, (_, index) => (
            <span className="generation-label" key={`g-${index}`}>
              g{index}
            </span>
          ))}
          {values.map((row, rowIndex) => (
            <GenerationRow
              active={policyCallActive(calls[rowIndex], timestep, rowIndex)}
              call={calls[rowIndex]}
              key={`row-${rowIndex}`}
              max={max}
              onClick={() => onTimestepChange(calls[rowIndex]?.env_timestep ?? rowIndex)}
              row={row}
              rowIndex={rowIndex}
            />
          ))}
        </div>
      ) : (
        <div className="empty-state">No generation actions recorded.</div>
      )}
    </section>
  );
}

function GenerationRow({
  active,
  call,
  max,
  onClick,
  row,
  rowIndex,
}: {
  active: boolean;
  call?: PolicyCall;
  max: number;
  onClick: () => void;
  row: (number | null)[];
  rowIndex: number;
}) {
  const label = call ? `c${call.index}` : `r${rowIndex}`;
  const timestepLabel = call ? `t${call.env_timestep}` : `idx ${rowIndex}`;
  const ariaPrefix = call
    ? `policy call ${call.index} at timestep ${call.env_timestep}`
    : `generation row ${rowIndex}`;
  return (
    <>
      <button className={active ? "generation-row-label active" : "generation-row-label"} type="button" onClick={onClick}>
        <span>{label}</span>
        <small>{timestepLabel}</small>
      </button>
      {row.map((value, columnIndex) => {
        const numericValue = typeof value === "number" && Number.isFinite(value) ? value : null;
        return (
          <button
            aria-label={`${ariaPrefix} generation step ${columnIndex} value ${
              numericValue === null ? "not captured" : numericValue.toFixed(3)
            }`}
            className={active ? "generation-cell active" : "generation-cell"}
            key={`${rowIndex}-${columnIndex}`}
            style={{ background: numericValue === null ? "transparent" : heatColor(numericValue / max) }}
            type="button"
            onClick={onClick}
          >
            {numericValue === null ? "-" : numericValue.toFixed(2)}
          </button>
        );
      })}
    </>
  );
}

function policyCallActive(call: PolicyCall | undefined, timestep: number, fallbackIndex: number): boolean {
  if (!call) {
    return fallbackIndex === timestep;
  }
  return timestep >= call.segment_start && timestep <= call.segment_end;
}


function EpisodeArtifactPanel({ artifacts }: { artifacts: Record<string, unknown>[] }) {
  return (
    <section className="episode-tool-panel">
      <header>
        <strong>Episode Artifacts</strong>
        <span className="panel-note">{artifacts.length} linked</span>
      </header>
      {artifacts.length ? (
        <table className="compact-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Type</th>
              <th>Group</th>
            </tr>
          </thead>
          <tbody>
            {artifacts.map((artifact) => (
              <tr key={String(artifact.artifact_id ?? artifact.name)}>
                <td>{String(artifact.name ?? artifact.artifact_id ?? "-")}</td>
                <td>{String(artifact.artifact_type ?? "-")}</td>
                <td>{String(artifact.group_id ?? "-")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <div className="empty-state">No linked artifacts for this episode.</div>
      )}
    </section>
  );
}
