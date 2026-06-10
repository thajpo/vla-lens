import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { Layers3, Maximize2, X } from "lucide-react";
import type { ActivationSite, ArchitectureMetadata } from "../../types/dataset";
import type {
  InspectionMode,
  PipelineNode,
  PipelineSiteChoice,
  PipelineStage,
  ProbeLayerRef,
} from "./shared";
import { formatMaybeNumber } from "./formatters";
import { formatProbeValue } from "./episodeProbeModel";
import { modelPipelineStages, pipelineDiagramLayout } from "./pipelineModel";
import {
  captureDescription,
  channelCountForSite,
  inspectionModeEmptyMessage,
  inspectionModeForSite,
  inspectionModeLabel,
  inspectionModes,
  isFeatureActivationSite,
  preferredSiteWithinNode,
  probeLayerTitle,
  probeRefsForNode,
  probeRefTone,
  probeToneForRefs,
} from "./siteModel";

export function ModelPipelineMap({
  axisControls,
  architecture,
  feature,
  inspectionMode,
  lensContextPanel,
  onInspectionModeChange,
  onProbeSelect,
  onSiteChange,
  probeLayerRefs,
  selectedProbeArtifactId,
  selectedSiteName,
  sites,
  topChannelPanel,
}: {
  axisControls?: ReactNode;
  architecture?: ArchitectureMetadata;
  feature: number;
  inspectionMode: InspectionMode;
  lensContextPanel?: ReactNode;
  onInspectionModeChange: (mode: InspectionMode) => void;
  onProbeSelect: (artifactId: string) => void;
  onSiteChange: (siteName: string) => void;
  probeLayerRefs: ProbeLayerRef[];
  selectedProbeArtifactId: string;
  selectedSiteName: string;
  sites: ActivationSite[];
  topChannelPanel?: ReactNode;
}) {
  const stages = useMemo(() => modelPipelineStages(sites), [sites]);
  const [showPipelineModal, setShowPipelineModal] = useState(false);
  const selectedNodeId = useMemo(
    () => stages.flatMap((stage) => stage.nodes).find((node) =>
      node.allSites.some((site) => site.name === selectedSiteName),
    )?.id,
    [stages, selectedSiteName],
  );
  const [expandedNodeId, setExpandedNodeId] = useState<string>("");
  const allNodes = useMemo(() => stages.flatMap((stage) => stage.nodes), [stages]);
  const expandedNodeStillVisible = allNodes
    .some((node) => node.id === expandedNodeId);
  const activeExpandedNodeId =
    selectedNodeId || (expandedNodeStillVisible ? expandedNodeId : "") || allNodes.find((node) => node.captured)?.id || "";
  const expandedNode = allNodes.find((node) => node.id === activeExpandedNodeId);

  if (!stages.length) {
    return <div className="empty-state">No activation sites captured in this trace.</div>;
  }
  return (
    <div className="model-pipeline">
      <div className="pipeline-toolbar">
        <PipelineLegend probeCount={probeLayerRefs.length} />
      </div>
      <PipelineMap2D
        cornerAction={
          <button
            className="pipeline-view-toggle"
            type="button"
            onClick={() => setShowPipelineModal(true)}
          >
            <Maximize2 size={14} />
            Fullsize
          </button>
        }
        feature={feature}
        inspectionMode={inspectionMode}
        onNodeSelect={(node) => {
          setExpandedNodeId(node.id);
          const preferred = preferredSiteWithinNode(
            inspectionMode === "advanced" ? node.allSites : node.sites,
            inspectionMode,
          );
          if (preferred) {
            if (inspectionMode !== "advanced" && inspectionModeForSite(preferred) !== inspectionMode) {
              onInspectionModeChange(inspectionModeForSite(preferred));
            }
            onSiteChange(preferred.name);
          }
        }}
        probeLayerRefs={probeLayerRefs}
        selectedProbeArtifactId={selectedProbeArtifactId}
        selectedSiteName={selectedSiteName}
        architecture={architecture}
        stages={stages}
      />
      {axisControls ? <div className="pipeline-channel-row">{axisControls}</div> : null}
      {showPipelineModal ? (
        <PipelineMapModal
          feature={feature}
          inspectionMode={inspectionMode}
          overlayControls={axisControls}
          onClose={() => setShowPipelineModal(false)}
          onNodeSelect={(node) => {
            setExpandedNodeId(node.id);
            const preferred = preferredSiteWithinNode(
              inspectionMode === "advanced" ? node.allSites : node.sites,
              inspectionMode,
            );
            if (preferred) {
              if (inspectionMode !== "advanced" && inspectionModeForSite(preferred) !== inspectionMode) {
                onInspectionModeChange(inspectionModeForSite(preferred));
              }
              onSiteChange(preferred.name);
            }
          }}
          probeLayerRefs={probeLayerRefs}
          selectedProbeArtifactId={selectedProbeArtifactId}
          selectedSiteName={selectedSiteName}
          architecture={architecture}
          stages={stages}
        />
      ) : null}
      <PipelineSelectedNodePanel
        inspectionMode={inspectionMode}
        node={expandedNode}
        onInspectionModeChange={onInspectionModeChange}
        onProbeSelect={onProbeSelect}
        onSiteChange={onSiteChange}
        probeLayerRefs={probeLayerRefs}
        selectedProbeArtifactId={selectedProbeArtifactId}
        selectedSiteName={selectedSiteName}
      />
      {lensContextPanel ? <div className="pipeline-lens-context">{lensContextPanel}</div> : null}
      {topChannelPanel ? <div className="pipeline-top-channels">{topChannelPanel}</div> : null}
    </div>
  );
}

function PipelineSelectedNodePanel({
  inspectionMode,
  node,
  onInspectionModeChange,
  onProbeSelect,
  onSiteChange,
  probeLayerRefs,
  selectedProbeArtifactId,
  selectedSiteName,
}: {
  inspectionMode: InspectionMode;
  node?: PipelineNode;
  onInspectionModeChange: (mode: InspectionMode) => void;
  onProbeSelect: (artifactId: string) => void;
  onSiteChange: (siteName: string) => void;
  probeLayerRefs: ProbeLayerRef[];
  selectedProbeArtifactId: string;
  selectedSiteName: string;
}) {
  const [rawQuery, setRawQuery] = useState("");
  if (!node) {
    return null;
  }
  const modeCounts = inspectionModes.map((mode) => ({
    mode,
    count: mode === "advanced"
      ? node.rawChoices.length
      : node.choices.filter((choice) => choice.mode === mode).length,
  })).filter(({ count }) => count > 0);
  const modeChoices =
    inspectionMode === "advanced"
      ? node.rawChoices
      : node.choices.filter((choice) => choice.mode === inspectionMode);
  const selectedChoice =
    modeChoices.find((choice) => choice.site.name === selectedSiteName) ?? modeChoices[0];
  const rawQueryText = rawQuery.trim().toLowerCase();
  const rawChoices = rawQueryText
    ? node.rawChoices.filter((choice) =>
        [choice.label, choice.site.name, choice.site.role, choice.site.tensor_type, choice.site.axes?.join(" ")]
          .filter(Boolean)
          .join(" ")
          .toLowerCase()
          .includes(rawQueryText),
      )
    : node.rawChoices;
  const chooseMode = (mode: InspectionMode) => {
    onInspectionModeChange(mode);
    const next = preferredSiteWithinNode(mode === "advanced" ? node.allSites : node.sites, mode);
    if (next) {
      onSiteChange(next.name);
    }
  };
  const nodeProbeRefs = probeRefsForNode(node, probeLayerRefs);

  return (
    <div className={`pipeline-site-detail ${node.family}`}>
      {node.rawChoices.length ? (
        <>
          <div className="pipeline-capture-heading">
            <span>Inspect</span>
            <small>{node.label}</small>
          </div>
          {modeCounts.length > 1 ? (
            <div className="inspection-mode-tabs" aria-label={`Inspection modes for ${node.label}`}>
              {modeCounts.map(({ mode, count }) => (
                <button
                  className={inspectionMode === mode ? "active" : ""}
                  key={mode}
                  title={inspectionModeHint(mode)}
                  type="button"
                  onClick={() => chooseMode(mode)}
                >
                  <span>{inspectionModeLabel(mode)}</span>
                  <small>{count}</small>
                </button>
              ))}
            </div>
          ) : null}
          {inspectionMode === "advanced" ? (
            <AdvancedRawCaptures
              choices={rawChoices}
              onQueryChange={setRawQuery}
              onChoiceSelect={(choice) => {
                onInspectionModeChange(choice.mode);
                onSiteChange(choice.site.name);
              }}
              query={rawQuery}
              selectedSiteName={selectedSiteName}
            />
          ) : modeChoices.length > 1 ? (
            <>
              <label className="capture-point-select">
                <span>{inspectionModeLabel(inspectionMode)}</span>
                <select
                  value={selectedChoice?.site.name ?? ""}
                  onChange={(event) => onSiteChange(event.target.value)}
                >
                  {modeChoices.map((choice) => (
                    <option key={choice.id} value={choice.site.name}>
                      {choice.label}
                    </option>
                  ))}
                </select>
              </label>
              {selectedChoice ? <CaptureDescription choice={selectedChoice} node={node} /> : null}
            </>
          ) : modeChoices.length === 1 && selectedChoice ? (
            <CaptureDescription choice={selectedChoice} node={node} />
          ) : (
            <div className="pipeline-site-empty">
              {inspectionModeEmptyMessage(inspectionMode, node)}
            </div>
          )}
        </>
      ) : (
        <div className="pipeline-site-empty">No captured inspectable site for this node in the current profile.</div>
      )}
      {nodeProbeRefs.length ? (
        <NodeProbeChips
          refs={nodeProbeRefs}
          selectedProbeArtifactId={selectedProbeArtifactId}
          onProbeSelect={onProbeSelect}
        />
      ) : null}
    </div>
  );
}

function NodeProbeChips({
  onProbeSelect,
  refs,
  selectedProbeArtifactId,
}: {
  onProbeSelect: (artifactId: string) => void;
  refs: ProbeLayerRef[];
  selectedProbeArtifactId: string;
}) {
  return (
    <div className="node-probe-chips">
      <div className="section-title">
        <span>Probes at this node</span>
        <small>{refs.length}</small>
      </div>
      <div className="node-probe-chip-row">
        {refs.map((ref) => (
          <button
            className={[
              probeRefTone(ref),
              ref.artifactId === selectedProbeArtifactId ? "active" : "",
              ref.selected || ref.default ? "current" : "",
            ].filter(Boolean).join(" ")}
            key={ref.artifactId}
            title={probeLayerTitle(ref)}
            type="button"
            onClick={() => onProbeSelect(ref.artifactId)}
          >
            <strong>{ref.target || ref.name}</strong>
            <span>
              {[
                ref.predicted === undefined ? "" : `Prediction ${formatProbeValue(ref.predicted)}`,
                ref.confidence === null || ref.confidence === undefined ? "" : `confidence ${formatMaybeNumber(ref.confidence)}`,
              ].filter(Boolean).join(" · ")}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}

function AdvancedRawCaptures({
  choices,
  onChoiceSelect,
  onQueryChange,
  query,
  selectedSiteName,
}: {
  choices: PipelineSiteChoice[];
  onChoiceSelect: (choice: PipelineSiteChoice) => void;
  onQueryChange: (query: string) => void;
  query: string;
  selectedSiteName: string;
}) {
  return (
    <div className="advanced-capture-drawer">
      <input
        aria-label="Search raw captures"
        placeholder="Search raw captures"
        type="search"
        value={query}
        onChange={(event) => onQueryChange(event.target.value)}
      />
      <div className="advanced-capture-list">
        {choices.length ? (
          choices.map((choice) => (
            <button
              className={choice.site.name === selectedSiteName ? "active" : ""}
              key={choice.id}
              title={choice.site.name}
              type="button"
              onClick={() => onChoiceSelect(choice)}
            >
              <strong>{choice.label}</strong>
              <span>{choice.site.name}</span>
              <small>{inspectionModeLabel(choice.mode)}</small>
            </button>
          ))
        ) : (
          <div className="pipeline-site-empty">No raw captures match this search.</div>
        )}
      </div>
    </div>
  );
}

function CaptureDescription({ choice, node }: { choice: PipelineSiteChoice; node: PipelineNode }) {
  const site = choice.site;
  const description = captureDescription(site, node);
  return (
    <div className="pipeline-capture-description">
      <div>
        <span>{choice.label}</span>
        <strong>{description}</strong>
      </div>
    </div>
  );
}

function PipelineLegend({ probeCount }: { probeCount: number }) {
  return (
    <div className="pipeline-legend" aria-label="Pipeline legend">
      <span><i className="legend-swatch vlm" />Vision path</span>
      <span><i className="legend-swatch expert" />Action denoiser</span>
      <span><i className="legend-swatch action" />Head / state</span>
      <span><i className="legend-swatch missing" />Uncaptured</span>
      <span><i className="legend-swatch active" />Current layer</span>
      {probeCount ? <span><i className="legend-probe source" />Probe trained here</span> : null}
      {probeCount ? <span><i className="legend-probe selected" />Current probe input</span> : null}
    </div>
  );
}

function inspectionModeHint(mode: InspectionMode): string {
  const hints: Record<InspectionMode, string> = {
    advanced: "Debug-only: show every captured tensor for this node.",
    attention: "Inspect attention heads and query tokens, then project their key mass onto the episode view.",
    computation: "Inspect MLP, normalization, and conditioning captures inside the layer.",
    features: "Inspect hidden-state features/channels and their image or token overlays.",
    saved_state: "Inspect cached keys, values, masks, positions, and RoPE state.",
  };
  return hints[mode];
}

function PipelineMapModal({
  architecture,
  feature,
  inspectionMode,
  overlayControls,
  onClose,
  onNodeSelect,
  probeLayerRefs,
  selectedProbeArtifactId,
  selectedSiteName,
  stages,
}: {
  architecture?: ArchitectureMetadata;
  feature: number;
  inspectionMode: InspectionMode;
  overlayControls?: ReactNode;
  onClose: () => void;
  onNodeSelect: (node: PipelineNode) => void;
  probeLayerRefs: ProbeLayerRef[];
  selectedProbeArtifactId: string;
  selectedSiteName: string;
  stages: PipelineStage[];
}) {
  useEffect(() => {
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [onClose]);

  return (
    <div className="pipeline-map-modal-backdrop" role="presentation" onMouseDown={onClose}>
      <div
        aria-label="Expanded PI0.5 model pipeline"
        aria-modal="true"
        className="pipeline-map-modal"
        role="dialog"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header>
          <div className="pipeline-map-modal-title">
            <Layers3 size={17} />
            <strong>PI0.5 Pipeline</strong>
          </div>
          <PipelineLegend probeCount={probeLayerRefs.length} />
          <button aria-label="Close expanded model pipeline" type="button" onClick={onClose}>
            <X size={17} />
          </button>
        </header>
        <PipelineMap2D
          architecture={architecture}
          className="large"
          feature={feature}
          inspectionMode={inspectionMode}
          onNodeSelect={onNodeSelect}
          probeLayerRefs={probeLayerRefs}
          selectedProbeArtifactId={selectedProbeArtifactId}
          selectedSiteName={selectedSiteName}
          stages={stages}
        />
        {overlayControls ? <div className="pipeline-channel-row modal">{overlayControls}</div> : null}
      </div>
    </div>
  );
}

function PipelineMap2D({
  architecture,
  className = "",
  cornerAction,
  feature,
  inspectionMode,
  onNodeSelect,
  probeLayerRefs,
  selectedProbeArtifactId,
  selectedSiteName,
  stages,
}: {
  architecture?: ArchitectureMetadata;
  className?: string;
  cornerAction?: ReactNode;
  feature: number;
  inspectionMode: InspectionMode;
  onNodeSelect: (node: PipelineNode) => void;
  probeLayerRefs: ProbeLayerRef[];
  selectedProbeArtifactId: string;
  selectedSiteName: string;
  stages: PipelineStage[];
}) {
  const layout = useMemo(() => pipelineDiagramLayout(stages, architecture), [architecture, stages]);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const isLargeMap = className.includes("large");
  const maxFitScale = isLargeMap ? 1.35 : 1;
  const minFitScale = isLargeMap ? 0.68 : 0.48;
  const [view, setView] = useState({ scale: isLargeMap ? 1 : 0.58, x: 0, y: 0 });
  const allSites = stages.flatMap((stage) => stage.nodes).flatMap((node) => node.allSites);
  const selectedSite = allSites.find((site) => site.name === selectedSiteName);
  const selectedNode = layout.nodes.find((entry) =>
    entry.node.allSites.some((site) => site.name === selectedSiteName),
  );
  const featureCount = channelCountForSite(selectedSite);
  const showFeatureSlice = Boolean(selectedNode && selectedSite && featureCount > 1 && isFeatureActivationSite(selectedSite));
  const featureFraction = Math.max(0, Math.min(1, feature / Math.max(1, featureCount - 1)));

  useEffect(() => {
    const target = viewportRef.current;
    if (!target) {
      return;
    }
    const observer = new ResizeObserver(([entry]) => {
      const padding = isLargeMap ? 56 : 24;
      const widthScale = (entry.contentRect.width - padding) / layout.width;
      const heightScale = (entry.contentRect.height - padding) / layout.height;
      const nextScale = Math.max(minFitScale, Math.min(maxFitScale, widthScale, heightScale));
      const nextView = {
        scale: nextScale,
        x: Math.max(12, (entry.contentRect.width - layout.width * nextScale) / 2),
        y: isLargeMap ? 24 : Math.max(12, (entry.contentRect.height - layout.height * nextScale) / 2),
      };
      setView(nextView);
    });
    observer.observe(target);
    return () => observer.disconnect();
  }, [isLargeMap, layout.height, layout.width, maxFitScale, minFitScale]);

  return (
    <div className={["pipeline-map2d", className].filter(Boolean).join(" ")}>
      {cornerAction ? <div className="pipeline-map2d-action">{cornerAction}</div> : null}
      <div className="pipeline-map2d-viewport" ref={viewportRef}>
        <div
          className="pipeline-map2d-canvas"
          style={{
            height: layout.height,
            transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})`,
            width: layout.width,
          }}
        >
          <svg
            aria-hidden="true"
            className="pipeline-map2d-svg"
            viewBox={`0 0 ${layout.width} ${layout.height}`}
          >
            <defs>
              <marker id="pipeline-forward-arrow" markerHeight="6" markerWidth="7" orient="auto" refX="6" refY="3">
                <path d="M0,0 L6,3 L0,6 Z" />
              </marker>
              <marker id="pipeline-conditioning-arrow" markerHeight="6" markerWidth="7" orient="auto" refX="6" refY="3">
                <path d="M0,0 L6,3 L0,6 Z" />
              </marker>
              <marker id="pipeline-loop-arrow" markerHeight="6" markerWidth="7" orient="auto" refX="6" refY="3">
                <path d="M0,0 L6,3 L0,6 Z" />
              </marker>
            </defs>
            {layout.bands.map((band) => (
              <g className={`pipeline-map2d-band ${band.className}`} key={band.id}>
                <rect height={band.height} width={band.width} x={band.x} y={band.y} />
                <text x={band.x + 10} y={band.y + 17}>{band.label}</text>
              </g>
            ))}
            {layout.arrows.map((arrow) => (
              <g className={`pipeline-map2d-arrow ${arrow.className}`} key={arrow.id}>
                <path d={arrow.path} />
                {arrow.label ? (
                  <text textAnchor={arrow.labelAnchor ?? "start"} x={arrow.labelX} y={arrow.labelY}>
                    {arrow.label}
                  </text>
                ) : null}
              </g>
            ))}
            {layout.ports.map((port) => (
              <g className={`pipeline-map2d-port ${port.className}`} key={port.id}>
                <circle cx={port.x} cy={port.y} r={port.radius ?? 4} />
                {port.label ? (
                  <text
                    textAnchor={port.textAnchor ?? "start"}
                    x={port.textAnchor === "middle" ? port.x : port.x + 9}
                    y={port.y + 4}
                  >
                    {port.label}
                  </text>
                ) : null}
              </g>
            ))}
          </svg>
          {layout.nodes.map((entry) => {
            const active = entry.node.allSites.some((site) => site.name === selectedSiteName);
            const preferred = preferredSiteWithinNode(
              inspectionMode === "advanced" ? entry.node.allSites : entry.node.sites,
              inspectionMode,
            );
            const nodeProbeRefs = probeRefsForNode(entry.node, probeLayerRefs);
            const nodeHasActiveProbe = Boolean(
              selectedProbeArtifactId &&
                nodeProbeRefs.some((ref) => ref.artifactId === selectedProbeArtifactId),
            );
            const nodeHasSelectedProbe = Boolean(
              selectedProbeArtifactId &&
                nodeProbeRefs.some(
                  (ref) =>
                    ref.artifactId === selectedProbeArtifactId &&
                    (ref.selected || ref.default),
                ),
            );
            const markerRefs =
              selectedProbeArtifactId && nodeHasActiveProbe
                ? nodeProbeRefs.filter((ref) => ref.artifactId === selectedProbeArtifactId)
                : nodeProbeRefs;
            return (
              <button
                className={[
                  "pipeline-map2d-node",
                  entry.node.family,
                  entry.node.captured ? "captured" : "missing",
                  active ? "active" : "",
                  nodeProbeRefs.length ? "probe-mapped" : "",
                  nodeHasSelectedProbe ? "probe-selected" : "",
                  entry.width < 42 ? "compact" : "",
                ].filter(Boolean).join(" ")}
                disabled={!entry.node.captured}
                key={entry.node.id}
                style={{
                  height: entry.height,
                  left: entry.x,
                  top: entry.y,
                  width: entry.width,
                }}
                title={entry.node.allSites.map((site) => site.name).join("\n") || entry.node.sublabel}
                type="button"
                onClick={() => {
                  if (preferred) {
                    onNodeSelect(entry.node);
                  }
                }}
              >
                <strong>{entry.node.label}</strong>
                {entry.width >= 74 ? <small>{entry.node.sublabel}</small> : null}
                {markerRefs.length ? (
                  <span
                    className={[
                      "pipeline-probe-marker",
                      nodeHasSelectedProbe ? "selected" : "",
                      probeToneForRefs(markerRefs),
                    ].filter(Boolean).join(" ")}
                    title={markerRefs.map(probeLayerTitle).join("\n")}
                  >
                    {nodeHasSelectedProbe ? "P" : markerRefs.length}
                  </span>
                ) : null}
                {showFeatureSlice && active ? (
                  <i
                    aria-hidden="true"
                    className={
                      entry.width >= 64
                        ? "pipeline-map2d-feature-slice"
                        : "pipeline-map2d-channel-marker"
                    }
                    style={{ left: `${6 + featureFraction * 88}%` }}
                    title={`Channel ${feature} / ${Math.max(0, featureCount - 1)}`}
                  />
                ) : null}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
