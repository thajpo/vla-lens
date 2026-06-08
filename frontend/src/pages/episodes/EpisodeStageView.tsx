import type { ComponentProps } from "react";

import { CameraGrid, FramePlaybackControls } from "./CameraTimeline";
import { EpisodeProbePanel } from "./EpisodeProbePanel";
import { PromptAttentionStrip } from "./InspectorPanels";
import { InteractionSummaryPanel } from "./InteractionSummary";
import { MetricPlotPanel } from "./MetricPlots";
import type { EpisodePlotTab } from "./shared";

type EpisodeStageViewProps = {
  cameraGrid: ComponentProps<typeof CameraGrid>;
  episodePlotTab: EpisodePlotTab;
  episodeProbePanel: ComponentProps<typeof EpisodeProbePanel>;
  frameControls: ComponentProps<typeof FramePlaybackControls>;
  interactionSummary: ComponentProps<typeof InteractionSummaryPanel>;
  metricPlot: ComponentProps<typeof MetricPlotPanel>;
  onEpisodePlotTabChange: (tab: EpisodePlotTab) => void;
  promptAttentionStrip: ComponentProps<typeof PromptAttentionStrip>;
  showProbePanel?: boolean;
};

export function EpisodeStageView({
  cameraGrid,
  episodePlotTab,
  episodeProbePanel,
  frameControls,
  interactionSummary,
  metricPlot,
  onEpisodePlotTabChange,
  promptAttentionStrip,
  showProbePanel = true,
}: EpisodeStageViewProps) {
  const activePlotTab = showProbePanel ? episodePlotTab : "episode";
  return (
    <section className="stage">
      <div className="stage-body stage-view">
        <div className="episode-workspace">
          <div className="workspace-main">
            <div className="viewer-layout">
              <div className="viewer-media">
                <PromptAttentionStrip {...promptAttentionStrip} />
                <CameraGrid {...cameraGrid} />
                <FramePlaybackControls {...frameControls} />
              </div>
              <aside className="viewer-plot-panel">
                {showProbePanel ? (
                  <div className="episode-plot-tabs" aria-label="Episode plot tabs">
                    <button
                      className={activePlotTab === "probes" ? "active" : ""}
                      type="button"
                      onClick={() => onEpisodePlotTabChange("probes")}
                    >
                      Probes
                    </button>
                    <button
                      className={activePlotTab === "episode" ? "active" : ""}
                      type="button"
                      onClick={() => onEpisodePlotTabChange("episode")}
                    >
                      Episode
                    </button>
                  </div>
                ) : null}
                {activePlotTab === "episode" ? (
                  <>
                    <MetricPlotPanel {...metricPlot} />
                    <InteractionSummaryPanel {...interactionSummary} />
                  </>
                ) : (
                  <EpisodeProbePanel {...episodeProbePanel} />
                )}
              </aside>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
