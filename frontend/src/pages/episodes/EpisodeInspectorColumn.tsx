import type { ComponentProps } from "react";
import { Layers3 } from "lucide-react";
import { ActivationSitePanel, InspectorDebugSections } from "./InspectorPanels";

type EpisodeInspectorColumnProps = {
  activationSitePanel: ComponentProps<typeof ActivationSitePanel>;
  debugSections: ComponentProps<typeof InspectorDebugSections>;
  hasModelSites: boolean;
  showDebugSections: boolean;
};

export function EpisodeInspectorColumn({
  activationSitePanel,
  debugSections,
  hasModelSites,
  showDebugSections,
}: EpisodeInspectorColumnProps) {
  return (
    <section className="inspector">
      <div className="head">
        <div className="inspector-title">
          <Layers3 size={17} />
          <span>Model Inspector</span>
        </div>
      </div>
      <div className="body">
        {hasModelSites ? (
          <ActivationSitePanel {...activationSitePanel} />
        ) : (
          <div className="empty-state">No model-site overlay is available for this dataset.</div>
        )}
        {showDebugSections ? <InspectorDebugSections {...debugSections} /> : null}
      </div>
    </section>
  );
}
