import type { ResolvedSelection } from "../../types/workbench";
import { PanelCard } from "../layout/PanelCard";

type EpisodeViewerPanelProps = {
  resolution?: ResolvedSelection | null;
};

export function EpisodeViewerPanel({ resolution }: EpisodeViewerPanelProps) {
  const example = Object.values(resolution?.examples ?? {})
    .flat()
    .find((row) => row.trace_id || row.episode_id);
  const traceId = String(example?.trace_id ?? "");
  const timestep = Number(example?.timestep ?? resolution?.valid_references.timestep ?? 0);
  const src = traceId
    ? `/api/frame?trace_id=${encodeURIComponent(traceId)}&camera=main&timestep=${Math.max(0, timestep)}&source=trace`
    : "";
  return (
    <PanelCard title="Episode Viewer">
      {!src ? <div className="empty-state">No episode reference.</div> : null}
      {src ? <img className="episode-frame" alt={`${traceId} timestep ${timestep}`} src={src} /> : null}
    </PanelCard>
  );
}
