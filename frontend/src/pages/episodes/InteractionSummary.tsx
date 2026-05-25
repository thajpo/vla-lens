import type { ReactNode } from "react";
import { Hand, Move3D, Target } from "lucide-react";
import type { EpisodeInteractionsResponse } from "../../types/dataset";

export function InteractionSummaryPanel({
  interactions,
  isError,
  isLoading,
  onTimestepChange,
}: {
  interactions?: EpisodeInteractionsResponse;
  isError: boolean;
  isLoading: boolean;
  onTimestepChange: (timestep: number) => void;
}) {
  const episode = interactions?.episode;
  const qualityFlags = Object.entries(interactions?.quality ?? {}).filter(([, enabled]) => enabled);
  return (
    <section className="episode-tool-panel interaction-summary">
      <header>
        <div className="icon-label">
          <Target size={16} />
          <strong>Interaction Labels</strong>
        </div>
        {interactions?.artifact_id ? <span>{interactions.artifact_id}</span> : null}
      </header>
      {isLoading ? <div className="empty-state">Loading interaction metrics.</div> : null}
      {!isLoading && isError ? (
        <div className="empty-state">Interaction metrics could not be loaded.</div>
      ) : null}
      {!isLoading && interactions && !interactions.available ? (
        <div className="empty-state">{interactions.reason ?? "No interaction metrics available."}</div>
      ) : null}
      {!isLoading && !isError && interactions?.available && episode ? (
        <>
          <div className="interaction-summary-grid">
            <InteractionField label="Target" value={episode.primary_target_object} />
            <InteractionField label="Scene" value={compactSceneLabel(episode)} />
            <InteractionField
              label="Parse"
              value={episode.target_parse_status || "unknown"}
              tone={episode.target_parse_status === "failed" ? "warn" : "ok"}
            />
          </div>
          <div className="interaction-event-row">
            <InteractionEventChip
              icon={<Move3D size={14} />}
              isTarget={episode.first_moved_is_target}
              label="Moved"
              objectName={episode.first_moved_object}
              timestep={episode.first_moved_timestep}
              onTimestepChange={onTimestepChange}
            />
            <InteractionEventChip
              icon={<Target size={14} />}
              isTarget={episode.first_lifted_is_target}
              label="Lifted"
              objectName={episode.first_lifted_object}
              timestep={episode.first_lifted_timestep}
              onTimestepChange={onTimestepChange}
            />
            <InteractionEventChip
              icon={<Hand size={14} />}
              label="Contact"
              objectName={episode.first_contacted_object}
              timestep={episode.first_contact_timestep}
              onTimestepChange={onTimestepChange}
            />
          </div>
          <div className="interaction-quality-row">
            {qualityFlags.length ? (
              qualityFlags.map(([key]) => (
                <span className="interaction-quality-badge" key={key}>
                  {formatInteractionKey(key)}
                </span>
              ))
            ) : (
              <span className="interaction-quality-badge ok">Quality ok</span>
            )}
          </div>
          <details className="interaction-object-details">
            <summary>
              <span>Objects</span>
              <small>{interactions.objects.length}</small>
            </summary>
            <div className="interaction-object-table">
              {interactions.objects.map((object) => (
                <div className="interaction-object-row" key={object.object_name}>
                  <strong>{object.object_name}</strong>
                  <span>{object.is_target_object ? "target" : object.object_kind || "object"}</span>
                  <span>{interactionObjectEvents(object)}</span>
                  <span>{formatDistance(object.max_displacement)}</span>
                </div>
              ))}
            </div>
          </details>
        </>
      ) : null}
    </section>
  );
}

function InteractionField({
  label,
  value,
  tone,
}: {
  label: string;
  value?: string;
  tone?: "ok" | "warn";
}) {
  return (
    <div className={["interaction-field", tone ? `tone-${tone}` : ""].filter(Boolean).join(" ")}>
      <span>{label}</span>
      <strong title={value || "unknown"}>{value || "unknown"}</strong>
    </div>
  );
}

function InteractionEventChip({
  icon,
  isTarget,
  label,
  objectName,
  timestep,
  onTimestepChange,
}: {
  icon: ReactNode;
  isTarget?: boolean;
  label: string;
  objectName?: string;
  timestep: number | null;
  onTimestepChange: (timestep: number) => void;
}) {
  const disabled = timestep === null || timestep === undefined;
  return (
    <button
      className={["interaction-event-chip", isTarget ? "target" : ""].filter(Boolean).join(" ")}
      disabled={disabled}
      title={disabled ? label : `${label} at t=${timestep}`}
      type="button"
      onClick={() => {
        if (!disabled) {
          onTimestepChange(timestep);
        }
      }}
    >
      {icon}
      <span>{label}</span>
      <strong>{objectName || "none"}</strong>
      <em>{disabled ? "-" : `t=${timestep}`}</em>
    </button>
  );
}

function compactSceneLabel(episode: EpisodeInteractionsResponse["episode"]): string {
  if (!episode) {
    return "";
  }
  return [episode.scene_family, episode.task_verb].filter(Boolean).join(" / ");
}

function formatInteractionKey(value: string): string {
  return value.replaceAll("_", " ");
}

function interactionObjectEvents(object: EpisodeInteractionsResponse["objects"][number]): string {
  const events = [
    object.moved ? `m${formatEventTimestep(object.movement_onset_timestep)}` : "",
    object.lifted ? `l${formatEventTimestep(object.lift_onset_timestep)}` : "",
    object.contacted ? `c${formatEventTimestep(object.contact_onset_timestep)}` : "",
  ].filter(Boolean);
  return events.length ? events.join(" ") : "no event";
}

function formatEventTimestep(timestep: number | null): string {
  return timestep === null ? "" : `:${timestep}`;
}

function formatDistance(value: number | null): string {
  return value === null ? "-" : value.toFixed(3);
}
