import { useState } from "react";
import { ChevronLeft, ChevronRight, Star } from "lucide-react";
import type {
  CounterfactualPair,
  DatasetEpisode,
  EpisodeAnnotation,
} from "../../types/dataset";
import { episodeMetadataItems, outcomeClass } from "./episodeData";

export function EpisodeNavigationBar({
  annotation,
  cohortReturnHref,
  counterfactualPair,
  episode,
  episodeCount,
  episodeIndex,
  hasNext,
  hasPrevious,
  isSavingAnnotation,
  onNext,
  onNavigateTrace,
  onPrevious,
  onSaveAnnotation,
}: {
  annotation?: EpisodeAnnotation;
  cohortReturnHref?: string;
  counterfactualPair?: CounterfactualPair;
  episode: DatasetEpisode;
  episodeCount: number;
  episodeIndex: number;
  hasNext: boolean;
  hasPrevious: boolean;
  isSavingAnnotation: boolean;
  onNext: () => void;
  onNavigateTrace: (traceId: string | undefined) => void;
  onPrevious: () => void;
  onSaveAnnotation: (annotation: Pick<EpisodeAnnotation, "trace_id" | "starred" | "notes">) => void;
}) {
  const metadataItems = episodeMetadataItems(episode);
  const outcome = String(episode.outcome || "unknown");
  const starred = Boolean(annotation?.starred);
  const savedNotes = annotation?.notes ?? "";
  const [draft, setDraft] = useState({
    notes: savedNotes,
    savedNotes,
    traceId: episode.trace_id,
  });
  const draftNotes =
    draft.traceId === episode.trace_id && draft.savedNotes === savedNotes
      ? draft.notes
      : savedNotes;
  const notesChanged = draftNotes !== savedNotes;
  const saveCurrentAnnotation = (next: { starred?: boolean; notes?: string }) => {
    onSaveAnnotation({
      trace_id: episode.trace_id,
      starred: next.starred ?? starred,
      notes: next.notes ?? draftNotes,
    });
  };
  return (
    <section className="episode-nav-bar" aria-label="Episode metadata and navigation">
      <div className="episode-nav-controls">
        <button
          aria-label="Previous episode"
          disabled={!hasPrevious}
          type="button"
          onClick={onPrevious}
        >
          <ChevronLeft size={16} />
        </button>
        <div className="episode-nav-position">
          <span>Episode</span>
          <strong>
            {episodeIndex >= 0 ? episodeIndex + 1 : "-"} / {episodeCount || "-"}
          </strong>
        </div>
        <button aria-label="Next episode" disabled={!hasNext} type="button" onClick={onNext}>
          <ChevronRight size={16} />
        </button>
      </div>

      <div className="episode-nav-main">
        <div className="episode-nav-meta">
          <span className={`outcome-pill ${outcomeClass(outcome)}`}>{outcome}</span>
          {metadataItems.length ? (
            <details className="episode-nav-details">
              <summary>Episode details</summary>
              <div>
                {metadataItems.map((item) => (
                  <span className="episode-meta-item" key={item.label}>
                    <b>{item.label}</b>
                    {item.value}
                  </span>
                ))}
              </div>
            </details>
          ) : null}
        </div>
        {cohortReturnHref ? (
          <a className="episode-cohort-link" href={cohortReturnHref}>
            Back to cohort
          </a>
        ) : null}
        {counterfactualPair ? (
          <CounterfactualPairStrip
            activeTraceId={episode.trace_id}
            pair={counterfactualPair}
            onNavigateTrace={onNavigateTrace}
          />
        ) : null}
      </div>
      <div className="episode-annotation-tools">
        <button
          aria-label={starred ? "Unstar episode" : "Star episode"}
          className={starred ? "episode-star-button active" : "episode-star-button"}
          type="button"
          onClick={() => saveCurrentAnnotation({ starred: !starred })}
        >
          <Star size={15} fill={starred ? "currentColor" : "none"} />
        </button>
        <textarea
          aria-label="Episode notes"
          placeholder="Notes"
          value={draftNotes}
          onBlur={() => {
            if (notesChanged) {
              saveCurrentAnnotation({ notes: draftNotes });
            }
          }}
          onChange={(event) =>
            setDraft({
              notes: event.target.value,
              savedNotes,
              traceId: episode.trace_id,
            })
          }
        />
        <button
          disabled={!notesChanged || isSavingAnnotation}
          type="button"
          onClick={() => saveCurrentAnnotation({ notes: draftNotes })}
        >
          {isSavingAnnotation ? "Saving" : "Save"}
        </button>
      </div>
    </section>
  );
}

function CounterfactualPairStrip({
  activeTraceId,
  onNavigateTrace,
  pair,
}: {
  activeTraceId: string;
  onNavigateTrace: (traceId: string | undefined) => void;
  pair: CounterfactualPair;
}) {
  const pairType = pair.type ? pair.type.replaceAll("_", " ") : "paired traces";
  return (
    <div className="counterfactual-pair-strip" aria-label="Counterfactual pair">
      <span className="counterfactual-pair-label">{pairType}</span>
      {pair.members.map((member) => {
        const role = member.role || "trace";
        const isActive = member.trace_id === activeTraceId;
        const target = member.target_object_id || member.counterfactual_target_object_id || "";
        return (
          <button
            className={isActive ? "active" : ""}
            disabled={isActive}
            key={member.trace_id}
            title={member.prompt || member.trace_id}
            type="button"
            onClick={() => onNavigateTrace(member.trace_id)}
          >
            <span>{role}</span>
            {target ? <small>{target}</small> : null}
          </button>
        );
      })}
    </div>
  );
}
