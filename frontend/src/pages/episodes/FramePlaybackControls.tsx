import { useEffect, useState } from "react";
import { Eye, EyeOff, Layers3, Pause, Play, RotateCcw } from "lucide-react";

import { episodeVideoUrl } from "../../api/dataset";
import type { PolicyCall } from "../../types/dataset";

export function FramePlaybackControls({
  cacheKey,
  fps,
  isPlaying,
  maxTimestep,
  policyCalls,
  onAttentionOverlayToggle,
  onFpsChange,
  onObjectOverlayToggle,
  onReset,
  onTimestepChange,
  onToggle,
  showAttentionOverlay,
  showObjectOverlay,
  timestep,
  traceId,
}: {
  cacheKey: string;
  fps: number;
  isPlaying: boolean;
  maxTimestep: number;
  policyCalls: PolicyCall[];
  onAttentionOverlayToggle: () => void;
  onFpsChange: (fps: number) => void;
  onObjectOverlayToggle: () => void;
  onReset: () => void;
  onTimestepChange: (timestep: number) => void;
  onToggle: () => void;
  showAttentionOverlay: boolean;
  showObjectOverlay: boolean;
  timestep: number;
  traceId: string;
}) {
  const activeCall = policyCalls.find(
    (call) => timestep >= call.segment_start && timestep <= call.segment_end,
  );
  return (
    <div className="frame-playback">
      <div className="playback-command-row">
        <button
          aria-label={isPlaying ? "Pause episode" : "Play episode"}
          className="playback-icon-button playback-primary"
          title={isPlaying ? "Pause" : "Play"}
          type="button"
          onClick={onToggle}
        >
          {isPlaying ? <Pause size={15} /> : <Play size={15} />}
        </button>
        <button
          aria-label="Reset episode playback"
          className="playback-icon-button frame-reset"
          title="Reset"
          type="button"
          onClick={onReset}
        >
          <RotateCcw size={15} />
        </button>
        <select
          aria-label="Playback speed"
          className="fps-select"
          value={fps}
          onChange={(event) => onFpsChange(Number(event.target.value))}
        >
          {[2, 5, 10, 15].map((option) => (
            <option key={option} value={option}>
              {option} fps
            </option>
          ))}
        </select>
        <div className="viewer-overlay-group" role="group" aria-label="Viewer overlays">
          <button
            aria-pressed={showObjectOverlay}
            className={`object-overlay-toggle${showObjectOverlay ? " active" : ""}`}
            title="Show object boxes and hover labels"
            type="button"
            onClick={onObjectOverlayToggle}
          >
            {showObjectOverlay ? <Eye size={15} /> : <EyeOff size={15} />}
            <span>Objects</span>
          </button>
          <button
            aria-pressed={showAttentionOverlay}
            className={`attention-overlay-toggle${showAttentionOverlay ? " active" : ""}`}
            title="Show the selected model activation or attention overlay"
            type="button"
            onClick={onAttentionOverlayToggle}
          >
            {showAttentionOverlay ? <Layers3 size={15} /> : <EyeOff size={15} />}
            <span>Model</span>
          </button>
        </div>
        <span className="playback-policy-readout">
          {activeCall
            ? `Policy call ${activeCall.index} / t=${activeCall.segment_start}-${activeCall.segment_end}`
            : `${policyCalls.length} policy calls`}
        </span>
        <a
          className="mp4-link"
          href={episodeVideoUrl(traceId, "all", cacheKey)}
          target="_blank"
          rel="noreferrer"
        >
          MP4
        </a>
      </div>
      <TimelineControl
        timestep={timestep}
        maxTimestep={maxTimestep}
        onChange={onTimestepChange}
        policyCalls={policyCalls}
      />
    </div>
  );
}

function TimelineControl({
  timestep,
  maxTimestep,
  onChange,
  policyCalls = [],
}: {
  timestep: number;
  maxTimestep: number;
  onChange: (timestep: number) => void;
  policyCalls?: PolicyCall[];
}) {
  const [draftTimestep, setDraftTimestep] = useState(timestep);
  const clampTimestep = (value: number) => Math.max(0, Math.min(maxTimestep, value));
  const commitImmediate = (value: number) => {
    const next = clampTimestep(value);
    setDraftTimestep(next);
    onChange(next);
  };

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => setDraftTimestep(timestep));
    return () => window.cancelAnimationFrame(frame);
  }, [timestep]);

  const boundedMax = Math.max(1, maxTimestep);
  return (
    <div className="timeline-control">
      <div className="timeline-main">
        <div className="timeline-readout">
          <span>Timestep {draftTimestep} / {maxTimestep}</span>
        </div>
        <div className="timeline-track-shell">
          <input
            aria-label="Episode timestep"
            max={maxTimestep}
            min={0}
            type="range"
            value={draftTimestep}
            onBlur={() => commitImmediate(draftTimestep)}
            onChange={(event) => setDraftTimestep(clampTimestep(Number(event.target.value)))}
            onKeyUp={() => commitImmediate(draftTimestep)}
            onPointerCancel={() => commitImmediate(draftTimestep)}
            onPointerUp={() => commitImmediate(draftTimestep)}
          />
          <div className="timeline-call-markers" aria-label="Policy call checkpoints">
            {policyCalls.map((call) => {
              const markerTimestep = call.env_timestep ?? call.segment_start;
              const left = Math.max(0, Math.min(100, (markerTimestep / boundedMax) * 100));
              const active =
                draftTimestep >= call.segment_start && draftTimestep <= call.segment_end;
              return (
                <button
                  aria-label={`Jump to policy call ${call.index}, timesteps ${call.segment_start} to ${call.segment_end}`}
                  className={active ? "timeline-call-marker active" : "timeline-call-marker"}
                  key={call.index}
                  style={{ left: `${left}%` }}
                  title={`Policy call ${call.index}: t=${call.segment_start}-${call.segment_end}`}
                  type="button"
                  onClick={() => commitImmediate(markerTimestep)}
                />
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
