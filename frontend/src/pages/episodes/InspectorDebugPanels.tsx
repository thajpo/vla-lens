import { memo } from "react";
import { BarChart3 } from "lucide-react";

import type { PolicyCall } from "../../types/dataset";
import type { InspectorContext } from "./shared";
import { heatColor } from "./formatters";

function InspectorDebugSectionsImpl({
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

export const InspectorDebugSections = memo(InspectorDebugSectionsImpl, (previous, next) => {
  if (
    previous.artifacts !== next.artifacts ||
    previous.inspectorContext !== next.inspectorContext
  ) {
    return false;
  }
  if (next.inspectorContext !== "expert") {
    return true;
  }
  return (
    previous.calls === next.calls &&
    previous.generationValues === next.generationValues &&
    previous.onTimestepChange === next.onTimestepChange &&
    previous.timestep === next.timestep
  );
});

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
