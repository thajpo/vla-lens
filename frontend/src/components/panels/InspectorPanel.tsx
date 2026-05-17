import type { ResolvedSelection } from "../../types/workbench";
import { PanelCard } from "../layout/PanelCard";

type InspectorPanelProps = {
  resolution?: ResolvedSelection | null;
};

export function InspectorPanel({ resolution }: InspectorPanelProps) {
  const axes = resolution?.selection.axis_values ?? {};
  const cell = resolution?.target_object_cell ?? resolution?.action_stabilization_cell ?? {};
  const provenance = resolution?.provenance ?? {};
  return (
    <PanelCard title="Inspector">
      {!resolution ? <div className="empty-state">Select a visual cell.</div> : null}
      {resolution ? (
        <dl className="inspector-grid">
          <Item label="Layer" value={first(axes.layer)} />
          <Item label="Timestep" value={first(axes.timestep)} />
          <Item label="Token" value={first(axes.token_kind)} />
          <Item label="Metric" value={first(axes.metric)} />
          <Item label="Score" value={cell.score} />
          <Item label="Baseline" value={cell.baseline_score} />
          <Item label="Delta" value={cell.delta} />
          <Item label="Arrays" value={resolution.lens_arrays.length} />
          <Item label="Panels" value={resolution.suggested_panels.length} />
          <Item label="Provenance" value={provenance.artifact_type ?? provenance.analysis_run} />
        </dl>
      ) : null}
    </PanelCard>
  );
}

function Item({ label, value }: { label: string; value: unknown }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{formatValue(value)}</dd>
    </div>
  );
}

function first(value: unknown): unknown {
  return Array.isArray(value) ? value[0] : value;
}

function formatValue(value: unknown) {
  if (typeof value === "number") {
    return Number.isFinite(value) ? value.toFixed(3) : "-";
  }
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  return String(value);
}
