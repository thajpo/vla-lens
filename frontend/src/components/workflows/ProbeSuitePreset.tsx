import { Fragment, useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchArtifact, fetchArtifacts } from "../../api/dataset";
import { PanelCard } from "../layout/PanelCard";

type ProbeSuitePresetProps = {
  activeRunId: string;
  onRunChange: (runId: string) => void;
};

type ProbeResult = Record<string, unknown> & {
  model?: string;
  split_value?: string;
  score?: number;
  baseline_score?: number;
  layer?: number | null;
  sweep_layer?: number | null;
  policy_call_index?: number | null;
  sweep_policy_call_index?: number | null;
  feature?: string;
  metadata_baseline?: string;
  sweep_value?: number | string | null;
};

type ProbeCell = {
  x: string;
  y: string;
  value: number | null;
  result?: ProbeResult;
};

const PROBE_METRICS = [
  { id: "score", label: "Score" },
  { id: "baseline", label: "Baseline" },
  { id: "delta", label: "Delta" },
];

export function ProbeSuitePreset({ activeRunId, onRunChange }: ProbeSuitePresetProps) {
  const artifacts = useQuery({
    queryKey: ["artifacts"],
    queryFn: fetchArtifacts,
    staleTime: 30_000,
  });
  const probeArtifacts = useMemo(
    () => (artifacts.data?.artifacts ?? []).filter((item) => item.artifact_type === "probe_suite"),
    [artifacts.data?.artifacts],
  );
  const selectedArtifactId =
    probeArtifacts.some((item) => item.artifact_id === activeRunId)
      ? activeRunId
      : probeArtifacts[0]?.artifact_id || "";

  useEffect(() => {
    if (!activeRunId && selectedArtifactId) {
      onRunChange(selectedArtifactId);
    }
  }, [activeRunId, onRunChange, selectedArtifactId]);

  const detail = useQuery({
    queryKey: ["artifact", selectedArtifactId],
    queryFn: () => fetchArtifact(selectedArtifactId),
    enabled: Boolean(selectedArtifactId),
  });
  const artifact =
    detail.data?.artifact ?? probeArtifacts.find((item) => item.artifact_id === selectedArtifactId);
  const display = objectValue(artifact?.display);
  const metrics = objectValue(artifact?.metrics);
  const results = arrayValue(display.results) as ProbeResult[];
  const models = unique(results.map((result) => stringValue(result.model)).filter(Boolean));
  const splits = unique(results.map((result) => stringValue(result.split_value)).filter(Boolean));
  const [metric, setMetric] = useState("delta");
  const [model, setModel] = useState("");
  const [split, setSplit] = useState("");
  const activeModel = model || stringValue(metrics.best_model) || models[0] || "";
  const activeSplit = split || stringValue(metrics.best_eval_split) || splits[0] || "";
  const filtered = results.filter(
    (result) =>
      (!activeModel || stringValue(result.model) === activeModel) &&
      (!activeSplit || stringValue(result.split_value) === activeSplit),
  );
  const cells = buildProbeCells(filtered, metric);
  const bestDetails = objectValue(display.best_result_details);
  const confusion = arrayValue(objectValue(bestDetails.details).confusion_matrix);
  const baselines = arrayValue(objectValue(bestDetails.details).metadata_baselines);
  const examples = arrayValue(objectValue(bestDetails.details).test_episode_summary);

  if (artifacts.isLoading) {
    return <div className="app-message">Loading probe artifacts...</div>;
  }
  if (!probeArtifacts.length) {
    return (
      <div className="workflow-empty">
        <h1>Probe Suites</h1>
        <p>No probe-suite artifact is registered yet.</p>
      </div>
    );
  }

  return (
    <div className="probe-suite-page">
      <header className="workflow-toolbar">
        <div>
          <h1>Probe Suites</h1>
          <p>{artifact?.name ?? selectedArtifactId}</p>
        </div>
        <label className="inline-select">
          Run
          <select value={selectedArtifactId} onChange={(event) => onRunChange(event.target.value)}>
            {probeArtifacts.map((item) => (
              <option key={item.artifact_id} value={item.artifact_id}>
                {item.name ?? item.artifact_id}
              </option>
            ))}
          </select>
        </label>
      </header>

      <section className="summary-grid">
        <Metric label="Target" value={stringValue(metrics.target) || stringValue(display.target)} />
        <Metric label="Best Score" value={formatNumber(numberValue(metrics.best_score))} />
        <Metric label="Best Delta" value={formatSigned(numberValue(metrics.best_delta))} />
        <Metric label="Null p" value={formatNumber(numberValue(metrics.null_p_value))} />
        <Metric label="Rows" value={formatCount(numberValue(display.row_count))} />
        <Metric label="Features" value={formatCount(numberValue(display.feature_dim))} />
      </section>

      <div className="probe-suite-grid">
        <PanelCard title="Probe Sweep">
          <div className="probe-controls">
            <label>
              Model
              <select value={activeModel} onChange={(event) => setModel(event.target.value)}>
                {models.map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Split
              <select value={activeSplit} onChange={(event) => setSplit(event.target.value)}>
                {splits.map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Metric
              <select value={metric} onChange={(event) => setMetric(event.target.value)}>
                {PROBE_METRICS.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.label}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <ProbeHeatmap cells={cells} />
        </PanelCard>

        <PanelCard title="Best Result">
          <dl className="probe-detail-list">
            <div>
              <dt>Feature</dt>
              <dd>{stringValue(bestDetails.feature) || stringValue(metrics.best_feature) || "-"}</dd>
            </div>
            <div>
              <dt>Model</dt>
              <dd>{stringValue(bestDetails.model) || stringValue(metrics.best_model) || "-"}</dd>
            </div>
            <div>
              <dt>Selection split</dt>
              <dd>{stringValue(bestDetails.eval_split) || stringValue(metrics.best_eval_split) || "-"}</dd>
            </div>
            <div>
              <dt>Metadata baseline</dt>
              <dd>{formatNumber(numberValue(metrics.best_baseline))}</dd>
            </div>
          </dl>
          <CompactRows
            columns={["actual", "predicted", "count"]}
            rows={confusion}
            empty="No confusion matrix saved."
          />
        </PanelCard>
      </div>

      <div className="probe-suite-grid lower">
        <PanelCard title="Metadata Baselines">
          <CompactRows columns={["baseline", "score"]} rows={baselines} empty="No baselines saved." />
        </PanelCard>
        <PanelCard title="Hard Examples">
          <CompactRows
            columns={["episode_id", "actual", "predicted", "accuracy", "mean_confidence"]}
            rows={examples.filter((item) => numberValue(objectValue(item).accuracy) === 0).slice(0, 12)}
            empty="No incorrect examples in the selected best-result summary."
          />
        </PanelCard>
      </div>
    </div>
  );
}

function ProbeHeatmap({ cells }: { cells: ProbeCell[] }) {
  const xValues = unique(cells.map((cell) => cell.x));
  const yValues = unique(cells.map((cell) => cell.y));
  const range = valueRange(cells.map((cell) => cell.value));
  if (!cells.length) {
    return <div className="empty-state">No sweep cells match this selection.</div>;
  }
  return (
    <div className="heatmap-wrap">
      <div
        className="heatmap-grid probe-heatmap"
        style={{ gridTemplateColumns: `90px repeat(${xValues.length}, minmax(48px, 1fr))` }}
      >
        <div className="axis-corner">layer</div>
        {xValues.map((x) => (
          <div className="axis-label" key={`x-${x}`}>
            call {x}
          </div>
        ))}
        {yValues.map((y) => (
          <Fragment key={`row-${y}`}>
            <div className="axis-label y-label" key={`y-${y}`}>
              {y}
            </div>
            {xValues.map((x) => {
              const cell = cells.find((item) => item.x === x && item.y === y);
              return (
                <div
                  className="heatmap-cell readonly"
                  key={`${y}-${x}`}
                  style={{ background: colorFor(cell?.value ?? null, range) }}
                  title={cell?.result?.feature ? String(cell.result.feature) : undefined}
                >
                  {cell?.value == null ? "" : formatNumber(cell.value)}
                </div>
              );
            })}
          </Fragment>
        ))}
      </div>
    </div>
  );
}

function CompactRows({
  columns,
  rows,
  empty,
}: {
  columns: string[];
  rows: unknown[];
  empty: string;
}) {
  if (!rows.length) {
    return <div className="empty-state compact">{empty}</div>;
  }
  return (
    <div className="table-scroll">
      <table className="compact-table">
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{column.replaceAll("_", " ")}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => {
            const item = objectValue(row);
            return (
              <tr key={index}>
                {columns.map((column) => (
                  <td key={column}>{formatCell(item[column])}</td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="summary-card">
      <span>{label}</span>
      <strong>{value || "-"}</strong>
    </div>
  );
}

function buildProbeCells(results: ProbeResult[], metric: string): ProbeCell[] {
  return results.map((result) => {
    const layer = result.sweep_layer ?? result.layer ?? "all";
    const policyCall = result.sweep_policy_call_index ?? result.policy_call_index ?? result.sweep_value ?? "all";
    return {
      x: String(policyCall),
      y: String(layer),
      value: metricValue(result, metric),
      result,
    };
  });
}

function metricValue(result: ProbeResult, metric: string): number | null {
  const score = numberValue(result.score);
  const baseline = numberValue(result.baseline_score);
  if (metric === "baseline") {
    return baseline;
  }
  if (metric === "delta") {
    return score === null || baseline === null ? null : score - baseline;
  }
  return score;
}

function colorFor(value: number | null, range: { min: number; max: number }) {
  if (value === null) {
    return "#eef2f7";
  }
  const midpoint = range.min < 0 && range.max > 0 ? 0 : (range.min + range.max) / 2;
  const span = Math.max(Math.abs(range.max - midpoint), Math.abs(range.min - midpoint), 1e-6);
  const t = Math.max(-1, Math.min(1, (value - midpoint) / span));
  if (t >= 0) {
    const alpha = 0.18 + t * 0.62;
    return `rgba(22, 101, 52, ${alpha})`;
  }
  const alpha = 0.18 + Math.abs(t) * 0.62;
  return `rgba(185, 28, 28, ${alpha})`;
}

function valueRange(values: (number | null)[]) {
  const finite = values.filter((value): value is number => Number.isFinite(value));
  if (!finite.length) {
    return { min: 0, max: 1 };
  }
  return { min: Math.min(...finite), max: Math.max(...finite) };
}

function unique(values: string[]): string[] {
  return Array.from(new Set(values)).sort((left, right) => left.localeCompare(right, undefined, { numeric: true }));
}

function objectValue(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function arrayValue(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function numberValue(value: unknown): number | null {
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) ? number : null;
}

function stringValue(value: unknown): string {
  return value === null || value === undefined ? "" : String(value);
}

function formatNumber(value: number | null): string {
  return value === null ? "-" : value.toFixed(3);
}

function formatSigned(value: number | null): string {
  if (value === null) {
    return "-";
  }
  return `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
}

function formatCount(value: number | null): string {
  return value === null ? "-" : value.toLocaleString();
}

function formatCell(value: unknown): string {
  const number = numberValue(value);
  if (number !== null && typeof value !== "string") {
    return Math.abs(number) < 10 && !Number.isInteger(number) ? number.toFixed(3) : number.toLocaleString();
  }
  return stringValue(value) || "-";
}
