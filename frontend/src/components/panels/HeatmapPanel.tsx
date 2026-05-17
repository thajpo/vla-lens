import { useQuery } from "@tanstack/react-query";
import { sliceLensArray } from "../../api/lensArrays";
import type { LensArraySpec, SelectionState } from "../../types/workbench";
import { PanelCard } from "../layout/PanelCard";

type HeatmapPanelProps = {
  panelId: string;
  array?: LensArraySpec;
  xAxis: string;
  yAxis: string;
  filterAxis?: string;
  filterValue?: string;
  analysisRunId: string;
  metric: string;
  selected?: SelectionState | null;
  onSelect: (selection: SelectionState) => void;
};

type HeatmapCell = {
  x: unknown;
  y: unknown;
  value: number | null;
};

export function HeatmapPanel({
  panelId,
  array,
  xAxis,
  yAxis,
  filterAxis,
  filterValue,
  analysisRunId,
  metric,
  selected,
  onSelect,
}: HeatmapPanelProps) {
  const selection =
    array && filterAxis && filterValue !== undefined && filterValue !== ""
      ? { [filterAxis]: [filterValue] }
      : {};

  const slice = useQuery({
    queryKey: ["lens-array-slice", array?.array_id, selection],
    queryFn: () => sliceLensArray(array?.array_id ?? "", selection, 8192),
    enabled: Boolean(array),
  });

  const cells = array && slice.data?.values ? buildCells(array, xAxis, yAxis, slice.data.values) : [];

  const xValues = uniqueValues(cells.map((cell) => cell.x));
  const yValues = uniqueValues(cells.map((cell) => cell.y));
  const range = valueRange(cells);

  return (
    <PanelCard title="Heatmap">
      {!array ? <div className="empty-state">No compatible LensArray.</div> : null}
      {slice.isError ? <div className="empty-state">Slice failed.</div> : null}
      {slice.isLoading ? <div className="empty-state">Loading slice...</div> : null}
      {cells.length ? (
        <div className="heatmap-wrap">
          <div
            className="heatmap-grid"
            style={{ gridTemplateColumns: `72px repeat(${xValues.length}, minmax(34px, 1fr))` }}
          >
            <div className="axis-corner">{yAxis}</div>
            {xValues.map((x) => (
              <div className="axis-label" key={`x-${String(x)}`}>
                {String(x)}
              </div>
            ))}
            {yValues.map((y) => (
              <Row
                key={`row-${String(y)}`}
                y={y}
                xValues={xValues}
                cells={cells}
                range={range}
                selected={selected}
                xAxis={xAxis}
                yAxis={yAxis}
                filterAxis={filterAxis}
                filterValue={filterValue}
                analysisRunId={analysisRunId}
                metric={metric}
                panelId={panelId}
                onSelect={onSelect}
              />
            ))}
          </div>
        </div>
      ) : null}
    </PanelCard>
  );
}

type RowProps = {
  y: unknown;
  xValues: unknown[];
  cells: HeatmapCell[];
  range: { min: number; max: number };
  selected?: SelectionState | null;
  xAxis: string;
  yAxis: string;
  filterAxis?: string;
  filterValue?: string;
  analysisRunId: string;
  metric: string;
  panelId: string;
  onSelect: (selection: SelectionState) => void;
};

function Row({
  y,
  xValues,
  cells,
  range,
  selected,
  xAxis,
  yAxis,
  filterAxis,
  filterValue,
  analysisRunId,
  metric,
  panelId,
  onSelect,
}: RowProps) {
  return (
    <>
      <div className="axis-label y-label">{String(y)}</div>
      {xValues.map((x) => {
        const cell = cells.find((item) => item.x === x && item.y === y);
        const isSelected =
          first(selected?.axis_values[xAxis]) === x &&
          first(selected?.axis_values[yAxis]) === y &&
          (!filterAxis || first(selected?.axis_values[filterAxis]) === filterValue);
        return (
          <button
            className={`heatmap-cell ${isSelected ? "selected" : ""}`}
            key={`${String(y)}-${String(x)}`}
            style={{ background: colorFor(cell?.value ?? null, range) }}
            type="button"
            onClick={() => {
              const axisValues: Record<string, unknown> = {
                [xAxis]: [x],
                [yAxis]: [y],
                analysis_run: [analysisRunId],
                metric: [metric],
              };
              if (filterAxis && filterValue !== undefined) {
                axisValues[filterAxis] = [filterValue];
              }
              onSelect({
                selection_id: `${panelId}_${String(y)}_${String(x)}_${filterValue ?? "all"}`,
                source_panel_id: panelId,
                axis_values: axisValues,
                unit_refs: [],
                cohort_refs: [],
                intent: "inspect",
              });
            }}
          >
            {cell?.value == null ? "" : formatNumber(cell.value)}
          </button>
        );
      })}
    </>
  );
}

function buildCells(
  array: LensArraySpec,
  xAxis: string,
  yAxis: string,
  values: unknown,
): HeatmapCell[] {
  const xIndex = array.dims.indexOf(xAxis);
  const yIndex = array.dims.indexOf(yAxis);
  if (xIndex < 0 || yIndex < 0) {
    return [];
  }
  const matrix = normalizeMatrix(values, xIndex, yIndex);
  const xCoords = coordsFor(array, xAxis, matrix.width);
  const yCoords = coordsFor(array, yAxis, matrix.height);
  const out: HeatmapCell[] = [];
  for (let y = 0; y < matrix.height; y += 1) {
    for (let x = 0; x < matrix.width; x += 1) {
      out.push({ x: xCoords[x], y: yCoords[y], value: matrix.values[y]?.[x] ?? null });
    }
  }
  return out;
}

function normalizeMatrix(values: unknown, xIndex: number, yIndex: number) {
  const arr = values as unknown[];
  if (xIndex === 1 && yIndex === 0) {
    const rows = arr as unknown[][];
    return { values: rows.map((row) => row.map(numberOrNull)), width: rows[0]?.length ?? 0, height: rows.length };
  }
  if (xIndex === 0 && yIndex === 1) {
    const rows = arr as unknown[][];
    const height = rows[0]?.length ?? 0;
    const width = rows.length;
    const transposed = Array.from({ length: height }, (_, y) =>
      Array.from({ length: width }, (_, x) => numberOrNull(rows[x]?.[y])),
    );
    return { values: transposed, width, height };
  }
  return { values: [], width: 0, height: 0 };
}

function coordsFor(array: LensArraySpec, axis: string, count: number): unknown[] {
  const coords = array.coords[axis];
  if (Array.isArray(coords)) {
    return coords.slice(0, count);
  }
  return Array.from({ length: count }, (_, index) => index);
}

function uniqueValues(values: unknown[]): unknown[] {
  return Array.from(new Set(values.map((value) => JSON.stringify(value)))).map((value) =>
    JSON.parse(value),
  );
}

function valueRange(cells: HeatmapCell[]) {
  const values = cells
    .map((cell) => cell.value)
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  return {
    min: values.length ? Math.min(...values) : 0,
    max: values.length ? Math.max(...values) : 1,
  };
}

function colorFor(value: number | null, range: { min: number; max: number }) {
  if (value == null || !Number.isFinite(value)) {
    return "#f1f5f9";
  }
  const span = Math.max(1e-8, range.max - range.min);
  const t = Math.max(0, Math.min(1, (value - range.min) / span));
  const hue = 205 - t * 165;
  return `hsl(${hue} 72% ${82 - t * 38}%)`;
}

function numberOrNull(value: unknown): number | null {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function formatNumber(value: number) {
  return Math.abs(value) >= 10 ? value.toFixed(1) : value.toFixed(2);
}

function first(value: unknown): unknown {
  return Array.isArray(value) ? value[0] : value;
}
