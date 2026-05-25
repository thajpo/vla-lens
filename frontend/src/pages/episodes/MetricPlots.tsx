import { useMemo, useState } from "react";
import { BarChart3, ChevronLeft, ChevronRight, GripVertical, Plus, X } from "lucide-react";
import type { EpisodeMetric } from "../../types/dataset";
import { DEFAULT_METRIC_ORDER, DEFAULT_METRIC_X_KEY, type MetricPlotConfig } from "./shared";

export function MetricPlotPanel({
  metrics,
  timestep,
  onTimestepChange,
}: {
  metrics: EpisodeMetric[];
  timestep: number;
  onTimestepChange: (timestep: number) => void;
}) {
  const [plots, setPlots] = useState<MetricPlotConfig[] | null>(null);
  const [addOpen, setAddOpen] = useState(false);
  const [draggedPlotId, setDraggedPlotId] = useState<string | null>(null);

  const metricByKey = useMemo(
    () => new Map(metrics.map((metric) => [metric.key, metric])),
    [metrics],
  );
  const resolvedPlots = useMemo(
    () => reconcileMetricPlots(plots ?? [], metrics),
    [metrics, plots],
  );
  const visiblePlots = resolvedPlots.filter((plot) => metricByKey.has(plot.yKey));

  const movePlot = (plotId: string, direction: -1 | 1) => {
    setPlots((current) => moveMetricPlot(reconcileMetricPlots(current ?? [], metrics), plotId, direction));
  };
  const removePlot = (plotId: string) => {
    setPlots((current) =>
      reconcileMetricPlots(current ?? [], metrics).filter((plot) => plot.id !== plotId),
    );
  };
  const addPlot = (plot: Omit<MetricPlotConfig, "id">) => {
    setPlots((current) => [
      ...reconcileMetricPlots(current ?? [], metrics),
      {
        ...plot,
        id: `custom:${plot.xKey}:${plot.yKey}:${Date.now()}`,
      },
    ]);
    setAddOpen(false);
  };
  const dropPlot = (targetPlotId: string) => {
    if (!draggedPlotId || draggedPlotId === targetPlotId) {
      setDraggedPlotId(null);
      return;
    }
    setPlots((current) =>
      reorderMetricPlots(reconcileMetricPlots(current ?? [], metrics), draggedPlotId, targetPlotId),
    );
    setDraggedPlotId(null);
  };

  return (
    <section className="episode-tool-panel metric-dashboard">
      <header>
        <div className="icon-label">
          <BarChart3 size={16} />
          <strong>Episode Metrics</strong>
        </div>
        <button
          className="metric-add-button"
          disabled={!metrics.length}
          type="button"
          onClick={() => setAddOpen(true)}
        >
          <Plus size={15} />
          <span>Add plot</span>
        </button>
      </header>
      {visiblePlots.length ? (
        <div className="metric-tile-grid">
          {visiblePlots.map((plot, index) => {
            const yMetric = metricByKey.get(plot.yKey);
            const xMetric =
              plot.xKey === DEFAULT_METRIC_X_KEY ? undefined : metricByKey.get(plot.xKey);
            if (!yMetric) {
              return null;
            }
            return (
              <article
                className="metric-tile"
                draggable
                key={plot.id}
                onDragEnd={() => setDraggedPlotId(null)}
                onDragOver={(event) => event.preventDefault()}
                onDragStart={() => setDraggedPlotId(plot.id)}
                onDrop={() => dropPlot(plot.id)}
              >
                <div className="metric-tile-head">
                  <span className="metric-drag-handle" title="Move plot">
                    <GripVertical size={15} />
                  </span>
                  <div className="metric-tile-title">
                    <strong>{yMetric.label}</strong>
                    <span>{metricAxisSummary(yMetric, xMetric)}</span>
                  </div>
                  <div className="metric-tile-actions">
                    <button
                      aria-label={`Move ${yMetric.label} left`}
                      disabled={index === 0}
                      title="Move left"
                      type="button"
                      onClick={() => movePlot(plot.id, -1)}
                    >
                      <ChevronLeft size={14} />
                    </button>
                    <button
                      aria-label={`Move ${yMetric.label} right`}
                      disabled={index === visiblePlots.length - 1}
                      title="Move right"
                      type="button"
                      onClick={() => movePlot(plot.id, 1)}
                    >
                      <ChevronRight size={14} />
                    </button>
                    <button
                      aria-label={`Remove ${yMetric.label}`}
                      title="Remove"
                      type="button"
                      onClick={() => removePlot(plot.id)}
                    >
                      <X size={14} />
                    </button>
                  </div>
                </div>
                <SeriesPlot
                  metric={yMetric}
                  xMetric={xMetric}
                  timestep={timestep}
                  onSelectIndex={onTimestepChange}
                />
                {yMetric.description ? <p>{yMetric.description}</p> : null}
              </article>
            );
          })}
        </div>
      ) : (
        <div className="empty-state">No metrics available for this episode.</div>
      )}
      {addOpen ? (
        <MetricPlotDialog
          metrics={metrics}
          onAdd={addPlot}
          onClose={() => setAddOpen(false)}
        />
      ) : null}
    </section>
  );
}

function MetricPlotDialog({
  metrics,
  onAdd,
  onClose,
}: {
  metrics: EpisodeMetric[];
  onAdd: (plot: Omit<MetricPlotConfig, "id">) => void;
  onClose: () => void;
}) {
  const [xKey, setXKey] = useState(DEFAULT_METRIC_X_KEY);
  const [yKey, setYKey] = useState(metrics[0]?.key ?? "");
  const yMetric = metrics.find((metric) => metric.key === yKey) ?? metrics[0];
  const resolvedYKey = yMetric?.key ?? "";
  const compatibleXMetrics = metrics.filter(
    (metric) => !yMetric || metric.values.length === yMetric.values.length,
  );
  const resolvedXKey =
    xKey === DEFAULT_METRIC_X_KEY ||
    compatibleXMetrics.some((metric) => metric.key === xKey)
      ? xKey
      : DEFAULT_METRIC_X_KEY;
  const canAdd = Boolean(yMetric);

  return (
    <div className="metric-dialog-backdrop" role="presentation" onMouseDown={onClose}>
      <div
        aria-label="Add metric plot"
        aria-modal="true"
        className="metric-dialog"
        role="dialog"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header>
          <div>
            <strong>Add plot</strong>
            <span>Select the series to use for each axis.</span>
          </div>
          <button aria-label="Close add plot" type="button" onClick={onClose}>
            <X size={16} />
          </button>
        </header>
        <div className="metric-dialog-grid">
          <label>
            X axis
            <select value={resolvedXKey} onChange={(event) => setXKey(event.target.value)}>
              <option value={DEFAULT_METRIC_X_KEY}>Metric timeline</option>
              {compatibleXMetrics.map((metric) => (
                <option key={metric.key} value={metric.key}>
                  {metric.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Y axis
            <select value={resolvedYKey} onChange={(event) => setYKey(event.target.value)}>
              {metrics.map((metric) => (
                <option key={metric.key} value={metric.key}>
                  {metric.label}
                </option>
              ))}
            </select>
          </label>
        </div>
        <footer>
          <button type="button" onClick={onClose}>
            Cancel
          </button>
          <button
            className="metric-dialog-primary"
            disabled={!canAdd}
            type="button"
            onClick={() => onAdd({ xKey: resolvedXKey, yKey: resolvedYKey })}
          >
            Add plot
          </button>
        </footer>
      </div>
    </div>
  );
}

function SeriesPlot({
  metric,
  xMetric,
  timestep,
  onSelectIndex,
}: {
  metric: EpisodeMetric;
  xMetric?: EpisodeMetric;
  timestep: number;
  onSelectIndex: (index: number) => void;
}) {
  const yValues = metric.values ?? [];
  const count = xMetric ? Math.min(yValues.length, xMetric.values.length) : yValues.length;
  const values = yValues.slice(0, count);
  const width = 640;
  const height = 168;
  const padLeft = 50;
  const padRight = 18;
  const padTop = 18;
  const padBottom = 34;
  const xValues = xMetric
    ? xMetric.values.slice(0, count)
    : metric.x_values?.length === values.length
      ? metric.x_values
      : values.map((_, index) => index);
  const plotPoints = values
    .map((value, index) => ({ index, value, x: xValues[index] ?? index }))
    .filter((point) => Number.isFinite(point.value) && Number.isFinite(point.x));
  const finiteValues = plotPoints.map((point) => point.value);
  const finiteX = plotPoints.map((point) => point.x);
  const minX = Math.min(...finiteX, 0);
  const maxX = Math.max(...finiteX, Math.max(1, values.length - 1));
  const min = Math.min(...finiteValues, 0);
  const max = Math.max(...finiteValues, 1);
  const span = max - min || 1;
  const xSpan = maxX - minX || 1;
  const xForValue = (value: number) =>
    padLeft + ((value - minX) / xSpan) * (width - padLeft - padRight);
  const xFor = (index: number) => xForValue(xValues[index] ?? index);
  const yFor = (value: number) =>
    height - padBottom - ((value - min) / span) * (height - padTop - padBottom);
  const points = plotPoints.map((point) => `${xForValue(point.x)},${yFor(point.value)}`).join(" ");
  const activeIndex = nearestMetricIndex(metric, timestep);
  const activeX = xFor(Math.min(activeIndex, Math.max(0, values.length - 1)));
  const xLabel = xMetric?.label || metric.x_label || "Environment timestep";
  const yLabel = metric.y_label || metric.label;
  const yUnit = metric.y_unit ? ` (${metric.y_unit})` : "";

  return (
    <svg
      className="episode-line-plot"
      role="img"
      viewBox={`0 0 ${width} ${height}`}
      onClick={(event) => {
        const rect = event.currentTarget.getBoundingClientRect();
        const svgX = ((event.clientX - rect.left) / rect.width) * width;
        const plotRatio = Math.max(
          0,
          Math.min(1, (svgX - padLeft) / (width - padLeft - padRight)),
        );
        const clickedX = minX + plotRatio * xSpan;
        const clickedIndex = nearestXIndex(xValues, clickedX);
        const timelineX = metric.x_values?.[clickedIndex] ?? clickedIndex;
        onSelectIndex(Math.round(timelineX));
      }}
    >
      <line className="plot-axis" x1={padLeft} x2={width - padRight} y1={height - padBottom} y2={height - padBottom} />
      <line className="plot-axis" x1={padLeft} x2={padLeft} y1={padTop} y2={height - padBottom} />
      <polyline className="plot-series" points={points} />
      <line className="plot-cursor" x1={activeX} x2={activeX} y1={padTop} y2={height - padBottom} />
      {plotPoints.map((point) => (
        <circle
          className={point.index === activeIndex ? "plot-point active" : "plot-point"}
          cx={xForValue(point.x)}
          cy={yFor(point.value)}
          key={`${metric.key}-${point.index}`}
          r={point.index === activeIndex ? 3.8 : 2.4}
        />
      ))}
      <text className="plot-label" x={padLeft} y={13}>
        {max.toFixed(3)}
      </text>
      <text className="plot-label" x={padLeft} y={height - padBottom + 15}>
        {min.toFixed(3)}
      </text>
      <text className="plot-label plot-x-label" x={width / 2} y={height - 8}>
        {xLabel}
      </text>
      <text
        className="plot-label plot-y-label"
        transform={`translate(14 ${height / 2}) rotate(-90)`}
      >
        {yLabel}
        {yUnit}
      </text>
    </svg>
  );
}

function reconcileMetricPlots(
  current: MetricPlotConfig[],
  metrics: EpisodeMetric[],
): MetricPlotConfig[] {
  const metricKeys = new Set(metrics.map((metric) => metric.key));
  const validCurrent = current.filter(
    (plot) =>
      metricKeys.has(plot.yKey) &&
      (plot.xKey === DEFAULT_METRIC_X_KEY || metricKeys.has(plot.xKey)),
  );
  if (validCurrent.length) {
    return validCurrent;
  }
  return defaultMetricPlots(metrics);
}

function defaultMetricPlots(metrics: EpisodeMetric[]): MetricPlotConfig[] {
  const metricByKey = new Map(metrics.map((metric) => [metric.key, metric]));
  const defaultMetricKeys = new Set<string>(DEFAULT_METRIC_ORDER);
  const ordered = [
    ...DEFAULT_METRIC_ORDER.flatMap((key) => {
      const metric = metricByKey.get(key);
      return metric ? [metric] : [];
    }),
    ...metrics.filter((metric) => !defaultMetricKeys.has(metric.key)),
  ];
  return ordered.slice(0, 6).map((metric) => ({
    id: `default:${metric.key}`,
    xKey: DEFAULT_METRIC_X_KEY,
    yKey: metric.key,
  }));
}

function moveMetricPlot(
  plots: MetricPlotConfig[],
  plotId: string,
  direction: -1 | 1,
): MetricPlotConfig[] {
  const index = plots.findIndex((plot) => plot.id === plotId);
  const nextIndex = index + direction;
  if (index < 0 || nextIndex < 0 || nextIndex >= plots.length) {
    return plots;
  }
  const next = [...plots];
  const [plot] = next.splice(index, 1);
  next.splice(nextIndex, 0, plot);
  return next;
}

function reorderMetricPlots(
  plots: MetricPlotConfig[],
  draggedPlotId: string,
  targetPlotId: string,
): MetricPlotConfig[] {
  const from = plots.findIndex((plot) => plot.id === draggedPlotId);
  const to = plots.findIndex((plot) => plot.id === targetPlotId);
  if (from < 0 || to < 0 || from === to) {
    return plots;
  }
  const next = [...plots];
  const [plot] = next.splice(from, 1);
  next.splice(to, 0, plot);
  return next;
}

function nearestXIndex(values: number[], target: number): number {
  if (!values.length) {
    return 0;
  }
  let best = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  values.forEach((value, index) => {
    const distance = Math.abs(value - target);
    if (distance < bestDistance) {
      best = index;
      bestDistance = distance;
    }
  });
  return best;
}

function metricAxisSummary(yMetric: EpisodeMetric, xMetric?: EpisodeMetric): string {
  const xLabel = xMetric?.label ?? yMetric.x_label ?? "Timeline";
  const yLabel = yMetric.y_label ?? yMetric.label;
  return `${xLabel} -> ${yLabel}`;
}

function nearestMetricIndex(metric: EpisodeMetric, target: number): number {
  const values = metric.values ?? [];
  const xValues =
    metric.x_values?.length === values.length
      ? metric.x_values
      : values.map((_, index) => index);
  if (!xValues.length) {
    return 0;
  }
  let best = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  xValues.forEach((value, index) => {
    const distance = Math.abs(value - target);
    if (distance < bestDistance) {
      best = index;
      bestDistance = distance;
    }
  });
  return best;
}
