import type { ActivationSliceResponse } from "../../types/dataset";
import { ACTIVATION_CLIP_OPTIONS } from "./shared";

export type FeatureTableRow = {
  detail?: string | null;
  direction?: string | null;
  index: number;
  label?: string | null;
  title?: string | null;
  value: number;
};

export function DetailItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="detail-item">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export function FeatureTable({
  activeFeature,
  clip,
  clipPercent,
  indexHeader = "Channel",
  indexPrefix = "channel ",
  loading = false,
  onClipPercentChange,
  onFeatureChange,
  onRowLimitChange,
  reserveHeight = false,
  rowLimit,
  rowLimitOptions,
  rows,
  selectable = true,
  stale = false,
  title,
  updating = false,
  valueHeader = "Value",
}: {
  activeFeature: number;
  clip?: ActivationSliceResponse["clip"];
  clipPercent?: number;
  indexHeader?: string;
  indexPrefix?: string;
  loading?: boolean;
  onClipPercentChange?: (clipPercent: number) => void;
  onFeatureChange: (feature: number) => void;
  onRowLimitChange?: (count: number) => void;
  reserveHeight?: boolean;
  rowLimit?: number;
  rowLimitOptions?: readonly number[];
  rows: FeatureTableRow[];
  selectable?: boolean;
  stale?: boolean;
  title: string;
  updating?: boolean;
  valueHeader?: string;
}) {
  const showClipControl = Boolean(onClipPercentChange);
  const showRowLimitControl = Boolean(onRowLimitChange && rowLimitOptions?.length);
  const rowsAreSelectable = selectable && activeFeature >= 0;
  const visibleRows = rows.slice(0, rowLimit ?? 12);
  return (
    <div
      className={[
        "feature-table-wrap",
        stale ? "updating" : "",
        reserveHeight ? "stable-height" : "",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <div className="section-title">
        <span>{title}</span>
        <div className="section-title-actions">
          {showClipControl ? (
            <div className="activation-clip-control" aria-label="Activation value clip">
              <span>Clip</span>
              {ACTIVATION_CLIP_OPTIONS.map((percent) => (
                <button
                  className={clipPercent === percent ? "active" : ""}
                  key={percent}
                  type="button"
                  onClick={() => onClipPercentChange?.(percent)}
                >
                  {percent === 0 ? "0" : `${percent}%`}
                </button>
              ))}
            </div>
          ) : null}
          {showRowLimitControl ? (
            <label className="feature-row-limit">
              <span>Top</span>
              <select
                aria-label="Number of top channels"
                value={rowLimit}
                onChange={(event) => onRowLimitChange?.(Number(event.target.value))}
              >
                {rowLimitOptions?.map((count) => (
                  <option key={count} value={count}>
                    {count}
                  </option>
                ))}
              </select>
            </label>
          ) : null}
          {updating || loading ? <small>{loading ? "loading" : "updating"}</small> : null}
        </div>
      </div>
      {showClipControl && clip?.enabled ? (
        <div className="activation-clip-note">
          Showing {clip.kept ?? 0} / {clip.total ?? 0} features after percentile trim.
        </div>
      ) : null}
      {rows.length ? (
        <table className="compact-table">
          <thead>
            <tr>
              <th>{indexHeader}</th>
              <th>{valueHeader}</th>
            </tr>
          </thead>
          <tbody>
            {visibleRows.map((row) => (
              <tr
                className={[
                  rowsAreSelectable ? "selectable-row" : "",
                  row.index === activeFeature ? "active" : "",
                ].filter(Boolean).join(" ")}
                key={row.index}
                onClick={rowsAreSelectable ? () => onFeatureChange(row.index) : undefined}
                title={row.title ?? undefined}
              >
                <td>
                  {rowsAreSelectable ? (
                    <button
                      className={row.index === activeFeature ? "feature-link active" : "feature-link"}
                      type="button"
                      onClick={(event) => {
                        event.stopPropagation();
                        onFeatureChange(row.index);
                      }}
                    >
                      {indexPrefix}
                      {row.index}
                    </button>
                  ) : (
                    <span className="feature-index-text">
                      {indexPrefix}
                      {row.index}
                    </span>
                  )}
                </td>
                <td className={row.value >= 0 ? "signed-positive" : "signed-negative"}>
                  <span>{row.value.toFixed(4)}</span>
                  {row.label ? <small className="feature-row-label">{row.label}</small> : null}
                  {row.detail ? <small className="feature-row-detail">{row.detail}</small> : null}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : loading ? (
        <div className="feature-table-skeleton" aria-label={`${title} loading`}>
          {Array.from({ length: 8 }, (_, index) => (
            <div className="skeleton-row" key={index}>
              <span />
              <span />
            </div>
          ))}
        </div>
      ) : (
        <div className="empty-state">No channel ranking available.</div>
      )}
    </div>
  );
}
