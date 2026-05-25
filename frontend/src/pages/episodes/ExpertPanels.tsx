import { useEffect, useMemo, useRef, useState } from "react";
import type {
  ExpertTokenActivationsResponse,
  ExpertTokenDetailsResponse,
  ImagePatchAttention,
  PromptTokenAttention,
  SelectedPatch,
} from "../../types/dataset";
import { DetailItem, FeatureTable } from "./InspectorTables";
import {
  displayTokenPiece,
  formatMaybeNumber,
  formatPercent,
  promptTokenTitle,
  signedActivationColor,
} from "./formatters";

export function ExpertTokenFlow({
  activeFeature,
  details,
  onFeatureChange,
  onPatchSelect,
  onPromptTokenSelect,
  onTokenChange,
  payload,
  selectedPatch,
  selectedPromptToken,
  selectedToken,
  tokenSiteName,
}: {
  activeFeature: number;
  details?: ExpertTokenDetailsResponse;
  onFeatureChange: (feature: number) => void;
  onPatchSelect: (patch: SelectedPatch | null) => void;
  onPromptTokenSelect: (tokenIndex: number | null) => void;
  onTokenChange: (tokenIndex: number | null) => void;
  payload?: ExpertTokenActivationsResponse;
  selectedPatch: SelectedPatch | null;
  selectedPromptToken: number | null;
  selectedToken: number | null;
  tokenSiteName: string;
}) {
  return (
    <>
      <div className="expert-token-panel">
        <ExpertTokenStrip
          payload={payload}
          selectedToken={selectedToken}
          onTokenChange={onTokenChange}
        />
        {!payload?.available ? (
          <div className="empty-state compact">
            {payload?.reason || tokenSiteName || "Select an expert token site to view action-token activations."}
          </div>
        ) : null}
        <ExpertTokenDetails
          activeFeature={activeFeature}
          details={details}
          selectedPatch={selectedPatch}
          selectedPromptToken={selectedPromptToken}
          selectedToken={selectedToken}
          onFeatureChange={onFeatureChange}
          onPatchSelect={onPatchSelect}
          onPromptTokenSelect={onPromptTokenSelect}
        />
      </div>
      <ExpertAttentionSummary details={details} selectedToken={selectedToken} />
    </>
  );
}

function ExpertAttentionSummary({
  details,
  selectedToken,
}: {
  details?: ExpertTokenDetailsResponse;
  selectedToken: number | null;
}) {
  const coarse = details?.attention_coarse;
  if (selectedToken === null) {
    return null;
  }
  return (
    <div className="expert-attention-summary">
      {details?.available && coarse ? (
        <div className="attention-split">
          <AttentionBar label="image" value={coarse.image} />
          <AttentionBar label="prompt" value={coarse.prompt} />
          <AttentionBar label="action" value={coarse.action_suffix} />
        </div>
      ) : (
        <div className="empty-state">
          {details?.reason || "No token-specific attention was found for this site."}
        </div>
      )}
    </div>
  );
}

function AttentionBar({ label, value }: { label: string; value: number | null | undefined }) {
  const numeric = typeof value === "number" && Number.isFinite(value) ? value : 0;
  const width = `${Math.max(0, Math.min(1, numeric)) * 100}%`;
  return (
    <div className="bar-row">
      <span>{label}</span>
      <div className="bar-track">
        <span style={{ width }} />
      </div>
      <span>{formatPercent(value)}</span>
    </div>
  );
}

function ExpertTokenStrip({
  onTokenChange,
  payload,
  selectedToken,
}: {
  onTokenChange: (tokenIndex: number | null) => void;
  payload?: ExpertTokenActivationsResponse;
  selectedToken: number | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const values = useMemo(() => payload?.values ?? [], [payload?.values]);
  const [layoutVersion, setLayoutVersion] = useState(0);

  useEffect(() => {
    const target = canvasRef.current;
    if (!target) {
      return;
    }
    const observer = new ResizeObserver(() => {
      setLayoutVersion((version) => version + 1);
    });
    observer.observe(target);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const scale = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(rect.width * scale));
    canvas.height = Math.max(1, Math.round(rect.height * scale));
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.fillStyle = "#fffefa";
    ctx.fillRect(0, 0, rect.width, rect.height);

    if (!payload?.available || !values.length) {
      ctx.fillStyle = "#61707b";
      ctx.font = "12px Inter, sans-serif";
      ctx.fillText("Select an expert layer to view action-token activations.", 10, 24);
      return;
    }

    const maxAbs = Math.max(...values.map((value) => Math.abs(value)).filter(Number.isFinite), 1e-6);
    const cellWidth = rect.width / values.length;
    values.forEach((value, index) => {
      ctx.fillStyle = signedActivationColor(value, maxAbs);
      ctx.fillRect(index * cellWidth, 0, Math.ceil(cellWidth), rect.height);
    });

    if (selectedToken !== null) {
      const index = Math.max(0, Math.min(values.length - 1, selectedToken));
      ctx.strokeStyle = "rgba(31, 33, 31, 0.82)";
      ctx.lineWidth = 2;
      ctx.strokeRect(index * cellWidth + 1, 1, Math.max(2, cellWidth - 2), rect.height - 2);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.95)";
      ctx.lineWidth = 1;
      ctx.strokeRect(index * cellWidth + 3, 3, Math.max(2, cellWidth - 6), rect.height - 6);
    }
  }, [layoutVersion, payload?.available, selectedToken, values]);

  return (
    <canvas
      ref={canvasRef}
      className="strip-canvas"
      aria-label="Expert action token activation strip"
      role="button"
      tabIndex={0}
      onClick={(event) => {
        if (!values.length) {
          return;
        }
        const rect = event.currentTarget.getBoundingClientRect();
        const index = Math.max(
          0,
          Math.min(values.length - 1, Math.floor(((event.clientX - rect.left) / rect.width) * values.length)),
        );
        onTokenChange(index);
      }}
      onKeyDown={(event) => {
        if ((event.key === "Enter" || event.key === " ") && values.length) {
          event.preventDefault();
          onTokenChange(selectedToken === null ? 0 : selectedToken);
        }
      }}
    />
  );
}

function ExpertTokenDetails({
  activeFeature,
  details,
  onFeatureChange,
  onPatchSelect,
  onPromptTokenSelect,
  selectedPatch,
  selectedPromptToken,
  selectedToken,
}: {
  activeFeature: number;
  details?: ExpertTokenDetailsResponse;
  onFeatureChange: (feature: number) => void;
  onPatchSelect: (patch: SelectedPatch | null) => void;
  onPromptTokenSelect: (tokenIndex: number | null) => void;
  selectedPatch: SelectedPatch | null;
  selectedPromptToken: number | null;
  selectedToken: number | null;
}) {
  if (selectedToken === null) {
    return (
      <div className="token-detail">
        <div className="empty-state">
          Click an action token to inspect its hidden vector and aligned action vector.
        </div>
      </div>
    );
  }
  if (!details?.available) {
    return (
      <div className="token-detail">
        <div className="empty-state">{details?.reason || "Loading token detail."}</div>
      </div>
    );
  }
  const action = details.action;
  const coarse = details.attention_coarse;
  return (
    <div className="token-detail">
      <div className="detail-grid">
        <DetailItem label="channel value" value={formatMaybeNumber(details.feature_value)} />
        <DetailItem
          label="channel rank"
          value={details.feature_rank_by_abs ? `#${details.feature_rank_by_abs}` : "-"}
        />
        <DetailItem label="action norm" value={formatMaybeNumber(action?.norm)} />
        <DetailItem
          label="token"
          value={`${details.token_index ?? selectedToken} / ${Math.max(0, (details.token_count ?? 1) - 1)}`}
        />
        <DetailItem
          label="attention split"
          value={
            coarse
              ? `img ${formatPercent(coarse.image)} / prompt ${formatPercent(coarse.prompt)}`
              : "-"
          }
        />
      </div>
      <FeatureTable
        activeFeature={activeFeature}
        onFeatureChange={onFeatureChange}
        rows={details.top_abs ?? []}
        title="Token Channel Ranking"
      />
      <ImageAttentionTable
        rows={details.top_image_patches ?? []}
        selectedPatch={selectedPatch}
        onPatchSelect={onPatchSelect}
      />
      <PromptAttentionTable
        rows={details.top_prompt_tokens ?? []}
        selectedPromptToken={selectedPromptToken}
        onPromptTokenSelect={onPromptTokenSelect}
      />
      <FeatureTable
        activeFeature={-1}
        onFeatureChange={() => undefined}
        rows={action?.top_abs ?? []}
        title={`Action Dimensions${action?.source ? ` / ${action.source}` : ""}`}
        indexHeader="Dim"
        indexPrefix="a"
        selectable={false}
      />
      {details.note ? <p className="note">{details.note}</p> : null}
    </div>
  );
}

function ImageAttentionTable({
  onPatchSelect,
  rows,
  selectedPatch,
}: {
  onPatchSelect: (patch: SelectedPatch | null) => void;
  rows: ImagePatchAttention[];
  selectedPatch: SelectedPatch | null;
}) {
  return (
    <div className="feature-table-wrap">
      <div className="section-title">
        <span>Image Attention Patches</span>
        <small>attention</small>
      </div>
      {rows.length ? (
        <table className="compact-table">
          <thead>
            <tr>
              <th>Patch</th>
              <th>Mass</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 12).map((row) => {
              const selected =
                selectedPatch?.camera === row.camera &&
                selectedPatch.row === row.row &&
                selectedPatch.col === row.col;
              return (
                <tr
                  className={selected ? "selectable-row active" : "selectable-row"}
                  key={`${row.camera}-${row.row}-${row.col}`}
                >
                  <td>
                    <button
                      className="table-row-button"
                      title={`Select ${row.camera} row ${row.row}, column ${row.col}`}
                      type="button"
                      onClick={() => onPatchSelect({ camera: row.camera, row: row.row, col: row.col })}
                    >
                      {row.camera} r{row.row} c{row.col}
                    </button>
                  </td>
                  <td>{formatPercent(row.attention)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      ) : (
        <div className="empty-state">No image attention patches found.</div>
      )}
    </div>
  );
}

function PromptAttentionTable({
  onPromptTokenSelect,
  rows,
  selectedPromptToken,
}: {
  onPromptTokenSelect: (tokenIndex: number | null) => void;
  rows: PromptTokenAttention[];
  selectedPromptToken: number | null;
}) {
  return (
    <div className="feature-table-wrap">
      <div className="section-title">
        <span>Prompt Attention Tokens</span>
        <small>attention</small>
      </div>
      {rows.length ? (
        <table className="compact-table">
          <thead>
            <tr>
              <th>Token</th>
              <th>Piece</th>
              <th>Mass</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 12).map((row) => {
              const selected = selectedPromptToken === row.local_index;
              return (
                <tr
                  className={selected ? "selectable-row active" : "selectable-row"}
                  key={`${row.prefix_index ?? row.local_index}-${row.token_id ?? ""}`}
                >
                  <td>
                    <button
                      className="table-row-button"
                      title={promptTokenTitle(row)}
                      type="button"
                      onClick={() => onPromptTokenSelect(row.local_index)}
                    >
                      #{row.local_index}
                    </button>
                  </td>
                  <td>{displayTokenPiece(row)}</td>
                  <td>{formatPercent(row.attention)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      ) : (
        <div className="empty-state">No active prompt tokens found.</div>
      )}
    </div>
  );
}
