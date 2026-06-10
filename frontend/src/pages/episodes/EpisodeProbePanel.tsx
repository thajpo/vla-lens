import { memo, type CSSProperties, useState } from "react";
import type {
  DiscoveryArtifactReadoutResponse,
  EpisodeProbePrediction,
  EpisodeProbeSummary,
  EpisodeProbesResponse,
  ObservationalComparisonCandidate,
  ObservationalComparisonsResponse,
} from "../../types/dataset";
import { researchCopy } from "../../copy/researchCopy";
import { EPISODE_PROBE_RESULT_LIMIT, type ProbeLayerRef } from "./shared";
import { formatLayerNumber, formatMaybeNumber } from "./formatters";
import {
  type EpisodeProbeCell,
  type EpisodeProbeMembership,
  type EpisodeProbeUsage,
  ensureSelectedProbeVisible,
  episodeProbeCells,
  episodeProbeColor,
  episodeProbeNumber,
  episodeProbeTemporalRows,
  episodeProbeUsage,
  episodeProbeValueLabel,
  formatProbeComparison,
  formatProbeCorrect,
  formatProbeValue,
  formatSignedProbeNumber,
  humanizeModelSite,
  probeDisplayedCorrect,
  probeEpisodeMembership,
  probeFeatureLabel,
  probeLayerReferences,
  probeModelLabel,
  probeQuestionLabel,
  probeReliabilityAudit,
  probeRowTone,
  probeSiteAudit,
  probeSourceLabel,
  probeTemporalLabel,
  probeToneFromCorrect,
  probeTrajectoryAudit,
  sortEpisodeProbesByInterestingness,
  uniqueStrings,
} from "./episodeProbeModel";

function EpisodeProbePanelImpl({
  artifactReadout,
  canInspectProbe,
  canIntervene,
  canJumpToProbeCall,
  comparisons,
  isComparisonError,
  isComparisonLoading,
  isError,
  isLoading,
  onInspectProbe,
  onIntervene,
  onOpenComparison,
  onJumpToProbeCall,
  onJumpToPolicyCall,
  onProbeChange,
  probes,
  selectedProbe,
  selectedProbeRef,
}: {
  artifactReadout?: DiscoveryArtifactReadoutResponse;
  canInspectProbe: boolean;
  canIntervene: boolean;
  canJumpToProbeCall: boolean;
  comparisons?: ObservationalComparisonsResponse;
  isComparisonError: boolean;
  isComparisonLoading: boolean;
  isError: boolean;
  isLoading: boolean;
  onInspectProbe: () => void;
  onIntervene: () => void;
  onOpenComparison: (traceId: string) => void;
  onJumpToProbeCall: () => void;
  onJumpToPolicyCall: (policyCallIndex: number) => void;
  onProbeChange: (artifactId: string) => void;
  probes?: EpisodeProbesResponse;
  selectedProbe?: EpisodeProbeSummary;
  selectedProbeRef?: ProbeLayerRef;
}) {
  const allProbes = probes?.probes ?? [];
  const [metric, setMetric] = useState<"confidence" | "correct">("confidence");
  const rows = selectedProbe?.rows ?? [];
  const bestModel = String(selectedProbe?.metrics.best_model ?? "");
  const filteredRows = bestModel ? rows.filter((row) => String(row.model ?? "") === bestModel) : rows;
  const cells = episodeProbeCells(filteredRows, metric);
  const usage = episodeProbeUsage(selectedProbe, filteredRows);
  const temporalRows = episodeProbeTemporalRows(filteredRows);
  const probeRefs = probeLayerReferences(probes);
  const bestRow = selectedProbe?.episode_summary.best_row;
  const bestFeature = selectedProbe?.episode_summary.best_feature || String(selectedProbe?.metrics.best_feature ?? "-");
  const predicted = selectedProbe?.episode_summary.predicted;
  const actual = selectedProbe?.episode_summary.actual;
  const displayedCorrect = probeDisplayedCorrect(predicted, actual, selectedProbe?.episode_summary.correct);
  const tone = probeToneFromCorrect(displayedCorrect, selectedProbe?.available);
  const showHeatmap = cells.length > 1;
  const trajectoryAudit = probeTrajectoryAudit(temporalRows, bestRow);
  const siteAudit = probeSiteAudit(bestRow, selectedProbeRef, bestFeature);
  const reliabilityAudit = probeReliabilityAudit(selectedProbe, usage);
  const episodeMembership = probeEpisodeMembership(selectedProbe, filteredRows);
  const probeSiteLabel = selectedProbeRef
    ? [
        selectedProbeRef.layer === null ? "" : `L${formatLayerNumber(selectedProbeRef.layer)}`,
        selectedProbeRef.policyCall === null ? "" : `call ${selectedProbeRef.policyCall}`,
      ].filter(Boolean).join(" / ") || bestFeature
    : bestFeature;

  if (isLoading) {
    return <div className="empty-state">Loading probe predictions.</div>;
  }
  if (isError) {
    return <div className="empty-state">Probe predictions could not be loaded.</div>;
  }
  if (!allProbes.length) {
    return <div className="empty-state">No trained probes found.</div>;
  }

  return (
    <div className="episode-probe-panel compact">
      <section className="episode-probe-header">
        <div className="episode-probe-selector-head">
          <span>{researchCopy.labels.probeShortlist}</span>
          <small>
            showing {Math.min(EPISODE_PROBE_RESULT_LIMIT, allProbes.length)}/{allProbes.length} · training-split episodes sorted last
          </small>
        </div>
        <EpisodeProbeStack
          probes={allProbes}
          refs={probeRefs}
          selectedArtifactId={selectedProbe?.artifact_id ?? ""}
          onProbeChange={onProbeChange}
        />
      </section>

      <section className={["episode-probe-summary", tone].filter(Boolean).join(" ")}>
        <div className="episode-probe-verdict">
          <span>{probeQuestionLabel(selectedProbe)}</span>
          <strong>{formatProbeComparison(predicted, actual)}</strong>
          <small>
            {[
              formatProbeCorrect(displayedCorrect),
              `conf ${formatMaybeNumber(selectedProbe?.episode_summary.confidence)}`,
              episodeMembership.label,
              probeSiteLabel,
              probeTemporalLabel(temporalRows),
            ].filter((item) => item && item !== "-").join(" · ")}
          </small>
        </div>
        <div className="episode-probe-actions">
          <button
            disabled={!canInspectProbe}
            title="Select the model layer this probe reads from"
            type="button"
            onClick={onInspectProbe}
          >
            Inspect
          </button>
          <button
            disabled={!canJumpToProbeCall}
            title="Move the episode timeline to the policy call this prediction came from"
            type="button"
            onClick={onJumpToProbeCall}
          >
            Jump
          </button>
          <button
            disabled={!canIntervene}
            title="Send this probe, episode, policy call, and probe input activation to the Intervention Lab"
            type="button"
            onClick={onIntervene}
          >
            Intervene
          </button>
        </div>
      </section>

      <ArtifactReadoutPanel readout={artifactReadout} />

      <section className="episode-probe-audit-strip" aria-label="Probe audit">
        <ProbeAuditPill label="Signal" value={trajectoryAudit.value} detail={trajectoryAudit.detail} />
        <ProbeAuditPill label="Input" value={siteAudit.value} detail={siteAudit.detail} />
        <ProbeAuditPill
          detail={`${episodeMembership.detail} · ${reliabilityAudit.detail}`}
          label="Trust"
          tone={episodeMembership.tone}
          value={episodeMembership.label}
        />
      </section>

      <ObservationalComparisonPanel
        comparisons={comparisons}
        isError={isComparisonError}
        isLoading={isComparisonLoading}
        onOpenComparison={onOpenComparison}
      />

      {temporalRows.length ? (
        <EpisodeProbeTimeline
          bestRow={bestRow}
          rows={temporalRows}
          onJumpToPolicyCall={onJumpToPolicyCall}
        />
      ) : null}

      {selectedProbe && !selectedProbe.available ? (
        <div className="empty-state compact">
          This probe could not be scored for this episode.
        </div>
      ) : null}

      {showHeatmap ? (
        <section className="episode-probe-plot">
          <div className="episode-probe-plot-head">
            <strong>Prediction grid</strong>
            <label>
              Plot
              <select value={metric} onChange={(event) => setMetric(event.target.value as "confidence" | "correct")}>
                <option value="confidence">Confidence</option>
                <option value="correct">Correct</option>
              </select>
            </label>
          </div>
          <EpisodeProbeHeatmap cells={cells} />
        </section>
      ) : null}

      <details className="episode-probe-details">
        <summary>
          <span>Training and scoring details</span>
          <small>{selectedProbe?.row_count ?? 0} predictions</small>
        </summary>
        <div className="episode-probe-context">
          <span>{researchCopy.labels.readSource}: {probeSourceLabel(bestRow, selectedProbeRef, bestFeature)}</span>
          <span>Probe model: {probeModelLabel(bestRow, selectedProbe)}</span>
          <span>{researchCopy.labels.inputFeature}: {probeFeatureLabel(bestRow, bestFeature)}</span>
          <span>Training: {usage.label} · {usage.detail}</span>
          <span>{researchCopy.labels.datasetScore}: {formatMaybeNumber(episodeProbeNumber(selectedProbe?.metrics.best_score))}</span>
          <span>{researchCopy.labels.datasetDelta}: {formatSignedProbeNumber(episodeProbeNumber(selectedProbe?.metrics.best_delta))}</span>
        </div>
        {filteredRows.length ? <EpisodeProbePredictionTable rows={filteredRows.slice(0, 12)} /> : null}
      </details>
    </div>
  );
}

export const EpisodeProbePanel = memo(EpisodeProbePanelImpl);

function ArtifactReadoutPanel({
  readout,
}: {
  readout?: DiscoveryArtifactReadoutResponse;
}) {
  if (!readout) {
    return null;
  }
  if (!readout.available) {
    return (
      <section className="episode-artifact-readout unavailable">
        <span>{researchCopy.labels.artifactReadout}</span>
        <strong>Unavailable</strong>
        <small>{readout.reason || researchCopy.unavailable.probeReadout}</small>
      </section>
    );
  }
  const summary = readout.summary ?? {};
  return (
    <section className="episode-artifact-readout">
      <div>
        <span>{researchCopy.labels.artifactReadout}</span>
        <strong>{readoutTypeLabel(readout.readout_type)}</strong>
      </div>
      <div>
        <span>Prediction</span>
        <strong>{formatProbeValue(summary.predicted)}</strong>
        <small>actual {formatProbeValue(summary.actual)}</small>
      </div>
      <div>
        <span>Confidence</span>
        <strong>{formatMaybeNumber(numberValue(summary.confidence))}</strong>
        <small>{summary.correct === false ? "incorrect" : summary.correct === true ? "correct" : "unknown"}</small>
      </div>
      <div>
        <span>Probe input</span>
        <strong>{humanizeModelSite(textValue(summary.model_site)) || "-"}</strong>
        <small>{[
          textValue(summary.feature),
          summary.policy_call_index === null || summary.policy_call_index === undefined
            ? ""
            : `call ${summary.policy_call_index}`,
        ].filter(Boolean).join(" · ")}</small>
      </div>
    </section>
  );
}

function numberValue(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function textValue(value: unknown): string {
  return value === null || value === undefined ? "" : String(value);
}

function readoutTypeLabel(value: string): string {
  const labels: Record<string, string> = {
    probe_episode: "Episode probe result",
    probe_suite: "Episode probe result",
  };
  return labels[value] ?? "Episode probe result";
}

function EpisodeProbeStack({
  onProbeChange,
  probes,
  refs,
  selectedArtifactId,
}: {
  onProbeChange: (artifactId: string) => void;
  probes: EpisodeProbeSummary[];
  refs: ProbeLayerRef[];
  selectedArtifactId: string;
}) {
  const refsByArtifact = new Map(refs.map((ref) => [ref.artifactId, ref]));
  const sortedProbes = sortEpisodeProbesByInterestingness(probes);
  const visibleProbes = ensureSelectedProbeVisible(
    sortedProbes.slice(0, EPISODE_PROBE_RESULT_LIMIT),
    sortedProbes,
    selectedArtifactId,
  );
  return (
    <div className="episode-probe-stack" aria-label="Episode probes">
      {visibleProbes.map((probe) => {
        const ref = refsByArtifact.get(probe.artifact_id);
        const predicted = probe.episode_summary.predicted;
        const actual = probe.episode_summary.actual;
        const correct = probeDisplayedCorrect(predicted, actual, probe.episode_summary.correct);
        const tone = probeToneFromCorrect(correct, probe.available);
        const membership = probeEpisodeMembership(probe, probe.rows);
        const call = ref?.policyCall;
        const layer = ref?.layer;
        return (
          <button
            className={[
              probe.artifact_id === selectedArtifactId ? "active" : "",
              tone,
              membership.tone === "train" ? "low-interest" : "",
            ].filter(Boolean).join(" ")}
            key={probe.artifact_id}
            type="button"
            onClick={() => onProbeChange(probe.artifact_id)}
          >
            <span>{probe.target || probe.name}</span>
            <strong>{formatProbeComparison(predicted, actual)}</strong>
            <small>
              {[
                `conf ${formatMaybeNumber(probe.episode_summary.confidence)}`,
                membership.label,
                call === null || call === undefined ? "" : `call ${call}`,
                layer === null || layer === undefined ? "" : `L${formatLayerNumber(layer)}`,
              ].filter(Boolean).join(" · ")}
            </small>
          </button>
        );
      })}
    </div>
  );
}

function ProbeAuditPill({
  detail,
  label,
  tone,
  value,
}: {
  detail: string;
  label: string;
  tone?: EpisodeProbeMembership["tone"] | EpisodeProbeUsage["tone"];
  value: string;
}) {
  return (
    <div className={["episode-probe-audit-pill", tone ?? ""].filter(Boolean).join(" ")}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  );
}

function ObservationalComparisonPanel({
  comparisons,
  isError,
  isLoading,
  onOpenComparison,
}: {
  comparisons?: ObservationalComparisonsResponse;
  isError: boolean;
  isLoading: boolean;
  onOpenComparison: (traceId: string) => void;
}) {
  const candidates = comparisons?.candidates ?? [];
  const visible = candidates.slice(0, 4);
  if (isLoading && !comparisons) {
    return <div className="empty-state compact">Finding comparable episodes.</div>;
  }
  if (isError && !comparisons) {
    return <div className="empty-state compact">Comparison candidates unavailable.</div>;
  }
  if (!visible.length) {
    return <div className="empty-state compact">No comparable episodes found yet.</div>;
  }
  return (
    <section className="observational-comparison-panel" aria-label="Observational comparison candidates">
      <header>
        <div>
          <span>Compare Episodes</span>
          <strong>{comparisons?.probe_name || "Nearest traces"}</strong>
        </div>
        <small>observational · not causal</small>
      </header>
      <div className="observational-comparison-list">
        {visible.map((candidate) => (
          <button
            className={[
              "observational-comparison-row",
              comparisonCandidateTone(candidate),
            ].filter(Boolean).join(" ")}
            key={candidate.trace_id}
            type="button"
            onClick={() => onOpenComparison(candidate.trace_id)}
          >
            <span>{comparisonOutcomeLabel(candidate)}</span>
            <strong>{candidate.episode.task_id || candidate.trace_id}</strong>
            <small>{comparisonReasonLine(candidate)}</small>
            <em>{comparisonProbeLine(candidate)}</em>
          </button>
        ))}
      </div>
    </section>
  );
}

function comparisonCandidateTone(candidate: ObservationalComparisonCandidate): string {
  const split = String(candidate.metrics.candidate_split_category ?? candidate.probe?.split_category ?? "");
  if (split === "train") {
    return "train";
  }
  const correct = candidate.metrics.candidate_probe_correct ?? candidate.probe?.correct;
  if (correct === false) {
    return "incorrect";
  }
  if (correct === true) {
    return "correct";
  }
  return "unscored";
}

function comparisonOutcomeLabel(candidate: ObservationalComparisonCandidate): string {
  const source = candidate.metrics.source_outcome || "current";
  const next = candidate.metrics.candidate_outcome || candidate.episode.outcome || "comparison";
  return candidate.metrics.different_outcome ? `${source} -> ${next}` : String(next);
}

function comparisonReasonLine(candidate: ObservationalComparisonCandidate): string {
  const reasons = candidate.reasons.filter((reason) => reason !== "probe scored").slice(0, 3);
  const lengthDelta = candidate.metrics.length_delta;
  if (typeof lengthDelta === "number" && lengthDelta !== 0) {
    reasons.push(`${lengthDelta > 0 ? "+" : ""}${lengthDelta} frames`);
  }
  return reasons.join(" · ") || "nearest existing trace";
}

function comparisonProbeLine(candidate: ObservationalComparisonCandidate): string {
  const split = candidate.metrics.candidate_split_category ?? candidate.probe?.split_category ?? "unknown";
  const correct = candidate.metrics.candidate_probe_correct ?? candidate.probe?.correct;
  const confidence = candidate.metrics.candidate_confidence ?? candidate.probe?.confidence;
  return [
    split === "train" ? "trained here" : `${split} split`,
    formatProbeCorrect(correct),
    confidence === null || confidence === undefined ? "" : `conf ${formatMaybeNumber(confidence)}`,
  ].filter(Boolean).join(" · ");
}

function EpisodeProbeTimeline({
  bestRow,
  onJumpToPolicyCall,
  rows,
}: {
  bestRow?: EpisodeProbePrediction;
  onJumpToPolicyCall: (policyCallIndex: number) => void;
  rows: EpisodeProbePrediction[];
}) {
  return (
    <section className="episode-probe-timeline" aria-label="Probe prediction by policy call">
      <div className="episode-probe-plot-head">
        <strong>Probe across policy calls</strong>
        <small>{rows.length === 1 ? "1 policy call scored" : `${rows.length} policy calls scored`}</small>
      </div>
      <div className="episode-probe-timeline-row">
        {rows.map((row, index) => {
          const policyCall = episodeProbeNumber(row.policy_call_index);
          const bestPolicyCall = episodeProbeNumber(bestRow?.policy_call_index);
          const isBest =
            policyCall !== null &&
            bestPolicyCall === policyCall &&
            String(bestRow?.model_site_id ?? "") === String(row.model_site_id ?? "");
          const label = policyCall === null ? `prediction ${index + 1}` : `call ${policyCall}`;
          const confidence = episodeProbeNumber(row.confidence);
          const styleValue = `${(confidence === null ? 0.08 : Math.max(0.08, Math.min(1, confidence))) * 100}%`;
          return (
            <button
              className={[
                probeRowTone(row),
                isBest ? "active" : "",
              ].filter(Boolean).join(" ")}
              disabled={policyCall === null}
              key={`${row.model_site_id ?? row.feature ?? "probe"}-${policyCall ?? index}`}
              style={{ "--probe-confidence": styleValue } as CSSProperties}
              title={[
                label,
                row.timestep === null || row.timestep === undefined ? "" : `t=${row.timestep}`,
                `predicted ${formatProbeValue(row.predicted ?? row.prediction_value)}`,
                `actual ${formatProbeValue(row.actual)}`,
                `confidence ${formatMaybeNumber(row.confidence)}`,
              ].filter(Boolean).join(" / ")}
              type="button"
              onClick={() => {
                if (policyCall !== null) {
                  onJumpToPolicyCall(policyCall);
                }
              }}
            >
              <span>{label}</span>
              <i />
              <small>{formatMaybeNumber(row.confidence)}</small>
            </button>
          );
        })}
      </div>
    </section>
  );
}


function EpisodeProbeHeatmap({ cells }: { cells: EpisodeProbeCell[] }) {
  const layers = uniqueStrings(cells.map((cell) => cell.layer));
  const calls = uniqueStrings(cells.map((cell) => cell.policyCall));
  const range = {
    min: Math.min(...cells.map((cell) => cell.value)),
    max: Math.max(...cells.map((cell) => cell.value)),
  };
  return (
    <div className="episode-probe-heatmap-wrap">
      <div
        className="heatmap-grid episode-probe-heatmap"
        style={{ gridTemplateColumns: `64px repeat(${calls.length}, minmax(42px, 1fr))` }}
      >
        <div className="axis-corner">layer</div>
        {calls.map((call) => (
          <div className="axis-label" key={`call-${call}`}>
            call {call}
          </div>
        ))}
        {layers.map((layer) => (
          <ProbeHeatmapRow
            calls={calls}
            cells={cells}
            key={`layer-${layer}`}
            layer={layer}
            range={range}
          />
        ))}
      </div>
    </div>
  );
}

function ProbeHeatmapRow({
  calls,
  cells,
  layer,
  range,
}: {
  calls: string[];
  cells: EpisodeProbeCell[];
  layer: string;
  range: { min: number; max: number };
}) {
  return (
    <>
      <div className="axis-label y-label">{layer}</div>
      {calls.map((call) => {
        const cell = cells.find((item) => item.layer === layer && item.policyCall === call);
        return (
          <div
            className="heatmap-cell readonly"
            key={`${layer}-${call}`}
            style={{ background: episodeProbeColor(cell?.value ?? null, range) }}
            title={cell ? `${cell.count} prediction${cell.count === 1 ? "" : "s"}` : undefined}
          >
            {cell ? episodeProbeValueLabel(cell.value) : ""}
          </div>
        );
      })}
    </>
  );
}

function EpisodeProbePredictionTable({ rows }: { rows: EpisodeProbeSummary["rows"] }) {
  return (
    <div className="table-scroll">
      <table className="compact-table">
        <thead>
          <tr>
            <th>Feature</th>
            <th>Actual</th>
            <th>Predicted</th>
            <th>Conf.</th>
            <th>OK</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={`${row.feature ?? "feature"}-${index}`}>
              <td>{probeFeatureLabel(row, `layer ${row.layer} call ${row.policy_call_index}`)}</td>
              <td>{String(row.actual ?? row.target_value ?? "-")}</td>
              <td>{String(row.predicted ?? row.prediction_value ?? "-")}</td>
              <td>{formatMaybeNumber(row.confidence)}</td>
              <td>{row.correct === null || row.correct === undefined ? "-" : row.correct ? "yes" : "no"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
