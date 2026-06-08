import type { ActivationSliceResponse, EpisodeLensView } from "../../types/dataset";
import { FeatureTable } from "./InspectorTables";
import { TOP_CHANNEL_COUNT_OPTIONS } from "./shared";
import { formatMaybeNumber, labelFromSnake } from "./formatters";
import {
  lensReadoutLine,
  probeSourceSitesFromLensView,
  probeSiteReadoutFromLensView,
  probeTrainingLine,
  type LensFeatureRow,
  type LensRankingMode,
} from "./episodeLensModel";

export function TopChannelPanel({
  activationSlice,
  activationSliceFetching,
  activationSlicePlaceholder,
  activationClipPercent,
  feature,
  lensRankingMode,
  lensRows,
  lensSiteReadout,
  onActivationClipPercentChange,
  onFeatureChange,
  onLensRankingModeChange,
  onTopChannelCountChange,
  selectedSiteHasFeatures,
  selectedSiteName,
  topChannelCount,
  topRows,
}: {
  activationSlice?: ActivationSliceResponse;
  activationSliceFetching: boolean;
  activationSlicePlaceholder: boolean;
  activationClipPercent: number;
  feature: number;
  lensRankingMode: LensRankingMode;
  lensRows: LensFeatureRow[];
  lensSiteReadout: ReturnType<typeof probeSiteReadoutFromLensView>;
  onActivationClipPercentChange: (clipPercent: number) => void;
  onFeatureChange: (feature: number) => void;
  onLensRankingModeChange: (mode: LensRankingMode) => void;
  onTopChannelCountChange: (count: number) => void;
  selectedSiteHasFeatures: boolean;
  selectedSiteName: string;
  topChannelCount: number;
  topRows: { index: number; value: number }[];
}) {
  if (!selectedSiteHasFeatures) {
    return null;
  }
  const lensMatchesSite = Boolean(
    lensSiteReadout?.model_site_id && lensSiteReadout.model_site_id === selectedSiteName,
  );
  const hasLensRanking = Boolean(lensSiteReadout && lensMatchesSite);
  const contributorUnavailable =
    lensRankingMode === "probe_contribution" &&
    hasLensRanking &&
    !lensSiteReadout?.probe_contribution_ranking_available;
  const rankingOptions = lensInspectorRankingsFromReadout(lensSiteReadout);
  const contributorOption = rankingOptions.find((option) => option.id === "probe_contributors");
  const rawOption = rankingOptions.find((option) => option.id === "raw_activations");
  const activeRows =
    hasLensRanking && lensRows.length
      ? lensRows
      : topRows;
  const activeTitle =
    hasLensRanking && lensRankingMode === "probe_contribution"
      ? "Probe-weighted features"
      : hasLensRanking
        ? "Raw feature activations"
        : "Top active features";
  const valueHeader =
    hasLensRanking && lensRankingMode === "probe_contribution"
      ? "Contribution"
      : "Activation";
  return (
    <div className="top-channel-panel">
      {hasLensRanking ? (
        <div className="lens-ranking-controls" aria-label="Feature ranking mode">
          <button
            className={lensRankingMode === "probe_contribution" ? "active" : ""}
            disabled={!contributorOption?.available}
            title={contributorOption?.unavailable_reason ?? "Rank by probe-weighted contribution"}
            type="button"
            onClick={() => onLensRankingModeChange("probe_contribution")}
          >
            {contributorOption?.label ?? "Probe contributors"}
          </button>
          <button
            className={lensRankingMode === "raw_activation" ? "active" : ""}
            disabled={!rawOption?.available}
            title={rawOption?.unavailable_reason ?? "Rank by raw activation magnitude"}
            type="button"
            onClick={() => onLensRankingModeChange("raw_activation")}
          >
            {rawOption?.label ?? "Raw activations"}
          </button>
        </div>
      ) : null}
      {contributorUnavailable ? (
        <div className="lens-readout-note">
          {lensSiteReadout?.feature_contributors_unavailable_reason ?? "Probe contributors unavailable."}
        </div>
      ) : null}
      <FeatureTable
        rows={activeRows}
        activeFeature={feature}
        title={activeTitle}
        indexHeader="Feature"
        indexPrefix="feature "
        loading={activationSliceFetching && !activeRows.length}
        reserveHeight
        updating={activationSliceFetching && Boolean(activeRows.length)}
        stale={activationSlicePlaceholder}
        onFeatureChange={onFeatureChange}
        clipPercent={hasLensRanking ? undefined : activationClipPercent}
        clip={hasLensRanking ? undefined : activationSlice?.clip}
        onClipPercentChange={hasLensRanking ? undefined : onActivationClipPercentChange}
        rowLimit={topChannelCount}
        rowLimitOptions={TOP_CHANNEL_COUNT_OPTIONS}
        onRowLimitChange={onTopChannelCountChange}
        valueHeader={valueHeader}
      />
    </div>
  );
}

export function LensCompactReadout({
  onJumpDefault,
  onSendToIntervention,
  view,
}: {
  onJumpDefault: () => void;
  onSendToIntervention: () => void;
  view?: EpisodeLensView;
}) {
  if (!view || view.family !== "probe_suite") {
    return null;
  }
  const readout = view.readout;
  const spec = view.lens.spec ?? {};
  const jumpAction = view.actions.find((action) => action.kind === "jump_to_lens_default");
  const interventionAction = view.actions.find((action) => action.kind === "send_to_intervention");
  const sourceSites = probeSourceSitesFromLensView(view);
  const visibleSources = sourceSites.filter((site) => site.trained).slice(0, 8);
  const selectedSources = sourceSites.filter((site) => site.selected || site.default || site.best);
  const trainingLine = probeTrainingLine(view);
  const prediction = String(spec.prediction ?? "this outcome");
  const input = String(spec.input ?? "model features");
  const output = String(spec.output ?? "False / True");
  const objective = String(spec.objective ?? "probe objective");
  const selectedSource = selectedSources[0];
  const sourceTitle = selectedSource?.label ?? selectedSource?.model_site_id ?? "";
  return (
    <section className="lens-compact-readout">
      <div className="section-title">
        <span>{view.lens.display_name}</span>
        <small>{lensReadoutLine(readout)}</small>
      </div>
      <div className="lens-spec-brief">
        <p>
          Predicts <strong>{prediction}</strong> from <strong>{input}</strong>.
        </p>
        <span>{objective}</span>
        <span>Output {output}</span>
      </div>
      {trainingLine ? <div className="lens-training-line">{trainingLine}</div> : null}
      {visibleSources.length ? (
        <div className="lens-source-sites" aria-label="Probe source sites">
          <strong>Probe layers</strong>
          {visibleSources.map((site) => (
            <span
              className={[
                site.selected ? "selected" : "",
                site.default ? "default" : "",
                site.best ? "best" : "",
              ].filter(Boolean).join(" ")}
              key={site.model_site_id}
              title={site.label ?? site.model_site_id}
            >
              {site.short_label ?? `L${site.layer ?? "-"}`}
            </span>
          ))}
        </div>
      ) : null}
      <div className="lens-readout-strip">
        <span>
          {readout?.predicted === undefined || readout?.predicted === null
            ? "Prediction -"
            : `Prediction ${String(readout.predicted)}`}
        </span>
        <span>
          {readout?.actual === undefined || readout?.actual === null
            ? "Actual -"
            : `Actual ${String(readout.actual)}`}
        </span>
        {readout?.confidence === undefined || readout?.confidence === null ? null : (
          <span>Confidence {formatMaybeNumber(readout.confidence)}</span>
        )}
        {readout?.split ? <span>{labelFromSnake(String(readout.split))} split</span> : null}
        {sourceTitle ? <span title={sourceTitle}>Site {selectedSource?.short_label ?? "selected"}</span> : null}
      </div>
      <div className="lens-action-row">
        <button
          disabled={!jumpAction?.enabled}
          title="Return the inspector to the model site and policy call this probe readout uses."
          type="button"
          onClick={onJumpDefault}
        >
          Use probe site
        </button>
        <button
          disabled={!interventionAction?.enabled}
          title="Seed the intervention workspace with the current probe site, call, and selected feature."
          type="button"
          onClick={onSendToIntervention}
        >
          Seed intervention
        </button>
      </div>
      {view.annotations.callouts.length ? (
        <div className="lens-callout-list">
          {view.annotations.callouts.slice(0, 3).map((callout, index) => (
            <div className="lens-readout-note" key={`${callout.severity}-${index}`}>
              {callout.text}
            </div>
          ))}
        </div>
      ) : null}
    </section>
  );
}

function lensInspectorRankingsFromReadout(
  siteReadout: ReturnType<typeof probeSiteReadoutFromLensView>,
) {
  return [
    {
      id: "probe_contributors",
      label: "Probe contributors",
      available: Boolean(siteReadout?.feature_contributors_available),
      unavailable_reason: siteReadout?.feature_contributors_unavailable_reason ?? null,
    },
    {
      id: "raw_activations",
      label: "Raw activations",
      available: Boolean(siteReadout?.raw_activation_ranking_available),
      unavailable_reason: siteReadout?.unavailable_reason ?? null,
    },
  ];
}
