import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import type { ActivationSliceResponse, EpisodeLensView } from "../../types/dataset";
import { researchCopy } from "../../copy/researchCopy";
import type { ProbeEvidenceBundle, ResearchSelectionState } from "../../types/probeEvidence";
import { saveEvidencePin } from "../../api/dataset";
import { FeatureTable } from "./InspectorTables";
import { TOP_CHANNEL_COUNT_OPTIONS } from "./shared";
import { formatMaybeNumber, labelFromSnake } from "./formatters";
import { probeLensViewDisplaySpec } from "../probeDisplayCopy";
import {
  lensReadoutLine,
  probeEvidenceCallouts,
  probeEvidenceReadout,
  probeEvidencePinPayload,
  probeEvidenceSpec,
  probeEvidenceTimelineMarks,
  probeSourceSitesFromLensView,
  probeSourceSitesFromEvidenceBundle,
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
  feature,
  isError = false,
  isLoading = false,
  lensRequested = false,
  onJumpDefault,
  onSendToIntervention,
  probeEvidenceBundle,
  probeEvidenceSelection,
  selectedSiteName,
  view,
}: {
  feature?: number;
  isError?: boolean;
  isLoading?: boolean;
  lensRequested?: boolean;
  onJumpDefault: () => void;
  onSendToIntervention: () => void;
  probeEvidenceBundle?: ProbeEvidenceBundle;
  probeEvidenceSelection?: ResearchSelectionState | null;
  selectedSiteName?: string;
  view?: EpisodeLensView;
}) {
  const queryClient = useQueryClient();
  const [pinStatus, setPinStatus] = useState<"idle" | "saving" | "saved" | "error">("idle");
  if (lensRequested && isLoading && !probeEvidenceBundle && !view) {
    return (
      <LensContextState
        tone="loading"
        title="Loading probe lens"
        detail="Fetching the selected probe evidence for this episode."
      />
    );
  }
  if (lensRequested && isError && !probeEvidenceBundle && !view) {
    return (
      <LensContextState
        tone="error"
        title="Probe lens unavailable"
        detail={researchCopy.unavailable.rawActivationFallback}
      />
    );
  }
  if (lensRequested && !probeEvidenceBundle && view?.available === false) {
    return (
      <LensContextState
        tone="warning"
        title="Probe lens unavailable"
        detail={researchCopy.unavailable.episodeLens}
      />
    );
  }
  if (probeEvidenceBundle && probeEvidenceSelection) {
    const readout = probeEvidenceReadout(probeEvidenceBundle, probeEvidenceSelection);
    const spec = probeEvidenceSpec(probeEvidenceBundle);
    const sourceSites = probeSourceSitesFromEvidenceBundle(probeEvidenceBundle, probeEvidenceSelection);
    const visibleSources = sourceSites.filter((site) => site.trained).slice(0, 8);
    const selectedSource = sourceSites.find((site) => site.selected || site.default || site.best);
    const sourceTitle = selectedSource?.label ?? selectedSource?.model_site_id ?? "";
    const callouts = probeEvidenceCallouts(probeEvidenceBundle);
    const topMoments = probeEvidenceTimelineMarks(probeEvidenceBundle, probeEvidenceSelection)
      .filter((mark) => mark.kind === "ranked_moment")
      .slice(0, 3);
    return (
      <section className="lens-compact-readout">
        <div className="section-title">
          <span>{probeEvidenceBundle.artifact.name}</span>
          <small>{lensReadoutLine(readout)}</small>
        </div>
        <div className="lens-spec-brief">
          <p>
            Predicts <strong>{spec.prediction}</strong> from <strong>{spec.input}</strong>.
          </p>
          <span>{spec.objective}</span>
          <span>Output {spec.output}</span>
        </div>
        {visibleSources.length ? (
          <div className="lens-source-sites" aria-label="Probe read sources">
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
            {readout.predicted === undefined || readout.predicted === null
              ? "Prediction -"
              : `Prediction ${String(readout.predicted)}`}
          </span>
          <span>
            {readout.actual === undefined || readout.actual === null
              ? "Actual -"
              : `Actual ${String(readout.actual)}`}
          </span>
          {readout.confidence === undefined || readout.confidence === null ? null : (
            <span>Confidence {formatMaybeNumber(readout.confidence)}</span>
          )}
          <span>Correct {readout.correct === null || readout.correct === undefined ? "-" : readout.correct ? "yes" : "no"}</span>
          <span>Policy call {probeEvidenceSelection.policy_call ?? readout.policy_call_index ?? "-"}</span>
          {sourceTitle ? <span title={sourceTitle}>Model site {selectedSource?.short_label ?? sourceTitle}</span> : null}
        </div>
        {topMoments.length ? (
          <div className="lens-moment-strip" aria-label="Top moments in episode">
            <strong>Episode moments</strong>
            {topMoments.map((mark, index) => (
              <span key={`${mark.kind}-${mark.policy_call_index ?? ""}-${mark.timestep ?? ""}-${index}`}>
                {mark.label ?? "moment"} · {mark.policy_call_index === null || mark.policy_call_index === undefined
                  ? `timestep ${mark.timestep ?? "-"}`
                  : `call ${mark.policy_call_index}`}
              </span>
            ))}
          </div>
        ) : null}
        <div className="lens-action-row">
          <button type="button" onClick={onJumpDefault}>
            Use probe site
          </button>
          <button
            disabled={pinStatus === "saving"}
            type="button"
            onClick={() => {
              setPinStatus("saving");
              saveEvidencePin(probeEvidencePinPayload(probeEvidenceBundle, probeEvidenceSelection, "", {
                feature,
                modelSiteId: selectedSiteName,
              }))
                .then(() => {
                  setPinStatus("saved");
                  queryClient.invalidateQueries({ queryKey: ["evidence-pins"] });
                })
                .catch(() => setPinStatus("error"));
            }}
          >
            {pinStatus === "saved" ? "Pinned" : "Pin evidence"}
          </button>
        </div>
        {pinStatus === "error" ? <div className="lens-readout-note">Unable to pin evidence.</div> : null}
        {callouts.length ? (
          <div className="lens-callout-list">
            {callouts.map((text, index) => (
              <div className="lens-readout-note" key={`${text}-${index}`}>
                {text}
              </div>
            ))}
          </div>
        ) : null}
      </section>
    );
  }
  if (!view || view.family !== "probe_suite") {
    if (lensRequested) {
      return (
        <LensContextState
          tone="warning"
          title="Probe lens selected"
          detail="No probe evidence is available for this episode yet."
        />
      );
    }
    return null;
  }
  const readout = view.readout;
  const spec = probeLensViewDisplaySpec(view);
  const jumpAction = view.actions.find((action) => action.kind === "jump_to_lens_default");
  const interventionAction = view.actions.find((action) => action.kind === "send_to_intervention");
  const sourceSites = probeSourceSitesFromLensView(view);
  const visibleSources = sourceSites.filter((site) => site.trained).slice(0, 8);
  const selectedSources = sourceSites.filter((site) => site.selected || site.default || site.best);
  const trainingLine = probeTrainingLine(view);
  const prediction = spec.prediction.value;
  const input = spec.input.value;
  const output = spec.output.value;
  const objective = spec.objective.value;
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
        <div className="lens-source-sites" aria-label="Probe read sources">
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
          title="Return the inspector to the layer and policy call this probe readout uses."
          type="button"
          onClick={onJumpDefault}
        >
          Use probe site
        </button>
        {interventionAction?.enabled ? (
          <button
            title="Seed the intervention workspace with the current probe site, call, and selected feature."
            type="button"
            onClick={onSendToIntervention}
          >
            Seed intervention
          </button>
        ) : null}
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

function LensContextState({
  detail,
  title,
  tone,
}: {
  detail: string;
  title: string;
  tone: "error" | "loading" | "warning";
}) {
  return (
    <section className={`lens-compact-readout lens-context-state ${tone}`}>
      <div className="section-title">
        <span>Selected probe lens</span>
        <small>{tone === "loading" ? "Loading" : tone === "error" ? "Unavailable" : "No evidence"}</small>
      </div>
      <div className="lens-readout-note">
        <strong>{title}</strong>
        <p>{detail}</p>
      </div>
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
