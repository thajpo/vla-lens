import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchProbeStudies } from "../../api/dataset";
import type { ProbeStudy, ProbeStudyControl, ProbeStudyReadout } from "../../types/dataset";

type ProbeSuitePresetProps = {
  activeRunId: string;
  onRunChange: (runId: string) => void;
};

type RecordColumn = {
  description?: string;
  key: string;
  label: string;
  render?: (row: Record<string, unknown>) => ReactNode;
};

type ReadoutSortMode =
  | "study_order"
  | "balanced_desc"
  | "balanced_asc"
  | "train_gap_desc"
  | "train_gap_asc"
  | "calls_desc"
  | "target"
  | "layer";

const READOUT_SORT_OPTIONS: Array<{ label: string; value: ReadoutSortMode }> = [
  { label: "Study order", value: "study_order" },
  { label: "Eval score high", value: "balanced_desc" },
  { label: "Eval score low", value: "balanced_asc" },
  { label: "Train-eval gap high", value: "train_gap_desc" },
  { label: "Train-eval gap low", value: "train_gap_asc" },
  { label: "Most policy calls", value: "calls_desc" },
  { label: "Target", value: "target" },
  { label: "Layer", value: "layer" },
];

const TOOLTIP = {
  callsBefore: "How many policy calls remain before the future contact or motion event.",
  classPrecision: "Of the rows where the probe predicted this label, the fraction that were correct.",
  classRecall: "Of the rows whose true label was this value, the fraction the probe recovered.",
  confidence: "The probe probability assigned to its top prediction.",
  contactLead: "How far this policy call is from first contact with the target object.",
  control: "The null-control comparison used to check whether the probe beats shuffled labels.",
  evalScore: "Balanced accuracy on this evaluation split. This averages recall over labels, so rare labels matter.",
  evalSplit: "Which data split this row was evaluated on, such as validation or test.",
  futureEvent: "The future object event used for the lead-time slice, such as contact or motion.",
  input: "The activation features used as probe input.",
  labels: "Number of distinct target labels present in this split.",
  macroF1: "Average F1 score across labels, giving each label equal weight.",
  nullControls: "Selection-aware shuffled-label fits used to compare against memorization or leakage.",
  objective: "The model family or loss used to train the probe.",
  output: "The label space the probe predicts.",
  pValue: "Fraction of shuffled-label controls that matched or beat the real score. Lower is better.",
  phase: "Object-flow phase assigned to this policy call.",
  policyCall: "One model decision point that produces an action chunk.",
  policyCalls: "Number of unique policy calls included in this row.",
  prediction: "The target variable this study is trying to decode from activations.",
  probeId: "Stable short identifier for this trained probe. Use this when referring to a probe in discussion.",
  probePrediction: "The label predicted by the trained probe.",
  readLayer: "The model layer whose activation vector was used for this trained probe.",
  readouts: "Individual trained probes for a target, layer, and split.",
  realScore: "The real probe score before label shuffling.",
  rows: "Activation rows used for this probe after filters. Usually policy calls times selected layers.",
  shuffleMean: "Average score from shuffled-label control fits.",
  shuffleStd: "Standard deviation of shuffled-label control scores.",
  skipped: "Requested probes that were not trained because required labels or columns were missing.",
  sort: "How the currently visible trained probes are ordered.",
  study: "A research-question lens artifact. One family groups trained probes and controls for the same hypothesis.",
  target: "The object role or phase being decoded by the probe.",
  top3: "Whether the true label appears in the probe's top three predictions.",
  trainEvalGap: "Train balanced accuracy minus this split's balanced accuracy. Larger values suggest overfitting.",
  trainScore: "Balanced accuracy on the training split for the same trained probe.",
  trueLabel: "The label assigned by the episode-derived object-flow data.",
} as const;

export function ProbeSuitePreset({ activeRunId, onRunChange }: ProbeSuitePresetProps) {
  const studiesQuery = useQuery({
    queryKey: ["probe-studies"],
    queryFn: fetchProbeStudies,
    staleTime: 15_000,
  });
  const studies = studiesQuery.data?.studies ?? [];
  const selectedStudyId = studies.some((study) => probeStudyId(study) === activeRunId)
    ? activeRunId
    : probeStudyId(studies[0]);
  const study = studies.find((item) => probeStudyId(item) === selectedStudyId);

  const [targetFilter, setTargetFilter] = useState("all");
  const [splitFilter, setSplitFilter] = useState("heldout");
  const [layerFilter, setLayerFilter] = useState("all");
  const [sortMode, setSortMode] = useState<ReadoutSortMode>("study_order");
  const [selectedReadoutId, setSelectedReadoutId] = useState("");

  useEffect(() => {
    if (selectedStudyId && selectedStudyId !== activeRunId) {
      onRunChange(selectedStudyId);
    }
  }, [activeRunId, onRunChange, selectedStudyId]);

  const targetOptions = useMemo(
    () => uniqueStrings((study?.readouts ?? []).map((readout) => readout.target).filter(Boolean)),
    [study],
  );
  const splitOptions = useMemo(
    () => uniqueStrings((study?.readouts ?? []).map((readout) => readout.split ?? "").filter(Boolean)),
    [study],
  );
  const layerOptions = useMemo(
    () => uniqueStrings((study?.readouts ?? []).map((readout) => layerLabel(readout.layer)).filter(Boolean)),
    [study],
  );

  const activeTarget = targetFilter === "all" || targetOptions.includes(targetFilter) ? targetFilter : "all";
  const activeSplit = splitFilter;
  const activeLayer = layerFilter === "all" || layerOptions.includes(layerFilter) ? layerFilter : "all";
  const filteredReadouts = useMemo(
    () =>
      (study?.readouts ?? []).filter(
        (readout) =>
          (activeTarget === "all" || readout.target === activeTarget) &&
          splitMatches(readout, activeSplit) &&
          (activeLayer === "all" || layerLabel(readout.layer) === activeLayer),
      ),
    [activeLayer, activeSplit, activeTarget, study],
  );
  const sortedReadouts = useMemo(
    () => sortReadouts(filteredReadouts, sortMode),
    [filteredReadouts, sortMode],
  );
  const selectedReadout = useMemo(
    () =>
      sortedReadouts.find((readout) => readout.readout_id === selectedReadoutId) ??
      chooseDefaultReadout(sortedReadouts),
    [selectedReadoutId, sortedReadouts],
  );

  useEffect(() => {
    if (selectedReadout && selectedReadout.readout_id !== selectedReadoutId) {
      setSelectedReadoutId(selectedReadout.readout_id);
    }
  }, [selectedReadout?.readout_id, selectedReadoutId]);

  if (studiesQuery.isLoading) {
    return <div className="app-message">Loading probe studies...</div>;
  }
  if (studiesQuery.isError) {
    return <div className="app-message">Probe studies could not be loaded.</div>;
  }
  if (!studies.length || !study) {
    return (
      <div className="workflow-empty">
        <h1>Probe studies</h1>
        <p>No probe-suite study is registered yet.</p>
      </div>
    );
  }

  return (
    <div className="probe-studies-page">
      <header className="probe-study-header">
        <div>
          <span className="probe-study-kicker">Probe family</span>
          <h1>{study.name}</h1>
          <p>{study.question_label || study.target || study.artifact_id}</p>
        </div>
        <label className="probe-study-select">
          <TooltipLabel label="Probe family" description={TOOLTIP.study} />
          <select value={selectedStudyId} onChange={(event) => onRunChange(event.target.value)}>
            {studies.map((item) => (
              <option key={probeStudyId(item)} value={probeStudyId(item)}>
                {item.name}
              </option>
            ))}
          </select>
        </label>
      </header>

      <section className="probe-study-spec-band">
        <SpecField
          description={TOOLTIP.prediction}
          label="Prediction"
          value={study.prediction || study.target || "Target"}
        />
        <SpecField description={TOOLTIP.input} label="Input" value={study.input || "Model activations"} />
        <SpecField description={TOOLTIP.output} label="Output" value={study.output || "Class label"} />
        <SpecField
          description={TOOLTIP.objective}
          label="Objective"
          value={study.objective || "Linear probe"}
        />
      </section>

      <section className="probe-study-count-band">
        <CountField
          description={TOOLTIP.readouts}
          label="Probes"
          value={formatCount(study.counts.readout_count)}
          detail={`${formatCount(study.counts.target_count)} targets, ${formatCount(study.counts.layer_count)} layers`}
        />
        <CountField
          description={TOOLTIP.policyCalls}
          label="Policy calls"
          value={formatCount(study.counts.policy_call_count)}
          detail={`${formatCount(study.counts.feature_rows)} activation rows`}
        />
        <CountField
          description="Episodes represented after this study's filters."
          label="Episodes"
          value={formatCount(study.counts.episode_count)}
          detail={`${formatCount(study.counts.class_count)} classes`}
        />
        <CountField
          description={TOOLTIP.nullControls}
          label="Null controls"
          value={formatCount(study.counts.null_run_count)}
          detail={`${formatCount(study.counts.null_eval_row_count)} eval rows`}
        />
        <CountField
          description={TOOLTIP.skipped}
          label="Skipped"
          value={formatCount(study.counts.skipped_readout_count)}
          detail={study.diagnostics_available ? "diagnostics loaded" : "artifact summary only"}
        />
      </section>

      <main className="probe-study-main">
        <section className="probe-readouts-section">
          <header className="probe-section-header">
            <div>
              <h2>Trained probes</h2>
              <span>
                {formatCount(sortedReadouts.length)} shown / {formatCount(study.readouts.length)} trained
              </span>
            </div>
            <div className="probe-readout-filters">
              <label>
                <TooltipLabel label="Target" description={TOOLTIP.target} />
                <select value={activeTarget} onChange={(event) => setTargetFilter(event.target.value)}>
                  <option value="all">All targets</option>
                  {targetOptions.map((target) => (
                    <option key={target} value={target}>
                      {formatTarget(target)}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <TooltipLabel label="Split" description={TOOLTIP.evalSplit} />
                <select value={activeSplit} onChange={(event) => setSplitFilter(event.target.value)}>
                  <option value="heldout">Heldout</option>
                  <option value="all">All splits</option>
                  {splitOptions.map((split) => (
                    <option key={split} value={split}>
                      {formatSplit(split)}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <TooltipLabel label="Layer" description={TOOLTIP.readLayer} />
                <select value={activeLayer} onChange={(event) => setLayerFilter(event.target.value)}>
                  <option value="all">All layers</option>
                  {layerOptions.map((layer) => (
                    <option key={layer} value={layer}>
                      {layer}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <TooltipLabel label="Sort" description={TOOLTIP.sort} />
                <select
                  value={sortMode}
                  onChange={(event) => setSortMode(event.target.value as ReadoutSortMode)}
                >
                  {READOUT_SORT_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </header>
          <ReadoutTable
            readouts={sortedReadouts}
            selectedReadoutId={selectedReadout?.readout_id ?? ""}
            study={study}
            onSelect={setSelectedReadoutId}
          />
        </section>

        <ReadoutInspector study={study} readout={selectedReadout} />
      </main>

      <section className="probe-study-lower-grid">
        <ControlSection controls={study.controls} skippedReadouts={study.skipped_readouts} />
        <ErrorExampleSection study={study} readout={selectedReadout} />
      </section>
    </div>
  );
}

function ReadoutTable({
  readouts,
  selectedReadoutId,
  study,
  onSelect,
}: {
  readouts: ProbeStudyReadout[];
  selectedReadoutId: string;
  study: ProbeStudy;
  onSelect: (readoutId: string) => void;
}) {
  if (!readouts.length) {
    return <div className="empty-state compact">No trained probes match these filters.</div>;
  }
  return (
    <div className="table-scroll probe-readout-table-wrap">
      <table className="compact-table probe-readout-table">
        <thead>
          <tr>
            <th><TooltipLabel label="Probe ID" description={TOOLTIP.probeId} /></th>
            <th><TooltipLabel label="Target" description={TOOLTIP.target} /></th>
            <th><TooltipLabel label="Read layer" description={TOOLTIP.readLayer} /></th>
            <th><TooltipLabel label="Eval split" description={TOOLTIP.evalSplit} /></th>
            <th><TooltipLabel label="Eval score" description={TOOLTIP.evalScore} /></th>
            <th><TooltipLabel label="Macro F1" description={TOOLTIP.macroF1} /></th>
            <th><TooltipLabel label="Top-3 acc" description={TOOLTIP.top3} /></th>
            <th><TooltipLabel label="Train-eval gap" description={TOOLTIP.trainEvalGap} /></th>
            <th><TooltipLabel label="Policy calls" description={TOOLTIP.policyCalls} /></th>
            <th><TooltipLabel label="Labels" description={TOOLTIP.labels} /></th>
          </tr>
        </thead>
        <tbody>
          {readouts.map((readout) => (
            <tr
              className={readout.readout_id === selectedReadoutId ? "selectable-row active" : "selectable-row"}
              key={readout.readout_id}
            >
              <td>
                <code className="probe-id-token" title={readout.readout_id}>
                  {trainedProbeDisplayId(readout, study)}
                </code>
              </td>
              <td>
                <button className="probe-readout-select" onClick={() => onSelect(readout.readout_id)}>
                  <strong>{formatTarget(readout.target)}</strong>
                  <small>{readoutRoleLabel(readout, study)}</small>
                </button>
              </td>
              <td>{layerLabel(readout.layer)}</td>
              <td>{formatSplit(readout.split)}</td>
              <MetricCell value={readout.balanced_accuracy} />
              <MetricCell value={readout.macro_f1} />
              <MetricCell value={readout.top3_accuracy} />
              <MetricCell value={readout.train_gap_balanced_accuracy} invert />
              <td>{formatCount(readout.policy_call_count)}</td>
              <td>{formatCount(readout.class_count)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ReadoutInspector({
  study,
  readout,
}: {
  study: ProbeStudy;
  readout?: ProbeStudyReadout;
}) {
  if (!readout) {
    return <aside className="probe-readout-inspector empty-state">Select a trained probe.</aside>;
  }
  const leadRows = matchingRecords(study.lead_time, readout).slice(0, 12);
  const classRows = matchingRecords(study.per_class, readout)
    .sort((left, right) => recordNumber(right, "policy_call_support") - recordNumber(left, "policy_call_support"))
    .slice(0, 10);
  const confusionRows = matchingRecords(study.confusion, readout)
    .sort((left, right) => {
      const leftWrong = recordString(left, "actual") !== recordString(left, "predicted") ? 1 : 0;
      const rightWrong = recordString(right, "actual") !== recordString(right, "predicted") ? 1 : 0;
      return rightWrong - leftWrong || recordNumber(right, "policy_call_count") - recordNumber(left, "policy_call_count");
    })
    .slice(0, 12);

  return (
    <aside className="probe-readout-inspector">
      <header>
        <div>
          <h2>Selected probe</h2>
          <span>
            <code className="probe-id-token" title={readout.readout_id}>
              {trainedProbeDisplayId(readout, study)}
            </code>
            {" · "}
            {formatTarget(readout.target)} · layer {layerLabel(readout.layer)} · {formatSplit(readout.split)}
          </span>
        </div>
      </header>
      <div className="probe-inspector-metrics">
        <MetricBlock
          description={TOOLTIP.evalScore}
          label="Eval score"
          value={formatMetric(readout.balanced_accuracy)}
        />
        <MetricBlock description={TOOLTIP.macroF1} label="Macro F1" value={formatMetric(readout.macro_f1)} />
        <MetricBlock description={TOOLTIP.top3} label="Top-3 acc" value={formatMetric(readout.top3_accuracy)} />
        <MetricBlock
          description={TOOLTIP.trainEvalGap}
          label="Train-eval gap"
          value={formatMetric(readout.train_gap_balanced_accuracy)}
        />
      </div>
      <dl className="probe-readout-facts">
        <div>
          <dt><TooltipLabel label="Probe ID" description={TOOLTIP.probeId} /></dt>
          <dd>
            <code className="probe-id-token" title={readout.readout_id}>
              {trainedProbeDisplayId(readout, study)}
            </code>
          </dd>
        </div>
        <div>
          <dt><TooltipLabel label="Train score" description={TOOLTIP.trainScore} /></dt>
          <dd>{formatMetric(readout.train_balanced_accuracy)}</dd>
        </div>
        <div>
          <dt><TooltipLabel label="Policy calls" description={TOOLTIP.policyCalls} /></dt>
          <dd>{formatCount(readout.policy_call_count)}</dd>
        </div>
        <div>
          <dt><TooltipLabel label="Rows" description={TOOLTIP.rows} /></dt>
          <dd>{formatCount(readout.row_count)}</dd>
        </div>
        <div>
          <dt><TooltipLabel label="Labels" description={TOOLTIP.labels} /></dt>
          <dd>{formatCount(readout.class_count)}</dd>
        </div>
      </dl>
      <MiniTable
        columns={[
          { description: TOOLTIP.futureEvent, key: "lead_kind", label: "Future event" },
          { description: TOOLTIP.callsBefore, key: "lead_bucket", label: "Calls before" },
          {
            description: TOOLTIP.evalScore,
            key: "balanced_accuracy",
            label: "Eval score",
            render: (row) => formatMetric(recordNumberOrNull(row, "balanced_accuracy")),
          },
          {
            description: TOOLTIP.policyCalls,
            key: "policy_call_count",
            label: "Policy calls",
            render: (row) => formatCount(recordNumberOrNull(row, "policy_call_count")),
          },
        ]}
        empty="No lead-time slice for this probe."
        rows={leadRows}
        title="Lead time"
      />
      <MiniTable
        columns={[
          { description: TOOLTIP.trueLabel, key: "class", label: "True label" },
          {
            description: TOOLTIP.policyCalls,
            key: "policy_call_support",
            label: "Policy calls",
            render: (row) => formatCount(recordNumberOrNull(row, "policy_call_support")),
          },
          { description: TOOLTIP.classRecall, key: "recall", label: "Recall", render: (row) => formatMetric(recordNumberOrNull(row, "recall")) },
          {
            description: TOOLTIP.classPrecision,
            key: "precision",
            label: "Precision",
            render: (row) => formatMetric(recordNumberOrNull(row, "precision")),
          },
        ]}
        empty="No per-class metrics for this probe."
        rows={classRows}
        title="Class behavior"
      />
      <MiniTable
        columns={[
          { description: TOOLTIP.trueLabel, key: "actual", label: "True label" },
          { description: TOOLTIP.probePrediction, key: "predicted", label: "Probe prediction" },
          {
            description: TOOLTIP.policyCalls,
            key: "policy_call_count",
            label: "Policy calls",
            render: (row) => formatCount(recordNumberOrNull(row, "policy_call_count")),
          },
        ]}
        empty="No confusion rows for this probe."
        rows={confusionRows}
        title="Confusions"
      />
    </aside>
  );
}

function ControlSection({
  controls,
  skippedReadouts,
}: {
  controls: ProbeStudyControl[];
  skippedReadouts: ProbeStudyReadout[];
}) {
  return (
    <section className="probe-study-support-section">
      <header className="probe-section-header compact">
        <div>
          <h2>Controls</h2>
          <span>{formatCount(controls.length)} null summaries</span>
        </div>
      </header>
      <MiniTable
        columns={[
          { description: TOOLTIP.control, key: "label", label: "Control" },
          {
            description: TOOLTIP.evalSplit,
            key: "split",
            label: "Eval split",
            render: (row) => formatSplit(recordString(row, "split")),
          },
          { description: TOOLTIP.realScore, key: "real_score", label: "Real score", render: (row) => formatMetric(recordNumberOrNull(row, "real_score")) },
          {
            description: TOOLTIP.shuffleMean,
            key: "null_score_mean",
            label: "Shuffle mean",
            render: (row) => formatMetric(recordNumberOrNull(row, "null_score_mean")),
          },
          {
            description: TOOLTIP.shuffleStd,
            key: "null_score_std",
            label: "Shuffle std",
            render: (row) => formatMetric(recordNumberOrNull(row, "null_score_std")),
          },
          { description: TOOLTIP.pValue, key: "p_value", label: "p-value", render: (row) => formatMetric(recordNumberOrNull(row, "p_value")) },
          { description: "Number of shuffled-label fits.", key: "runs", label: "Runs", render: (row) => formatCount(recordNumberOrNull(row, "runs")) },
        ]}
        empty="No null controls saved for this study."
        rows={controls as unknown as Record<string, unknown>[]}
      />
      {skippedReadouts.length ? (
        <div className="probe-skipped-readouts">
          <h3>Skipped probes</h3>
          <MiniTable
            columns={[
              {
                description: TOOLTIP.target,
                key: "target",
                label: "Target",
                render: (row) => formatTarget(recordString(row, "target")),
              },
              { description: "Why this requested probe was not trained.", key: "reason", label: "Reason" },
            ]}
            empty="No skipped probes."
            rows={skippedReadouts as unknown as Record<string, unknown>[]}
          />
        </div>
      ) : null}
    </section>
  );
}

function ErrorExampleSection({
  study,
  readout,
}: {
  study: ProbeStudy;
  readout?: ProbeStudyReadout;
}) {
  const rows = readout?.is_primary_target
    ? matchingRecords(study.error_examples, readout).slice(0, 40)
    : [];
  return (
    <section className="probe-study-support-section">
      <header className="probe-section-header compact">
        <div>
          <h2>Error browser</h2>
          <span>{readout ? `${formatTarget(readout.target)} · layer ${layerLabel(readout.layer)}` : "No probe selected"}</span>
        </div>
      </header>
      <MiniTable
        columns={[
          {
            description: "Episode containing this probe row.",
            key: "trace_id",
            label: "Episode",
            render: (row) => (
              <a className="probe-episode-open" href={episodeHref(study, row)}>
                {shortTrace(recordString(row, "trace_id"))}
              </a>
            ),
          },
          {
            description: TOOLTIP.policyCall,
            key: "policy_call_index",
            label: "Policy call",
            render: (row) => formatCount(recordNumberOrNull(row, "policy_call_index")),
          },
          { description: TOOLTIP.trueLabel, key: "actual", label: "True label" },
          { description: TOOLTIP.probePrediction, key: "predicted", label: "Probe prediction" },
          { description: TOOLTIP.confidence, key: "confidence", label: "Confidence", render: (row) => formatMetric(recordNumberOrNull(row, "confidence")) },
          { description: TOOLTIP.phase, key: "task_phase", label: "Task phase" },
          { description: TOOLTIP.contactLead, key: "contact_lead_bucket", label: "Contact lead", render: (row) => formatBucket(recordString(row, "contact_lead_bucket")) },
          { description: "Object-flow events after this policy call.", key: "events_after", label: "Next events" },
        ]}
        empty={readout?.is_primary_target === false ? "Error rows are saved for the primary target only." : "No error rows for this probe."}
        rows={rows}
      />
    </section>
  );
}

function MiniTable({
  columns,
  empty,
  rows,
  title,
}: {
  columns: RecordColumn[];
  empty: string;
  rows: Record<string, unknown>[];
  title?: string;
}) {
  return (
    <div className="probe-mini-table">
      {title ? <h3>{title}</h3> : null}
      {!rows.length ? <div className="empty-state compact">{empty}</div> : null}
      {rows.length ? (
        <div className="table-scroll">
          <table className="compact-table">
            <thead>
              <tr>
                {columns.map((column) => (
                  <th key={column.key}>
                    <TooltipLabel label={column.label} description={column.description} />
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, index) => (
                <tr key={`${recordString(row, "trace_id") || recordString(row, "class") || index}-${index}`}>
                  {columns.map((column) => (
                    <td key={column.key}>{column.render ? column.render(row) : formatRecordValue(row[column.key])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  );
}

function SpecField({
  description,
  label,
  value,
}: {
  description?: string;
  label: string;
  value: string;
}) {
  return (
    <div className="probe-spec-field">
      <TooltipLabel label={label} description={description} />
      <strong>{value}</strong>
    </div>
  );
}

function CountField({
  description,
  label,
  value,
  detail,
}: {
  description?: string;
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <div className="probe-count-field">
      <TooltipLabel label={label} description={description} />
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  );
}

function MetricBlock({
  description,
  label,
  value,
}: {
  description?: string;
  label: string;
  value: string;
}) {
  return (
    <div>
      <TooltipLabel label={label} description={description} />
      <strong>{value}</strong>
    </div>
  );
}

function TooltipLabel({
  description,
  label,
}: {
  description?: string;
  label: string;
}) {
  if (!description) {
    return <span>{label}</span>;
  }
  return (
    <span
      aria-label={`${label}: ${description}`}
      className="probe-tooltip-label"
      data-tooltip={description}
      tabIndex={0}
      title={description}
    >
      {label}
    </span>
  );
}

function MetricCell({ value, invert = false }: { value?: number | null; invert?: boolean }) {
  const number = numericOrNull(value);
  const width = number === null ? 0 : Math.max(0, Math.min(100, number * 100));
  const tone = metricTone(number, invert);
  return (
    <td className="probe-metric-cell">
      <span className={`probe-score ${tone}`}>{formatMetric(number)}</span>
      <span className="probe-score-bar" aria-hidden="true">
        <span style={{ width: `${width}%` }} />
      </span>
    </td>
  );
}

function chooseDefaultReadout(readouts: ProbeStudyReadout[]): ProbeStudyReadout | undefined {
  return [...readouts].sort((left, right) => readoutRank(right) - readoutRank(left))[0];
}

function sortReadouts(
  readouts: ProbeStudyReadout[],
  sortMode: ReadoutSortMode,
): ProbeStudyReadout[] {
  const sorted = [...readouts];
  if (sortMode === "study_order") {
    return sorted;
  }
  return sorted.sort((left, right) => {
    if (sortMode === "balanced_desc") {
      return compareNumberDesc(left.balanced_accuracy, right.balanced_accuracy) || readoutTieBreak(left, right);
    }
    if (sortMode === "balanced_asc") {
      return compareNumberAsc(left.balanced_accuracy, right.balanced_accuracy) || readoutTieBreak(left, right);
    }
    if (sortMode === "train_gap_desc") {
      return compareNumberDesc(left.train_gap_balanced_accuracy, right.train_gap_balanced_accuracy) ||
        readoutTieBreak(left, right);
    }
    if (sortMode === "train_gap_asc") {
      return compareNumberAsc(left.train_gap_balanced_accuracy, right.train_gap_balanced_accuracy) ||
        readoutTieBreak(left, right);
    }
    if (sortMode === "calls_desc") {
      return compareNumberDesc(left.policy_call_count, right.policy_call_count) || readoutTieBreak(left, right);
    }
    if (sortMode === "target") {
      return compareText(left.target, right.target) ||
        compareText(left.split ?? "", right.split ?? "") ||
        compareLayer(left.layer, right.layer);
    }
    return compareLayer(left.layer, right.layer) ||
      compareText(left.target, right.target) ||
      compareText(left.split ?? "", right.split ?? "");
  });
}

function readoutRank(readout: ProbeStudyReadout): number {
  const splitBonus = readout.split_category === "test" ? 4 : readout.split_category === "validation" ? 3 : 0;
  const primaryBonus = readout.is_primary_target ? 2 : 0;
  const selectedBonus = readout.is_selected_layer ? 1 : 0;
  return (readout.balanced_accuracy ?? -1) + splitBonus + primaryBonus + selectedBonus;
}

function readoutTieBreak(left: ProbeStudyReadout, right: ProbeStudyReadout): number {
  return compareText(left.target, right.target) ||
    compareLayer(left.layer, right.layer) ||
    compareText(left.split ?? "", right.split ?? "");
}

function compareNumberDesc(left: unknown, right: unknown): number {
  return compareNullableNumbers(left, right, "desc");
}

function compareNumberAsc(left: unknown, right: unknown): number {
  return compareNullableNumbers(left, right, "asc");
}

function compareNullableNumbers(left: unknown, right: unknown, direction: "asc" | "desc"): number {
  const leftNumber = numericOrNull(left);
  const rightNumber = numericOrNull(right);
  if (leftNumber === null && rightNumber === null) {
    return 0;
  }
  if (leftNumber === null) {
    return 1;
  }
  if (rightNumber === null) {
    return -1;
  }
  return direction === "asc" ? leftNumber - rightNumber : rightNumber - leftNumber;
}

function compareLayer(left: ProbeStudyReadout["layer"], right: ProbeStudyReadout["layer"]): number {
  return compareNullableNumbers(left, right, "asc");
}

function compareText(left: string, right: string): number {
  return left.localeCompare(right, undefined, { numeric: true });
}

function matchingRecords(rows: Record<string, unknown>[], readout: ProbeStudyReadout): Record<string, unknown>[] {
  return rows.filter((row) => {
    const rowTarget = recordString(row, "target");
    if (rowTarget && rowTarget !== readout.target) {
      return false;
    }
    const rowLayer = recordString(row, "layer");
    if (rowLayer && layerLabel(readout.layer) !== rowLayer) {
      return false;
    }
    const rowSplit = recordString(row, "split");
    return !rowSplit || rowSplit === readout.split;
  });
}

function splitMatches(readout: ProbeStudyReadout, splitFilter: string): boolean {
  if (splitFilter === "all") {
    return true;
  }
  if (splitFilter === "heldout") {
    return readout.split_category !== "train";
  }
  return readout.split === splitFilter;
}

function readoutRoleLabel(readout: ProbeStudyReadout, study: ProbeStudy): string {
  if (readout.target === study.target) {
    return "family target";
  }
  return "related target";
}

function trainedProbeDisplayId(readout: ProbeStudyReadout, study: ProbeStudy): string {
  return `${studyArtifactCode(study)}-${readout.trained_probe_id || trainedProbeScopeId(readout)}`;
}

function trainedProbeScopeId(readout: ProbeStudyReadout): string {
  return [
    targetCode(readout.target),
    `L${idPiece(layerLabel(readout.layer), "ALL")}`,
    idPiece(readout.split || readout.split_category || "", "NOSPLIT"),
  ].join("-");
}

function studyArtifactCode(study: ProbeStudy): string {
  const text = String(study.artifact_id || study.study_id || "probe");
  const hash = text.match(/[a-f0-9]{6,}$/i)?.[0]?.slice(-6) ?? idPiece(text, "PROBE").slice(-6);
  return `P${hash.toUpperCase()}`;
}

function targetCode(target: string): string {
  const labels: Record<string, string> = {
    active_manipulated_object: "AMO",
    active_receptacle_object: "ARO",
    next_manipulated_object: "NMO",
    task_phase: "TPH",
  };
  if (labels[target]) {
    return labels[target];
  }
  const pieces = target.split("_").filter(Boolean);
  if (pieces.length >= 2) {
    return pieces.slice(0, 4).map((piece) => piece[0]?.toUpperCase() ?? "").join("");
  }
  return idPiece(target, "TARGET").slice(0, 10);
}

function idPiece(value: string, fallback: string): string {
  return value.trim().toUpperCase().replace(/[^A-Z0-9]+/g, "-").replace(/^-|-$/g, "") || fallback;
}

function episodeHref(study: ProbeStudy, row: Record<string, unknown>): string {
  const traceId = recordString(row, "trace_id");
  const params = new URLSearchParams();
  params.set("probe_id", study.artifact_id);
  const policyCall = recordNumberOrNull(row, "policy_call_index");
  const timestep = recordNumberOrNull(row, "timestep");
  const site = recordString(row, "model_site_id");
  if (policyCall !== null) {
    params.set("call", String(policyCall));
  }
  if (timestep !== null) {
    params.set("timestep", String(timestep));
  }
  if (site) {
    params.set("site", site);
  }
  return `#episode/${encodeURIComponent(traceId)}?${params.toString()}`;
}

function uniqueStrings(values: string[]): string[] {
  return Array.from(new Set(values)).sort((left, right) => left.localeCompare(right, undefined, { numeric: true }));
}

function probeStudyId(study: ProbeStudy | undefined): string {
  return study?.study_id || study?.artifact_id || "";
}

function layerLabel(value: ProbeStudyReadout["layer"]): string {
  return value === null || value === undefined || value === "" ? "all" : String(value);
}

function recordString(row: Record<string, unknown>, key: string): string {
  const value = row[key];
  return value === null || value === undefined ? "" : String(value);
}

function recordNumber(row: Record<string, unknown>, key: string): number {
  return recordNumberOrNull(row, key) ?? 0;
}

function recordNumberOrNull(row: Record<string, unknown>, key: string): number | null {
  return numericOrNull(row[key]);
}

function numericOrNull(value: unknown): number | null {
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) ? number : null;
}

function metricTone(value: number | null, invert: boolean): string {
  if (value === null) {
    return "missing";
  }
  const score = invert ? 1 - value : value;
  if (score >= 0.45) {
    return "good";
  }
  if (score >= 0.25) {
    return "warn";
  }
  return "bad";
}

function formatMetric(value: number | null | undefined): string {
  const number = numericOrNull(value);
  return number === null ? "-" : number.toFixed(3);
}

function formatCount(value: number | null | undefined): string {
  const number = numericOrNull(value);
  return number === null ? "-" : number.toLocaleString();
}

function formatTarget(value: string | null | undefined): string {
  if (!value) {
    return "-";
  }
  return value.replaceAll("_", " ");
}

function formatSplit(value: string | null | undefined): string {
  if (!value) {
    return "-";
  }
  return value
    .replaceAll("_", " ")
    .replace("heldout", "heldout")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatBucket(value: string): string {
  return value ? value.replaceAll("_", " ") : "-";
}

function shortTrace(value: string): string {
  if (!value) {
    return "-";
  }
  const parts = value.split("_");
  return parts.length > 4 ? parts.slice(-4).join("_") : value;
}

function formatRecordValue(value: unknown): string {
  const number = typeof value === "number" ? value : null;
  if (number !== null) {
    return Math.abs(number) < 10 && !Number.isInteger(number) ? number.toFixed(3) : number.toLocaleString();
  }
  const text = value === null || value === undefined ? "" : String(value);
  return text || "-";
}
