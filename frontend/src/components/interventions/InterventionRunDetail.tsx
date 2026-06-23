import { AlertTriangle, CheckCircle2, CircleDot, XCircle } from "lucide-react";
import type { InterventionRunRecord } from "../../types/interventions";
import { statusClass, summarizeInterventionRun } from "./interventionDisplay";

type InterventionRunDetailProps = {
  run?: InterventionRunRecord;
};

export function InterventionRunDetail({ run }: InterventionRunDetailProps) {
  if (!run) {
    return (
      <section className="intervention-detail empty">
        <p>Select a saved record.</p>
      </section>
    );
  }
  const summary = summarizeInterventionRun(run);
  const readouts = record(run.readouts);
  const trials = arrayValue(readouts.trials);
  const outcomes = arrayValue(readouts.outcomes);
  const controls = arrayValue(readouts.controls);
  return (
    <section className="intervention-detail">
      <header className="intervention-detail-header">
        <div>
          <span className={statusClass(summary.status)}>
            <StatusIcon status={summary.status} />
            {summary.status}
          </span>
          <h1>{summary.title}</h1>
          <p>
            {summary.traceId || "unknown trace"} · policy call {summary.policyCall}
          </p>
        </div>
        {summary.status === "inspected_only" ? (
          <div className="intervention-notice">Not causal evidence</div>
        ) : null}
      </header>

      <div className="intervention-detail-grid">
        <SummaryBlock label="Claim" value={summary.claimLabels.join(", ") || "unlabeled"} />
        <SummaryBlock label="Target" value={summary.target} />
        <SummaryBlock label="Source" value={summary.sourceLabel || "manual"} />
        <SummaryBlock label="Intervention" value={summary.operator || "operator pending"} />
        <SummaryBlock label="Outcome" value={summary.outcomeKind || "outcome pending"} />
        <SummaryBlock label="Trials" value={String(trials.length)} />
        <SummaryBlock label="Controls" value={String(controls.length)} />
      </div>

      <PayloadSection title="Context" payload={summary.context} />
      <PayloadSection title="Target" payload={run.target} />
      <PayloadSection title="Intervention" payload={run.intervention} />
      <PayloadSection title="Readouts" payload={run.readouts} />
      <PayloadSection title="Outcomes" payload={outcomes} />
      <PayloadSection title="Controls" payload={controls} />
      <PayloadSection title="Outputs" payload={run.outputs} />
      <PayloadSection title="Provenance" payload={run.provenance} />
    </section>
  );
}

function SummaryBlock({ label, value }: { label: string; value: string }) {
  return (
    <div className="intervention-summary-block">
      <span>{label}</span>
      <strong>{value || "-"}</strong>
    </div>
  );
}

function PayloadSection({ title, payload }: { title: string; payload: unknown }) {
  return (
    <details className="intervention-payload-section">
      <summary>{title}</summary>
      <pre>{JSON.stringify(payload, null, 2)}</pre>
    </details>
  );
}

function StatusIcon({ status }: { status: string }) {
  if (status === "ok") {
    return <CheckCircle2 size={14} />;
  }
  if (status === "partial") {
    return <CircleDot size={14} />;
  }
  if (status === "failed") {
    return <XCircle size={14} />;
  }
  return <AlertTriangle size={14} />;
}

function record(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function arrayValue(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}
