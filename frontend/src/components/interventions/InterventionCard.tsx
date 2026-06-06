import { FileText } from "lucide-react";
import type { InterventionRunRecord } from "../../types/interventions";
import { statusClass, summarizeInterventionRun } from "./interventionDisplay";

type InterventionCardProps = {
  run: InterventionRunRecord;
  selected?: boolean;
  onOpen?: (runId: string) => void;
};

export function InterventionCard({ run, selected = false, onOpen }: InterventionCardProps) {
  const summary = summarizeInterventionRun(run);
  return (
    <button
      className={selected ? "intervention-card selected" : "intervention-card"}
      type="button"
      onClick={() => onOpen?.(run.run_id)}
    >
      <span className={statusClass(summary.status)}>{summary.status}</span>
      <span className="intervention-card-title">
        <FileText size={16} />
        {summary.title}
      </span>
      <span className="intervention-card-meta">
        {summary.traceId || "unknown trace"} · call {summary.policyCall}
      </span>
      <span className="intervention-card-meta">{summary.target}</span>
      <span className="intervention-card-meta">
        {summary.operator || "operator pending"} · {summary.outcomeKind || "outcome pending"}
      </span>
      <span className="intervention-label-row">
        {summary.claimLabels.length ? (
          summary.claimLabels.map((label) => (
            <span className="evidence-label" key={label}>
              {label}
            </span>
          ))
        ) : (
          <span className="evidence-label muted">unlabeled</span>
        )}
      </span>
    </button>
  );
}
