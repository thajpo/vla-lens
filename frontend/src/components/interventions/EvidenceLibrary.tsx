import { Search } from "lucide-react";
import { useMemo, useState } from "react";
import type { InterventionRunRecord } from "../../types/interventions";
import { InterventionCard } from "./InterventionCard";
import { summarizeInterventionRun } from "./interventionDisplay";

type EvidenceLibraryProps = {
  runs: InterventionRunRecord[];
  selectedRunId?: string;
  onOpenRun: (runId: string) => void;
};

export function EvidenceLibrary({ runs, selectedRunId, onOpenRun }: EvidenceLibraryProps) {
  const [query, setQuery] = useState("");
  const visibleRuns = useMemo(() => filterRuns(runs, query), [query, runs]);
  return (
    <aside className="evidence-library">
      <header>
        <h2>Interventions</h2>
        <span>{runs.length}</span>
      </header>
      <label className="evidence-search">
        <Search size={15} />
        <input
          aria-label="Search intervention records"
          placeholder="Search interventions"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
      </label>
      <div className="evidence-list">
        {visibleRuns.length ? (
          visibleRuns.map((run) => (
            <InterventionCard
              key={run.run_id}
              run={run}
              selected={run.run_id === selectedRunId}
              onOpen={onOpenRun}
            />
          ))
        ) : (
          <p className="empty-state compact">No saved records.</p>
        )}
      </div>
    </aside>
  );
}

function filterRuns(runs: InterventionRunRecord[], query: string): InterventionRunRecord[] {
  const needle = query.trim().toLowerCase();
  if (!needle) {
    return runs;
  }
  return runs.filter((run) => {
    const summary = summarizeInterventionRun(run);
    return [
      run.run_id,
      summary.title,
      summary.status,
      summary.traceId,
      summary.policyCall,
      summary.target,
      summary.operator,
      summary.outcomeKind,
      summary.sourceLabel,
      ...summary.claimLabels,
    ]
      .join(" ")
      .toLowerCase()
      .includes(needle);
  });
}
