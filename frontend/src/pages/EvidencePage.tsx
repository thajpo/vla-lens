import { useEffect, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchInterventionRun, fetchInterventionRuns } from "../api/interventions";
import { EvidenceLibrary } from "../components/interventions/EvidenceLibrary";
import { InterventionLab } from "../components/interventions/InterventionLab";
import { InterventionRunDetail } from "../components/interventions/InterventionRunDetail";
import type { InterventionLabSeed } from "../types/interventions";

type EvidencePageProps = {
  interventionSeed?: InterventionLabSeed;
  selectedRunId: string;
  onRunChange: (runId: string) => void;
};

export function EvidencePage({ interventionSeed, selectedRunId, onRunChange }: EvidencePageProps) {
  const runs = useQuery({
    queryKey: ["intervention-runs"],
    queryFn: fetchInterventionRuns,
    staleTime: 15_000,
  });
  const records = useMemo(() => runs.data?.intervention_runs ?? [], [runs.data]);
  const activeRunId = selectedRunId || records[0]?.run_id || "";
  const detail = useQuery({
    queryKey: ["intervention-run", activeRunId],
    queryFn: () => fetchInterventionRun(activeRunId),
    enabled: Boolean(activeRunId),
    staleTime: 15_000,
  });
  const selectedRun =
    detail.data?.intervention_run ?? records.find((run) => run.run_id === activeRunId);

  useEffect(() => {
    if (!selectedRunId && records[0]?.run_id) {
      onRunChange(records[0].run_id);
    }
  }, [onRunChange, records, selectedRunId]);

  return (
    <main className="evidence-page">
      <EvidenceLibrary
        runs={records}
        selectedRunId={activeRunId}
        onOpenRun={onRunChange}
      />
      <section className="evidence-workspace">
        <InterventionLab initialDraft={interventionSeed} onSavedRun={onRunChange} />
        {runs.isLoading ? <p className="app-message">Loading saved records.</p> : null}
        {runs.isError ? <p className="app-message">Unable to load saved records.</p> : null}
        {!records.length && !runs.isLoading ? (
          <p className="app-message">No saved intervention records.</p>
        ) : null}
        {detail.isError ? <p className="app-message">Unable to open selected record.</p> : null}
        {records.length ? <InterventionRunDetail run={selectedRun} /> : null}
      </section>
    </main>
  );
}
