import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchEvidencePins } from "../api/dataset";
import { fetchInterventionRun, fetchInterventionRuns } from "../api/interventions";
import { EvidenceLibrary } from "../components/interventions/EvidenceLibrary";
import { InterventionLab } from "../components/interventions/InterventionLab";
import { InterventionRunDetail } from "../components/interventions/InterventionRunDetail";
import { InterventionTargetPicker } from "../components/interventions/InterventionTargetPicker";
import { evidencePinHash, evidencePinSummary } from "./evidencePinsModel";
import type { InterventionLabSeed } from "../types/interventions";

type InterventionsPageProps = {
  interventionSeed?: InterventionLabSeed;
  selectedRunId: string;
  onRunChange: (runId: string) => void;
};

export function InterventionsPage({
  interventionSeed,
  selectedRunId,
  onRunChange,
}: InterventionsPageProps) {
  const [targetDraft, setTargetDraft] = useState<InterventionLabSeed>(interventionSeed ?? {});
  const interventionSeedKey = [
    targetDraft.artifactId ?? "",
    targetDraft.traceId ?? "",
    targetDraft.policyCallIndex ?? "",
    targetDraft.modelSite ?? "",
    targetDraft.layer ?? "",
    targetDraft.feature ?? "",
    targetDraft.rankingMode ?? "",
    targetDraft.tokenSpace ?? "",
  ].join("|");
  const runs = useQuery({
    queryKey: ["intervention-runs"],
    queryFn: fetchInterventionRuns,
    staleTime: 15_000,
  });
  const pins = useQuery({
    queryKey: ["evidence-pins"],
    queryFn: fetchEvidencePins,
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
      <section className="interventions-workspace">
        <section className="interventions-center">
          <InterventionTargetPicker
            initialDraft={targetDraft}
            onDraftChange={setTargetDraft}
          />
          <section className="intervention-card">
            <div className="section-title">
              <span>Pinned probe evidence</span>
              <small>{pins.data?.total ?? 0} saved</small>
            </div>
            {(pins.data?.pins ?? []).map((pin) => (
              <div className="evidence-library-row" key={pin.pin_id}>
                <div>
                  <strong>{pin.label}</strong>
                  <small>{evidencePinSummary(pin)}</small>
                </div>
                <button type="button" onClick={() => { window.location.hash = evidencePinHash(pin); }}>
                  Open
                </button>
              </div>
            ))}
            {pins.isLoading ? <p className="app-message">Loading pinned evidence.</p> : null}
            {!pins.isLoading && !(pins.data?.pins ?? []).length ? <p className="app-message">No pinned probe evidence.</p> : null}
          </section>
        </section>
        <aside className="interventions-recipe">
          <InterventionLab
            initialDraft={targetDraft}
            key={interventionSeedKey || "manual-intervention"}
            onSavedRun={onRunChange}
          />
          {runs.isLoading ? <p className="app-message">Loading saved records.</p> : null}
          {runs.isError ? <p className="app-message">Unable to load saved records.</p> : null}
          {!records.length && !runs.isLoading ? (
            <p className="app-message">No saved intervention records.</p>
          ) : null}
          {detail.isError ? <p className="app-message">Unable to open selected record.</p> : null}
          {records.length ? <InterventionRunDetail run={selectedRun} /> : null}
        </aside>
      </section>
    </main>
  );
}

export { InterventionsPage as EvidencePage };
