import { ChevronDown, ChevronRight, FlaskConical } from "lucide-react";
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchResearchRun, fetchResearchRuns } from "../api/researchRuns";
import type { ResearchRunRecord } from "../types/researchRuns";
import {
  formatResearchProgress,
  formatResearchResult,
  formatResearchStage,
  groupResearchRuns,
  researchArtifactDestination,
} from "./researchRunsModel";

type ResearchRunsPageProps = {
  selectedRunId: string;
  onOpenIntervention: (runId: string) => void;
  onOpenProbe: (artifactId: string) => void;
  onRunChange: (runId: string) => void;
};

export function ResearchRunsPage({
  selectedRunId,
  onOpenIntervention,
  onOpenProbe,
  onRunChange,
}: ResearchRunsPageProps) {
  const runsQuery = useQuery({
    queryKey: ["research-runs"],
    queryFn: fetchResearchRuns,
    refetchInterval: 5_000,
  });
  const runs = useMemo(() => runsQuery.data?.research_runs ?? [], [runsQuery.data]);
  const groups = useMemo(() => groupResearchRuns(runs), [runs]);
  const activeRunId = selectedRunId || runs[0]?.run_id || "";
  const detailQuery = useQuery({
    queryKey: ["research-run", activeRunId],
    queryFn: () => fetchResearchRun(activeRunId),
    enabled: Boolean(activeRunId),
    refetchInterval: 5_000,
  });
  const selected = detailQuery.data?.research_run ?? runs.find((run) => run.run_id === activeRunId);
  const [collapsed, setCollapsed] = useState<Set<string>>(() => new Set());

  if (runsQuery.isLoading) {
    return <div className="app-message">Loading research runs.</div>;
  }
  if (runsQuery.isError) {
    return <div className="app-message">Research runs could not be loaded.</div>;
  }

  return (
    <main className="research-runs-page">
      <header className="research-runs-header">
        <div>
          <span>Research</span>
          <h1>Research runs</h1>
          <p>Experiments, their controls, and what they found.</p>
        </div>
        <strong>{runs.length} runs</strong>
      </header>

      {!runs.length ? (
        <section className="research-runs-empty">
          <FlaskConical size={22} />
          <h2>No research runs yet</h2>
          <p>Campaigns and experiments will appear here as agents save lifecycle records.</p>
        </section>
      ) : (
        <div className="research-runs-table-wrap">
          <table className="research-runs-table">
            <thead>
              <tr>
                <th>Question</th>
                <th>Stage / status</th>
                <th>Progress</th>
                <th>Result vs baseline</th>
                <th>Updated</th>
              </tr>
            </thead>
            <tbody>
              {groups.flatMap(({ run, children }) => {
                const isCollapsed = collapsed.has(run.run_id);
                return [
                  <ResearchRunRow
                    childCount={children.length}
                    collapsed={isCollapsed}
                    key={run.run_id}
                    run={run}
                    selected={run.run_id === activeRunId}
                    onSelect={onRunChange}
                    onToggle={() => setCollapsed(toggleSet(collapsed, run.run_id))}
                  />,
                  ...(!isCollapsed
                    ? children.map((child) => (
                      <ResearchRunRow
                        child
                        childCount={0}
                        collapsed={false}
                        key={child.run_id}
                        run={child}
                        selected={child.run_id === activeRunId}
                        onSelect={onRunChange}
                        onToggle={() => undefined}
                      />
                    ))
                    : []),
                ];
              })}
            </tbody>
          </table>
        </div>
      )}

      {selected ? (
        <ResearchRunDetail
          run={selected}
          onOpenArtifact={(artifactId) => {
            if (researchArtifactDestination(selected) === "interventions") {
              onOpenIntervention(artifactId);
            } else {
              onOpenProbe(artifactId);
            }
          }}
        />
      ) : null}
    </main>
  );
}

function ResearchRunRow({
  child = false,
  childCount,
  collapsed,
  run,
  selected,
  onSelect,
  onToggle,
}: {
  child?: boolean;
  childCount: number;
  collapsed: boolean;
  run: ResearchRunRecord;
  selected: boolean;
  onSelect: (runId: string) => void;
  onToggle: () => void;
}) {
  return (
    <tr className={`${child ? "research-run-child " : ""}${selected ? "selected" : ""}`}>
      <td>
        <div className="research-question-cell">
          {childCount ? (
            <button aria-label={`${collapsed ? "Expand" : "Collapse"} ${run.name}`} onClick={onToggle}>
              {collapsed ? <ChevronRight size={15} /> : <ChevronDown size={15} />}
            </button>
          ) : <span className="research-run-indent" />}
          <button className="research-run-select" onClick={() => onSelect(run.run_id)}>
            <strong>{run.question}</strong>
            <small>{run.name}{childCount ? ` · ${childCount} experiments` : ""}</small>
          </button>
        </div>
      </td>
      <td>
        <span className={`research-run-status ${run.status}`}>{run.status}</span>
        <small className="research-stage">{formatResearchStage(run.stage)}</small>
      </td>
      <td>{formatResearchProgress(run.progress)}</td>
      <td>
        <strong className="research-result">{formatResearchResult(run.result)}</strong>
        {run.result.verdict ? <small>{run.result.verdict}</small> : null}
      </td>
      <td>{formatUpdated(run.updated_utc)}</td>
    </tr>
  );
}

function ResearchRunDetail({
  run,
  onOpenArtifact,
}: {
  run: ResearchRunRecord;
  onOpenArtifact: (artifactId: string) => void;
}) {
  return (
    <section className="research-run-detail">
      <div>
        <span>Selected run</span>
        <h2>{run.name}</h2>
        {run.error ? <p className="research-run-error">{run.error}</p> : null}
      </div>
      {run.artifact_ids.length ? (
        <div className="research-artifact-links">
          <span>Evidence</span>
          {run.artifact_ids.map((artifactId) => (
            <button key={artifactId} onClick={() => onOpenArtifact(artifactId)}>
              Open {researchArtifactDestination(run) === "interventions" ? "intervention" : "probe study"}
            </button>
          ))}
        </div>
      ) : <p className="research-no-artifact">No saved evidence yet.</p>}
      <details className="research-provenance">
        <summary>Reproduction details</summary>
        <pre>{JSON.stringify(run.provenance, null, 2)}</pre>
      </details>
    </section>
  );
}

function toggleSet(current: Set<string>, value: string): Set<string> {
  const next = new Set(current);
  if (next.has(value)) {
    next.delete(value);
  } else {
    next.add(value);
  }
  return next;
}

function formatUpdated(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value || "-";
  }
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}
