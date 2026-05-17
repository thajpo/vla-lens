import type { AnalysisRunSpec, WorkflowPreset } from "../../types/workbench";

type LeftRailProps = {
  workflows: WorkflowPreset[];
  runs: AnalysisRunSpec[];
  activeWorkflowId: string;
  activeRunId: string;
  activeMetric: string;
  activeTokenKind: string;
  tokenKinds: string[];
  onWorkflowChange: (workflowId: string) => void;
  onRunChange: (runId: string) => void;
  onMetricChange: (metric: string) => void;
  onTokenKindChange: (tokenKind: string) => void;
};

const metrics = [
  { id: "metric_cube", label: "Score" },
  { id: "baseline_cube", label: "Baseline" },
  { id: "delta_cube", label: "Delta" },
];

export function LeftRail({
  workflows,
  runs,
  activeWorkflowId,
  activeRunId,
  activeMetric,
  activeTokenKind,
  tokenKinds,
  onWorkflowChange,
  onRunChange,
  onMetricChange,
  onTokenKindChange,
}: LeftRailProps) {
  return (
    <aside className="left-rail">
      <label>
        Workflow
        <select value={activeWorkflowId} onChange={(event) => onWorkflowChange(event.target.value)}>
          {workflows.map((workflow) => (
            <option key={workflow.workflow_id} value={workflow.workflow_id}>
              {workflow.label}
            </option>
          ))}
        </select>
      </label>
      <label>
        Analysis Run
        <select value={activeRunId} onChange={(event) => onRunChange(event.target.value)}>
          <option value="">Select a run</option>
          {runs.map((run) => (
            <option key={run.run_id} value={run.run_id}>
              {run.run_id}
            </option>
          ))}
        </select>
      </label>
      <label>
        Metric
        <select value={activeMetric} onChange={(event) => onMetricChange(event.target.value)}>
          {metrics.map((metric) => (
            <option key={metric.id} value={metric.id}>
              {metric.label}
            </option>
          ))}
        </select>
      </label>
      <label>
        Token Kind
        <select
          value={activeTokenKind}
          onChange={(event) => onTokenKindChange(event.target.value)}
        >
          {tokenKinds.map((tokenKind) => (
            <option key={tokenKind} value={tokenKind}>
              {tokenKind}
            </option>
          ))}
        </select>
      </label>
    </aside>
  );
}
