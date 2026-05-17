import { useEffect, useMemo } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Save, Users } from "lucide-react";
import { saveCohortFromSelection } from "../../api/cohorts";
import { resolveSelection } from "../../api/selections";
import { saveWorkspace } from "../../api/workspaces";
import { useWorkbenchStore } from "../../store/workbenchStore";
import type { AnalysisRunSpec, LensArraySpec, WorkbenchManifest } from "../../types/workbench";
import { ConfusionMatrixPanel } from "../panels/ConfusionMatrixPanel";
import { EpisodeViewerPanel } from "../panels/EpisodeViewerPanel";
import { ExamplesPanel } from "../panels/ExamplesPanel";
import { HeatmapPanel } from "../panels/HeatmapPanel";
import { InspectorPanel } from "../panels/InspectorPanel";

type TargetObjectEncodingPresetProps = {
  manifest: WorkbenchManifest;
};

export function TargetObjectEncodingPreset({ manifest }: TargetObjectEncodingPresetProps) {
  const {
    activeRunId,
    activeMetric,
    activeTokenKind,
    activeSelection,
    resolvedSelection,
    activeCohortId,
    setActiveRunId,
    setActiveTokenKind,
    setActiveSelection,
    setResolvedSelection,
    setActiveCohortId,
  } = useWorkbenchStore();

  const runs = useMemo(
    () => manifest.analysis_runs.filter((run) => run.workflow === "target_object_encoding"),
    [manifest.analysis_runs],
  );
  const runId = activeRunId || runs[0]?.run_id || "";
  const tokenKinds = tokenKindsForRun(manifest, runId, activeMetric);
  const tokenKind = activeTokenKind || tokenKinds[0] || "";
  const array = lensArrayForRun(manifest.lens_arrays, runId, activeMetric);

  useEffect(() => {
    if (!activeRunId && runId) {
      setActiveRunId(runId);
    }
  }, [activeRunId, runId, setActiveRunId]);

  useEffect(() => {
    if (!activeTokenKind && tokenKind) {
      setActiveTokenKind(tokenKind);
    }
  }, [activeTokenKind, tokenKind, setActiveTokenKind]);

  const resolution = useQuery({
    queryKey: ["selection-resolution", activeSelection],
    queryFn: () => resolveSelection(activeSelection!),
    enabled: Boolean(activeSelection),
  });

  useEffect(() => {
    if (resolution.data) {
      setResolvedSelection(resolution.data);
    }
  }, [resolution.data, setResolvedSelection]);

  const cohortMutation = useMutation({
    mutationFn: () => {
      if (!activeSelection) {
        throw new Error("No selection");
      }
      return saveCohortFromSelection(activeSelection, cohortLabel(activeSelection));
    },
    onSuccess: (payload) => {
      if (payload.cohort?.cohort_id) {
        setActiveCohortId(payload.cohort.cohort_id);
      }
    },
  });

  const workspaceMutation = useMutation({
    mutationFn: () => {
      if (!activeSelection) {
        throw new Error("No selection");
      }
      const workspaceId = `target_object_${Date.now()}`;
      return saveWorkspace({
        workspace_id: workspaceId,
        dataset_id: manifest.dataset_id,
        panels: [
          {
            panel_type: "heatmap",
            array_id: array?.array_id,
            encoding: { x: "timestep", y: "layer", color: activeMetric, facet: "token_kind" },
          },
          { panel_type: "examples.table" },
          { panel_type: "confusion_matrix" },
          { panel_type: "episode.viewer" },
          { panel_type: "inspector" },
        ],
        selection: activeSelection,
        cohorts: activeCohortId ? [activeCohortId] : [],
        analysis_runs: runId ? [runId] : [],
      });
    },
  });

  if (!runs.length) {
    return (
      <div className="workflow-empty">
        <h1>Target Object Encoding</h1>
        <p>No target-object encoding analysis run is registered yet.</p>
      </div>
    );
  }

  return (
    <div className="workflow-grid">
      <main className="center-panels">
        <div className="workflow-toolbar">
          <div>
            <h1>Target Object Encoding</h1>
            <p>{runLabel(runs, runId)}</p>
          </div>
          <div className="toolbar-actions">
            <button
              type="button"
              disabled={!activeSelection || cohortMutation.isPending}
              onClick={() => cohortMutation.mutate()}
            >
              <Users size={16} />
              Save Cohort
            </button>
            <button
              type="button"
              disabled={!activeSelection || workspaceMutation.isPending}
              onClick={() => workspaceMutation.mutate()}
            >
              <Save size={16} />
              Save Workspace
            </button>
          </div>
        </div>
        <HeatmapPanel
          panelId="target_object.heatmap"
          array={array}
          xAxis="timestep"
          yAxis="layer"
          filterAxis="token_kind"
          filterValue={tokenKind}
          analysisRunId={runId}
          metric={activeMetric}
          selected={activeSelection}
          onSelect={setActiveSelection}
        />
        <div className="panel-grid two-col">
          <ExamplesPanel resolution={resolvedSelection} />
          <ConfusionMatrixPanel resolution={resolvedSelection} />
        </div>
      </main>
      <aside className="right-rail">
        <InspectorPanel resolution={resolvedSelection} />
        <EpisodeViewerPanel resolution={resolvedSelection} />
        <div className="saved-state">
          <strong>Saved state</strong>
          <span>Cohort: {activeCohortId || "-"}</span>
          <span>
            Workspace:{" "}
            {workspaceMutation.data?.workspace?.workspace_id ??
              (workspaceMutation.isPending ? "saving" : "-")}
          </span>
        </div>
      </aside>
    </div>
  );
}

function lensArrayForRun(
  arrays: LensArraySpec[],
  runId: string,
  metric: string,
): LensArraySpec | undefined {
  return arrays.find((array) => array.array_id === `artifact.${runId}.${metric}`);
}

function tokenKindsForRun(manifest: WorkbenchManifest, runId: string, metric: string): string[] {
  const array = lensArrayForRun(manifest.lens_arrays, runId, metric);
  const coords = array?.coords.token_kind;
  return Array.isArray(coords) ? coords.map(String) : [];
}

function runLabel(runs: AnalysisRunSpec[], runId: string) {
  const run = runs.find((item) => item.run_id === runId);
  return run ? `${run.workflow} / ${run.outputs.join(", ")}` : "No run selected";
}

function cohortLabel(selection: { axis_values: Record<string, unknown> }) {
  const layer = first(selection.axis_values.layer);
  const timestep = first(selection.axis_values.timestep);
  const tokenKind = first(selection.axis_values.token_kind);
  return `target_object_l${layer}_t${timestep}_${tokenKind}`;
}

function first(value: unknown): unknown {
  return Array.isArray(value) ? value[0] : value;
}
