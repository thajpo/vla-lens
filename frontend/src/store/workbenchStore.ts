import { create } from "zustand";
import type { ResolvedSelection, SelectionState } from "../types/workbench";

type WorkbenchState = {
  activeWorkflowId: string;
  activeRunId: string;
  activeMetric: string;
  activeTokenKind: string;
  activeSelection: SelectionState | null;
  resolvedSelection: ResolvedSelection | null;
  activeCohortId: string;
  setActiveWorkflowId: (workflowId: string) => void;
  setActiveRunId: (runId: string) => void;
  setActiveMetric: (metric: string) => void;
  setActiveTokenKind: (tokenKind: string) => void;
  setActiveSelection: (selection: SelectionState | null) => void;
  setResolvedSelection: (resolution: ResolvedSelection | null) => void;
  setActiveCohortId: (cohortId: string) => void;
};

export const useWorkbenchStore = create<WorkbenchState>((set) => ({
  activeWorkflowId: "target_object_encoding",
  activeRunId: "",
  activeMetric: "metric_cube",
  activeTokenKind: "",
  activeSelection: null,
  resolvedSelection: null,
  activeCohortId: "",
  setActiveWorkflowId: (workflowId) => set({ activeWorkflowId: workflowId }),
  setActiveRunId: (runId) =>
    set({ activeRunId: runId, activeSelection: null, resolvedSelection: null }),
  setActiveMetric: (metric) => set({ activeMetric: metric }),
  setActiveTokenKind: (tokenKind) => set({ activeTokenKind: tokenKind }),
  setActiveSelection: (selection) => set({ activeSelection: selection }),
  setResolvedSelection: (resolution) => set({ resolvedSelection: resolution }),
  setActiveCohortId: (cohortId) => set({ activeCohortId: cohortId }),
}));
