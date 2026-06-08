import { create } from "zustand";

type WorkbenchState = {
  activeRunId: string;
  setActiveRunId: (runId: string) => void;
};

export const useWorkbenchStore = create<WorkbenchState>((set) => ({
  activeRunId: "",
  setActiveRunId: (runId) => set({ activeRunId: runId }),
}));
