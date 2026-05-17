import { postJson } from "./client";
import type { SavedCohortResponse, SelectionState } from "../types/workbench";

export function saveCohortFromSelection(
  selection: SelectionState,
  label: string,
): Promise<SavedCohortResponse> {
  return postJson<SavedCohortResponse>("/api/cohorts/from-selection", { selection, label });
}
