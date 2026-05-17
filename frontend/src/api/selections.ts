import { postJson } from "./client";
import type { ResolvedSelection, SelectionState } from "../types/workbench";

export function resolveSelection(selection: SelectionState): Promise<ResolvedSelection> {
  return postJson<ResolvedSelection>("/api/selections/resolve", { selection });
}
