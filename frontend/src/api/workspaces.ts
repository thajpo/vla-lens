import { postJson } from "./client";
import type { SavedWorkspace, SavedWorkspaceResponse } from "../types/workbench";

export function saveWorkspace(workspace: SavedWorkspace): Promise<SavedWorkspaceResponse> {
  return postJson<SavedWorkspaceResponse>("/api/workspaces", { workspace });
}
