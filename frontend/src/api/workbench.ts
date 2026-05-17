import { getJson } from "./client";
import type { WorkbenchManifest } from "../types/workbench";

export function fetchWorkbench(): Promise<WorkbenchManifest> {
  return getJson<WorkbenchManifest>("/api/workbench");
}
