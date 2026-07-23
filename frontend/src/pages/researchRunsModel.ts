import type { ResearchProgress, ResearchRunRecord, ResearchResultSummary } from "../types/researchRuns.ts";

export type ResearchRunGroup = {
  run: ResearchRunRecord;
  children: ResearchRunRecord[];
};

export function groupResearchRuns(runs: ResearchRunRecord[]): ResearchRunGroup[] {
  const ids = new Set(runs.map((run) => run.run_id));
  const children = new Map<string, ResearchRunRecord[]>();
  for (const run of runs) {
    if (run.parent_run_id && ids.has(run.parent_run_id)) {
      children.set(run.parent_run_id, [...(children.get(run.parent_run_id) ?? []), run]);
    }
  }
  return runs
    .filter((run) => !run.parent_run_id || !ids.has(run.parent_run_id))
    .map((run) => ({
      run,
      children: sortByUpdated(children.get(run.run_id) ?? []),
    }));
}

export function formatResearchProgress(progress: ResearchProgress): string {
  if (!progress.total) {
    return "Not started";
  }
  const unit = progress.unit && progress.unit !== "steps" ? ` ${progress.unit}` : "";
  const percent = Math.round((progress.completed / progress.total) * 100);
  return `${progress.completed} / ${progress.total}${unit} · ${percent}%`;
}

export function formatResearchResult(result: ResearchResultSummary): string {
  if (typeof result.score !== "number") {
    return result.verdict || "No result yet";
  }
  if (typeof result.baseline !== "number") {
    return `${formatScore(result.score)}${result.metric ? ` ${result.metric}` : ""}`;
  }
  const delta = typeof result.delta === "number" ? result.delta : result.score - result.baseline;
  return `${formatScore(result.score)} vs ${formatScore(result.baseline)} (${formatSigned(delta)})`;
}

export function researchArtifactDestination(run: ResearchRunRecord): "interventions" | "probes" {
  return run.kind.toLowerCase().includes("intervention") ? "interventions" : "probes";
}

export function formatResearchStage(stage: string): string {
  return stage.replace(/[_-]+/g, " ").replace(/^./, (value) => value.toUpperCase());
}

function sortByUpdated(runs: ResearchRunRecord[]): ResearchRunRecord[] {
  return [...runs].sort((left, right) => right.updated_utc.localeCompare(left.updated_utc));
}

function formatScore(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(3);
}

function formatSigned(value: number): string {
  return `${value >= 0 ? "+" : ""}${formatScore(value)}`;
}
