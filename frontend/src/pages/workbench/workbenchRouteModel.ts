import {
  emptyEpisodeRouteState,
  parseEpisodeRoute,
  type EpisodeRouteState,
} from "./episodeRouteModel.ts";

export type WorkbenchPageName = "dataset" | "episode" | "interventions" | "probes";

export type WorkbenchRoute = {
  episodeState: EpisodeRouteState;
  interventionRunId: string;
  page: WorkbenchPageName;
  traceId: string;
};

export function parseWorkbenchHash(hash: string): WorkbenchRoute {
  const page = hash.replace(/^#/, "");
  if (page.startsWith("episode/")) {
    const episodeState = parseEpisodeRoute(page.slice("episode/".length));
    return {
      episodeState,
      interventionRunId: "",
      page: "episode",
      traceId: episodeState.traceId,
    };
  }
  if (page.startsWith("dataset/")) {
    const episodeState = parseEpisodeRoute(page.slice("dataset/".length));
    return {
      episodeState,
      interventionRunId: "",
      page: "episode",
      traceId: episodeState.traceId,
    };
  }
  if (page.startsWith("interventions/")) {
    return {
      episodeState: emptyEpisodeRouteState(),
      interventionRunId: decodeURIComponent(page.slice("interventions/".length)),
      page: "interventions",
      traceId: "",
    };
  }
  if (page.startsWith("evidence/")) {
    return {
      episodeState: emptyEpisodeRouteState(),
      interventionRunId: decodeURIComponent(page.slice("evidence/".length)),
      page: "interventions",
      traceId: "",
    };
  }
  if (page === "dataset" || page === "probes") {
    return { episodeState: emptyEpisodeRouteState(), interventionRunId: "", page, traceId: "" };
  }
  if (page === "interventions" || page === "evidence") {
    return {
      episodeState: emptyEpisodeRouteState(),
      interventionRunId: "",
      page: "interventions",
      traceId: "",
    };
  }
  if (page === "episode" || page === "episodes") {
    return {
      episodeState: emptyEpisodeRouteState(),
      interventionRunId: "",
      page: "episode",
      traceId: "",
    };
  }
  return {
    episodeState: emptyEpisodeRouteState(),
    interventionRunId: "",
    page: "dataset",
    traceId: "",
  };
}

export function buildInterventionsHash(runId = ""): string {
  return runId ? `#interventions/${encodeURIComponent(runId)}` : "#interventions";
}
