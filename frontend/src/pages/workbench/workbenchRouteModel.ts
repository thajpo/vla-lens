import {
  emptyEpisodeRouteState,
  parseEpisodeRoute,
  type EpisodeRouteState,
} from "./episodeRouteModel.ts";

export type WorkbenchPageName = "dataset" | "episode" | "interventions" | "probes" | "research";

export type WorkbenchRoute = {
  episodeState: EpisodeRouteState;
  interventionRunId: string;
  page: WorkbenchPageName;
  probeRunId: string;
  researchRunId: string;
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
      probeRunId: "",
      researchRunId: "",
      traceId: episodeState.traceId,
    };
  }
  if (page.startsWith("dataset/")) {
    const episodeState = parseEpisodeRoute(page.slice("dataset/".length));
    return {
      episodeState,
      interventionRunId: "",
      page: "episode",
      probeRunId: "",
      researchRunId: "",
      traceId: episodeState.traceId,
    };
  }
  if (page.startsWith("interventions/")) {
    return {
      episodeState: emptyEpisodeRouteState(),
      interventionRunId: decodeURIComponent(page.slice("interventions/".length)),
      page: "interventions",
      probeRunId: "",
      researchRunId: "",
      traceId: "",
    };
  }
  if (page.startsWith("evidence/")) {
    return {
      episodeState: emptyEpisodeRouteState(),
      interventionRunId: decodeURIComponent(page.slice("evidence/".length)),
      page: "interventions",
      probeRunId: "",
      researchRunId: "",
      traceId: "",
    };
  }
  if (page.startsWith("probes/")) {
    return {
      episodeState: emptyEpisodeRouteState(),
      interventionRunId: "",
      page: "probes",
      probeRunId: decodeURIComponent(page.slice("probes/".length)),
      researchRunId: "",
      traceId: "",
    };
  }
  if (page.startsWith("research/")) {
    return {
      episodeState: emptyEpisodeRouteState(),
      interventionRunId: "",
      page: "research",
      probeRunId: "",
      researchRunId: decodeURIComponent(page.slice("research/".length)),
      traceId: "",
    };
  }
  if (page === "dataset" || page === "probes" || page === "research") {
    return {
      episodeState: emptyEpisodeRouteState(),
      interventionRunId: "",
      page,
      probeRunId: "",
      researchRunId: "",
      traceId: "",
    };
  }
  if (page === "interventions" || page === "evidence") {
    return {
      episodeState: emptyEpisodeRouteState(),
      interventionRunId: "",
      page: "interventions",
      probeRunId: "",
      researchRunId: "",
      traceId: "",
    };
  }
  if (page === "episode" || page === "episodes") {
    return {
      episodeState: emptyEpisodeRouteState(),
      interventionRunId: "",
      page: "episode",
      probeRunId: "",
      researchRunId: "",
      traceId: "",
    };
  }
  return {
    episodeState: emptyEpisodeRouteState(),
    interventionRunId: "",
    page: "dataset",
    probeRunId: "",
    researchRunId: "",
    traceId: "",
  };
}

export function buildInterventionsHash(runId = ""): string {
  return runId ? `#interventions/${encodeURIComponent(runId)}` : "#interventions";
}

export function buildProbeHash(runId = ""): string {
  return runId ? `#probes/${encodeURIComponent(runId)}` : "#probes";
}

export function buildResearchHash(runId = ""): string {
  return runId ? `#research/${encodeURIComponent(runId)}` : "#research";
}
