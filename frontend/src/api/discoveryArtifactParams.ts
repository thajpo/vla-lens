import type { DiscoveryArtifactEpisodeParams, EpisodeLensViewParams } from "./dataset";

export function discoveryArtifactEpisodeSearchParams(
  params: DiscoveryArtifactEpisodeParams = {},
): URLSearchParams {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === "" || value === "all") {
      continue;
    }
    search.set(key, String(value));
  }
  return search;
}

export function episodeLensViewSearchParams(params: EpisodeLensViewParams): URLSearchParams {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === "") {
      continue;
    }
    search.set(key, String(value));
  }
  return search;
}
