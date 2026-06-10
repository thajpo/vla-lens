import { getJson, noStore, postJson } from "./client";
import {
  discoveryArtifactEpisodeSearchParams,
  episodeLensViewSearchParams,
} from "./discoveryArtifactParams";
import type {
  ActivationSitesResponse,
  ArtifactDetailResponse,
  ArtifactListResponse,
  AttentionMapResponse,
  ExpertTokenActivationsResponse,
  ExpertTokenDetailsResponse,
  DatasetDiagnostics,
  DatasetPayload,
  DiscoveryArtifactEpisodesResponse,
  DiscoveryArtifactFamiliesResponse,
  EpisodeLensViewResponse,
  DiscoveryArtifactReadoutResponse,
  DiscoveryArtifactTargetResponse,
  EpisodeAnnotation,
  EpisodeAnnotationResponse,
  EpisodeDetail,
  EpisodeNeighborsResponse,
  EpisodePageResponse,
  EpisodeInteractionsResponse,
  EpisodeMetricsResponse,
  EpisodeProbesResponse,
  ActivationSliceResponse,
  ImageTokenMapResponse,
  MatrixSeriesResponse,
  ObjectCameraOverlayResponse,
  ObservationalComparisonsResponse,
  PatchFeaturesResponse,
  PromptAttentionResponse,
  ProbeEvidenceResponse,
  ProbeIndexResponse,
  NumericSeriesResponse,
  PolicyCallsResponse,
  SelectedPatch,
} from "../types/dataset";
import type { ProbeEvidenceBundle } from "../types/probeEvidence";

export type EpisodePageParams = {
  benchmark?: string;
  dataset_id?: string;
  limit?: number;
  offset?: number;
  outcome?: string;
  profile?: string;
  probe_cohort_preset?: string;
  probe_id?: string;
  probe_prediction?: string;
  probe_split?: string;
  q?: string;
  sort?: string;
  task_id?: string;
};

export type DiscoveryArtifactEpisodeParams = EpisodePageParams & {
  cohort_preset?: string;
  prediction?: string;
  rank_by?: string;
  split?: string;
};

export type DiscoveryArtifactTargetParams = {
  model_site?: string;
  policy_call?: number | string | null;
  site?: string;
  token_space?: string | null;
  trace_id?: string;
};

export type DiscoveryArtifactReadoutParams = DiscoveryArtifactTargetParams & {
  trace_id: string;
};

export type EpisodeLensViewParams = {
  trace_id: string;
  timestep?: number | string | null;
  policy_call_index?: number | string | null;
  model_site_id?: string | null;
  feature?: number | string | null;
  ranking_mode?: "probe_contribution" | "raw_activation";
  top_k?: number | string | null;
};

export function fetchDataset(): Promise<DatasetPayload> {
  return getJson<DatasetPayload>("/api/dataset", noStore());
}

export function cachedDatasetSnapshot(): DatasetPayload | undefined {
  return undefined;
}

function freshJsonInit(signal?: AbortSignal): RequestInit {
  return signal ? noStore({ signal }) : noStore();
}

export function fetchDatasetDiagnostics(): Promise<DatasetDiagnostics> {
  return getJson<DatasetDiagnostics>("/api/dataset-diagnostics", freshJsonInit());
}

export function fetchEpisodesPage(
  params: EpisodePageParams = {},
  signal?: AbortSignal,
): Promise<EpisodePageResponse> {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === "" || value === "all") {
      continue;
    }
    search.set(key, String(value));
  }
  return getJson<EpisodePageResponse>(
    `/api/episodes?${search.toString()}`,
    freshJsonInit(signal),
  );
}

export function fetchEpisodeNeighbors(traceId: string): Promise<EpisodeNeighborsResponse> {
  return getJson<EpisodeNeighborsResponse>(
    `/api/episodes/${encodeURIComponent(traceId)}/neighbors`,
    freshJsonInit(),
  );
}

export function fetchArtifacts(): Promise<ArtifactListResponse> {
  return getJson<ArtifactListResponse>("/api/artifacts", freshJsonInit());
}

export function fetchArtifact(artifactId: string): Promise<ArtifactDetailResponse> {
  return getJson<ArtifactDetailResponse>(
    `/api/artifacts/${encodeURIComponent(artifactId)}`,
    freshJsonInit(),
  );
}

export function fetchEpisode(traceId: string): Promise<EpisodeDetail> {
  return getJson<EpisodeDetail>(`/api/episodes/${encodeURIComponent(traceId)}`, freshJsonInit());
}

export function fetchEpisodeAnnotation(traceId: string): Promise<EpisodeAnnotationResponse> {
  return getJson<EpisodeAnnotationResponse>(
    `/api/episode-annotations?trace_id=${encodeURIComponent(traceId)}`,
    freshJsonInit(),
  );
}

export function saveEpisodeAnnotation(
  annotation: Pick<EpisodeAnnotation, "trace_id" | "starred" | "notes">,
): Promise<EpisodeAnnotationResponse> {
  return postJson<EpisodeAnnotationResponse>("/api/episode-annotations", annotation);
}

export function fetchPolicyCalls(traceId: string): Promise<PolicyCallsResponse> {
  return getJson<PolicyCallsResponse>(
    `/api/policy-calls?trace_id=${encodeURIComponent(traceId)}`,
    freshJsonInit(),
  );
}

export function fetchActionNorm(traceId: string): Promise<NumericSeriesResponse> {
  return getJson<NumericSeriesResponse>(
    `/api/action-norm?trace_id=${encodeURIComponent(traceId)}`,
    freshJsonInit(),
  );
}

export function fetchGenerationCommitment(traceId: string): Promise<MatrixSeriesResponse> {
  return getJson<MatrixSeriesResponse>(
    `/api/generation-commitment?trace_id=${encodeURIComponent(traceId)}`,
    freshJsonInit(),
  );
}

export function fetchEpisodeMetrics(traceId: string): Promise<EpisodeMetricsResponse> {
  return getJson<EpisodeMetricsResponse>(
    `/api/episode-metrics?trace_id=${encodeURIComponent(traceId)}`,
    freshJsonInit(),
  );
}

export function fetchEpisodeInteractions(traceId: string): Promise<EpisodeInteractionsResponse> {
  return getJson<EpisodeInteractionsResponse>(
    `/api/episode-interactions?trace_id=${encodeURIComponent(traceId)}`,
    freshJsonInit(),
  );
}

export function fetchEpisodeProbes(traceId: string): Promise<EpisodeProbesResponse> {
  return getJson<EpisodeProbesResponse>(
    `/api/episode-probes?trace_id=${encodeURIComponent(traceId)}`,
    freshJsonInit(),
  );
}

export function fetchProbeIndex(): Promise<ProbeIndexResponse> {
  return getJson<ProbeIndexResponse>("/api/probe-index", freshJsonInit());
}

export function fetchProbeEvidence(
  probeId: string,
  params: EpisodePageParams = {},
  signal?: AbortSignal,
): Promise<ProbeEvidenceResponse> {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === "" || value === "all") {
      continue;
    }
    search.set(key, String(value));
  }
  return getJson<ProbeEvidenceResponse>(
    `/api/probes/${encodeURIComponent(probeId)}/evidence?${search.toString()}`,
    freshJsonInit(signal),
  );
}

export function fetchProbeEvidenceBundle(
  probeId: string,
  params: { dataset_id?: string; limit?: number } = {},
  signal?: AbortSignal,
): Promise<ProbeEvidenceBundle> {
  const search = new URLSearchParams();
  if (params.dataset_id && params.dataset_id !== "all") {
    search.set("dataset_id", params.dataset_id);
  }
  if (params.limit !== undefined && params.limit !== null) {
    search.set("limit", String(params.limit));
  }
  return getJson<ProbeEvidenceBundle>(
    `/api/probes/${encodeURIComponent(probeId)}/evidence-bundle?${search.toString()}`,
    freshJsonInit(signal),
  );
}

export { discoveryArtifactEpisodeSearchParams };
export { episodeLensViewSearchParams };

export function fetchDiscoveryArtifactFamilies(): Promise<DiscoveryArtifactFamiliesResponse> {
  return getJson<DiscoveryArtifactFamiliesResponse>(
    "/api/discovery-artifact-families",
    freshJsonInit(),
  );
}

export function fetchDiscoveryArtifactEpisodes(
  artifactId: string,
  params: DiscoveryArtifactEpisodeParams = {},
  signal?: AbortSignal,
): Promise<DiscoveryArtifactEpisodesResponse> {
  const search = discoveryArtifactEpisodeSearchParams(params);
  return getJson<DiscoveryArtifactEpisodesResponse>(
    `/api/discovery-artifacts/${encodeURIComponent(artifactId)}/episodes?${search.toString()}`,
    freshJsonInit(signal),
  );
}

export function fetchDiscoveryArtifactReadout(
  artifactId: string,
  params: DiscoveryArtifactReadoutParams,
): Promise<DiscoveryArtifactReadoutResponse> {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === "") {
      continue;
    }
    search.set(key, String(value));
  }
  return getJson<DiscoveryArtifactReadoutResponse>(
    `/api/discovery-artifacts/${encodeURIComponent(artifactId)}/readout?${search.toString()}`,
    freshJsonInit(),
  );
}

export function fetchDiscoveryArtifactEpisodeLensView(
  artifactId: string,
  params: EpisodeLensViewParams,
): Promise<EpisodeLensViewResponse> {
  const search = episodeLensViewSearchParams(params);
  return getJson<EpisodeLensViewResponse>(
    `/api/discovery-artifacts/${encodeURIComponent(artifactId)}/episode-lens-view?${search.toString()}`,
    freshJsonInit(),
  );
}

export function fetchDiscoveryArtifactTarget(
  artifactId: string,
  params: DiscoveryArtifactTargetParams = {},
): Promise<DiscoveryArtifactTargetResponse> {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === "") {
      continue;
    }
    search.set(key, String(value));
  }
  return getJson<DiscoveryArtifactTargetResponse>(
    `/api/discovery-artifacts/${encodeURIComponent(artifactId)}/target?${search.toString()}`,
    freshJsonInit(),
  );
}

export function fetchObservationalComparisons(
  traceId: string,
  probeId?: string,
  limit = 6,
): Promise<ObservationalComparisonsResponse> {
  const params = new URLSearchParams({
    trace_id: traceId,
    limit: String(limit),
  });
  if (probeId) {
    params.set("probe_id", probeId);
  }
  return getJson<ObservationalComparisonsResponse>(
    `/api/observational-comparisons?${params.toString()}`,
    freshJsonInit(),
  );
}

export function fetchActivationSites(traceId: string): Promise<ActivationSitesResponse> {
  return getJson<ActivationSitesResponse>(
    `/api/activation-sites?trace_id=${encodeURIComponent(traceId)}`,
    freshJsonInit(),
  );
}

export function fetchActivationSlice(
  traceId: string,
  callIndex: number,
  siteName: string,
  feature: number,
  generationStep?: number,
  clipPercent = 0,
  topK = 12,
  signal?: AbortSignal,
): Promise<ActivationSliceResponse> {
  const params = new URLSearchParams({
    trace_id: traceId,
    call_index: String(callIndex),
    name: siteName,
    feature: String(feature),
  });
  if (generationStep !== undefined) {
    params.set("generation_step", String(generationStep));
  }
  if (clipPercent > 0) {
    params.set("clip_percent", String(clipPercent));
  }
  params.set("top_k", String(topK));
  return getJson<ActivationSliceResponse>(
    `/api/activation-slice?${params.toString()}`,
    freshJsonInit(signal),
  );
}

export function fetchImageTokenMap(
  traceId: string,
  callIndex: number,
  siteName: string,
  feature: number,
  signal?: AbortSignal,
): Promise<ImageTokenMapResponse> {
  const params = new URLSearchParams({
    trace_id: traceId,
    call_index: String(callIndex),
    name: siteName,
    feature: String(feature),
  });
  return getJson<ImageTokenMapResponse>(
    `/api/image-token-map?${params.toString()}`,
    freshJsonInit(signal),
  );
}

export function fetchObjectCameraOverlay(
  traceId: string,
  camera: string,
  timestep: number,
  signal?: AbortSignal,
): Promise<ObjectCameraOverlayResponse> {
  const params = new URLSearchParams({
    trace_id: traceId,
    camera,
    timestep: String(Math.max(0, timestep)),
  });
  return getJson<ObjectCameraOverlayResponse>(
    `/api/object-camera-overlay?${params.toString()}`,
    freshJsonInit(signal),
  );
}

export function fetchAttentionMap(
  traceId: string,
  callIndex: number,
  kind: "vlm" | "expert",
  generationStep: number,
  siteName?: string,
  head?: number | null,
  queryToken?: number | null,
  signal?: AbortSignal,
): Promise<AttentionMapResponse> {
  const params = new URLSearchParams({
    trace_id: traceId,
    call_index: String(callIndex),
    kind,
    generation_step: String(generationStep),
  });
  if (siteName) {
    params.set("name", siteName);
  }
  if (head !== undefined && head !== null) {
    params.set("head", String(head));
  }
  if (queryToken !== undefined && queryToken !== null) {
    params.set("query_token", String(queryToken));
  }
  return getJson<AttentionMapResponse>(
    `/api/attention-map?${params.toString()}`,
    freshJsonInit(signal),
  );
}

export function fetchPromptAttention(
  traceId: string,
  callIndex: number,
  generationStep: number,
  kind: "vlm" | "expert",
  siteName?: string,
  head?: number | null,
  queryToken?: number | null,
  signal?: AbortSignal,
): Promise<PromptAttentionResponse> {
  const params = new URLSearchParams({
    trace_id: traceId,
    call_index: String(callIndex),
    generation_step: String(generationStep),
    kind,
  });
  if (siteName) {
    params.set("name", siteName);
  }
  if (head !== undefined && head !== null) {
    params.set("head", String(head));
  }
  if (queryToken !== undefined && queryToken !== null) {
    params.set("query_token", String(queryToken));
  }
  return getJson<PromptAttentionResponse>(
    `/api/prompt-attention?${params.toString()}`,
    freshJsonInit(signal),
  );
}

export function fetchPromptFeatureMap(
  traceId: string,
  callIndex: number,
  siteName: string,
  feature: number,
  signal?: AbortSignal,
): Promise<PromptAttentionResponse> {
  const params = new URLSearchParams({
    trace_id: traceId,
    call_index: String(callIndex),
    name: siteName,
    feature: String(feature),
  });
  return getJson<PromptAttentionResponse>(
    `/api/prompt-feature-map?${params.toString()}`,
    freshJsonInit(signal),
  );
}

export function fetchPatchFeatures(
  traceId: string,
  callIndex: number,
  siteName: string,
  feature: number,
  patch: SelectedPatch,
  signal?: AbortSignal,
): Promise<PatchFeaturesResponse> {
  const params = new URLSearchParams({
    trace_id: traceId,
    call_index: String(callIndex),
    name: siteName,
    feature: String(feature),
    camera: patch.camera,
    row: String(patch.row),
    col: String(patch.col),
  });
  return getJson<PatchFeaturesResponse>(
    `/api/patch-features?${params.toString()}`,
    freshJsonInit(signal),
  );
}

export function fetchExpertTokenActivations(
  traceId: string,
  callIndex: number,
  siteName: string,
  feature: number,
  generationStep: number,
  signal?: AbortSignal,
): Promise<ExpertTokenActivationsResponse> {
  const params = new URLSearchParams({
    trace_id: traceId,
    call_index: String(callIndex),
    name: siteName,
    feature: String(feature),
    generation_step: String(generationStep),
  });
  return getJson<ExpertTokenActivationsResponse>(
    `/api/expert-token-activations?${params.toString()}`,
    freshJsonInit(signal),
  );
}

export function fetchExpertTokenDetails(
  traceId: string,
  callIndex: number,
  siteName: string,
  feature: number,
  tokenIndex: number,
  generationStep: number,
  signal?: AbortSignal,
): Promise<ExpertTokenDetailsResponse> {
  const params = new URLSearchParams({
    trace_id: traceId,
    call_index: String(callIndex),
    name: siteName,
    feature: String(feature),
    token_index: String(tokenIndex),
    generation_step: String(generationStep),
  });
  return getJson<ExpertTokenDetailsResponse>(
    `/api/expert-token-details?${params.toString()}`,
    freshJsonInit(signal),
  );
}

export function frameUrl(
  traceId: string,
  camera: string,
  timestep: number,
  cacheKey = "trace_frames_v3",
): string {
  const params = new URLSearchParams({
    trace_id: traceId,
    camera,
    timestep: String(Math.max(0, timestep)),
    source: "auto",
    v: cacheKey,
  });
  return `/api/frame?${params.toString()}`;
}

export function episodeVideoUrl(
  traceId: string,
  camera: string,
  cacheKey = "full_episode_v2",
): string {
  const params = new URLSearchParams({
    trace_id: traceId,
    camera,
    fps: "10",
    max_width: "640",
    v: cacheKey,
  });
  return `/api/episode-video?${params.toString()}`;
}
