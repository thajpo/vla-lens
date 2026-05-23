import { getJson, postJson } from "./client";
import type {
  ActivationSitesResponse,
  ArtifactDetailResponse,
  ArtifactListResponse,
  AttentionMapResponse,
  ExpertTokenActivationsResponse,
  ExpertTokenDetailsResponse,
  DatasetDiagnostics,
  DatasetPayload,
  EpisodeAnnotation,
  EpisodeAnnotationResponse,
  EpisodeDetail,
  EpisodeInteractionsResponse,
  EpisodeMetricsResponse,
  EpisodeProbesResponse,
  ActivationSliceResponse,
  ImageTokenMapResponse,
  MatrixSeriesResponse,
  ObjectCameraOverlayResponse,
  PatchFeaturesResponse,
  PromptAttentionResponse,
  ProbeIndexResponse,
  NumericSeriesResponse,
  PolicyCallsResponse,
  SelectedPatch,
} from "../types/dataset";

const DATASET_CACHE_KEY = "vla-lens.dataset.v1";

export function fetchDataset(): Promise<DatasetPayload> {
  return getJson<DatasetPayload>("/api/dataset").then((payload) => {
    cacheDatasetSnapshot(payload);
    return payload;
  });
}

export function cachedDatasetSnapshot(): DatasetPayload | undefined {
  if (typeof window === "undefined") {
    return undefined;
  }
  try {
    const raw = window.localStorage.getItem(DATASET_CACHE_KEY);
    return raw ? (JSON.parse(raw) as DatasetPayload) : undefined;
  } catch {
    return undefined;
  }
}

function cacheDatasetSnapshot(payload: DatasetPayload): void {
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.localStorage.setItem(DATASET_CACHE_KEY, JSON.stringify(payload));
  } catch {
    // Local storage is a paint-speed hint only; network data remains authoritative.
  }
}

export function fetchDatasetDiagnostics(): Promise<DatasetDiagnostics> {
  return getJson<DatasetDiagnostics>("/api/dataset-diagnostics");
}

export function fetchArtifacts(): Promise<ArtifactListResponse> {
  return getJson<ArtifactListResponse>("/api/artifacts");
}

export function fetchArtifact(artifactId: string): Promise<ArtifactDetailResponse> {
  return getJson<ArtifactDetailResponse>(`/api/artifacts/${encodeURIComponent(artifactId)}`);
}

export function fetchEpisode(traceId: string): Promise<EpisodeDetail> {
  return getJson<EpisodeDetail>(`/api/episodes/${encodeURIComponent(traceId)}`);
}

export function fetchEpisodeAnnotation(traceId: string): Promise<EpisodeAnnotationResponse> {
  return getJson<EpisodeAnnotationResponse>(
    `/api/episode-annotations?trace_id=${encodeURIComponent(traceId)}`,
  );
}

export function saveEpisodeAnnotation(
  annotation: Pick<EpisodeAnnotation, "trace_id" | "starred" | "notes">,
): Promise<EpisodeAnnotationResponse> {
  return postJson<EpisodeAnnotationResponse>("/api/episode-annotations", annotation);
}

export function fetchPolicyCalls(traceId: string): Promise<PolicyCallsResponse> {
  return getJson<PolicyCallsResponse>(`/api/policy-calls?trace_id=${encodeURIComponent(traceId)}`);
}

export function fetchActionNorm(traceId: string): Promise<NumericSeriesResponse> {
  return getJson<NumericSeriesResponse>(`/api/action-norm?trace_id=${encodeURIComponent(traceId)}`);
}

export function fetchGenerationCommitment(traceId: string): Promise<MatrixSeriesResponse> {
  return getJson<MatrixSeriesResponse>(
    `/api/generation-commitment?trace_id=${encodeURIComponent(traceId)}`,
  );
}

export function fetchEpisodeMetrics(traceId: string): Promise<EpisodeMetricsResponse> {
  return getJson<EpisodeMetricsResponse>(
    `/api/episode-metrics?trace_id=${encodeURIComponent(traceId)}`,
  );
}

export function fetchEpisodeInteractions(traceId: string): Promise<EpisodeInteractionsResponse> {
  return getJson<EpisodeInteractionsResponse>(
    `/api/episode-interactions?trace_id=${encodeURIComponent(traceId)}`,
  );
}

export function fetchEpisodeProbes(traceId: string): Promise<EpisodeProbesResponse> {
  return getJson<EpisodeProbesResponse>(
    `/api/episode-probes?trace_id=${encodeURIComponent(traceId)}`,
  );
}

export function fetchProbeIndex(): Promise<ProbeIndexResponse> {
  return getJson<ProbeIndexResponse>("/api/probe-index");
}

export function fetchActivationSites(traceId: string): Promise<ActivationSitesResponse> {
  return getJson<ActivationSitesResponse>(
    `/api/activation-sites?trace_id=${encodeURIComponent(traceId)}`,
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
  return getJson<ActivationSliceResponse>(`/api/activation-slice?${params.toString()}`);
}

export function fetchImageTokenMap(
  traceId: string,
  callIndex: number,
  siteName: string,
  feature: number,
): Promise<ImageTokenMapResponse> {
  const params = new URLSearchParams({
    trace_id: traceId,
    call_index: String(callIndex),
    name: siteName,
    feature: String(feature),
  });
  return getJson<ImageTokenMapResponse>(`/api/image-token-map?${params.toString()}`);
}

export function fetchObjectCameraOverlay(
  traceId: string,
  camera: string,
  timestep: number,
): Promise<ObjectCameraOverlayResponse> {
  const params = new URLSearchParams({
    trace_id: traceId,
    camera,
    timestep: String(Math.max(0, timestep)),
  });
  return getJson<ObjectCameraOverlayResponse>(`/api/object-camera-overlay?${params.toString()}`);
}

export function fetchAttentionMap(
  traceId: string,
  callIndex: number,
  kind: "vlm" | "expert",
  generationStep: number,
  siteName?: string,
  head?: number | null,
  queryToken?: number | null,
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
  return getJson<AttentionMapResponse>(`/api/attention-map?${params.toString()}`);
}

export function fetchPromptAttention(
  traceId: string,
  callIndex: number,
  generationStep: number,
  kind: "vlm" | "expert",
  siteName?: string,
  head?: number | null,
  queryToken?: number | null,
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
  return getJson<PromptAttentionResponse>(`/api/prompt-attention?${params.toString()}`);
}

export function fetchPromptFeatureMap(
  traceId: string,
  callIndex: number,
  siteName: string,
  feature: number,
): Promise<PromptAttentionResponse> {
  const params = new URLSearchParams({
    trace_id: traceId,
    call_index: String(callIndex),
    name: siteName,
    feature: String(feature),
  });
  return getJson<PromptAttentionResponse>(`/api/prompt-feature-map?${params.toString()}`);
}

export function fetchPatchFeatures(
  traceId: string,
  callIndex: number,
  siteName: string,
  feature: number,
  patch: SelectedPatch,
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
  return getJson<PatchFeaturesResponse>(`/api/patch-features?${params.toString()}`);
}

export function fetchExpertTokenActivations(
  traceId: string,
  callIndex: number,
  siteName: string,
  feature: number,
  generationStep: number,
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
  );
}

export function fetchExpertTokenDetails(
  traceId: string,
  callIndex: number,
  siteName: string,
  feature: number,
  tokenIndex: number,
  generationStep: number,
): Promise<ExpertTokenDetailsResponse> {
  const params = new URLSearchParams({
    trace_id: traceId,
    call_index: String(callIndex),
    name: siteName,
    feature: String(feature),
    token_index: String(tokenIndex),
    generation_step: String(generationStep),
  });
  return getJson<ExpertTokenDetailsResponse>(`/api/expert-token-details?${params.toString()}`);
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
    source: "trace",
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
