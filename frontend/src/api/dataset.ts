import { getJson, noStore, postJson } from "./client";
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

export type DatasetSnapshotLookup = {
  datasetId?: string;
  fingerprint?: string;
  root?: string;
};

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

export function fetchDataset(_identity: DatasetSnapshotLookup = {}): Promise<DatasetPayload> {
  return getJson<DatasetPayload>("/api/dataset", noStore());
}

export function cachedDatasetSnapshot(_identity: DatasetSnapshotLookup = {}): DatasetPayload | undefined {
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
