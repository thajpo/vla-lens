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
  ProbeIndexResponse,
  NumericSeriesResponse,
  PolicyCallsResponse,
  SelectedPatch,
} from "../types/dataset";

const LEGACY_DATASET_CACHE_KEY = "vla-lens.dataset.v1";
const DATASET_CACHE_PREFIX = "vla-lens.dataset.v2";
const DATASET_CACHE_LATEST_KEY = `${DATASET_CACHE_PREFIX}.latest`;
const DATASET_CACHE_VERSION = 2;

export type DatasetSnapshotLookup = {
  datasetId?: string;
  fingerprint?: string;
  root?: string;
};

type DatasetSnapshotIdentity = {
  datasetIds: string[];
  fingerprint?: string;
  root?: string;
};

type DatasetSnapshotPointer = {
  identity: DatasetSnapshotIdentity;
  key: string;
  storedAt: number;
  version: typeof DATASET_CACHE_VERSION;
};

type DatasetSnapshotEnvelope = DatasetSnapshotPointer & {
  payload: DatasetPayload;
};

export function fetchDataset(identity: DatasetSnapshotLookup = {}): Promise<DatasetPayload> {
  return getJson<DatasetPayload>("/api/dataset", noStore()).then((payload) => {
    scheduleDatasetSnapshotCache(payload, identity);
    return payload;
  });
}

export function cachedDatasetSnapshot(identity: DatasetSnapshotLookup = {}): DatasetPayload | undefined {
  if (typeof window === "undefined") {
    return undefined;
  }
  for (const aliasKey of cacheAliasKeys(identity)) {
    const envelope = readSnapshotEnvelope(aliasKey);
    if (envelope && snapshotMatchesLookup(envelope, identity)) {
      return envelope.payload;
    }
  }
  if (hasScopedSnapshotLookup(identity)) {
    return undefined;
  }
  const latest = readSnapshotEnvelope(DATASET_CACHE_LATEST_KEY);
  if (latest) {
    return latest.payload;
  }
  return readLegacyDatasetSnapshot();
}

function cacheDatasetSnapshot(payload: DatasetPayload, lookup: DatasetSnapshotLookup): void {
  if (typeof window === "undefined") {
    return;
  }
  try {
    const identity = snapshotIdentity(payload, lookup);
    const key = snapshotStorageKey(identity, payload);
    const envelope: DatasetSnapshotEnvelope = {
      identity,
      key,
      payload,
      storedAt: Date.now(),
      version: DATASET_CACHE_VERSION,
    };
    const pointer: DatasetSnapshotPointer = {
      identity,
      key,
      storedAt: envelope.storedAt,
      version: DATASET_CACHE_VERSION,
    };
    window.localStorage.setItem(key, JSON.stringify(envelope));
    window.localStorage.setItem(DATASET_CACHE_LATEST_KEY, JSON.stringify(pointer));
    for (const aliasKey of cacheAliasKeys(identity)) {
      window.localStorage.setItem(aliasKey, JSON.stringify(pointer));
    }
    window.localStorage.removeItem(LEGACY_DATASET_CACHE_KEY);
  } catch {
    // Local storage is a paint-speed hint only; network data remains authoritative.
  }
}

function scheduleDatasetSnapshotCache(payload: DatasetPayload, identity: DatasetSnapshotLookup): void {
  if (typeof window === "undefined") {
    return;
  }
  window.setTimeout(() => cacheDatasetSnapshot(payload, identity), 0);
}

function readSnapshotEnvelope(pointerOrEnvelopeKey: string): DatasetSnapshotEnvelope | undefined {
  try {
    const raw = window.localStorage.getItem(pointerOrEnvelopeKey);
    if (!raw) {
      return undefined;
    }
    const value = JSON.parse(raw) as unknown;
    if (!isSnapshotPointer(value)) {
      return isSnapshotEnvelope(value) ? value : undefined;
    }
    const snapshotRaw = window.localStorage.getItem(value.key);
    if (!snapshotRaw) {
      return undefined;
    }
    const snapshot = JSON.parse(snapshotRaw) as unknown;
    return isSnapshotEnvelope(snapshot) ? snapshot : undefined;
  } catch {
    return undefined;
  }
}

function readLegacyDatasetSnapshot(): DatasetPayload | undefined {
  try {
    const raw = window.localStorage.getItem(LEGACY_DATASET_CACHE_KEY);
    return raw ? (JSON.parse(raw) as DatasetPayload) : undefined;
  } catch {
    return undefined;
  }
}

function snapshotIdentity(
  payload: DatasetPayload,
  lookup: DatasetSnapshotLookup,
): DatasetSnapshotIdentity {
  return {
    datasetIds: datasetIdsForSnapshot(payload, lookup.datasetId),
    fingerprint: normalizedIdentityValue(
      lookup.fingerprint ?? recordString(payload, "fingerprint") ?? recordString(payload, "dataset_fingerprint"),
    ),
    root: normalizedIdentityValue(lookup.root ?? payload.root),
  };
}

function datasetIdsForSnapshot(payload: DatasetPayload, explicitDatasetId?: string): string[] {
  const values = new Set<string>();
  const explicit = normalizedIdentityValue(explicitDatasetId);
  if (explicit) {
    values.add(explicit);
  }
  const topLevel = normalizedIdentityValue(recordString(payload, "dataset_id"));
  if (topLevel) {
    values.add(topLevel);
  }
  const workbench = recordValue(payload, "workbench");
  const workbenchDatasetId = isRecord(workbench)
    ? normalizedIdentityValue(recordString(workbench, "dataset_id"))
    : undefined;
  if (workbenchDatasetId) {
    values.add(workbenchDatasetId);
  }
  for (const episode of payload.episodes) {
    const episodeDatasetId = normalizedIdentityValue(recordString(episode.metadata, "dataset_id"));
    if (episodeDatasetId) {
      values.add(episodeDatasetId);
    }
  }
  return [...values].sort();
}

function snapshotStorageKey(identity: DatasetSnapshotIdentity, payload: DatasetPayload): string {
  if (identity.fingerprint) {
    return cacheKey("snapshot", "fingerprint", identity.fingerprint);
  }
  if (identity.root) {
    return cacheKey("snapshot", "root", identity.root);
  }
  if (identity.datasetIds.length) {
    return cacheKey("snapshot", "dataset-id", identity.datasetIds.join("|"));
  }
  return cacheKey(
    "snapshot",
    "content",
    `${payload.episodes.length}:${payload.episodes[0]?.trace_id ?? ""}:${payload.activation_sites}`,
  );
}

function cacheAliasKeys(identity: DatasetSnapshotLookup | DatasetSnapshotIdentity): string[] {
  const keys: string[] = [];
  const fingerprint = normalizedIdentityValue(identity.fingerprint);
  const root = normalizedIdentityValue(identity.root);
  if (fingerprint) {
    keys.push(cacheKey("alias", "fingerprint", fingerprint));
  }
  if (root) {
    keys.push(cacheKey("alias", "root", root));
  }
  if ("datasetIds" in identity) {
    for (const datasetId of identity.datasetIds) {
      keys.push(cacheKey("alias", "dataset-id", datasetId));
    }
  } else {
    const datasetId = normalizedIdentityValue(identity.datasetId);
    if (datasetId) {
      keys.push(cacheKey("alias", "dataset-id", datasetId));
    }
  }
  return keys;
}

function cacheKey(kind: "alias" | "snapshot", type: string, value: string): string {
  return `${DATASET_CACHE_PREFIX}.${kind}.${type}.${hashString(value)}`;
}

function snapshotMatchesLookup(
  envelope: DatasetSnapshotEnvelope,
  lookup: DatasetSnapshotLookup,
): boolean {
  const fingerprint = normalizedIdentityValue(lookup.fingerprint);
  if (fingerprint && envelope.identity.fingerprint !== fingerprint) {
    return false;
  }
  const root = normalizedIdentityValue(lookup.root);
  if (root && envelope.identity.root !== root) {
    return false;
  }
  const datasetId = normalizedIdentityValue(lookup.datasetId);
  if (datasetId && !envelope.identity.datasetIds.includes(datasetId)) {
    return false;
  }
  return true;
}

function hasScopedSnapshotLookup(identity: DatasetSnapshotLookup): boolean {
  return Boolean(
    normalizedIdentityValue(identity.fingerprint) ||
      normalizedIdentityValue(identity.root) ||
      normalizedIdentityValue(identity.datasetId),
  );
}

function isSnapshotPointer(value: unknown): value is DatasetSnapshotPointer {
  return (
    isRecord(value) &&
    value.version === DATASET_CACHE_VERSION &&
    typeof value.key === "string" &&
    isRecord(value.identity)
  );
}

function isSnapshotEnvelope(value: unknown): value is DatasetSnapshotEnvelope {
  const record = isRecord(value) ? value : undefined;
  return Boolean(record && isSnapshotPointer(value) && isRecord(record.payload));
}

function normalizedIdentityValue(value: unknown): string | undefined {
  if (typeof value !== "string") {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed || undefined;
}

function recordValue(record: unknown, key: string): unknown {
  return isRecord(record) ? record[key] : undefined;
}

function recordString(record: unknown, key: string): string | undefined {
  const value = recordValue(record, key);
  return typeof value === "string" ? value : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function hashString(value: string): string {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(36);
}

function freshJsonInit(signal?: AbortSignal): RequestInit {
  return signal ? noStore({ signal }) : noStore();
}

export function fetchDatasetDiagnostics(): Promise<DatasetDiagnostics> {
  return getJson<DatasetDiagnostics>("/api/dataset-diagnostics", freshJsonInit());
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
