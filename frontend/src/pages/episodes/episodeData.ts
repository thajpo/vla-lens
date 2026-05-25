import type { ActivationSite, DatasetEpisode } from "../../types/dataset";
import type { WorkbenchManifest } from "../../types/workbench";
import type { CameraOverlayStatus, InspectorContext } from "./shared";
import { siteOptionLabel } from "./siteModel";

export function episodesFromManifest(manifest: WorkbenchManifest | undefined): DatasetEpisode[] {
  if (!manifest) {
    return [];
  }
  const byTrace = new Map<string, DatasetEpisode>();
  manifest.image_frames.forEach((frame) => {
    if (!byTrace.has(frame.trace_id)) {
      byTrace.set(frame.trace_id, {
        trace_id: frame.trace_id,
        episode_id: frame.episode_id,
        length: frame.frame_count,
        metadata: frame.provenance,
      });
    }
  });
  return Array.from(byTrace.values());
}

export function camerasFromManifest(manifest: WorkbenchManifest | undefined, traceId: string): string[] {
  if (!manifest) {
    return [];
  }
  return manifest.image_frames
    .filter((frame) => frame.trace_id === traceId)
    .map((frame) => frame.camera);
}

export function frameVersionKey(episode: DatasetEpisode | undefined, datasetFingerprint: unknown): string {
  const metadata = episode?.metadata ?? {};
  const captureStamp = metadata.timestamp_utc ?? metadata.capture_started_utc ?? metadata.capture_granularity;
  return encodeURIComponent(
    String(datasetFingerprint ?? captureStamp ?? episode?.trace_id ?? "trace_frames_v3"),
  );
}

export function episodeMetadataItems(episode: DatasetEpisode): Array<{ label: string; value: string }> {
  return [
    ["Dataset", metadataString(episode, "dataset_id")],
    ["Type", episodeDatasetType(episode)],
    ["Suite", episodeTaskSuite(episode)],
    ["Task", String(episode.task_id ?? metadataString(episode, "task_number") ?? "")],
    ["Seed", metadataString(episode, "seed")],
    ["Length", episode.length === undefined || episode.length === null ? "" : String(episode.length)],
    ["Profile", metadataString(episode, "actual_profile") || metadataString(episode, "capture_profile")],
    ["Batch", metadataString(episode, "batch_id")],
  ]
    .filter(([, value]) => Boolean(value))
    .map(([label, value]) => ({ label, value }));
}

export function episodeDatasetType(episode: DatasetEpisode): string {
  const explicit =
    metadataString(episode, "dataset_type") ||
    metadataString(episode, "benchmark") ||
    metadataString(episode, "environment");
  if (explicit) {
    return explicit;
  }
  const env = String(episode.env_id || "");
  if (env.toLowerCase().includes("libero")) {
    return "LIBERO";
  }
  return env;
}

export function episodeTaskSuite(episode: DatasetEpisode): string {
  const explicit =
    metadataString(episode, "task_suite") ||
    metadataString(episode, "suite") ||
    metadataString(episode, "benchmark");
  if (explicit) {
    return explicit;
  }
  return String(episode.env_id || "");
}

export function metadataString(episode: DatasetEpisode, key: string): string {
  const value = episode.metadata?.[key];
  if (value === undefined || value === null || value === "") {
    return "";
  }
  if (typeof value === "object") {
    return "";
  }
  return String(value);
}

export function outcomeClass(outcome: string): string {
  const normalized = outcome.toLowerCase();
  if (["success", "passed", "pass", "true"].includes(normalized)) {
    return "success";
  }
  if (["failure", "failed", "fail", "false", "timeout"].includes(normalized)) {
    return "failure";
  }
  return "";
}

export function imageTokenMapQueryKey(
  traceId: string,
  callIndex: number | undefined,
  siteName: string,
  feature: number,
) {
  return ["image-token-map", traceId, callIndex, siteName, feature] as const;
}

export function overlayStatusForSelection({
  attentionMapFetching,
  attentionMapPlaceholder,
  attentionHead,
  attentionQueryToken,
  expertTokenDetailsFetching,
  feature,
  imageTokenMapFetching,
  imageTokenMapPlaceholder,
  inspectorContext,
  selectedExpertToken,
  selectedSite,
}: {
  attentionMapFetching: boolean;
  attentionMapPlaceholder: boolean;
  attentionHead: number | null;
  attentionQueryToken: number | null;
  expertTokenDetailsFetching: boolean;
  feature: number;
  imageTokenMapFetching: boolean;
  imageTokenMapPlaceholder: boolean;
  inspectorContext: InspectorContext;
  selectedExpertToken: number | null;
  selectedSite?: ActivationSite;
}): CameraOverlayStatus | undefined {
  const siteLabel = selectedSite ? siteOptionLabel(selectedSite) : "Selected site";
  if (inspectorContext === "vlm") {
    return {
      detail: `${siteLabel} / c${feature}`,
      isStale: imageTokenMapPlaceholder,
      isUpdating: imageTokenMapFetching,
      label: "Channel Map",
      mode: inspectorContext,
    };
  }
  if (inspectorContext === "attention") {
    return {
      detail: `head ${attentionHead === null ? "avg" : attentionHead} / slot ${attentionQueryToken === null ? "avg" : attentionQueryToken}`,
      isStale: attentionMapPlaceholder,
      isUpdating: attentionMapFetching,
      label: "Attention Map",
      mode: inspectorContext,
    };
  }
  if (inspectorContext === "expert") {
    return {
      detail: selectedExpertToken === null ? `${siteLabel} / token -` : `${siteLabel} / token ${selectedExpertToken} / c${feature}`,
      isStale: false,
      isUpdating: expertTokenDetailsFetching,
      label: "Expert Token",
      mode: inspectorContext,
    };
  }
  return undefined;
}
