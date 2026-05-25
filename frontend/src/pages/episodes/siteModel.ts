import type { ActivationSite } from "../../types/dataset";
import {
  type CaptureGroupId,
  type InspectionMode,
  type InspectorContext,
  type PipelineNode,
  type ProbeLayerRef,
  type ProbeTone,
} from "./shared";
import { episodeProbeNumber } from "./episodeProbeModel";
import { formatLayerNumber, formatMaybeNumber, labelFromSnake } from "./formatters";

export function preferredPipelineSite(sites: ActivationSite[]): ActivationSite | undefined {
  return (
    sites.find(
      (site) =>
        site.name === "pi05.vlm.prefix.image_hidden_tokens" &&
        site.axes?.includes("token"),
    ) ??
    sites.find(
      (site) =>
        site.name.includes(".vlm.") &&
        site.token_kind === "image" &&
        site.tensor_type === "hidden_tokens",
    ) ??
    sites.find(
      (site) =>
        site.name.includes(".vlm.") &&
        site.tensor_type === "hidden_tokens" &&
        site.axes?.includes("token"),
    ) ??
    sites.find(
      (site) =>
        site.name.includes(".expert.") &&
        site.tensor_type === "hidden_tokens" &&
        Number(site.layer) === 17,
    ) ??
    sites.find((site) => site.name.includes(".expert.") && site.tensor_type === "hidden_tokens") ??
    sites[0]
  );
}

export function inspectorContextForSite(site?: ActivationSite): InspectorContext {
  if (!site) {
    return "other";
  }
  if (isAttentionSite(site)) {
    return "attention";
  }
  if (site.name.includes(".expert.") || site.segment === "action_expert") {
    return "expert";
  }
  if (site.name.includes(".vlm.") || site.segment === "vlm_prefix" || site.token_space_id === "pi05.prefix") {
    return "vlm";
  }
  return "other";
}

export function isFeatureActivationSite(site?: ActivationSite): boolean {
  return Boolean(
    site?.axes?.includes("channel") &&
      !isAttentionSite(site),
  );
}

export function isAttentionSite(site?: ActivationSite): boolean {
  return Boolean(
    site &&
      (site.tensor_type === "attention" ||
        site.tensor_type === "attention_probs" ||
        site.family === "attention" ||
        site.role === "attention_probs" ||
        site.name.includes("attention")),
  );
}

export function isAttentionMapSite(site?: ActivationSite): boolean {
  return Boolean(
    site &&
      (site.tensor_type === "attention" ||
        site.tensor_type === "attention_probs" ||
        site.role === "attention_probs" ||
        site.name.endsWith(".by_step.attention") ||
        site.name.endsWith(".prefix.attention") ||
        site.name.endsWith(".attention.attention_probs")),
  );
}

export function siteLayerNumber(site?: ActivationSite): number | null {
  const value = episodeProbeNumber(site?.layer);
  return value === null ? null : value;
}

export function siteForProbeLayer(sites: ActivationSite[], layer: number | null): ActivationSite | undefined {
  if (layer === null) {
    return undefined;
  }
  const sameLayer = sites.filter((site) => siteLayerNumber(site) === Number(layer));
  return (
    sameLayer.find(
      (site) =>
        (site.segment === "action_expert" || site.name.includes(".expert.")) &&
        site.tensor_type === "hidden_tokens" &&
        site.axes?.includes("channel"),
    ) ??
    sameLayer.find((site) => site.tensor_type === "hidden_tokens" && site.axes?.includes("channel")) ??
    sameLayer[0]
  );
}

export function siteForProbeRef(sites: ActivationSite[], ref?: ProbeLayerRef): ActivationSite | undefined {
  if (!ref) {
    return undefined;
  }
  if (ref.modelSiteId) {
    const exact = sites.find((site) => site.name === ref.modelSiteId);
    if (exact) {
      return exact;
    }
  }
  return siteForProbeLayer(sites, ref.layer);
}

export function probeRefsForNode(node: PipelineNode, refs: ProbeLayerRef[]): ProbeLayerRef[] {
  const exact = refs.filter(
    (ref) => ref.modelSiteId && node.allSites.some((site) => site.name === ref.modelSiteId),
  );
  if (exact.length) {
    return exact;
  }
  const nodeLayers = new Set(
    node.allSites
      .map(siteLayerNumber)
      .filter((layer): layer is number => layer !== null)
      .map(Number),
  );
  if (!nodeLayers.size) {
    return [];
  }
  const nodeIsExpert = node.allSites.some(
    (site) => site.segment === "action_expert" || site.name.includes(".expert."),
  );
  return refs.filter(
    (ref) =>
      ref.layer !== null &&
      nodeLayers.has(Number(ref.layer)) &&
      (!ref.modelSiteId || !nodeIsExpert || ref.modelSiteId.includes(".expert.")),
  );
}

export function probeRefTone(ref: ProbeLayerRef): ProbeTone {
  if (ref.correct === false) {
    return "incorrect";
  }
  if (ref.correct === true) {
    return "correct";
  }
  return "unscored";
}

export function probeToneForRefs(refs: ProbeLayerRef[]): ProbeTone {
  if (refs.some((ref) => probeRefTone(ref) === "incorrect")) {
    return "incorrect";
  }
  if (refs.some((ref) => probeRefTone(ref) === "correct")) {
    return "correct";
  }
  return "unscored";
}

export function probeLayerTitle(ref: ProbeLayerRef): string {
  return [
    ref.target || ref.name,
    ref.layer === null ? "" : `L${formatLayerNumber(ref.layer)}`,
    ref.policyCall === null ? "" : `call ${ref.policyCall}`,
    ref.confidence === null || ref.confidence === undefined ? "" : `conf ${formatMaybeNumber(ref.confidence)}`,
  ].filter(Boolean).join(" / ");
}


export function siteUsesGenerationStep(site?: ActivationSite): boolean {
  return Boolean(site?.axes?.includes("generation_step"));
}

export function expertTokenSiteForSite(
  sites: ActivationSite[],
  selectedSite?: ActivationSite,
): ActivationSite | undefined {
  if (selectedSite?.name.includes(".expert.") && selectedSite.tensor_type === "hidden_tokens") {
    return selectedSite;
  }
  const selectedLayer = selectedSite?.layer;
  if (selectedLayer !== null && selectedLayer !== undefined) {
    const layer = Number(selectedLayer);
    const layerTokenSite = sites.find(
      (site) =>
        site.name.includes(".expert.") &&
        site.tensor_type === "hidden_tokens" &&
        Number(site.layer) === layer,
    );
    if (layerTokenSite) {
      return layerTokenSite;
    }
  }
  return sites.find((site) => site.name.includes(".expert.") && site.tensor_type === "hidden_tokens");
}

export function attentionSiteForSite(
  sites: ActivationSite[],
  selectedSite: ActivationSite | undefined,
): ActivationSite | undefined {
  if (isAttentionMapSite(selectedSite)) {
    return selectedSite;
  }
  const context = selectedSite?.name.includes(".vlm.")
    ? "vlm"
    : selectedSite?.name.includes(".expert.")
      ? "expert"
      : inspectorContextForSite(selectedSite);
  const selectedLayer = selectedSite?.layer;
  const sameLayer = (site: ActivationSite) =>
    selectedLayer !== null &&
    selectedLayer !== undefined &&
    Number(site.layer) === Number(selectedLayer);
  const attentionSites = sites.filter(
    (site) =>
      isAttentionSite(site) &&
      (context === "vlm"
        ? site.name.includes(".vlm.") || site.segment === "vlm_prefix"
        : context === "expert"
          ? site.name.includes(".expert.") || site.segment === "action_expert"
          : true),
  );
  return (
    attentionSites.find(
      (site) => sameLayer(site) && site.name.endsWith(".attention.attention_probs"),
    ) ??
    attentionSites.find((site) => sameLayer(site) && site.name.endsWith(".prefix.attention")) ??
    attentionSites.find((site) => sameLayer(site) && site.name.endsWith(".by_step.attention")) ??
    attentionSites.find(sameLayer) ??
    attentionSites[0]
  );
}

export function generationStepCountForSite(site?: ActivationSite): number {
  if (!site?.axes?.length || !site.shape?.length) {
    return 1;
  }
  const index = site.axes.indexOf("generation_step");
  if (index < 0) {
    return 1;
  }
  return Math.max(1, Number(site.shape[index]) || 1);
}

export function channelCountForSite(site?: ActivationSite): number {
  return axisCountForSite(site, "channel");
}

export function axisCountForSite(site: ActivationSite | undefined, axis: string): number {
  if (!site?.axes?.length || !site.shape?.length) {
    return 0;
  }
  const index = site.axes.indexOf(axis);
  if (index < 0) {
    return 0;
  }
  return Math.max(0, Number(site.shape[index]) || 0);
}


export function isInspectableSite(site: ActivationSite): boolean {
  const role = String(site.role || "");
  return Boolean(
    isAttentionSite(site) ||
      site.family === "attention" ||
      (site.tensor_type === "hidden_tokens" && site.axes?.includes("channel")) ||
      (site.tensor_type === "hidden_mean" && site.axes?.includes("channel")) ||
      site.role === "input_embeddings" ||
      site.tensor_type === "embedding" ||
      site.family === "residual" ||
      site.family === "normalization" ||
      site.family === "mlp" ||
      site.family === "cache" ||
      site.tensor_type === "cache" ||
      site.name.includes(".kv_cache.") ||
      site.name.includes("mask") ||
      site.name.includes("position_ids") ||
      site.name.includes(".rope.") ||
      role.includes("mask") ||
      role.includes("position") ||
      role.includes("rope") ||
      site.family === "action_head" ||
      site.tensor_type === "action_head" ||
      site.name.includes("action_head"),
  );
}

export function filterVisibleCaptureSites(sites: ActivationSite[]): ActivationSite[] {
  const hasTokenFeatures = sites.some(
    (site) => site.tensor_type === "hidden_tokens" && site.axes?.includes("channel"),
  );
  if (!hasTokenFeatures) {
    return sites;
  }
  return sites.filter((site) => site.tensor_type !== "hidden_mean");
}

export function preferredSiteWithinNode(sites: ActivationSite[], mode: InspectionMode = "features"): ActivationSite | undefined {
  const modeSites = mode === "advanced" ? sites : sites.filter((site) => inspectionModeForSite(site) === mode);
  const candidates = modeSites.length ? modeSites : sites;
  return (
    candidates.find((site) => site.token_kind === "image" && site.tensor_type === "hidden_tokens") ??
    candidates.find((site) => site.tensor_type === "hidden_tokens" && !isAttentionSite(site)) ??
    candidates.find(isAttentionMapSite) ??
    candidates.find((site) => String(site.role || "") === "q") ??
    candidates.find((site) => String(site.role || "") === "k") ??
    candidates.find((site) => String(site.role || "") === "v") ??
    candidates.find((site) => site.role === "input_embeddings" || site.tensor_type === "embedding") ??
    candidates[0]
  );
}

export function sortSites(sites: ActivationSite[]): ActivationSite[] {
  const rank = (site: ActivationSite) => {
    if (site.tensor_type === "hidden_tokens") return 0;
    if (site.role === "input_embeddings" || site.tensor_type === "embedding") return 1;
    if (site.family === "residual") return 2;
    if (site.family === "normalization") return 3;
    if (isAttentionMapSite(site)) return 4;
    if (String(site.role || "").startsWith("q") || site.tensor_type === "attention_q") return 5;
    if (String(site.role || "").startsWith("k") || site.tensor_type === "attention_k") return 6;
    if (String(site.role || "").startsWith("v") || site.tensor_type === "attention_v") return 7;
    if (isAttentionSite(site)) return 8;
    if (site.family === "mlp") return 9;
    if (site.family === "cache") return 10;
    if (site.tensor_type === "hidden_mean") return 11;
    return 12;
  };
  return [...sites].sort((left, right) => rank(left) - rank(right) || left.name.localeCompare(right.name));
}

export function siteOptionLabel(site: ActivationSite): string {
  const role = String(site.role || "");
  if (role === "input_embeddings" || site.tensor_type === "embedding") {
    return "Input features";
  }
  if (site.tensor_type === "hidden_mean") {
    return "Average features";
  }
  if (site.tensor_type === "hidden_tokens") {
    if (site.token_kind === "image") {
      return "Image features";
    }
    if (site.token_kind === "prefix") {
      return "Text + image features";
    }
    return "Action features";
  }
  if (["q", "k", "v"].includes(role)) {
    if (role === "q") return "Query vectors";
    if (role === "k") return "Key vectors";
    return "Value vectors";
  }
  if (role.includes("logits") || role.includes("scores")) {
    return "Attention scores";
  }
  if (role.includes("o_proj")) {
    return "Attention output";
  }
  if (isAttentionMapSite(site)) {
    return site.name.includes("key_mass") ? "Attention summary" : "Attention map";
  }
  if (isAttentionSite(site)) {
    return labelFromSnake(role || site.tensor_type || "Attention capture");
  }
  if (site.family === "residual") {
    return residualCaptureLabel(role);
  }
  if (site.family === "normalization") {
    return role.includes("adarms") ? adarmsCaptureLabel(role) : "Normalized features";
  }
  if (site.family === "mlp") {
    return mlpCaptureLabel(role);
  }
  if (site.family === "cache") {
    return role.includes("key") ? "KV cache key" : role.includes("value") ? "KV cache value" : "Cache";
  }
  if (site.name.includes("action_head")) {
    return role === "action_head_output" ? "Action output" : "Head input";
  }
  return labelFromSnake(site.tensor_type || site.token_kind || site.module?.split(".").at(-1) || "Site");
}

export function residualCaptureLabel(role: string): string {
  if (role.includes("pre_attention")) return "Layer input";
  if (role.includes("post_attention")) return "After attention";
  if (role.includes("pre_mlp")) return "Before MLP";
  if (role.includes("post_mlp")) return "Layer output";
  return "Layer state";
}

export function adarmsCaptureLabel(role: string): string {
  if (role.includes("scale")) return "Conditioning scale";
  if (role.includes("shift")) return "Conditioning shift";
  if (role.includes("gate")) return "Conditioning gate";
  return "Conditioning";
}

export function mlpCaptureLabel(role: string): string {
  if (role.includes("gate")) return "MLP gate";
  if (role.includes("up")) return "MLP up";
  if (role.includes("intermediate")) return "MLP hidden";
  if (role.includes("down")) return "MLP down";
  if (role.includes("output")) return "MLP output";
  return "MLP";
}

export function captureGroupForSite(site: ActivationSite): CaptureGroupId {
  const role = String(site.role || "");
  if (site.family === "mlp") return "mlp";
  if (site.family === "normalization" && role.includes("adarms")) return "mlp";
  if (site.family === "cache" || site.tensor_type === "cache" || site.name.includes(".kv_cache.")) return "saved_state";
  if (
    isAttentionSite(site) ||
    site.family === "attention" ||
    role.startsWith("q") ||
    role.startsWith("k") ||
    role.startsWith("v")
  ) {
    return "attention";
  }
  if (site.family === "action_head" || site.tensor_type === "action_head" || site.name.includes("action_head")) {
    return "action";
  }
  if (
    site.tensor_type === "hidden_tokens" ||
    site.tensor_type === "hidden_mean" ||
    site.tensor_type === "embedding" ||
    site.family === "residual" ||
    site.family === "normalization"
  ) {
    return "features";
  }
  return "other";
}

export const inspectionModes: InspectionMode[] = [
  "features",
  "attention",
  "computation",
  "saved_state",
  "advanced",
];

export function inspectionModeForSite(site?: ActivationSite): InspectionMode {
  if (!site) {
    return "features";
  }
  const group = captureGroupForSite(site);
  if (group === "attention") {
    return "attention";
  }
  if (group === "mlp") {
    return "computation";
  }
  if (group === "saved_state") {
    return "saved_state";
  }
  if (group === "features" || group === "action") {
    return "features";
  }
  return "advanced";
}

export function inspectionModeLabel(mode: InspectionMode): string {
  const labels: Record<InspectionMode, string> = {
    advanced: "All captures",
    attention: "Attention",
    computation: "Computation",
    features: "Features",
    saved_state: "Cache",
  };
  return labels[mode];
}

export function inspectionModeEmptyMessage(mode: InspectionMode, node: PipelineNode): string {
  if (!node.captured) {
    return "This layer is not captured in the current profile.";
  }
  if (mode === "attention") {
    return "No attention captures are available for this layer in the current profile.";
  }
  if (mode === "computation") {
    return "No MLP, normalization, or conditioning captures are available for this layer in the current profile.";
  }
  if (mode === "saved_state") {
    return "No saved cache, mask, position, or RoPE captures are available for this layer.";
  }
  if (mode === "advanced") {
    return "No raw captures are available for this node.";
  }
  return "No feature captures are available for this layer in the current profile.";
}

export function summarizeSiteKinds(sites: ActivationSite[]): string {
  const labels = Array.from(new Set(sites.filter(isInspectableSite).map(siteOptionLabel)));
  return labels.slice(0, 2).join(" + ") || "site";
}

export function captureDescription(site: ActivationSite, node: PipelineNode): string {
  const role = String(site.role || "");
  const stack = node.family === "expert" ? "expert" : node.family === "vlm" ? "vlm" : "other";
  if (site.tensor_type === "hidden_tokens") {
    if (site.token_kind === "image") {
      return "One feature vector for each image patch. Pick a channel to see where that feature is strong in the camera view.";
    }
    if (site.token_kind === "action") {
      return "One feature vector for each action slot while the model is refining its action plan.";
    }
    return "Feature vectors for the text and image slots after this layer. Pick a channel to inspect one coordinate.";
  }
  if (site.tensor_type === "hidden_mean") {
    return "A smaller average of the layer features. Useful as a quick summary, but it loses token-level detail.";
  }
  if (role === "input_embeddings" || site.tensor_type === "embedding") {
    return stack === "expert"
      ? "The current action state after it has been converted into model features."
      : "The prompt and camera inputs after they have been converted into model features.";
  }
  if (role === "q") {
    return "The vectors each slot uses to ask for information before attention is computed.";
  }
  if (role === "k") {
    return "The vectors each slot exposes so other slots can match against it. Same shape as values, different contents.";
  }
  if (role === "v") {
    return "The content vectors that get mixed together after attention decides what to read. Same shape as keys, different contents.";
  }
  if (role.includes("logits") || role.includes("scores")) {
    return "Attention scores before they are turned into final attention weights.";
  }
  if (role.includes("o_proj")) {
    return "The attention result after the output projection, ready to be added back into the layer state.";
  }
  if (role.includes("attn_output_pre_o_proj")) {
    return "The attention result before the output projection.";
  }
  if (isAttentionMapSite(site)) {
    if (site.name.includes("key_mass")) {
      return "A compact attention summary. It is cheaper to view, but it is not the full attention map.";
    }
    return stack === "expert"
      ? "Pick a looking action slot, then see which saved scene/text slots and action slots it reads from."
      : "Pick a looking prompt/image slot, then see which prompt and image slots it reads from.";
  }
  if (isAttentionSite(site)) {
    return "An attention-family capture from this layer.";
  }
  if (site.family === "cache" || site.tensor_type === "cache" || site.name.includes(".kv_cache.")) {
    return "One layer's saved prefix keys or values. The action denoiser receives the full list of these per-layer tensors as past_key_values.";
  }
  if (site.family === "residual") {
    return "The layer's running state at this boundary. These are useful points for probes because they show what the layer has added.";
  }
  if (site.family === "normalization") {
    return role.includes("adarms")
      ? "A conditioning signal that scales, shifts, or gates the expert layer."
      : "The layer state after normalization, before the next major computation.";
  }
  if (site.family === "mlp") {
    return "The feed-forward part of the layer. These captures show the non-attention computation inside the block.";
  }
  if (site.family === "action_head" || site.tensor_type === "action_head" || site.name.includes("action_head")) {
    return role === "action_head_output"
      ? "The action chunk predicted from the expert state."
      : "The expert features just before they are projected into action values.";
  }
  return "A captured tensor from this part of the model.";
}
