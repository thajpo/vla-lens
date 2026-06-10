import type { EvidencePin } from "../types/probeEvidence";

export function evidencePinHash(pin: EvidencePin): string {
  const selection = pin.selection ?? {};
  const episodeId = selection.episode_id ?? "";
  if (!episodeId) {
    return "#evidence";
  }
  const params = new URLSearchParams();
  if (selection.lens_id) params.set("probe_id", selection.lens_id);
  if (selection.lens_run_id) params.set("lens_run_id", selection.lens_run_id);
  if (selection.dataset_id) params.set("dataset_id", selection.dataset_id);
  if (selection.ranking) params.set("rank", selection.ranking);
  if (typeof selection.policy_call === "number") params.set("call", String(selection.policy_call));
  if (typeof selection.timestep === "number") params.set("timestep", String(selection.timestep));
  const site = pin.evidence?.model_site_id ?? selection.model_locus?.model_site_id ?? selection.model_locus?.module;
  if (site) params.set("site", site);
  const contributor = selection.feature_id ?? pin.evidence?.selected_contributor;
  const feature = numericContributor(contributor);
  if (feature !== null) params.set("feature", String(feature));
  if (contributor && feature === null) params.set("contributor", contributor);
  const query = params.toString();
  return `#episode/${encodeURIComponent(episodeId)}${query ? `?${query}` : ""}`;
}

export function evidencePinSummary(pin: EvidencePin): string {
  const parts = [
    pin.selection.ranking ? pin.selection.ranking.replaceAll("_", " ") : "manual evidence",
    typeof pin.selection.policy_call === "number" ? `call ${pin.selection.policy_call}` : "",
    typeof pin.selection.timestep === "number" ? `timestep ${pin.selection.timestep}` : "",
    pin.evidence?.score === null || pin.evidence?.score === undefined ? "" : `score ${pin.evidence.score.toFixed(3)}`,
  ];
  return parts.filter(Boolean).join(" · ");
}

function numericContributor(value?: string | null): number | null {
  const match = String(value ?? "").match(/^(?:dim|feature|sae|head)_([0-9]+)$/);
  if (!match) return null;
  const parsed = Number(match[1]);
  return Number.isFinite(parsed) ? parsed : null;
}
