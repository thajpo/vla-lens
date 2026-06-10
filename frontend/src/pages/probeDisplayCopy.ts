import type { EpisodeLensView } from "../types/dataset";
import type { EvidenceClaimLevel, ModelLocusRef, ProbeEvidenceBundle } from "../types/probeEvidence";
import { primitivesByKind } from "../types/probeEvidence";

export type ProbeDisplayField = {
  detail?: string;
  label: string;
  value: string;
};

export type ProbeDisplaySpec = {
  input: ProbeDisplayField;
  objective: ProbeDisplayField;
  output: ProbeDisplayField;
  prediction: ProbeDisplayField;
};

export function probeEvidenceDisplaySpec(bundle: ProbeEvidenceBundle): ProbeDisplaySpec {
  const provenance = primitivesByKind(bundle, "provenance")[0];
  const fields = provenance?.kind === "provenance" ? provenance.fields ?? {} : {};
  const inputValue = evidenceInputLabel(fields.Input, bundle);
  return {
    input: {
      detail: nonRepeatingDetail(inputValue, humanizeProbeText(bundle.geometry.input_basis)),
      label: "Input",
      value: inputValue,
    },
    objective: {
      label: "Objective",
      value: evidenceObjectiveLabel(fields.Objective, bundle),
    },
    output: {
      label: "Output",
      value: humanizeProbeText(String(fields.Output ?? bundle.geometry.output_kind)),
    },
    prediction: {
      label: "Prediction",
      value: humanizeProbeText(String(fields.Prediction ?? bundle.artifact.target ?? bundle.artifact.name)),
    },
  };
}

export function probeLensViewDisplaySpec(view: EpisodeLensView): ProbeDisplaySpec {
  const spec = view.lens.spec ?? {};
  return {
    input: {
      label: "Input",
      value: humanizeProbeText(String(spec.input ?? "model features")),
    },
    objective: {
      label: "Objective",
      value: humanizeProbeText(String(spec.objective ?? "probe")),
    },
    output: {
      label: "Output",
      value: humanizeProbeText(String(spec.output ?? "False / True")),
    },
    prediction: {
      label: "Prediction",
      value: humanizeProbeText(String(spec.prediction ?? view.lens.display_name)),
    },
  };
}

export function contributionCaveat(claimLevel?: EvidenceClaimLevel | string | null): string {
  if (claimLevel === "human_labeled_feature" || claimLevel === "semantic_hypothesis") {
    return "semantic label";
  }
  if (claimLevel === "grouped_model_locus") {
    return "model-site grouped";
  }
  return "not semantic-labeled";
}

export function contributionFeatureLabel(value: string | null | undefined, fallback: number): string {
  const text = String(value ?? "").trim();
  const match = text.match(/(?:dim|feature|sae|head)_?([0-9]+)/i);
  if (match) {
    return `dim ${match[1]}`;
  }
  return text ? humanizeProbeText(text) : `dim ${fallback}`;
}

export function conciseModelSiteLabel(value: string | null | undefined): string {
  const text = String(value ?? "").trim();
  const normalized = normalizeProbeText(text);
  if (!text) {
    return "";
  }
  if (normalized.includes("action head")) {
    return normalized.includes("output") ? "Action head output" : "Action head";
  }
  if (normalized.includes("expert")) {
    return "Expert activations";
  }
  if (normalized.includes("vlm")) {
    return "VLM activations";
  }
  return humanizeProbeText(text.replace(/^pi05\./, ""));
}

export function modelLocusDisplayLabel(locus?: ModelLocusRef | null): string {
  if (!locus) {
    return "";
  }
  return conciseModelSiteLabel(locus.model_site_id ?? locus.module ?? "") ||
    (locus.layer === null || locus.layer === undefined ? "" : `Layer ${locus.layer}`);
}

export function humanizeProbeText(value: string): string {
  const cleaned = value
    .replace(/^probe for\s+/i, "")
    .replace(/^selected\s+/i, "")
    .replace(/[._-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (!cleaned) {
    return "";
  }
  return cleaned.replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function nonRepeatingDetail(value: string, detail?: string | null): string {
  const text = String(detail ?? "").trim();
  return text && normalizeProbeText(value) !== normalizeProbeText(text) ? text : "";
}

export function sameDisplayText(left: string | null | undefined, right: string | null | undefined): boolean {
  return normalizeProbeText(left) === normalizeProbeText(right);
}

function evidenceInputLabel(value: unknown, bundle: ProbeEvidenceBundle): string {
  const raw = String(value ?? "").trim();
  if (raw && !genericInputLabel(raw)) {
    return humanizeProbeText(raw);
  }
  const source = String(bundle.artifact.source?.module ?? "").trim();
  const locus = primitivesByKind(bundle, "model_locus")[0]?.locus;
  return conciseModelSiteLabel(source) || modelLocusDisplayLabel(locus) || humanizeProbeText(bundle.geometry.input_basis);
}

function evidenceObjectiveLabel(value: unknown, bundle: ProbeEvidenceBundle): string {
  const raw = String(value ?? bundle.artifact.training?.objective ?? "").trim();
  if (!raw) {
    return "Probe";
  }
  const target = String(bundle.artifact.target ?? bundle.artifact.name ?? "");
  const normalizedRaw = normalizeProbeText(raw);
  const normalizedTarget = normalizeProbeText(target);
  if (
    normalizedRaw === normalizedTarget ||
    normalizedRaw === `probe for ${normalizedTarget}` ||
    normalizedRaw.startsWith(`probe for ${normalizedTarget}`)
  ) {
    return "Probe";
  }
  return humanizeProbeText(raw);
}

function genericInputLabel(value: string): boolean {
  const normalized = normalizeProbeText(value);
  return normalized === "selected model sites" || normalized === "model sites" || normalized === "selected source";
}

function normalizeProbeText(value: unknown): string {
  return String(value ?? "")
    .replace(/[._-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();
}
