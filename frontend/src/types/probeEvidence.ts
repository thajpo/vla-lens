export type LensCapability =
  | "score_series"
  | "ranked_moments"
  | "thresholding"
  | "prediction"
  | "uncertainty"
  | "contribution_breakdown"
  | "model_locus_view"
  | "visual_heatmap"
  | "cohort_summary"
  | "failure_cases"
  | "comparison";

export type LensGeometry = {
  temporal_scope:
    | "episode"
    | "timestep"
    | "window"
    | "event"
    | "token"
    | "frame"
    | "policy_call";
  output_kind:
    | "scalar"
    | "class_label"
    | "class_distribution"
    | "vector"
    | "ranked_features"
    | "attribution_map"
    | "heatmap";
  input_basis:
    | "layer_activation"
    | "pooled_layer_activation"
    | "sae_feature"
    | "attention_head_output"
    | "token_state"
    | "image_patch"
    | "action_state"
    | "custom";
  locus_kind:
    | "none"
    | "model_locus"
    | "visual_locus"
    | "action_locus"
    | "token_locus"
    | "mixed_locus";
  capabilities: LensCapability[];
};

export type ModelLocusRef = {
  model_site_id?: string | null;
  layer?: number | null;
  module?: string | null;
  stream?: string | null;
  head_index?: number | null;
  token_index?: number | null;
  channel_index?: number | null;
};

export type ProbeLensArtifact = {
  lens_id: string;
  lens_version: string;
  lens_type: "probe";
  name: string;
  target?: string | null;
  source_model?: Record<string, unknown>;
  source?: Record<string, unknown>;
  training?: Record<string, unknown>;
  created_at?: string | null;
};

export type LensRun = {
  lens_run_id: string;
  lens_id: string;
  lens_version: string;
  dataset_id: string;
  episode_ids?: string[];
  capture_profile_id?: string | null;
  computed_at: string;
  result_version: string;
  status: "complete" | "partial" | "failed";
  evidence_bundle_id?: string | null;
};

export type EvidenceClaimLevel =
  | "numeric_only"
  | "grouped_model_locus"
  | "human_labeled_feature"
  | "semantic_hypothesis";

export type EvidencePrimitiveKind =
  | "provenance"
  | "score_series"
  | "ranked_moments"
  | "prediction"
  | "contribution"
  | "model_locus"
  | "cohort_summary"
  | "failure_case";

export type RankingKind =
  | "top"
  | "bottom"
  | "uncertain"
  | "false_positive"
  | "false_negative"
  | "largest_delta";

export type ArrayRef = {
  uri: string;
  format?: string;
  shape?: number[];
  dtype?: string | null;
};

export type LensProvenanceEvidence = {
  kind: "provenance";
  lens_id: string;
  lens_run_id: string;
  fields?: Record<string, unknown>;
};

export type ScoreSeriesEvidence = {
  kind: "score_series";
  lens_id: string;
  lens_run_id: string;
  episode_id: string;
  time_axis: "timestep" | "frame" | "token" | "window" | "policy_call";
  values_ref: ArrayRef;
  summary: Record<string, number>;
  threshold?: number | null;
};

export type RankedMoment = {
  episode_id: string;
  timestep?: number | null;
  policy_call?: number | null;
  frame_idx?: number | null;
  score?: number | null;
  prediction?: string | boolean | number | null;
  label?: string | boolean | number | null;
  confidence?: number | null;
  thumbnail_ref?: string | null;
};

export type RankedMomentsEvidence = {
  kind: "ranked_moments";
  lens_id: string;
  lens_run_id: string;
  ranking: RankingKind;
  moments: RankedMoment[];
};

export type PredictionEvidence = {
  kind: "prediction";
  lens_id: string;
  lens_run_id: string;
  episode_id: string;
  timestep?: number | null;
  policy_call?: number | null;
  prediction: string | boolean | number;
  label?: string | boolean | number | null;
  confidence?: number | null;
  correct?: boolean | null;
  split?: "train" | "validation" | "test" | "missing" | null;
};

export type ContributionItem = {
  key: string;
  value: number;
  rank: number;
  sign?: "positive" | "negative" | null;
  model_locus?: ModelLocusRef | null;
  label?: string | null;
  description?: string | null;
};

export type ContributionEvidence = {
  kind: "contribution";
  lens_id: string;
  lens_run_id: string;
  episode_id: string;
  timestep?: number | null;
  policy_call?: number | null;
  basis:
    | "raw_activation_dimension"
    | "sae_feature"
    | "attention_head_output"
    | "token_state"
    | "action_dimension"
    | "custom";
  claim_level: EvidenceClaimLevel;
  items: ContributionItem[];
};

export type ModelLocusEvidence = {
  kind: "model_locus";
  lens_id: string;
  lens_run_id: string;
  episode_id?: string | null;
  timestep?: number | null;
  policy_call?: number | null;
  locus: ModelLocusRef;
  source_label?: string | null;
};

export type ResearchSelectionState = {
  dataset_id?: string | null;
  lens_id?: string | null;
  lens_run_id?: string | null;
  episode_id?: string | null;
  timestep?: number | null;
  policy_call?: number | null;
  time_window?: { start: number; end: number } | null;
  ranking?: RankingKind | null;
  cohort_id?: string | null;
  model_locus?: ModelLocusRef | null;
  feature_id?: string | null;
};

export type EvidenceCohortRef = {
  cohort_id: string;
  source: "ranking" | "filter" | "manual" | "saved";
  selection: ResearchSelectionState;
  count: number;
};

export type CohortSummaryEvidence = {
  kind: "cohort_summary";
  lens_id: string;
  lens_run_id: string;
  cohort: EvidenceCohortRef;
  summary: Record<string, number>;
};

export type FailureCaseEvidence = {
  kind: "failure_case";
  lens_id: string;
  lens_run_id: string;
  ranking: "false_positive" | "false_negative" | "high_confidence_wrong";
  moments: RankedMoment[];
};

export type EvidencePrimitive =
  | LensProvenanceEvidence
  | ScoreSeriesEvidence
  | RankedMomentsEvidence
  | PredictionEvidence
  | ContributionEvidence
  | ModelLocusEvidence
  | CohortSummaryEvidence
  | FailureCaseEvidence;

export type UnavailableReason = {
  capability: LensCapability;
  panel_id?: string | null;
  reason:
    | "missing_scores"
    | "missing_labels"
    | "missing_contribution_basis"
    | "pooled_representation"
    | "missing_model_locus"
    | "unsupported_probe_type"
    | "not_computed";
  message: string;
};

export type ProbeEvidenceBundle = {
  bundle_id: string;
  family: "probe";
  artifact: ProbeLensArtifact;
  run: LensRun;
  geometry: LensGeometry;
  capabilities: LensCapability[];
  primitives: EvidencePrimitive[];
  unavailable: UnavailableReason[];
};

export type PanelSpec = {
  panel_id: string;
  consumes: EvidencePrimitiveKind[];
  requires_capabilities?: LensCapability[];
  requires_geometry?: Partial<Pick<LensGeometry, "temporal_scope" | "output_kind" | "input_basis" | "locus_kind">>;
  unavailable_copy: string;
};

export type PanelAvailability = {
  panel_id: string;
  available: boolean;
  reason?: string | null;
  message?: string | null;
};

export function primitiveKinds(bundle: ProbeEvidenceBundle): Set<EvidencePrimitiveKind> {
  return new Set(bundle.primitives.map((primitive) => primitive.kind));
}

export function primitivesByKind<K extends EvidencePrimitiveKind>(
  bundle: ProbeEvidenceBundle,
  kind: K,
): Extract<EvidencePrimitive, { kind: K }>[] {
  return bundle.primitives.filter(
    (primitive): primitive is Extract<EvidencePrimitive, { kind: K }> => primitive.kind === kind,
  );
}

export function rankedMoments(
  bundle: ProbeEvidenceBundle,
  ranking: RankingKind,
): RankedMoment[] {
  const primitive = bundle.primitives.find(
    (item): item is RankedMomentsEvidence => item.kind === "ranked_moments" && item.ranking === ranking,
  );
  return primitive?.moments ?? [];
}

export function selectAvailablePanels(
  bundle: ProbeEvidenceBundle,
  panelSpecs: PanelSpec[],
): PanelAvailability[] {
  const present = primitiveKinds(bundle);
  const capabilities = new Set(bundle.capabilities);
  return panelSpecs.map((spec) => {
    const missingCapability = (spec.requires_capabilities ?? []).find(
      (capability) => !capabilities.has(capability),
    );
    const missingPrimitive = spec.consumes.find((kind) => !present.has(kind));
    const geometryMismatch = firstGeometryMismatch(bundle.geometry, spec.requires_geometry ?? {});
    if (!missingCapability && !missingPrimitive && !geometryMismatch) {
      return { panel_id: spec.panel_id, available: true };
    }
    const matchedReason = matchingUnavailable(bundle, spec, missingCapability);
    return {
      panel_id: spec.panel_id,
      available: false,
      reason:
        matchedReason?.reason ??
        (missingCapability
          ? `missing capability: ${missingCapability}`
          : missingPrimitive
            ? `missing evidence primitive: ${missingPrimitive}`
            : geometryMismatch),
      message: matchedReason?.message ?? spec.unavailable_copy,
    };
  });
}

function firstGeometryMismatch(
  geometry: LensGeometry,
  required: PanelSpec["requires_geometry"],
): string | null {
  for (const [key, expected] of Object.entries(required ?? {})) {
    if (geometry[key as keyof typeof required] !== expected) {
      return `geometry ${key}=${String(geometry[key as keyof typeof required])} does not match required ${String(expected)}`;
    }
  }
  return null;
}

function matchingUnavailable(
  bundle: ProbeEvidenceBundle,
  spec: PanelSpec,
  missingCapability?: LensCapability,
): UnavailableReason | undefined {
  return (
    bundle.unavailable.find((reason) => reason.panel_id === spec.panel_id) ??
    bundle.unavailable.find((reason) => reason.capability === missingCapability)
  );
}

export function defaultProbePanelSpecs(): PanelSpec[] {
  return [
    {
      panel_id: "probe_provenance",
      consumes: ["provenance"],
      unavailable_copy: "Probe provenance is unavailable for this run.",
    },
    {
      panel_id: "score_series",
      consumes: ["score_series"],
      requires_capabilities: ["score_series"],
      unavailable_copy: "Score series is unavailable for this probe run.",
    },
    {
      panel_id: "ranked_moments",
      consumes: ["ranked_moments"],
      requires_capabilities: ["ranked_moments"],
      unavailable_copy: "Ranked moments are unavailable for this probe run.",
    },
    {
      panel_id: "prediction",
      consumes: ["prediction"],
      requires_capabilities: ["prediction"],
      unavailable_copy: "Predictions are unavailable for this probe run.",
    },
    {
      panel_id: "contribution",
      consumes: ["contribution"],
      requires_capabilities: ["contribution_breakdown"],
      unavailable_copy: "Contribution breakdown is unavailable for this probe run.",
    },
    {
      panel_id: "model_locus",
      consumes: ["model_locus"],
      requires_capabilities: ["model_locus_view"],
      requires_geometry: { locus_kind: "model_locus" },
      unavailable_copy: "Model locus is unavailable for this probe run.",
    },
    {
      panel_id: "failure_cases",
      consumes: ["failure_case"],
      requires_capabilities: ["failure_cases"],
      unavailable_copy: "Failure cases are unavailable for this probe run.",
    },
  ];
}
