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
  | "failure_case"
  | "manual";

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

export type EvidencePinEvidence = {
  primitive_kind?: EvidencePrimitiveKind | null;
  score?: number | null;
  prediction?: string | boolean | number | null;
  confidence?: number | null;
  claim_level?: EvidenceClaimLevel | null;
  model_site_id?: string | null;
  selected_contributor?: string | null;
};

export type EvidencePin = {
  pin_id: string;
  created_utc: string;
  label: string;
  note?: string;
  selection: ResearchSelectionState;
  evidence: EvidencePinEvidence;
};

export type EvidencePinSavePayload = {
  label?: string;
  note?: string;
  selection: ResearchSelectionState;
  evidence: EvidencePinEvidence;
};

export type EvidencePinsResponse = {
  pins: EvidencePin[];
  total: number;
};

export type EvidencePinSaveResponse = EvidencePinsResponse & {
  pin: EvidencePin;
};

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

export type CurrentMomentEvidence = {
  selection: ResearchSelectionState;
  score_series: ScoreSeriesEvidence[];
  ranked_moments: RankedMoment[];
  predictions: PredictionEvidence[];
  contributions: ContributionEvidence[];
  model_loci: ModelLocusEvidence[];
  failure_moments: RankedMoment[];
  unavailable: UnavailableReason[];
};

export type LensFeatureContribution = ContributionItem & {
  claim_level: EvidenceClaimLevel;
  basis: ContributionEvidence["basis"];
};

export type PipelineLensAnnotation = {
  annotation_id: string;
  source: "model_locus" | "contribution";
  label: string;
  model_locus: ModelLocusRef;
  episode_id?: string | null;
  timestep?: number | null;
  policy_call?: number | null;
  claim_level?: EvidenceClaimLevel | null;
};

export type LensTemporalRow = {
  row_id: string;
  source: "ranked_moment" | "prediction" | "failure_case";
  episode_id: string;
  timestep?: number | null;
  policy_call?: number | null;
  ranking?: string | null;
  score?: number | null;
  confidence?: number | null;
  prediction?: string | boolean | number | null;
  label?: string | boolean | number | null;
};

export type EpisodeLensAdapter = {
  family: "probe";
  defaultSelection: (bundle: ProbeEvidenceBundle) => ResearchSelectionState;
  pipelineAnnotations: (
    bundle: ProbeEvidenceBundle,
    selection?: ResearchSelectionState | null,
  ) => PipelineLensAnnotation[];
  channelRanking: (
    bundle: ProbeEvidenceBundle,
    selection: ResearchSelectionState,
  ) => LensFeatureContribution[];
  timelineRows: (bundle: ProbeEvidenceBundle) => LensTemporalRow[];
  interventionSeed: (
    bundle: ProbeEvidenceBundle,
    selection: ResearchSelectionState,
  ) => null;
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

export function selectTopMoments(
  bundle: ProbeEvidenceBundle,
  ranking: RankingKind = "top",
  limit?: number | null,
): RankedMoment[] {
  return limitItems(rankedMoments(bundle, ranking), limit);
}

export function selectCurrentMomentEvidence(
  bundle: ProbeEvidenceBundle,
  selection: ResearchSelectionState,
): CurrentMomentEvidence {
  const ranked_moments: RankedMoment[] = [];
  const failure_moments: RankedMoment[] = [];
  for (const primitive of bundle.primitives) {
    if (primitive.kind === "ranked_moments") {
      if (selection.ranking && primitive.ranking !== selection.ranking) {
        continue;
      }
      ranked_moments.push(...primitive.moments.filter((moment) => momentMatchesSelection(moment, selection)));
    } else if (primitive.kind === "failure_case") {
      failure_moments.push(...primitive.moments.filter((moment) => momentMatchesSelection(moment, selection)));
    }
  }
  return {
    selection,
    score_series: primitivesByKind(bundle, "score_series").filter((primitive) =>
      evidenceMatchesSelection(primitive, selection),
    ),
    ranked_moments,
    predictions: primitivesByKind(bundle, "prediction").filter((primitive) =>
      evidenceMatchesSelection(primitive, selection),
    ),
    contributions: primitivesByKind(bundle, "contribution").filter((primitive) =>
      evidenceMatchesSelection(primitive, selection),
    ),
    model_loci: primitivesByKind(bundle, "model_locus").filter((primitive) =>
      evidenceMatchesSelection(primitive, selection),
    ),
    failure_moments,
    unavailable: selectUnavailableReasons(bundle),
  };
}

export function selectContributionRows(
  bundle: ProbeEvidenceBundle,
  selection?: ResearchSelectionState | null,
  limit?: number | null,
): ContributionItem[] {
  const rows = primitivesByKind(bundle, "contribution").flatMap((primitive) => {
    if (selection && !evidenceMatchesSelection(primitive, selection)) {
      return [];
    }
    return primitive.items.filter((item) => !selection?.feature_id || item.key === selection.feature_id);
  });
  rows.sort((left, right) => left.rank - right.rank);
  return limitItems(rows, limit);
}

export function selectContributionClaimLevel(
  bundle: ProbeEvidenceBundle,
  selection?: ResearchSelectionState | null,
): EvidenceClaimLevel | null {
  const levels = primitivesByKind(bundle, "contribution")
    .filter((primitive) => !selection || evidenceMatchesSelection(primitive, selection))
    .filter((primitive) => {
      if (!selection?.feature_id) {
        return true;
      }
      return primitive.items.some((item) => item.key === selection.feature_id);
    })
    .map((primitive) => primitive.claim_level);
  if (!levels.length) {
    return null;
  }
  return levels.sort((left, right) => claimLevelStrength[left] - claimLevelStrength[right])[0] ?? null;
}

export function selectUnavailableReasons(
  bundle: ProbeEvidenceBundle,
  options: { panel_id?: string | null; capability?: LensCapability | null } = {},
): UnavailableReason[] {
  return bundle.unavailable.filter(
    (reason) =>
      (!options.panel_id || reason.panel_id === options.panel_id) &&
      (!options.capability || reason.capability === options.capability),
  );
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
      unavailable_copy: "Activation source is unavailable for this probe run.",
    },
    {
      panel_id: "failure_cases",
      consumes: ["failure_case"],
      requires_capabilities: ["failure_cases"],
      unavailable_copy: "Failure cases are unavailable for this probe run.",
    },
    {
      panel_id: "unavailable_reasons",
      consumes: [],
      unavailable_copy: "Unavailable reason explanations are unavailable.",
    },
  ];
}

export const probeEpisodeLensAdapter: EpisodeLensAdapter = {
  family: "probe",

  defaultSelection(bundle) {
    const base = {
      dataset_id: bundle.run.dataset_id,
      lens_id: bundle.artifact.lens_id,
      lens_run_id: bundle.run.lens_run_id,
    };
    const rankedDefault = firstAvailableMoment(bundle);
    if (rankedDefault) {
      return {
        ...base,
        episode_id: rankedDefault.moment.episode_id,
        timestep: rankedDefault.moment.timestep ?? null,
        policy_call: rankedDefault.moment.policy_call ?? null,
        ranking: rankedDefault.ranking,
      };
    }
    const prediction = primitivesByKind(bundle, "prediction")[0];
    if (prediction) {
      return {
        ...base,
        episode_id: prediction.episode_id,
        timestep: prediction.timestep ?? null,
        policy_call: prediction.policy_call ?? null,
      };
    }
    const locus = primitivesByKind(bundle, "model_locus")[0];
    if (locus) {
      return {
        ...base,
        episode_id: locus.episode_id ?? null,
        timestep: locus.timestep ?? null,
        policy_call: locus.policy_call ?? null,
        model_locus: locus.locus,
      };
    }
    return base;
  },

  pipelineAnnotations(bundle, selection = null) {
    const annotations: PipelineLensAnnotation[] = [];
    for (const primitive of primitivesByKind(bundle, "model_locus")) {
      if (selection && !evidenceMatchesSelection(primitive, selection)) {
        continue;
      }
      annotations.push({
        annotation_id: `model_locus:${annotations.length}`,
        source: "model_locus",
        label: primitive.source_label ?? modelLocusLabel(primitive.locus),
        model_locus: primitive.locus,
        episode_id: primitive.episode_id ?? null,
        timestep: primitive.timestep ?? null,
        policy_call: primitive.policy_call ?? null,
      });
    }
    for (const primitive of primitivesByKind(bundle, "contribution")) {
      if (selection && !evidenceMatchesSelection(primitive, selection)) {
        continue;
      }
      for (const item of primitive.items) {
        if (!item.model_locus) {
          continue;
        }
        if (selection?.feature_id && item.key !== selection.feature_id) {
          continue;
        }
        annotations.push({
          annotation_id: `contribution:${item.key}`,
          source: "contribution",
          label: item.label ?? item.key,
          model_locus: item.model_locus,
          episode_id: primitive.episode_id,
          timestep: primitive.timestep ?? null,
          policy_call: primitive.policy_call ?? null,
          claim_level: primitive.claim_level,
        });
      }
    }
    return annotations;
  },

  channelRanking(bundle, selection) {
    const rows: LensFeatureContribution[] = [];
    for (const primitive of primitivesByKind(bundle, "contribution")) {
      if (!evidenceMatchesSelection(primitive, selection)) {
        continue;
      }
      for (const item of primitive.items) {
        if (selection.feature_id && item.key !== selection.feature_id) {
          continue;
        }
        rows.push({
          ...item,
          claim_level: primitive.claim_level,
          basis: primitive.basis,
        });
      }
    }
    return rows.sort((left, right) => left.rank - right.rank);
  },

  timelineRows(bundle) {
    const rows: LensTemporalRow[] = [];
    for (const primitive of bundle.primitives) {
      if (primitive.kind === "ranked_moments") {
        primitive.moments.forEach((moment, index) => {
          rows.push({
            row_id: `ranked:${primitive.ranking}:${index}`,
            source: "ranked_moment",
            episode_id: moment.episode_id,
            timestep: moment.timestep ?? null,
            policy_call: moment.policy_call ?? null,
            ranking: primitive.ranking,
            score: moment.score ?? null,
            confidence: moment.confidence ?? null,
            prediction: moment.prediction ?? null,
            label: moment.label ?? null,
          });
        });
      } else if (primitive.kind === "prediction") {
        rows.push({
          row_id: `prediction:${rows.length}`,
          source: "prediction",
          episode_id: primitive.episode_id,
          timestep: primitive.timestep ?? null,
          policy_call: primitive.policy_call ?? null,
          confidence: primitive.confidence ?? null,
          prediction: primitive.prediction,
          label: primitive.label ?? null,
        });
      } else if (primitive.kind === "failure_case") {
        primitive.moments.forEach((moment, index) => {
          rows.push({
            row_id: `failure:${primitive.ranking}:${index}`,
            source: "failure_case",
            episode_id: moment.episode_id,
            timestep: moment.timestep ?? null,
            policy_call: moment.policy_call ?? null,
            ranking: primitive.ranking,
            score: moment.score ?? null,
            confidence: moment.confidence ?? null,
            prediction: moment.prediction ?? null,
            label: moment.label ?? null,
          });
        });
      }
    }
    return rows;
  },

  interventionSeed() {
    return null;
  },
};

const claimLevelStrength: Record<EvidenceClaimLevel, number> = {
  numeric_only: 0,
  grouped_model_locus: 1,
  human_labeled_feature: 2,
  semantic_hypothesis: 3,
};

function limitItems<T>(items: T[], limit?: number | null): T[] {
  if (limit === undefined || limit === null) {
    return items;
  }
  if (limit < 0) {
    throw new Error("limit must be non-negative");
  }
  return items.slice(0, limit);
}

function firstAvailableMoment(
  bundle: ProbeEvidenceBundle,
): { ranking: RankingKind; moment: RankedMoment } | undefined {
  for (const ranking of ["top", "uncertain", "bottom"] as RankingKind[]) {
    const [moment] = rankedMoments(bundle, ranking);
    if (moment) {
      return { ranking, moment };
    }
  }
  return undefined;
}

function momentMatchesSelection(moment: RankedMoment, selection: ResearchSelectionState): boolean {
  if (selection.episode_id && moment.episode_id !== selection.episode_id) {
    return false;
  }
  return timeMatchesSelection(selection, {
    timestep: moment.timestep ?? null,
    policy_call: moment.policy_call ?? null,
  });
}

function evidenceMatchesSelection(
  evidence:
    | ScoreSeriesEvidence
    | PredictionEvidence
    | ContributionEvidence
    | ModelLocusEvidence,
  selection: ResearchSelectionState,
): boolean {
  if (selection.episode_id && evidence.episode_id && evidence.episode_id !== selection.episode_id) {
    return false;
  }
  return timeMatchesSelection(selection, {
    timestep: "timestep" in evidence ? evidence.timestep ?? null : null,
    policy_call: "policy_call" in evidence ? evidence.policy_call ?? null : null,
  });
}

function timeMatchesSelection(
  selection: ResearchSelectionState,
  value: { timestep?: number | null; policy_call?: number | null },
): boolean {
  if (selection.timestep !== undefined && selection.timestep !== null && value.timestep !== undefined && value.timestep !== null) {
    if (selection.timestep !== value.timestep) {
      return false;
    }
  }
  if (selection.policy_call !== undefined && selection.policy_call !== null && value.policy_call !== undefined && value.policy_call !== null) {
    if (selection.policy_call !== value.policy_call) {
      return false;
    }
  }
  return true;
}

function modelLocusLabel(locus: ModelLocusRef): string {
  if (locus.model_site_id) {
    return locus.model_site_id;
  }
  if (locus.layer !== undefined && locus.layer !== null) {
    return `layer ${locus.layer}`;
  }
  return "activation tensor";
}
