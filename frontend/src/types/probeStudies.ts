export type ProbeStudyReadout = {
  readout_id: string;
  trained_probe_id?: string | null;
  target: string;
  status: string;
  source: string;
  layer?: number | string | null;
  split?: string | null;
  split_category?: string | null;
  row_count?: number | null;
  policy_call_count?: number | null;
  class_count?: number | null;
  balanced_accuracy?: number | null;
  accuracy?: number | null;
  macro_f1?: number | null;
  top1_accuracy?: number | null;
  top2_accuracy?: number | null;
  top3_accuracy?: number | null;
  train_balanced_accuracy?: number | null;
  train_gap_balanced_accuracy?: number | null;
  reason?: string | null;
  is_primary_target?: boolean;
  is_selected_layer?: boolean;
  is_selection_split?: boolean;
  is_test_split?: boolean;
};

export type ProbeStudyControl = {
  kind: string;
  label: string;
  split?: string | null;
  runs?: number | null;
  selected_layer?: number | string | null;
  real_score?: number | null;
  null_score_mean?: number | null;
  null_score_std?: number | null;
  p_value?: number | null;
  selected_layer_counts?: Record<string, number>;
};

export type ProbeStudyTrainingSummary = {
  objective?: string | null;
  target_type?: string | null;
  estimator?: string | null;
  library?: string | null;
  preprocessing?: string | null;
  hyperparameters?: string[];
  trained_on?: string | null;
  selected_on?: string | null;
  metric?: string | null;
};

export type ProbeStudy = {
  study_id?: string;
  artifact_id: string;
  artifact_type: "probe_suite" | string;
  source_artifact_id?: string | null;
  source_artifact_name?: string | null;
  name: string;
  created_utc?: string | null;
  target?: string | null;
  question_label?: string | null;
  prediction?: string | null;
  input?: string | null;
  output?: string | null;
  objective?: string | null;
  training_summary?: ProbeStudyTrainingSummary;
  diagnostics_available: boolean;
  source: "diagnostics" | "artifact" | string;
  counts: {
    readout_count: number;
    skipped_readout_count: number;
    target_count?: number | null;
    layer_count?: number | null;
    feature_rows?: number | null;
    policy_call_count?: number | null;
    episode_count?: number | null;
    class_count?: number | null;
    null_run_count?: number | null;
    null_eval_row_count?: number | null;
    split_policy_call_counts?: Record<string, number>;
  };
  summary: Record<string, unknown>;
  readouts: ProbeStudyReadout[];
  skipped_readouts: ProbeStudyReadout[];
  controls: ProbeStudyControl[];
  lead_time: Record<string, unknown>[];
  per_class: Record<string, unknown>[];
  confusion: Record<string, unknown>[];
  class_support: Record<string, unknown>[];
  error_examples: Record<string, unknown>[];
};

export type ProbeStudyResponse = {
  studies: ProbeStudy[];
  total: number;
};

export type ProbeStudyEpisodeSummary = {
  policy_call_count: number;
  episode_count: number;
  scored: number;
  unscored: number;
  correct: number;
  wrong: number;
  high_confidence: number;
  high_conf_wrong: number;
  split_counts?: Record<
    string,
    {
      policy_call_count: number;
      scored: number;
      correct: number;
      wrong: number;
      high_confidence: number;
      high_conf_wrong: number;
    }
  >;
};
