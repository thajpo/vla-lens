"""Shared probe workflow schemas and constants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from vla_lens.artifacts import LensArtifact

PROBE_ARTIFACT_SCHEMA_VERSION = 3

INTERACTION_METRICS_ARTIFACT_TYPE = "pi05_interaction_metrics"
OBJECT_FLOW_ARTIFACT_TYPE = "pi05_object_flow"
POLICY_CALL_LABELS_ARTIFACT_TYPE = "pi05_policy_call_labels"


@dataclass(frozen=True, slots=True)
class SavedProbeSuite:
    artifact: LensArtifact
    results: pd.DataFrame
    rows: pd.DataFrame


DEFAULT_PROBE_SPEC: dict[str, Any] = {
    "name": "Outcome probe over expert action features",
    "target": {"kind": "outcome"},
    "features": {
        "module": "pi05.expert.layers.*",
        "tensor_type": "hidden_mean",
        "token_kind": "action",
        "layers": None,
        "timesteps": "all",
        "generation_step": None,
        "reduction": "mean",
    },
    "split": {"kind": "heldout_benchmark"},
    "baseline": ["majority_class", "benchmark", "target_object"],
    "sweep": "layer",
}
