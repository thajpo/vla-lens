"""PI0.5 capture schema helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from vla_lens.pi05.token_metadata import (
    PI05TokenMetadata,
)

LANDMARK_5_LAYERS = (0, 4, 8, 12, 17)

AUDIT_WINDOWED_LAYERS = (0, 1, 4, 5, 8, 9, 12, 13, 16, 17)

ALL_PI05_LAYERS = tuple(range(18))

PROFILE_ALIASES = {
    "representation": "features",
    "mechanistic_light": "mechanistic_sampled",
    "mechanistic_heavy": "mechanistic_all",
    "full": "audit_full",
}

CANONICAL_PROFILES = (
    "rollout",
    "features",
    "mechanistic_sampled",
    "mechanistic_all",
    "internals_sampled",
    "audit_sampled",
    "audit_windowed",
    "audit_full",
    "custom",
)

PROFILE_CHOICES = (*CANONICAL_PROFILES, *PROFILE_ALIASES)

PROFILE_LAYERS = {
    "rollout": (),
    "features": LANDMARK_5_LAYERS,
    "mechanistic_sampled": LANDMARK_5_LAYERS,
    "mechanistic_all": ALL_PI05_LAYERS,
    "internals_sampled": LANDMARK_5_LAYERS,
    "audit_sampled": LANDMARK_5_LAYERS,
    "audit_windowed": AUDIT_WINDOWED_LAYERS,
    "audit_full": ALL_PI05_LAYERS,
    "custom": LANDMARK_5_LAYERS,
}

PROFILE_VLM_LAYERS = PROFILE_LAYERS

PROFILE_EXPERT_LAYERS = PROFILE_LAYERS

HIDDEN_RESOLUTIONS = ("profile", "none", "mean", "tokens")

ATTENTION_RESOLUTIONS = ("profile", "none", "key_mass", "full")

STORAGE_DTYPES = ("float16", "float32")

LIBERO_ACTION_DIM_NAMES = (
    "eef_delta_x",
    "eef_delta_y",
    "eef_delta_z",
    "eef_rotvec_delta_x",
    "eef_rotvec_delta_y",
    "eef_rotvec_delta_z",
    "gripper",
)

LIBERO_ACTION_DIM_LABELS = (
    "EEF delta x",
    "EEF delta y",
    "EEF delta z",
    "EEF rotvec delta x",
    "EEF rotvec delta y",
    "EEF rotvec delta z",
    "Gripper command",
)

LIBERO_ACTION_DIM_UNITS = (
    "normalized OSC command",
    "normalized OSC command",
    "normalized OSC command",
    "normalized OSC command",
    "normalized OSC command",
    "normalized OSC command",
    "normalized gripper command",
)

@dataclass(frozen=True, slots=True)
class CapturePlan:
    """Concrete tensor budget derived from a PI0.5 capture profile.

    Profiles are researcher-facing names such as ``mechanistic_sampled`` or
    ``audit_windowed``. A plan resolves those names into layer sets, hidden and
    attention resolutions, bridge-site capture, and storage dtype so the capture
    runner can make deterministic hook/writer decisions.
    """

    profile: str
    vlm_layers: tuple[int, ...]
    expert_layers: tuple[int, ...]
    vlm_hidden: str
    vlm_attention: str
    expert_hidden: str
    expert_attention: str
    storage_dtype: str

    @property
    def np_dtype(self) -> np.dtype:
        """Return the numpy dtype used for persisted model internals."""
        return np.dtype(self.storage_dtype)

    @property
    def capture_bridge_sites(self) -> bool:
        """Return whether VLM cache, expert input, and action-head sites are captured."""
        return canonical_profile(self.profile) in {
            "mechanistic_sampled",
            "mechanistic_all",
            "internals_sampled",
            "audit_sampled",
            "audit_windowed",
            "audit_full",
        }

    @property
    def capture_audit_full_sites(self) -> bool:
        """Return whether all declared raw/debug sites are required."""
        return canonical_profile(self.profile) == "audit_full"

    @property
    def capture_audit_sampled_sites(self) -> bool:
        """Return whether sampled-layer raw audit internals are captured."""
        return canonical_profile(self.profile) == "audit_sampled"

    @property
    def capture_windowed_audit_sites(self) -> bool:
        """Return whether adjacent-layer audit windows are captured."""
        return canonical_profile(self.profile) == "audit_windowed"

    @property
    def capture_internals_sites(self) -> bool:
        """Return whether operation-level internals beyond normal views are captured."""
        return canonical_profile(self.profile) in {
            "internals_sampled",
            "audit_sampled",
            "audit_windowed",
            "audit_full",
        }

    def to_metadata(self) -> dict[str, Any]:
        """Serialize the plan into provenance stored in the overlay manifest."""
        payload = asdict(self)
        payload["profile"] = canonical_profile(self.profile)
        if self.profile != payload["profile"]:
            payload["requested_profile"] = self.profile
            payload["profile_alias"] = self.profile
        payload["layers"] = sorted(set(self.vlm_layers) | set(self.expert_layers))
        payload["vlm_layers"] = list(self.vlm_layers)
        payload["expert_layers"] = list(self.expert_layers)
        payload["profile_dimensions"] = _plan_dimensions(self)
        payload["captures_bridge_sites"] = self.capture_bridge_sites
        payload["bridge_sites"] = (
            [
                "vlm_kv_cache_key_value",
                "expert_generation_input_embeddings",
                "action_head_input_output",
            ]
            if self.capture_bridge_sites
            else []
        )
        payload["runtime_collections"] = (
            [_past_key_values_collection_metadata(self)] if self.capture_bridge_sites else []
        )
        payload["axis_strategy"] = "policy_call"
        from vla_lens.pi05.patch_sites import pi05_runtime_patch_sites

        payload["runtime_hook_sites"] = list(pi05_runtime_patch_sites())
        payload["runtime_hook_sites_materialized"] = False
        payload["attention_full_semantics"] = "head x query_token x key_token"
        payload["attention_key_mass_semantics"] = "head x key_token, mean over query_token"
        return payload

def _past_key_values_collection_metadata(plan: "CapturePlan") -> dict[str, Any]:
    return {
        "id": "pi05.vlm.past_key_values",
        "label": "Layer-wise prefix K/V",
        "kind": "runtime_collection",
        "materialized": False,
        "aggregation": "none",
        "members": [
            {
                "layer": int(layer),
                "component": component,
                "site_name": f"pi05.vlm.layers.{layer}.kv_cache.{component}",
            }
            for layer in plan.vlm_layers
            for component in ("key", "value")
        ],
    }

def canonical_profile(profile: str) -> str:
    """Resolve historical profile aliases to canonical profile names."""
    return PROFILE_ALIASES.get(str(profile), str(profile))

def profile_dimensions(profile: str) -> dict[str, Any]:
    """Return researcher-facing coverage labels for a canonical profile."""
    profile = canonical_profile(profile)
    if profile == "rollout":
        return {
            "layer_coverage": {"vlm": "none", "expert": "none"},
            "families": {
                "representations": "none",
                "attention": "none",
                "cache": "none",
                "action_head": "none",
                "internals": "none",
                "state_setup": "none",
            },
        }
    if profile == "features":
        return {
            "layer_coverage": {"vlm": "landmark_5", "expert": "landmark_5"},
            "families": {
                "representations": "tokens",
                "attention": "none",
                "cache": "none",
                "action_head": "none",
                "internals": "none",
                "state_setup": "none",
            },
        }
    if profile == "mechanistic_all":
        layer_coverage = {"vlm": "all", "expert": "all"}
    elif profile == "audit_windowed":
        layer_coverage = {"vlm": "audit_windows_10", "expert": "audit_windows_10"}
    elif profile == "audit_full":
        layer_coverage = {"vlm": "all", "expert": "all"}
    elif profile == "custom":
        return {
            "layer_coverage": {"vlm": "sampled_5", "expert": "sampled_5"},
            "families": {
                "representations": "tokens",
                "attention": "none",
                "cache": "none",
                "action_head": "none",
                "internals": "none",
                "state_setup": "none",
            },
        }
    else:
        layer_coverage = {"vlm": "sampled_5", "expert": "sampled_5"}
    return {
        "layer_coverage": layer_coverage,
        "families": {
            "representations": "tokens",
            "attention": "full_probs",
            "cache": "layer_kv",
            "action_head": "io",
            "internals": "full_raw"
            if profile == "audit_full"
            else (
                _sampled_audit_internals_label(profile)
                if profile in {"audit_sampled", "audit_windowed"}
                else ("selected_ops" if profile == "internals_sampled" else "none")
            ),
            "state_setup": "full_raw" if profile == "audit_full" else "none",
        },
    }

def _plan_dimensions(plan: CapturePlan) -> dict[str, Any]:
    profile = canonical_profile(plan.profile)

    def coverage_label(layers: tuple[int, ...]) -> str | list[int]:
        if not layers:
            return "none"
        if layers == ALL_PI05_LAYERS:
            return "all"
        if layers == AUDIT_WINDOWED_LAYERS:
            return "audit_windows_10"
        if layers == LANDMARK_5_LAYERS:
            return "landmark_5" if profile == "features" else "sampled_5"
        return [int(layer) for layer in layers]

    def attention_label(vlm_attention: str, expert_attention: str) -> str | dict[str, str]:
        labels = {
            "none": "none",
            "key_mass": "key_mass",
            "full": "full_probs",
        }
        vlm = labels.get(vlm_attention, vlm_attention)
        expert = labels.get(expert_attention, expert_attention)
        if vlm == expert:
            return vlm
        return {"vlm": vlm, "expert": expert}

    def representation_label(vlm_hidden: str, expert_hidden: str) -> str | dict[str, str]:
        if vlm_hidden == expert_hidden:
            return vlm_hidden
        return {"vlm": vlm_hidden, "expert": expert_hidden}

    return {
        "layer_coverage": {
            "vlm": coverage_label(plan.vlm_layers),
            "expert": coverage_label(plan.expert_layers),
        },
        "families": {
            "representations": representation_label(plan.vlm_hidden, plan.expert_hidden),
            "attention": attention_label(plan.vlm_attention, plan.expert_attention),
            "cache": "layer_kv" if plan.capture_bridge_sites else "none",
            "action_head": "io" if plan.capture_bridge_sites else "none",
            "internals": "full_raw"
            if plan.capture_audit_full_sites
            else (
                _sampled_audit_internals_label(profile)
                if profile in {"audit_sampled", "audit_windowed"}
                else ("selected_ops" if profile == "internals_sampled" else "none")
            ),
            "state_setup": "full_raw" if plan.capture_audit_full_sites else "none",
        },
    }

def _sampled_audit_internals_label(profile: str) -> str:
    return "windowed_audit" if profile == "audit_windowed" else "sampled_audit"

@dataclass
class CaptureCall:
    """Captured tensors and metadata for one policy call/action chunk."""

    call_index: int
    env_timestep: int
    final_action_chunk: np.ndarray
    denoising_actions: np.ndarray
    suffix_hidden: np.ndarray
    initial_noise: np.ndarray | None = None
    prefix_image_hidden: np.ndarray | None = None
    prefix_patches_per_image: int | None = None
    prefix_image_slots: int | None = None
    attention_mass: np.ndarray | None = None
    denoise_velocities: np.ndarray | None = None
    vlm_hidden_by_layer: dict[int, np.ndarray] = field(default_factory=dict)
    vlm_attention_by_layer: dict[int, np.ndarray] = field(default_factory=dict)
    expert_hidden_by_layer: dict[int, np.ndarray] = field(default_factory=dict)
    expert_attention_by_layer: dict[int, np.ndarray] = field(default_factory=dict)
    vlm_kv_key_by_layer: dict[int, np.ndarray] = field(default_factory=dict)
    vlm_kv_value_by_layer: dict[int, np.ndarray] = field(default_factory=dict)
    generation_input_embeddings: np.ndarray | None = None
    action_head_input: np.ndarray | None = None
    action_head_output: np.ndarray | None = None
    token_metadata: PI05TokenMetadata | None = None
    policy_call_metadata: dict[str, Any] = field(default_factory=dict)
    full_site_arrays: dict[str, np.ndarray] = field(default_factory=dict)

@dataclass
class EpisodeBuffer:
    """Mutable in-memory rollout buffer before writing one captured episode."""

    trace_id: str
    task_id: int
    task_name: str
    prompt: str
    seed: int
    frames: list[np.ndarray] = field(default_factory=list)
    wrist_frames: list[np.ndarray] = field(default_factory=list)
    executed_actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    calls: list[CaptureCall] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    scene_snapshots: list[dict[str, Any]] = field(default_factory=list)
    camera_snapshots: list[dict[str, Any]] = field(default_factory=list)
    infos: list[dict[str, Any]] = field(default_factory=list)
    terminated: list[bool] = field(default_factory=list)
    truncated: list[bool] = field(default_factory=list)
    scene_mutation_report: dict[str, Any] = field(default_factory=dict)
    success: bool = False

@dataclass
class PI05CaptureRuntime:
    """Heavy PI0.5 runtime objects loaded only in the capture environment."""

    torch: Any
    policy_cfg: Any
    policy: Any
    preprocessor: Any
    postprocessor: Any
    make_env: Any
    make_env_config: Any
    make_env_pre_post_processors: Any
    add_envs_task: Any
    preprocess_observation: Any
    get_benchmark: Any
