"""Factory for policy backends."""

from __future__ import annotations

from omegaconf import DictConfig

from openvla_steering.model.minivla import MiniVLAPolicy, MiniVLAPolicyConfig
from openvla_steering.model.openvla import OpenVLAPolicy, OpenVLAPolicyConfig
from openvla_steering.model.scripted import ScriptedPickPolicy, ScriptedPickPolicyConfig


def build_policy_from_config(cfg: DictConfig):
    backend = str(cfg.model.backend)
    if backend == "scripted_pick":
        return ScriptedPickPolicy(
            ScriptedPickPolicyConfig(
                target_object=str(cfg.model.scripted_pick.target_object),
                approach_height=float(cfg.model.scripted_pick.approach_height),
                grasp_offset=float(cfg.model.scripted_pick.grasp_offset),
                lift_height=float(cfg.model.scripted_pick.lift_height),
                position_gain=float(cfg.model.scripted_pick.position_gain),
                open_gripper=float(cfg.model.scripted_pick.open_gripper),
                close_gripper=float(cfg.model.scripted_pick.close_gripper),
                approach_steps=int(cfg.model.scripted_pick.phase_steps.approach),
                descend_steps=int(cfg.model.scripted_pick.phase_steps.descend),
                close_steps=int(cfg.model.scripted_pick.phase_steps.close),
                lift_steps=int(cfg.model.scripted_pick.phase_steps.lift),
            )
        )
    if backend == "openvla":
        return OpenVLAPolicy(
            OpenVLAPolicyConfig(
                model_id=str(cfg.model.openvla.model_id),
                instruction=str(cfg.model.openvla.instruction),
                trust_remote_code=bool(cfg.model.openvla.trust_remote_code),
                device=str(cfg.model.openvla.device),
                torch_dtype=str(cfg.model.openvla.torch_dtype),
                attn_implementation=(
                    str(cfg.model.openvla.attn_implementation)
                    if cfg.model.openvla.attn_implementation is not None
                    else None
                ),
                unnorm_key=(
                    str(cfg.model.openvla.unnorm_key)
                    if cfg.model.openvla.unnorm_key is not None
                    else None
                ),
                action_dim=int(cfg.model.openvla.action_dim),
                camera_name=str(cfg.model.openvla.camera_name),
            )
        )
    if backend == "minivla":
        return MiniVLAPolicy(
            MiniVLAPolicyConfig(
                model_id=str(cfg.model.minivla.model_id),
                instruction=str(cfg.model.minivla.instruction),
                trust_remote_code=bool(cfg.model.minivla.trust_remote_code),
                device=str(cfg.model.minivla.device),
                torch_dtype=str(cfg.model.minivla.torch_dtype),
                attn_implementation=(
                    str(cfg.model.minivla.attn_implementation)
                    if cfg.model.minivla.attn_implementation is not None
                    else None
                ),
                unnorm_key=(
                    str(cfg.model.minivla.unnorm_key)
                    if cfg.model.minivla.unnorm_key is not None
                    else None
                ),
                action_dim=int(cfg.model.minivla.action_dim),
                camera_name=str(cfg.model.minivla.camera_name),
            )
        )
    raise ValueError(f"Unsupported model backend: {backend}")
