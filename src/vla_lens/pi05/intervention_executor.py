"""Concrete deterministic PI0.5 action intervention executor.

Importing this module is safe in the normal development environment. Heavy
Torch, LeRobot, and LIBERO imports occur only when the builder loads the
capture-specific runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from vla_lens.interventions import (
    InterventionOperatorSpec,
    InterventionScheduleSpec,
    RuntimeTrialOutput,
    TargetSpec,
)
from vla_lens.pi05.capture_runner import (
    load_pi05_capture_runtime,
    namespace_for_capture_args,
)
from vla_lens.pi05.capture_schema import PI05CaptureRuntime
from vla_lens.pi05.replay import PolicyCallReplayInputs, policy_call_replay_inputs
from vla_lens.traces import TraceDataset

SYNTHETIC_HOOK_MODE = "synthetic_hook_smoke"
SUPPORTED_MODEL_SITE = "pi05.action_head.input"


@dataclass(slots=True)
class PI05ActionInterventionExecutor:
    """Replay one PI0.5 policy call and optionally perturb its action-head input."""

    runtime: PI05CaptureRuntime
    replay_inputs: PolicyCallReplayInputs
    observation: Mapping[str, Any]
    initial_noise: Any

    def run_noop(self, request: Mapping[str, Any]) -> RuntimeTrialOutput:
        del request
        action = self._predict()
        return RuntimeTrialOutput(
            trial_id="trial_noop",
            trial_kind="noop_rerun",
            action_chunk=action,
            runtime={
                "executor": "pi05_action_intervention",
                "replay_noise_ref": self.replay_inputs.initial_noise_ref,
                "replay_noise_exactness": self.replay_inputs.initial_noise_exactness,
            },
        )

    def run_intervention(self, request: Mapping[str, Any]) -> RuntimeTrialOutput:
        target, operator, schedule = _resolve_request(request)
        _validate_synthetic_hook_request(
            target,
            operator,
            schedule,
            policy_call_index=self.replay_inputs.policy_call_index,
        )
        action, applied_steps, hidden_dim = self._predict_with_direction(
            operator=operator,
            schedule=schedule,
            direction_kind="one_hot",
        )
        return RuntimeTrialOutput(
            trial_id="trial_intervention",
            trial_kind="intervention",
            action_chunk=action,
            metrics={"strength": float(operator.strength or 0.0)},
            runtime={
                "executor": "pi05_action_intervention",
                "purpose": "hook_wiring_smoke",
                "claim_eligible": False,
                "model_site": SUPPORTED_MODEL_SITE,
                "direction": "one_hot",
                "dimension": int(operator.parameters.get("dimension", 0)),
                "hidden_dim": hidden_dim,
                "applied_generation_steps": applied_steps,
            },
        )

    def run_control(
        self,
        request: Mapping[str, Any],
        *,
        control_kind: str,
    ) -> RuntimeTrialOutput:
        if control_kind not in {"random_direction", "random_direction_control"}:
            raise ValueError(
                "PI0.5 synthetic hook smoke currently supports only random_direction control"
            )
        target, operator, schedule = _resolve_request(request)
        _validate_synthetic_hook_request(
            target,
            operator,
            schedule,
            policy_call_index=self.replay_inputs.policy_call_index,
        )
        action, applied_steps, hidden_dim = self._predict_with_direction(
            operator=operator,
            schedule=schedule,
            direction_kind="random",
        )
        return RuntimeTrialOutput(
            trial_id="trial_random_direction_control",
            trial_kind="random_direction_control",
            control_kind="random_direction",
            action_chunk=action,
            metrics={"strength": float(operator.strength or 0.0)},
            runtime={
                "executor": "pi05_action_intervention",
                "purpose": "hook_wiring_smoke_control",
                "claim_eligible": False,
                "model_site": SUPPORTED_MODEL_SITE,
                "direction": "random_unit",
                "seed": int(operator.parameters.get("control_seed", 0)),
                "hidden_dim": hidden_dim,
                "applied_generation_steps": applied_steps,
            },
        )

    def close(self) -> None:
        """Match resource-owning executor interfaces; replay setup closes its environment."""

    def _predict(self) -> np.ndarray:
        torch = self.runtime.torch
        self.runtime.policy.reset()
        with torch.no_grad():
            action = self.runtime.policy.predict_action_chunk(
                self.observation,
                noise=self.initial_noise.clone(),
            )
        return _action_numpy(action)

    def _predict_with_direction(
        self,
        *,
        operator: InterventionOperatorSpec,
        schedule: InterventionScheduleSpec,
        direction_kind: str,
    ) -> tuple[np.ndarray, list[int], int]:
        torch = self.runtime.torch
        policy = self.runtime.policy
        projection = policy.model.action_out_proj
        original_forward = projection.forward
        requested_steps = _generation_step_indices(schedule.generation_steps)
        strength = float(operator.strength or 0.0)
        dimension = int(operator.parameters.get("dimension", 0))
        control_seed = int(operator.parameters.get("control_seed", 0))
        applied_steps: list[int] = []
        call_index = 0
        hidden_dim = int(getattr(projection, "in_features", 0))

        if hidden_dim and not 0 <= dimension < hidden_dim:
            raise ValueError(
                f"Synthetic direction dimension {dimension} is outside "
                f"action-head width {hidden_dim}"
            )

        def forward_with_direction(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_index, hidden_dim
            step = call_index
            call_index += 1
            if requested_steps is not None and step not in requested_steps:
                return original_forward(*args, **kwargs)
            if not args:
                raise RuntimeError(
                    "PI0.5 action_out_proj did not receive a positional input tensor"
                )
            activation = args[0]
            hidden_dim = int(activation.shape[-1])
            if not 0 <= dimension < hidden_dim:
                raise ValueError(
                    f"Synthetic direction dimension {dimension} is outside action-head width "
                    f"{hidden_dim}"
                )
            direction = _direction_tensor(
                torch,
                activation,
                hidden_dim=hidden_dim,
                dimension=dimension,
                direction_kind=direction_kind,
                seed=control_seed,
            )
            modified = activation + strength * direction
            applied_steps.append(step)
            return original_forward(modified, *args[1:], **kwargs)

        policy.reset()
        projection.forward = forward_with_direction
        try:
            with torch.no_grad():
                action = policy.predict_action_chunk(
                    self.observation,
                    noise=self.initial_noise.clone(),
                )
        finally:
            projection.forward = original_forward
        if not applied_steps:
            raise RuntimeError(
                "The requested generation-step schedule never reached the action head"
            )
        return _action_numpy(action), applied_steps, hidden_dim


def build_pi05_action_intervention_executor(
    dataset: TraceDataset,
    payload: Mapping[str, Any],
    *,
    device: str,
    dtype: str,
    model_id: str | None = None,
) -> PI05ActionInterventionExecutor:
    """Load the dedicated PI0.5 runtime and reconstruct the selected raw observation."""
    trace_id, policy_call_index = _context_selection(payload)
    bundle = dataset.bundle(trace_id)
    inputs = policy_call_replay_inputs(bundle, policy_call_index)
    selected_model_id = str(model_id or bundle.manifest.model_id or "").strip()
    if not selected_model_id:
        raise ValueError(f"Trace {trace_id} does not declare a PI0.5 model_id")
    args = namespace_for_capture_args(
        model_id=selected_model_id,
        benchmark=inputs.config.benchmark,
        task_id=inputs.config.task_id,
        start_seed=inputs.config.seed,
        obs_size=inputs.config.obs_size,
        device=device,
        dtype=dtype,
    )
    runtime = load_pi05_capture_runtime(args)
    observation = replay_policy_call_observation(runtime, bundle.actions(mmap=True), inputs)
    initial_noise = runtime.torch.as_tensor(
        np.asarray(inputs.initial_noise, dtype=np.float32),
        device=runtime.policy.config.device,
        dtype=runtime.torch.float32,
    )
    if initial_noise.ndim == 2:
        initial_noise = initial_noise.unsqueeze(0)
    return PI05ActionInterventionExecutor(
        runtime=runtime,
        replay_inputs=inputs,
        observation=observation,
        initial_noise=initial_noise,
    )


def replay_policy_call_observation(
    runtime: PI05CaptureRuntime,
    executed_actions: np.ndarray,
    inputs: PolicyCallReplayInputs,
) -> Mapping[str, Any]:
    """Recreate and preprocess the observation seen at a captured policy-call boundary."""
    config = inputs.config
    env_cfg = runtime.make_env_config(
        "libero",
        task=config.benchmark,
        task_ids=[config.task_id],
        observation_height=config.obs_size,
        observation_width=config.obs_size,
        camera_name=config.camera_name,
        control_mode=config.control_mode,
    )
    envs = runtime.make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs[config.benchmark][config.task_id]
    base_env = env.envs[0] if getattr(env, "envs", None) else None
    if config.layout_id is not None and base_env is not None:
        base_env.episode_index = config.layout_id
        base_env.init_state_id = config.layout_id
    env_preprocessor, _env_postprocessor = runtime.make_env_pre_post_processors(
        env_cfg=env_cfg,
        policy_cfg=runtime.policy_cfg,
    )
    if inputs.observation_timestep > len(executed_actions):
        env.close()
        raise IndexError(
            f"Policy call {inputs.policy_call_index} requires observation timestep "
            f"{inputs.observation_timestep}, but the trace has {len(executed_actions)} actions"
        )
    try:
        observation, _ = env.reset(seed=[config.seed])
        for timestep in range(inputs.observation_timestep):
            env_action = np.expand_dims(
                np.asarray(executed_actions[timestep], dtype=np.float32),
                axis=0,
            )
            observation, _reward, terminated, truncated, _info = env.step(env_action)
            if bool(np.all(np.asarray(terminated) | np.asarray(truncated))):
                raise RuntimeError(
                    "LIBERO replay terminated before the selected policy call at "
                    f"timestep {timestep + 1}"
                )
        processed = runtime.preprocess_observation(observation)
        processed = runtime.add_envs_task(env, processed)
        processed = env_preprocessor(processed)
        return runtime.preprocessor(processed)
    finally:
        env.close()


def _resolve_request(
    payload: Mapping[str, Any],
) -> tuple[TargetSpec, InterventionOperatorSpec, InterventionScheduleSpec]:
    target_payload = _mapping(payload.get("target"))
    intervention = _mapping(payload.get("intervention"))
    request = _mapping(payload.get("request")) or _mapping(intervention.get("request"))
    if not target_payload or not request:
        raise ValueError("PI0.5 executor requires target and intervention.request")
    return (
        TargetSpec.from_dict(target_payload),
        InterventionOperatorSpec.from_dict(_mapping(request.get("operator"))),
        InterventionScheduleSpec.from_dict(_mapping(request.get("schedule"))),
    )


def _validate_synthetic_hook_request(
    target: TargetSpec,
    operator: InterventionOperatorSpec,
    schedule: InterventionScheduleSpec,
    *,
    policy_call_index: int,
) -> None:
    if target.model_site != SUPPORTED_MODEL_SITE:
        raise ValueError(
            f"Synthetic hook smoke supports model_site={SUPPORTED_MODEL_SITE!r}, "
            f"not {target.model_site!r}"
        )
    if operator.operator != "add_direction":
        raise ValueError("Synthetic hook smoke requires operator=add_direction")
    if operator.parameters.get("mode") != SYNTHETIC_HOOK_MODE:
        raise ValueError(
            f"Synthetic hook smoke requires operator.parameters.mode={SYNTHETIC_HOOK_MODE!r}"
        )
    if operator.strength is None:
        raise ValueError("Synthetic hook smoke requires an explicit operator strength")
    if isinstance(schedule.policy_calls, tuple) and policy_call_index not in schedule.policy_calls:
        raise ValueError(
            f"Schedule does not include selected policy call {policy_call_index}: "
            f"{schedule.policy_calls}"
        )
    if schedule.tokens not in {"action", "all", "target_tokens"}:
        raise ValueError("Synthetic action-head smoke currently applies to the full action suffix")


def _generation_step_indices(selector: Mapping[str, Any] | str) -> set[int] | None:
    if selector == "all":
        return None
    if not isinstance(selector, Mapping):
        raise ValueError(f"Unsupported generation-step selector: {selector!r}")
    raw_indices = selector.get("indices")
    if raw_indices is not None:
        return {int(value) for value in raw_indices}
    if "start" in selector or "end" in selector:
        start = int(selector.get("start", 0))
        end = int(selector.get("end", start + 1))
        return set(range(start, end))
    raise ValueError(f"Unsupported generation-step selector: {dict(selector)!r}")


def _direction_tensor(
    torch: Any,
    activation: Any,
    *,
    hidden_dim: int,
    dimension: int,
    direction_kind: str,
    seed: int,
) -> Any:
    if direction_kind == "one_hot":
        direction = np.zeros(hidden_dim, dtype=np.float32)
        direction[dimension] = 1.0
    elif direction_kind == "random":
        direction = np.random.default_rng(seed).normal(size=hidden_dim).astype(np.float32)
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            raise RuntimeError("Random control direction unexpectedly has zero norm")
        direction /= norm
    else:
        raise ValueError(f"Unknown synthetic direction kind: {direction_kind}")
    return torch.as_tensor(direction, device=activation.device, dtype=activation.dtype)


def _action_numpy(action: Any) -> np.ndarray:
    array = action.detach().float().cpu().numpy()
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    return np.asarray(array, dtype=np.float32)


def _context_selection(payload: Mapping[str, Any]) -> tuple[str, int]:
    baseline = _mapping(payload.get("baseline"))
    context = _mapping(payload.get("context")) or _mapping(baseline.get("context"))
    trace_id = str(context.get("trace_id") or "").strip()
    if not trace_id:
        raise ValueError("PI0.5 intervention requires context.trace_id")
    call_index = int(context.get("policy_call_index") or 0)
    return trace_id, call_index


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = [
    "PI05ActionInterventionExecutor",
    "SYNTHETIC_HOOK_MODE",
    "SUPPORTED_MODEL_SITE",
    "build_pi05_action_intervention_executor",
    "replay_policy_call_observation",
]
