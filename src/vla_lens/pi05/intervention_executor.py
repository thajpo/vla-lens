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
from vla_lens.pi05.probe_direction import (
    ResolvedProbeDirection,
    resolve_object_roi_probe_direction,
)
from vla_lens.pi05.replay import PolicyCallReplayInputs, policy_call_replay_inputs
from vla_lens.traces import TraceDataset

SYNTHETIC_HOOK_MODE = "synthetic_hook_smoke"
PROBE_DIRECTION_MODE = "artifact_probe_direction"
SUPPORTED_MODEL_SITE = "pi05.action_head.input"


@dataclass(slots=True)
class PI05ActionInterventionExecutor:
    """Replay one PI0.5 policy call and optionally perturb a validated internal site."""

    runtime: PI05CaptureRuntime
    replay_inputs: PolicyCallReplayInputs
    observation: Mapping[str, Any]
    initial_noise: Any
    probe_direction: ResolvedProbeDirection | None = None

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
        mode = str(operator.parameters.get("mode") or "")
        if mode == SYNTHETIC_HOOK_MODE:
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
        direction = self._validated_probe_request(target, operator, schedule)
        action, hook = self._predict_with_probe_direction(
            operator=operator,
            direction=direction,
        )
        return RuntimeTrialOutput(
            trial_id="trial_intervention",
            trial_kind="intervention",
            action_chunk=action,
            metrics={
                "strength": float(operator.strength or 0.0),
                **hook["metrics"],
            },
            runtime={
                "executor": "pi05_action_intervention",
                "purpose": "artifact_probe_direction",
                "claim_eligible": True,
                "model_site": direction.model_site,
                "operator_semantics": _operator_semantics(operator.operator),
                "direction_resolution": dict(direction.provenance),
                **hook["runtime"],
            },
        )

    def run_control(
        self,
        request: Mapping[str, Any],
        *,
        control_kind: str,
    ) -> RuntimeTrialOutput:
        target, operator, schedule = _resolve_request(request)
        mode = str(operator.parameters.get("mode") or "")
        if mode == SYNTHETIC_HOOK_MODE:
            if control_kind not in {"random_direction", "random_direction_control"}:
                raise ValueError(
                    "PI0.5 synthetic hook smoke supports only random_direction control"
                )
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

        primary = self._validated_probe_request(target, operator, schedule)
        parameters = _control_parameters(request, control_kind)
        token_indices = primary.token_indices
        if control_kind in {"random_direction", "random_direction_control"}:
            seed = int(parameters.get("seed", operator.parameters.get("control_seed", 0)))
            direction = primary.random_control(seed)
            resolved_kind = "matched_random"
        elif control_kind in {"wrong_feature", "wrong_identity"}:
            direction = primary.with_class_pair(
                int(parameters["target_class"]),
                parameters.get("contrast_class", "class_mean"),
                purpose="wrong_identity_control",
            )
            resolved_kind = "wrong_identity"
        elif control_kind in {"wrong_token", "wrong_roi"}:
            direction = primary
            raw_tokens = parameters.get("indices") or primary.provenance.get(
                "wrong_roi_token_indices"
            )
            if not isinstance(raw_tokens, (list, tuple)) or not raw_tokens:
                raise ValueError(
                    "Wrong-token control requires explicit or artifact wrong-ROI tokens"
                )
            token_indices = tuple(int(value) for value in raw_tokens)
            resolved_kind = "wrong_roi"
        else:
            raise ValueError(f"Unsupported artifact probe control {control_kind!r}")
        action, hook = self._predict_with_probe_direction(
            operator=operator,
            direction=direction,
            token_indices=token_indices,
            norm_reference=primary if direction is not primary else None,
        )
        return RuntimeTrialOutput(
            trial_id=f"trial_{resolved_kind}_control",
            trial_kind={
                "matched_random": "random_direction_control",
                "wrong_identity": "control",
                "wrong_roi": "wrong_token_control",
            }[resolved_kind],
            control_kind=resolved_kind,
            action_chunk=action,
            metrics={
                "strength": float(operator.strength or 0.0),
                **hook["metrics"],
            },
            runtime={
                "executor": "pi05_action_intervention",
                "purpose": f"artifact_probe_{resolved_kind}_control",
                "claim_eligible": True,
                "model_site": primary.model_site,
                "control_kind": resolved_kind,
                "token_indices": list(token_indices),
                "direction_resolution": dict(direction.provenance),
                **hook["runtime"],
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

    def _validated_probe_request(
        self,
        target: TargetSpec,
        operator: InterventionOperatorSpec,
        schedule: InterventionScheduleSpec,
    ) -> ResolvedProbeDirection:
        direction = self.probe_direction
        if direction is None:
            raise ValueError("Artifact probe direction was not resolved before runtime loading")
        _validate_probe_direction_request(
            target,
            operator,
            schedule,
            direction=direction,
            policy_call_index=self.replay_inputs.policy_call_index,
        )
        return direction

    def _predict_with_probe_direction(
        self,
        *,
        operator: InterventionOperatorSpec,
        direction: ResolvedProbeDirection,
        token_indices: tuple[int, ...] | None = None,
        norm_reference: ResolvedProbeDirection | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        torch = self.runtime.torch
        policy = self.runtime.policy
        layer = _pi05_vlm_layer(policy, direction.layer)
        original_forward = layer.forward
        selected_tokens = tuple(token_indices or direction.token_indices)
        strength = float(operator.strength or 0.0)
        hook_calls = 0
        realized: list[dict[str, float]] = []
        observed_runtime: dict[str, Any] = {}
        expected_shape = tuple(
            int(value) for value in direction.provenance.get("source_tensor_shape") or ()
        )

        def forward_with_probe_direction(*args: Any, **kwargs: Any) -> Any:
            nonlocal hook_calls
            output = original_forward(*args, **kwargs)
            hidden, rebuild = _layer_hidden_output(output)
            shape = tuple(int(value) for value in hidden.shape)
            if len(shape) != 3 or shape[0] != 1:
                raise ValueError(
                    "PI0.5 VLM layer output must have [1, prefix_token, hidden] shape, "
                    f"found {shape}"
                )
            if shape[-1] != direction.hidden_dim:
                raise ValueError(
                    f"Runtime hidden width {shape[-1]} does not match {direction.hidden_dim}"
                )
            if expected_shape and shape[1] != expected_shape[1]:
                raise ValueError(
                    f"Runtime prefix length {shape[1]} does not match saved source "
                    f"length {expected_shape[1]}"
                )
            if not selected_tokens or min(selected_tokens) < 0 or max(selected_tokens) >= shape[1]:
                raise ValueError("Intervention token selector is outside the runtime prefix")

            modified = hidden.clone()
            selected = hidden[:, list(selected_tokens), :]
            before_mean = selected.mean(dim=1)
            before_numpy = _tensor_numpy(before_mean)[0]
            if operator.operator == "add_direction":
                raw_delta = direction.add_delta(strength)
            else:
                raw_delta = direction.project_out_delta(before_numpy, strength)
                if norm_reference is not None:
                    reference_delta = norm_reference.project_out_delta(before_numpy, strength)
                    raw_delta = _match_nonzero_norm(raw_delta, reference_delta)
            delta_tensor = torch.as_tensor(
                raw_delta,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            modified[:, list(selected_tokens), :] = selected + delta_tensor
            after_mean = modified[:, list(selected_tokens), :].mean(dim=1)
            after_numpy = _tensor_numpy(after_mean)[0]
            measurement_direction = norm_reference or direction
            probe_before = float(measurement_direction.score_difference(before_numpy))
            probe_after = float(measurement_direction.score_difference(after_numpy))
            direction_before = float(direction.direction_coordinate(before_numpy))
            direction_after = float(direction.direction_coordinate(after_numpy))
            observed_runtime.update(
                {
                    "resolved_tensor_shape": list(shape),
                    "runtime_prefix_length": shape[1],
                    "runtime_dtype": str(getattr(before_mean, "dtype", "unknown")),
                    "runtime_device": str(getattr(before_mean, "device", "unknown")),
                }
            )
            realized.append(
                {
                    "probe_margin_before": probe_before,
                    "probe_margin_after": probe_after,
                    "probe_margin_delta": probe_after - probe_before,
                    "applied_direction_coordinate_before": direction_before,
                    "applied_direction_coordinate_after": direction_after,
                    "applied_direction_coordinate_delta": direction_after - direction_before,
                    "raw_perturbation_l2": float(np.linalg.norm(raw_delta)),
                    "raw_perturbation_rms": float(
                        np.sqrt(np.mean(np.square(raw_delta)))
                    ),
                }
            )
            hook_calls += 1
            return rebuild(modified)

        policy.reset()
        layer.forward = forward_with_probe_direction
        try:
            with torch.no_grad():
                action = policy.predict_action_chunk(
                    self.observation,
                    noise=self.initial_noise.clone(),
                )
        finally:
            layer.forward = original_forward
        if hook_calls != 1 or len(realized) != 1:
            raise RuntimeError(
                "Expected the selected PI0.5 VLM prefix layer hook exactly once, "
                f"observed {hook_calls} calls"
            )
        metrics = realized[0]
        return _action_numpy(action), {
            "metrics": metrics,
            "runtime": {
                "hook_calls": hook_calls,
                "hidden_dim": direction.hidden_dim,
                "token_indices": list(selected_tokens),
                **observed_runtime,
            },
        }

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
    target, operator, _schedule = _resolve_request(payload)
    probe_direction = None
    if str(operator.parameters.get("mode") or "") == PROBE_DIRECTION_MODE:
        probe_direction = resolve_object_roi_probe_direction(
            dataset,
            target,
            trace_id=trace_id,
            policy_call_index=policy_call_index,
        )
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
        probe_direction=probe_direction,
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


def _validate_probe_direction_request(
    target: TargetSpec,
    operator: InterventionOperatorSpec,
    schedule: InterventionScheduleSpec,
    *,
    direction: ResolvedProbeDirection,
    policy_call_index: int,
) -> None:
    if operator.parameters.get("mode") != PROBE_DIRECTION_MODE:
        raise ValueError(
            "Artifact probe intervention requires operator.parameters.mode="
            f"{PROBE_DIRECTION_MODE!r}"
        )
    if operator.operator not in {"add_direction", "project_out_direction"}:
        raise ValueError(
            "Artifact probe intervention supports add_direction or project_out_direction"
        )
    if operator.strength is None or not np.isfinite(operator.strength):
        raise ValueError("Artifact probe intervention requires a finite explicit strength")
    if target.model_site != direction.model_site or target.layer != direction.layer:
        raise ValueError("Runtime target disagrees with the resolved probe direction")
    if isinstance(schedule.policy_calls, tuple) and policy_call_index not in schedule.policy_calls:
        raise ValueError(
            f"Schedule does not include selected policy call {policy_call_index}: "
            f"{schedule.policy_calls}"
        )
    if schedule.generation_steps != "all":
        raise ValueError(
            "VLM prefix direction is computed once before denoising; generation_steps must be 'all'"
        )
    if isinstance(schedule.tokens, Mapping):
        raw_indices = schedule.tokens.get("indices")
        if tuple(int(value) for value in raw_indices or ()) != direction.token_indices:
            raise ValueError("Schedule token indices disagree with target.token_selector.indices")
    elif schedule.tokens != "target_tokens":
        raise ValueError("Artifact probe intervention schedule.tokens must be 'target_tokens'")


def _pi05_vlm_layer(policy: Any, layer_index: int) -> Any:
    root = getattr(getattr(policy, "model", None), "paligemma_with_expert", None)
    paligemma = getattr(root, "paligemma", None)
    candidates = (
        getattr(paligemma, "language_model", None),
        getattr(getattr(paligemma, "model", None), "language_model", None),
    )
    for language_model in candidates:
        layers = getattr(language_model, "layers", None)
        if layers is not None and 0 <= int(layer_index) < len(layers):
            return layers[int(layer_index)]
    raise ValueError(
        f"Loaded PI0.5 runtime does not expose VLM decoder layer {layer_index}"
    )


def _layer_hidden_output(output: Any) -> tuple[Any, Any]:
    if isinstance(output, tuple):
        if not output:
            raise RuntimeError("PI0.5 VLM layer returned an empty tuple")
        return output[0], lambda hidden: (hidden, *output[1:])
    if isinstance(output, list):
        if not output:
            raise RuntimeError("PI0.5 VLM layer returned an empty list")
        return output[0], lambda hidden: [hidden, *output[1:]]
    if not hasattr(output, "shape"):
        raise RuntimeError("PI0.5 VLM layer output does not expose a hidden tensor")
    return output, lambda hidden: hidden


def _tensor_numpy(value: Any) -> np.ndarray:
    tensor = value.detach() if hasattr(value, "detach") else value
    tensor = tensor.float() if hasattr(tensor, "float") else tensor
    tensor = tensor.cpu() if hasattr(tensor, "cpu") else tensor
    tensor = tensor.numpy() if hasattr(tensor, "numpy") else tensor
    return np.asarray(tensor, dtype=np.float64)


def _match_nonzero_norm(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    candidate_norm = float(np.linalg.norm(candidate))
    reference_norm = float(np.linalg.norm(reference))
    if reference_norm == 0.0:
        return np.zeros_like(candidate, dtype=np.float64)
    if candidate_norm == 0.0:
        raise ValueError("Specificity control has zero realized perturbation before norm matching")
    return np.asarray(candidate, dtype=np.float64) * (reference_norm / candidate_norm)


def _control_parameters(payload: Mapping[str, Any], requested_kind: str) -> Mapping[str, Any]:
    _target, _operator, _schedule = _resolve_request(payload)
    intervention = _mapping(payload.get("intervention"))
    request = _mapping(payload.get("request")) or _mapping(intervention.get("request"))
    controls = request.get("controls") or request.get("control") or ()
    if isinstance(controls, (str, Mapping)):
        controls = (controls,)
    matches: list[Mapping[str, Any]] = []
    for item in controls if isinstance(controls, (list, tuple)) else ():
        record = _mapping(item) if isinstance(item, Mapping) else {"kind": str(item)}
        kind = str(record.get("kind") or "")
        aliases = {kind}
        if kind == "random_direction":
            aliases.add("random_direction_control")
        parameters = _mapping(record.get("parameters"))
        if kind == "wrong_feature" and parameters.get("role") == "wrong_identity":
            aliases.add("wrong_identity")
        if kind == "wrong_token" and parameters.get("role") == "wrong_roi":
            aliases.add("wrong_roi")
        if requested_kind in aliases:
            matches.append(_mapping(record.get("parameters")))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one declared control matching {requested_kind!r}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _operator_semantics(operator: str) -> str:
    if operator == "project_out_direction":
        return "remove feature-dependent direction from ROI mean, leaving classifier intercept"
    return "add signed standardized probe-space direction to every selected ROI token"


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
