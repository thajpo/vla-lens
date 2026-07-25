from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

from vla_lens.pi05.intervention_executor import (
    PI05ActionInterventionExecutor,
    replay_policy_call_observation,
)
from vla_lens.pi05.replay import PolicyCallReplayInputs, ReplayConfig


class FakeVectorEnv:
    def __init__(self):
        self.envs = [SimpleNamespace()]
        self.steps: list[np.ndarray] = []
        self.closed = False
        self.reset_seed = None

    def reset(self, *, seed):
        self.reset_seed = seed
        return {"value": 0}, {}

    def step(self, action):
        self.steps.append(np.asarray(action))
        observation = {"value": len(self.steps)}
        return observation, np.array([0.0]), np.array([False]), np.array([False]), {}

    def close(self):
        self.closed = True


def _replay_inputs(
    *,
    layout_id: int | None = 4,
    trace_id: str = "trace-a",
) -> PolicyCallReplayInputs:
    return PolicyCallReplayInputs(
        config=ReplayConfig(
            benchmark="libero_object",
            task_id=2,
            layout_id=layout_id,
            seed=99,
            horizon=6,
            obs_size=224,
        ),
        trace_id=trace_id,
        policy_call_index=0,
        observation_timestep=2,
        policy_call={"policy_call_index": 0},
        stored_action_chunk=np.zeros((2, 2), dtype=np.float32),
        initial_noise=np.zeros((2, 4), dtype=np.float32),
        initial_noise_ref="flow_initial_noise[0]",
        initial_noise_exactness="exact",
    )


def test_replay_policy_call_observation_replays_actions_and_capture_preprocessing_order():
    env = FakeVectorEnv()

    def append_stage(stage):
        def apply(observation):
            return {**observation, "stages": [*observation.get("stages", []), stage]}

        return apply

    runtime = SimpleNamespace(
        make_env_config=lambda *args, **kwargs: {"args": args, "kwargs": kwargs},
        make_env=lambda *args, **kwargs: {"libero_object": {2: env}},
        make_env_pre_post_processors=lambda **kwargs: (append_stage("env"), object()),
        policy_cfg=object(),
        preprocess_observation=append_stage("raw"),
        add_envs_task=lambda selected_env, observation: append_stage("task")(observation),
        preprocessor=append_stage("policy"),
    )
    actions = np.arange(12, dtype=np.float32).reshape(6, 2)

    observation = replay_policy_call_observation(runtime, actions, _replay_inputs())

    assert observation == {"value": 2, "stages": ["raw", "task", "env", "policy"]}
    assert env.reset_seed == [99]
    assert len(env.steps) == 2
    np.testing.assert_array_equal(env.steps[0], actions[0][None, :])
    assert env.envs[0].episode_index == 4
    assert env.envs[0].init_state_id == 4
    assert env.closed is True


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)
        self.device = "fake"
        self.dtype = "float32"

    @property
    def shape(self):
        return self.value.shape

    def clone(self):
        return FakeTensor(self.value.copy())

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value.copy()

    def __add__(self, other):
        value = other.value if isinstance(other, FakeTensor) else other
        return FakeTensor(self.value + value)

    def __rmul__(self, other):
        return FakeTensor(other * self.value)

    def __mul__(self, other):
        return FakeTensor(self.value * other)

    def __sub__(self, other):
        value = other.value if isinstance(other, FakeTensor) else other
        return FakeTensor(self.value - value)

    def __getitem__(self, index):
        return FakeTensor(self.value[index])

    def __setitem__(self, index, value):
        self.value[index] = value.value if isinstance(value, FakeTensor) else value

    def to(self, *, device, dtype):
        del device, dtype
        return FakeTensor(self.value.copy())


class FakeProjection:
    in_features = 4

    def forward(self, activation):
        return FakeTensor(activation.value[..., :2])


class FakePolicy:
    def __init__(self):
        self.model = SimpleNamespace(action_out_proj=FakeProjection())
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1

    def predict_action_chunk(self, observation, *, noise):
        del observation, noise
        output = None
        for _ in range(3):
            output = self.model.action_out_proj.forward(FakeTensor(np.zeros((1, 2, 4))))
        return output


class FakeTorch:
    @staticmethod
    def no_grad():
        return nullcontext()

    @staticmethod
    def as_tensor(value, *, device, dtype):
        del device, dtype
        return FakeTensor(value)


def _hook_request() -> dict:
    return {
        "target": {
            "kind": "manual",
            "model_family": "pi05",
            "model_site": "pi05.action_head.input",
            "token_space": "pi05.action_suffix",
        },
        "intervention": {
            "request": {
                "operator": {
                    "operator": "add_direction",
                    "strength": 0.5,
                    "parameters": {"mode": "synthetic_hook_smoke", "dimension": 0},
                },
                "schedule": {
                    "policy_calls": [0],
                    "generation_steps": "all",
                    "tokens": "action",
                },
            }
        },
    }


def test_pi05_executor_applies_and_restores_non_claiming_synthetic_hook():
    policy = FakePolicy()
    runtime = SimpleNamespace(torch=FakeTorch(), policy=policy)
    executor = PI05ActionInterventionExecutor(
        runtime=runtime,
        replay_inputs=_replay_inputs(),
        observation={},
        initial_noise=FakeTensor(np.zeros((1, 2, 4))),
    )
    original_forward = policy.model.action_out_proj.forward

    noop = executor.run_noop(_hook_request())
    intervention = executor.run_intervention(_hook_request())

    np.testing.assert_array_equal(noop.action_chunk, np.zeros((2, 2), dtype=np.float32))
    np.testing.assert_array_equal(
        intervention.action_chunk,
        np.array([[0.5, 0.0], [0.5, 0.0]], dtype=np.float32),
    )
    assert intervention.runtime["applied_generation_steps"] == [0, 1, 2]
    assert intervention.runtime["claim_eligible"] is False
    assert policy.model.action_out_proj.forward == original_forward


class FakeVLMLayer:
    def forward(self, hidden):
        return (hidden, {"cache": "unchanged"})


class FakeSourcePatchPolicy:
    def __init__(self):
        self.layer = FakeVLMLayer()
        language_model = SimpleNamespace(layers=[self.layer])
        paligemma = SimpleNamespace(language_model=language_model)
        self.model = SimpleNamespace(
            paligemma_with_expert=SimpleNamespace(paligemma=paligemma)
        )
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1

    def predict_action_chunk(self, observation, *, noise):
        del noise
        hidden, _cache = self.layer.forward(FakeTensor(observation["hidden"]))
        return hidden[:, 1:3, :2]


def _source_patch_request() -> dict:
    return {
        "target": {
            "kind": "activation_slice",
            "model_family": "pi05",
            "model_site": "pi05.vlm.layers.0.prefix.hidden_tokens",
            "layer": 0,
            "token_space": "pi05.prefix",
            "token_selector": {"indices": [1, 2]},
        },
        "donor": {
            "trace": {"trace_id": "trace-b"},
            "policy_call": {"trace_id": "trace-b", "policy_call_index": 0},
        },
        "intervention": {
            "request": {
                "operator": {
                    "operator": "source_patch",
                    "strength": 1.0,
                    "parameters": {"mode": "donor_source_patch"},
                },
                "schedule": {
                    "policy_calls": [0],
                    "generation_steps": "all",
                    "tokens": "target_tokens",
                },
            }
        },
    }


def test_pi05_executor_caches_donor_and_patches_declared_prefix_tokens():
    policy = FakeSourcePatchPolicy()
    runtime = SimpleNamespace(torch=FakeTorch(), policy=policy)
    recipient_hidden = np.zeros((1, 4, 3), dtype=np.float32)
    donor_hidden = np.ones((1, 4, 3), dtype=np.float32)
    executor = PI05ActionInterventionExecutor(
        runtime=runtime,
        replay_inputs=_replay_inputs(trace_id="trace-a"),
        observation={"hidden": recipient_hidden},
        initial_noise=FakeTensor(np.zeros((1, 2, 2))),
        donor_replay_inputs=_replay_inputs(trace_id="trace-b"),
        donor_observation={"hidden": donor_hidden},
        pair_compatibility={"model_id": True, "prompt": True},
    )
    original_forward = policy.layer.forward

    noop = executor.run_noop(_source_patch_request())
    patched = executor.run_intervention(_source_patch_request())
    patched_again = executor.run_intervention(_source_patch_request())

    np.testing.assert_array_equal(noop.action_chunk, np.zeros((2, 2)))
    np.testing.assert_array_equal(patched.action_chunk, np.ones((2, 2)))
    np.testing.assert_array_equal(
        patched.array_outputs["donor_shared_noise"],
        np.ones((2, 2)),
    )
    assert patched.runtime["hook_calls"] == 1
    assert patched.runtime["recipient_token_indices"] == [1, 2]
    assert patched.runtime["donor_token_indices"] == [1, 2]
    assert patched.runtime["shared_noise_ref"] == "flow_initial_noise[0]"
    assert sorted(executor.donor_hidden_cache) == [0]
    assert policy.reset_calls == 4  # no-op, one donor capture, and two patched recipients
    assert policy.layer.forward == original_forward
    np.testing.assert_array_equal(patched_again.action_chunk, patched.action_chunk)


def test_pi05_source_patch_controls_are_explicit_and_norm_matched():
    policy = FakeSourcePatchPolicy()
    donor_hidden = np.ones((1, 4, 3), dtype=np.float32)
    donor_hidden[:, [0, 3], :] = 4.0
    executor = PI05ActionInterventionExecutor(
        runtime=SimpleNamespace(torch=FakeTorch(), policy=policy),
        replay_inputs=_replay_inputs(trace_id="trace-a"),
        observation={"hidden": np.zeros((1, 4, 3), dtype=np.float32)},
        initial_noise=FakeTensor(np.zeros((1, 2, 2))),
        donor_replay_inputs=_replay_inputs(trace_id="trace-b"),
        donor_observation={"hidden": donor_hidden},
        pair_compatibility={"model_id": True, "prompt": True},
    )
    original_forward = policy.layer.forward
    request = _source_patch_request()
    request["intervention"]["request"]["controls"] = [
        {"kind": "recipient_self_patch"},
        {"kind": "donor_self_patch"},
        {"kind": "shuffled_donor", "parameters": {"seed": 4}},
        {"kind": "random_matched_norm", "parameters": {"seed": 7}},
        {
            "kind": "wrong_region",
            "parameters": {
                "recipient_indices": [0, 3],
                "donor_indices": [0, 3],
            },
        },
    ]

    recipient_self = executor.run_control(request, control_kind="recipient_self_patch")
    donor_self = executor.run_control(request, control_kind="donor_self_patch")
    shuffled = executor.run_control(request, control_kind="shuffled_donor")
    random = executor.run_control(request, control_kind="random_matched_norm")
    wrong_region = executor.run_control(request, control_kind="wrong_region")

    np.testing.assert_array_equal(recipient_self.action_chunk, np.zeros((2, 2)))
    np.testing.assert_array_equal(donor_self.action_chunk, np.ones((2, 2)))
    np.testing.assert_array_equal(shuffled.action_chunk, np.ones((2, 2)))
    np.testing.assert_array_equal(wrong_region.action_chunk, np.zeros((2, 2)))
    expected_source_norm = np.sqrt(2 * 3)
    for control in (shuffled, random, wrong_region):
        assert np.isclose(control.metrics["source_delta_l2"], expected_source_norm)
        assert control.runtime["norm_matched_to_intended"] is True
    assert np.isclose(
        random.metrics["realized_perturbation_l2"],
        random.metrics["source_delta_l2"],
    )
    assert random.runtime["token_mapping_sha256"]
    assert random.runtime["donor_values_sha256"]
    assert random.runtime["shared_noise_sha256"]
    assert policy.layer.forward == original_forward


class FakeExpertSourcePatchPolicy:
    def __init__(self):
        self.layer = FakeVLMLayer()
        expert = SimpleNamespace(model=SimpleNamespace(layers=[self.layer]))
        self.model = SimpleNamespace(
            paligemma_with_expert=SimpleNamespace(gemma_expert=expert)
        )
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1

    def predict_action_chunk(self, observation, *, noise):
        del noise
        rows = []
        for hidden_at_step in observation["hidden_by_step"]:
            hidden, _cache = self.layer.forward(FakeTensor(hidden_at_step))
            rows.append(hidden[:, 1, :2].value)
        return FakeTensor(np.concatenate(rows, axis=0))


def _expert_source_patch_request(*, generation_steps="all") -> dict:
    request = _source_patch_request()
    request["target"] = {
        "kind": "activation_slice",
        "model_family": "pi05",
        "model_site": "pi05.expert.layers.0.by_step.hidden_tokens",
        "layer": 0,
        "token_space": "pi05.action_suffix",
        "token_selector": {"indices": [0, 1, 2, 3]},
    }
    request["intervention"]["request"]["operator"]["parameters"][
        "donor_token_indices"
    ] = [0, 1, 2, 3]
    request["intervention"]["request"]["schedule"][
        "generation_steps"
    ] = generation_steps
    return request


def test_pi05_executor_patches_expert_tokens_at_every_denoising_step():
    policy = FakeExpertSourcePatchPolicy()
    recipient = np.zeros((3, 1, 4, 3), dtype=np.float32)
    donor = np.stack(
        [np.full((1, 4, 3), step, dtype=np.float32) for step in (1, 2, 3)]
    )
    executor = PI05ActionInterventionExecutor(
        runtime=SimpleNamespace(torch=FakeTorch(), policy=policy),
        replay_inputs=_replay_inputs(trace_id="trace-a"),
        observation={"hidden_by_step": recipient},
        initial_noise=FakeTensor(np.zeros((1, 2, 2))),
        donor_replay_inputs=_replay_inputs(trace_id="trace-b"),
        donor_observation={"hidden_by_step": donor},
        pair_compatibility={"model_id": True, "prompt": True},
    )
    original_forward = policy.layer.forward

    patched = executor.run_intervention(_expert_source_patch_request())

    np.testing.assert_array_equal(
        patched.action_chunk,
        np.asarray([[1, 1], [2, 2], [3, 3]], dtype=np.float32),
    )
    assert patched.runtime["hook_calls"] == 3
    assert patched.runtime["expected_hook_calls"] == 3
    assert patched.runtime["hook_valid"] is True
    assert patched.runtime["applied_generation_steps"] == [0, 1, 2]
    assert patched.runtime["patch_site"]["stack"] == "expert_action"
    assert len(executor.donor_site_cache[patched.runtime["model_site"]]) == 3
    assert policy.layer.forward == original_forward


def test_pi05_expert_patch_supports_step_slices_and_alpha_zero_control():
    policy = FakeExpertSourcePatchPolicy()
    recipient = np.zeros((3, 1, 4, 3), dtype=np.float32)
    donor = np.ones((3, 1, 4, 3), dtype=np.float32)
    executor = PI05ActionInterventionExecutor(
        runtime=SimpleNamespace(torch=FakeTorch(), policy=policy),
        replay_inputs=_replay_inputs(trace_id="trace-a"),
        observation={"hidden_by_step": recipient},
        initial_noise=FakeTensor(np.zeros((1, 2, 2))),
        donor_replay_inputs=_replay_inputs(trace_id="trace-b"),
        donor_observation={"hidden_by_step": donor},
        pair_compatibility={"model_id": True, "prompt": True},
    )
    request = _expert_source_patch_request(generation_steps={"indices": [0]})
    request["intervention"]["request"]["controls"] = [{"kind": "alpha_zero"}]

    patched = executor.run_intervention(request)
    alpha_zero = executor.run_control(request, control_kind="alpha_zero")

    np.testing.assert_array_equal(
        patched.action_chunk,
        np.asarray([[1, 1], [0, 0], [0, 0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(alpha_zero.action_chunk, np.zeros((3, 2)))
    assert patched.runtime["hook_calls"] == 3
    assert patched.runtime["applied_generation_steps"] == [0]
    assert alpha_zero.metrics["realized_perturbation_l2"] == 0.0
