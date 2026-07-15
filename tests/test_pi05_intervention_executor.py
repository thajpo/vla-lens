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


def _replay_inputs(*, layout_id: int | None = 4) -> PolicyCallReplayInputs:
    return PolicyCallReplayInputs(
        config=ReplayConfig(
            benchmark="libero_object",
            task_id=2,
            layout_id=layout_id,
            seed=99,
            horizon=6,
            obs_size=224,
        ),
        trace_id="trace-a",
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
