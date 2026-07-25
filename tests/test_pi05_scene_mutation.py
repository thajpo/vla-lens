from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from vla_lens.pi05.scene_mutation import (
    SceneMutationSpec,
    apply_scene_mutation,
    scene_mutation_from_json,
    scene_mutation_from_metadata,
)


class FakeModel:
    def get_joint_qpos_addr(self, name):
        return {"book_joint": (2, 9), "mug_joint": (9, 16)}[name]


class FakeData:
    def __init__(self):
        self.qpos = np.arange(20, dtype=np.float64)
        self.qvel = np.arange(10, dtype=np.float64)

    def get_joint_qpos(self, name):
        start, end = FakeModel().get_joint_qpos_addr(name)
        return self.qpos[start:end]

    def set_joint_qpos(self, name, value):
        start, end = FakeModel().get_joint_qpos_addr(name)
        self.qpos[start:end] = value


class FakeInnerEnv:
    def __init__(self, data):
        self.objects_dict = {
            "black_book_1": SimpleNamespace(joints=["book_joint"]),
            "white_yellow_mug_1": SimpleNamespace(joints=["mug_joint"]),
        }
        self.data = data

    def _get_observations(self):
        return {
            "pixels": {"image": np.zeros((4, 4, 3), dtype=np.uint8)},
            "robot": np.array([1.0, 2.0], dtype=np.float32),
        }


class FakeRawEnv:
    def __init__(self):
        self.sim = SimpleNamespace(model=FakeModel(), data=FakeData(), forward=lambda: None)
        self.env = FakeInnerEnv(self.sim.data)

    def check_success(self):
        return False

    def _post_process(self):
        return None

    def _update_observables(self, force=False):
        assert force is True


class FakeBaseEnv:
    def __init__(self):
        self._env = FakeRawEnv()

    def _format_raw_obs(self, observation):
        return observation


def test_pose_exchange_changes_only_the_two_free_joint_poses_and_batches_observation():
    base = FakeBaseEnv()
    vector = SimpleNamespace(envs=[base])
    before = base._env.sim.data.qpos.copy()
    first = before[2:9].copy()
    second = before[9:16].copy()

    observation, report = apply_scene_mutation(
        vector,
        SceneMutationSpec(
            kind="pose_exchange",
            objects=("black_book_1", "white_yellow_mug_1"),
        ),
    )

    after = base._env.sim.data.qpos
    np.testing.assert_array_equal(after[2:9], second)
    np.testing.assert_array_equal(after[9:16], first)
    np.testing.assert_array_equal(after[:2], before[:2])
    np.testing.assert_array_equal(after[16:], before[16:])
    assert observation["pixels"]["image"].shape == (1, 4, 4, 3)
    assert observation["robot"].shape == (1, 2)
    assert report["outside_object_qpos_max_abs"] == 0.0
    assert report["qvel_max_abs"] == 0.0
    assert report["observation_refreshed_without_step"] is True
    assert report["before_qpos_sha256"] != report["after_qpos_sha256"]


def test_scene_mutation_roundtrips_inline_file_and_saved_report(tmp_path):
    payload = {
        "kind": "pose_exchange",
        "objects": ["black_book_1", "white_yellow_mug_1"],
    }
    path = tmp_path / "mutation.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    inline = scene_mutation_from_json(json.dumps(payload))
    from_file = scene_mutation_from_json(str(path))
    from_report = scene_mutation_from_metadata({"spec": payload, "qvel_max_abs": 0.0})

    assert inline == from_file == from_report


def test_identity_scene_mutation_refreshes_without_changing_state():
    base = FakeBaseEnv()
    before_qpos = base._env.sim.data.qpos.copy()
    before_qvel = base._env.sim.data.qvel.copy()

    observation, report = apply_scene_mutation(
        SimpleNamespace(envs=[base]),
        SceneMutationSpec(kind="identity"),
    )

    np.testing.assert_array_equal(base._env.sim.data.qpos, before_qpos)
    np.testing.assert_array_equal(base._env.sim.data.qvel, before_qvel)
    assert observation["robot"].shape == (1, 2)
    assert report["changed_qpos_indices"] == []
    assert report["before_qpos_sha256"] == report["after_qpos_sha256"]
