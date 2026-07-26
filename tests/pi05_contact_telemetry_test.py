from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vla_lens.pi05.capture import (
    CapturePlan,
    EpisodeBuffer,
    _write_episode,
    namespace_for_capture_args,
)
from vla_lens.pi05.context_capture import capture_contact_snapshot, capture_libero_context
from vla_lens.pi05.context_capture_contacts import (
    CONTACT_SAMPLING_LIMITATION,
    contact_capability_audit,
)
from vla_lens.traces import TraceDataset


class _FakeObject:
    def __init__(self, root_body: str):
        self.root_body = root_body


class _FakeContact:
    def __init__(
        self,
        geom1: int,
        geom2: int,
        dist: float,
        pos: list[float],
        frame: list[float],
    ):
        self.geom1 = geom1
        self.geom2 = geom2
        self.dist = dist
        self.pos = np.asarray(pos, dtype=np.float64)
        self.frame = np.asarray(frame, dtype=np.float64)


class _FakeModel:
    nbody = 4
    body_names = ["world", "robot0_finger", "mug_root", "table_root"]
    geom_names = ["finger_collision", "mug_collision", "table_collision"]
    body_parentid = np.array([0, 0, 0, 0], dtype=np.int32)
    geom_bodyid = np.array([1, 2, 3], dtype=np.int32)

    def body_name2id(self, name: str) -> int:
        return self.body_names.index(name)

    def body_id2name(self, index: int) -> str:
        return self.body_names[index]

    def geom_id2name(self, index: int) -> str:
        return self.geom_names[index]


class _FakeDataWithForce:
    def __init__(self, contacts: list[_FakeContact]):
        self.contact = contacts
        self.ncon = len(contacts)

    def get_contact_force(self, contact_index: int) -> np.ndarray:
        return np.asarray(
            [10.0 + contact_index, 1.0, 2.0, 0.1, 0.2, 0.3],
            dtype=np.float64,
        )


class _FakeDataWithoutForce:
    def __init__(self, contacts: list[_FakeContact]):
        self.contact = contacts
        self.ncon = len(contacts)


class _FakeSim:
    mujoco_version = "2.3.7"

    def __init__(self, data: object):
        self.model = _FakeModel()
        self.data = data


class _InnerEnv:
    def __init__(self, sim: _FakeSim):
        self.sim = sim
        self.obj_body_id = {"mug": 2, "table": 3}
        self.objects_dict = {"mug": _FakeObject("mug_root")}
        self.fixtures_dict = {"table": _FakeObject("table_root")}


class _WrappedEnv:
    def __init__(self, sim: _FakeSim):
        self.env = _InnerEnv(sim)


def _contacts() -> list[_FakeContact]:
    identity = np.eye(3, dtype=np.float64).reshape(-1).tolist()
    return [
        _FakeContact(0, 1, -0.002, [0.1, 0.2, 0.3], identity),
        _FakeContact(1, 2, 0.004, [0.4, 0.5, 0.6], identity),
    ]


def test_contact_snapshot_captures_raw_manifold_ownership_and_force() -> None:
    env = _WrappedEnv(_FakeSim(_FakeDataWithForce(_contacts())))

    snapshot = capture_contact_snapshot(env, timestep=12)

    assert snapshot["capability"]["available"] is True
    assert snapshot["capability"]["mujoco_version"] == "2.3.7"
    assert snapshot["capability"]["mujoco_version_exact"] is True
    assert snapshot["capability"]["sample_phase"] == "pre_action_control_step"
    assert snapshot["capability"]["exhaustive_physics_substeps"] is False

    contact, proximity = snapshot["contacts"]
    assert contact["timestep"] == 12
    assert contact["geom1_id"] == 0
    assert contact["geom1_name"] == "finger_collision"
    assert contact["geom1_body_name"] == "robot0_finger"
    assert contact["geom1_owner_kind"] == "body"
    assert contact["geom2_name"] == "mug_collision"
    assert contact["geom2_owner_name"] == "mug"
    assert contact["geom2_owner_kind"] == "object"
    assert contact["signed_distance_m"] == -0.002
    assert contact["distance_class"] == "penetrating"
    assert contact["physical_contact"] is True
    assert contact["positive_gap_proximity"] is False
    assert contact["position_world_m"] == [0.1, 0.2, 0.3]
    assert contact["frame_world_row_major"] == np.eye(3).reshape(-1).tolist()
    assert contact["force_torque_contact_frame"] == [10.0, 1.0, 2.0, 0.1, 0.2, 0.3]
    assert contact["force_available"] is True

    assert proximity["signed_distance_m"] == 0.004
    assert proximity["distance_class"] == "positive_gap_within_contact_margin"
    assert proximity["physical_contact"] is False
    assert proximity["positive_gap_proximity"] is True


def test_contact_force_is_omitted_without_a_robust_simulator_api() -> None:
    env = _WrappedEnv(_FakeSim(_FakeDataWithoutForce(_contacts()[:1])))

    snapshot = capture_contact_snapshot(env, timestep=0)
    audit = contact_capability_audit([snapshot])

    assert snapshot["contacts"][0]["force_torque_contact_frame"] is None
    assert snapshot["contacts"][0]["force_available"] is False
    assert snapshot["contacts"][0]["force_source"] == ""
    assert audit["available"] is True
    assert audit["force_capture_status"] == "unavailable"


def test_zero_contacts_is_available_telemetry_not_missing_telemetry() -> None:
    env = _WrappedEnv(_FakeSim(_FakeDataWithForce([])))

    snapshot = capture_contact_snapshot(env, timestep=4)
    audit = contact_capability_audit([snapshot])

    assert snapshot["contacts"] == []
    assert audit["available"] is True
    assert audit["verdict"] == "available_control_step_contact_manifold"
    assert audit["contact_count"] == 0
    assert audit["force_capture_status"] == "api_available_unobserved"
    assert audit["sampling_limitation"] == CONTACT_SAMPLING_LIMITATION
    assert audit["legacy_interaction_metrics_relabelled"] is False


def test_contact_context_and_capability_persist_in_scene_state(tmp_path) -> None:
    env = _WrappedEnv(_FakeSim(_FakeDataWithForce(_contacts()[:1])))
    snapshot = capture_contact_snapshot(env, timestep=0)
    context = capture_libero_context(contact_snapshots=[snapshot])

    assert context.tables["contacts"]["geom2_owner_name"].tolist() == ["mug"]
    assert context.tables["contact_capability"].iloc[0]["available"]

    buffer = EpisodeBuffer(
        trace_id="contact-trace",
        task_id=0,
        task_name="pick up mug",
        prompt="pick up mug",
        seed=17,
        executed_actions=[np.zeros(7, dtype=np.float32)],
        rewards=[0.0],
        observations=[{"robot0_joint_pos": np.zeros(7, dtype=np.float32)}],
        contact_snapshots=[snapshot],
        infos=[{}],
        terminated=[True],
        truncated=[False],
    )
    args = namespace_for_capture_args(
        capture_profile="rollout",
        output_root=tmp_path,
        device="cpu",
    )
    plan = CapturePlan(
        profile="rollout",
        vlm_layers=(),
        expert_layers=(),
        vlm_hidden="none",
        vlm_attention="none",
        expert_hidden="none",
        expert_attention="none",
        storage_dtype="float16",
    )
    policy = SimpleNamespace(config=SimpleNamespace(device="cpu"))

    _write_episode(buffer, args, policy, plan, env=env)

    bundle = TraceDataset.open(tmp_path).bundle("contact-trace")
    contact_rows = bundle.scene_state.loc[
        bundle.scene_state["context_kind"].eq("mujoco_contact")
    ]
    capability_rows = bundle.scene_state.loc[
        bundle.scene_state["context_kind"].eq("contact_capability")
    ]
    assert contact_rows["geom1_name"].tolist() == ["finger_collision"]
    assert contact_rows["signed_distance_m"].tolist() == [-0.002]
    assert capability_rows["verdict"].tolist() == [
        "available_control_step_contact_manifold"
    ]
    telemetry = bundle.capture_report["simulator_contact_telemetry"]
    assert telemetry["mujoco_version"] == "2.3.7"
    assert telemetry["sample_phase"] == "pre_action_control_step"
    assert telemetry["exhaustive_physics_substeps"] is False
