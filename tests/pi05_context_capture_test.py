from __future__ import annotations

import numpy as np

from vla_lens.pi05.context_capture import capture_libero_context, capture_scene_snapshot
from vla_lens.traces import TraceBundle, TraceManifest


def test_context_capture_extracts_robot_arrays_and_missing_reasons():
    observations = [
        {
            "robot0_joint_pos": np.array([[0.0, 0.1, 0.2]], dtype=np.float32),
            "robot0_joint_vel": np.array([[1.0, 1.1, 1.2]], dtype=np.float32),
            "robot0_eef_pos": np.array([[0.2, 0.3, 0.4]], dtype=np.float32),
            "robot0_eef_quat": np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            "robot0_gripper_qpos": np.array([[0.01, 0.02]], dtype=np.float32),
            "agentview_image": np.zeros((1, 8, 9, 3), dtype=np.uint8),
        },
        {
            "robot0_joint_pos": np.array([[0.3, 0.4, 0.5]], dtype=np.float32),
            "robot0_joint_vel": np.array([[1.3, 1.4, 1.5]], dtype=np.float32),
            "robot0_eef_pos": np.array([[0.5, 0.6, 0.7]], dtype=np.float32),
            "robot0_eef_quat": np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32),
            "robot0_gripper_qpos": np.array([[0.03, 0.04]], dtype=np.float32),
            "agentview_image": np.zeros((1, 8, 9, 3), dtype=np.uint8),
        },
    ]

    result = capture_libero_context(observations)

    assert result.arrays["robot_joint_pos"].array.shape == (2, 3)
    assert result.arrays["robot_joint_vel"].axes == ["timestep", "joint"]
    assert result.arrays["eef_pos"].array.shape == (2, 3)
    assert result.arrays["eef_mat"].array.shape == (2, 3, 3)
    np.testing.assert_allclose(result.arrays["eef_mat"].array[0], np.eye(3), atol=1e-6)
    assert result.arrays["gripper_qpos"].array.shape == (2, 2)
    assert result.arrays["camera_resolution"].array.tolist() == [[8, 9]]

    unavailable = result.unavailable
    assert set(unavailable["field"]) >= {"gripper_qvel", "intrinsics", "extrinsics"}
    qvel_reason = unavailable.loc[unavailable["field"] == "gripper_qvel", "reason"].iloc[0]
    assert "observation keys" in qvel_reason


def test_context_capture_extracts_lerobot_nested_robot_state():
    observations = [
        {
            "robot_state": {
                "joints": {
                    "pos": np.array([[0.0, 0.1, 0.2]], dtype=np.float32),
                    "vel": np.array([[1.0, 1.1, 1.2]], dtype=np.float32),
                },
                "eef": {
                    "pos": np.array([[0.2, 0.3, 0.4]], dtype=np.float32),
                    "quat": np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                    "mat": np.eye(3, dtype=np.float32)[None, ...],
                },
                "gripper": {
                    "qpos": np.array([[0.01, -0.01]], dtype=np.float32),
                    "qvel": np.array([[0.02, -0.02]], dtype=np.float32),
                },
            },
            "pixels": {
                "image": np.zeros((1, 8, 9, 3), dtype=np.uint8),
                "wrist_image": np.zeros((1, 8, 9, 3), dtype=np.uint8),
            },
        },
        {
            "robot_state": {
                "joints": {
                    "pos": np.array([[0.3, 0.4, 0.5]], dtype=np.float32),
                    "vel": np.array([[1.3, 1.4, 1.5]], dtype=np.float32),
                },
                "eef": {
                    "pos": np.array([[0.5, 0.6, 0.7]], dtype=np.float32),
                    "quat": np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
                    "mat": (np.eye(3, dtype=np.float32) * 2.0)[None, ...],
                },
                "gripper": {
                    "qpos": np.array([[0.03, -0.03]], dtype=np.float32),
                    "qvel": np.array([[0.04, -0.04]], dtype=np.float32),
                },
            },
            "pixels": {
                "image": np.zeros((1, 8, 9, 3), dtype=np.uint8),
                "wrist_image": np.zeros((1, 8, 9, 3), dtype=np.uint8),
            },
        },
    ]

    result = capture_libero_context(observations)

    np.testing.assert_allclose(
        result.arrays["robot_joint_pos"].array,
        np.array([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]], dtype=np.float32),
    )
    assert result.arrays["robot_joint_vel"].array.shape == (2, 3)
    assert result.arrays["eef_pos"].array.shape == (2, 3)
    assert result.arrays["eef_quat"].array.shape == (2, 4)
    assert result.arrays["eef_mat"].array.shape == (2, 3, 3)
    assert result.arrays["gripper_qpos"].array.shape == (2, 2)
    assert result.arrays["gripper_qvel"].array.shape == (2, 2)

    captured = result.availability.loc[
        result.availability["component"].eq("robot") & result.availability["available"],
        "field",
    ].tolist()
    assert set(captured) >= {
        "robot_joint_pos",
        "robot_joint_vel",
        "eef_pos",
        "eef_quat",
        "eef_mat",
        "gripper_qpos",
        "gripper_qvel",
    }


def test_context_capture_extracts_env_metadata_objects_and_predicates():
    class InnerLiberoEnv:
        task_id = 7
        task_name = "pick up the mug"
        task_description = "pick up the mug and place it on the plate"
        layout_id = "layout_a"
        bddl_file_name = "libero_object/pick_mug.bddl"
        seed = 123
        init_state = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        reset_state = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        object_names = ["mug", "plate"]
        object_poses = {
            "mug": {"pos": [0.1, 0.2, 0.3], "quat": [0.0, 0.0, 0.0, 1.0], "joints": [0.5]},
            "plate": {"pos": [0.4, 0.5, 0.6], "quat": [0.0, 0.0, 1.0, 0.0], "joints": [0.0]},
        }
        predicates = {"mug_grasped": False, "plate_contact": True}
        camera_names = ["agentview", "robot0_eye_in_hand"]
        camera_heights = {"agentview": 64, "robot0_eye_in_hand": 32}
        camera_widths = {"agentview": 96, "robot0_eye_in_hand": 48}

    class LeRobotWrapper:
        env = InnerLiberoEnv()

    result = capture_libero_context([], LeRobotWrapper())

    metadata = result.tables["episode_context"]
    assert set(metadata.loc[metadata["available"], "field"]) >= {
        "task_id",
        "task_name",
        "task_description",
        "layout_id",
        "bddl_file",
        "seed",
        "reset_state",
        "init_state",
    }
    assert result.arrays["scene_init_state"].array.tolist() == [1.0, 2.0, 3.0]
    assert result.arrays["scene_reset_state"].array.tolist() == [4.0, 5.0, 6.0]

    objects = result.tables["objects"]
    assert objects["object_name"].tolist() == ["mug", "plate"]
    assert result.arrays["scene_object_pos"].array.shape == (2, 3)
    assert result.arrays["scene_object_quat"].array.shape == (2, 4)
    assert result.arrays["scene_object_joints"].array.shape == (2, 1)
    assert result.arrays["scene_predicates"].metadata["predicate_names"] == [
        "mug_grasped",
        "plate_contact",
    ]
    assert result.arrays["scene_predicates"].array.tolist() == [0.0, 1.0]

    cameras = result.tables["cameras"]
    assert cameras["camera_name"].tolist() == ["agentview", "robot0_eye_in_hand"]
    assert result.arrays["camera_resolution"].array.tolist() == [[64, 96], [32, 48]]
    assert not cameras["intrinsics_available"].any()
    assert cameras["intrinsics_reason"].iloc[0]


def test_scene_snapshot_samples_mujoco_object_and_fixture_poses():
    class FakeModel:
        nbody = 3
        ngeom = 2
        body_names = ["world", "mug_root", "table_root"]
        body_parentid = np.array([0, 0, 0], dtype=np.int32)
        geom_bodyid = np.array([1, 2], dtype=np.int32)
        geom_size = np.array(
            [
                [0.01, 0.02, 0.03],
                [0.02, 0.03, 0.04],
            ],
            dtype=np.float32,
        )

        def body_name2id(self, name):
            return self.body_names.index(name)

        def body_id2name(self, index):
            return self.body_names[index]

    class FakeData:
        body_xpos = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ],
            dtype=np.float32,
        )
        # MuJoCo body quaternions are wxyz; the trace stores xyzw.
        body_xquat = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.7071068, 0.0, 0.0, 0.7071068],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        geom_xpos = np.array(
            [
                [0.11, 0.21, 0.31],
                [0.42, 0.52, 0.62],
            ],
            dtype=np.float32,
        )
        geom_xmat = np.stack([np.eye(3).reshape(-1), np.eye(3).reshape(-1)]).astype(np.float32)

    class FakeSim:
        model = FakeModel()
        data = FakeData()

    class FakeObject:
        def __init__(self, root_body):
            self.root_body = root_body

    class InnerLiberoEnv:
        sim = FakeSim()
        obj_body_id = {"mug": 1, "table": 2}
        objects_dict = {"mug": FakeObject("mug_root")}
        fixtures_dict = {"table": FakeObject("table_root")}

    class LeRobotWrapper:
        env = InnerLiberoEnv()

    snapshot = capture_scene_snapshot(LeRobotWrapper())

    assert [item["object_name"] for item in snapshot["objects"]] == ["mug", "table"]
    assert [item["object_kind"] for item in snapshot["objects"]] == ["object", "fixture"]
    np.testing.assert_allclose(snapshot["objects"][0]["pos"], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(
        snapshot["objects"][0]["quat"],
        [0.0, 0.0, 0.7071068, 0.7071068],
    )
    np.testing.assert_allclose(snapshot["objects"][0]["geom_center"], [0.11, 0.21, 0.31])
    np.testing.assert_allclose(snapshot["objects"][0]["bbox_min"], [0.10, 0.19, 0.28])
    np.testing.assert_allclose(snapshot["objects"][0]["bbox_max"], [0.12, 0.23, 0.34])
    assert snapshot["objects"][0]["geom_count"] == 1


def test_context_capture_prefers_time_aligned_scene_snapshots():
    class FakeModel:
        body_names = ["world", "mug_root", "table_root"]

        def body_name2id(self, name):
            return self.body_names.index(name)

        def body_id2name(self, index):
            return self.body_names[index]

    class FakeData:
        body_xpos = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ],
            dtype=np.float32,
        )
        body_xquat = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

    class FakeSim:
        model = FakeModel()
        data = FakeData()

    class FakeObject:
        def __init__(self, root_body):
            self.root_body = root_body

    class InnerLiberoEnv:
        sim = FakeSim()
        obj_body_id = {"mug": 1, "table": 2}
        objects_dict = {"mug": FakeObject("mug_root")}
        fixtures_dict = {"table": FakeObject("table_root")}
        object_poses = {
            "mug": {"pos": [9.0, 9.0, 9.0], "quat": [0.0, 0.0, 0.0, 1.0]},
            "table": {"pos": [8.0, 8.0, 8.0], "quat": [0.0, 0.0, 0.0, 1.0]},
        }

    class LeRobotWrapper:
        env = InnerLiberoEnv()

    first = capture_scene_snapshot(LeRobotWrapper())
    InnerLiberoEnv.sim.data.body_xpos[1] = [0.2, 0.3, 0.4]
    InnerLiberoEnv.sim.data.body_xpos[2] = [0.5, 0.6, 0.7]
    second = capture_scene_snapshot(LeRobotWrapper())

    result = capture_libero_context([], LeRobotWrapper(), scene_snapshots=[first, second])

    objects = result.tables["objects"]
    assert objects["object_name"].tolist() == ["mug", "table"]
    assert objects["object_kind"].tolist() == ["object", "fixture"]
    assert objects["pos_array_id"].tolist() == ["scene_object_pos", "scene_object_pos"]
    assert objects["quat_array_id"].tolist() == ["scene_object_quat", "scene_object_quat"]
    assert result.arrays["scene_object_pos"].axes == ["timestep", "object", "xyz"]
    assert result.arrays["scene_object_quat"].axes == ["timestep", "object", "xyzw"]
    np.testing.assert_allclose(
        result.arrays["scene_object_pos"].array[:, 0],
        [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],
    )
    np.testing.assert_allclose(
        result.arrays["scene_object_pos"].array[:, 1],
        [[0.4, 0.5, 0.6], [0.5, 0.6, 0.7]],
    )


def test_context_capture_uses_time_aligned_camera_snapshots():
    first = np.eye(4, dtype=np.float32)
    second = np.eye(4, dtype=np.float32)
    second[0, 3] = 0.25
    snapshots = [
        {
            "cameras": [
                {
                    "camera_name": "robot0_eye_in_hand",
                    "height": 8,
                    "width": 10,
                    "intrinsic": np.eye(3, dtype=np.float32),
                    "extrinsic": first,
                    "object_bboxes": [
                        {"object_name": "mug", "bbox_pixel_xyxy": [1.0, 2.0, 5.0, 6.0]}
                    ],
                }
            ]
        },
        {
            "cameras": [
                {
                    "camera_name": "robot0_eye_in_hand",
                    "height": 8,
                    "width": 10,
                    "intrinsic": np.eye(3, dtype=np.float32),
                    "extrinsic": second,
                    "object_bboxes": [
                        {"object_name": "mug", "bbox_pixel_xyxy": [2.0, 3.0, 6.0, 7.0]}
                    ],
                }
            ]
        },
    ]

    result = capture_libero_context(camera_snapshots=snapshots)

    assert result.arrays["camera_extrinsics"].axes == ["timestep", "camera", "row", "col"]
    assert result.arrays["camera_extrinsics"].array.shape == (2, 1, 4, 4)
    np.testing.assert_allclose(result.arrays["camera_extrinsics"].array[0, 0], first)
    np.testing.assert_allclose(result.arrays["camera_extrinsics"].array[1, 0], second)
    assert result.arrays["camera_object_bbox"].axes == [
        "timestep",
        "camera",
        "object",
        "bbox_xyxy",
    ]
    np.testing.assert_allclose(
        result.arrays["camera_object_bbox"].array[:, 0, 0],
        [[1.0, 2.0, 5.0, 6.0], [2.0, 3.0, 6.0, 7.0]],
    )
    assert result.arrays["camera_object_visible"].array[:, 0, 0].tolist() == [1, 1]
    assert result.tables["cameras"]["extrinsics_time_varying"].tolist() == [True]

    availability = result.availability
    captured = set(
        f"{row.component}.{row.field}"
        for row in availability.loc[availability["available"]].itertuples()
    )
    assert {
        "camera.names",
        "camera.resolution",
        "camera.intrinsics",
        "camera.extrinsics",
        "camera.object_bbox",
    } <= captured


def test_context_capture_reports_missing_object_pose_source():
    class EnvWithNamesOnly:
        object_names = ["mug"]

    result = capture_libero_context([], EnvWithNamesOnly())

    unavailable = result.unavailable
    reasons = {
        row.field: row.reason
        for row in unavailable.loc[unavailable["component"] == "object"].itertuples()
    }
    assert "pos" in reasons
    assert "no object pos values were exposed" in reasons["pos"]
    assert "quat" in reasons
    assert "no object quat values were exposed" in reasons["quat"]


def test_context_capture_extracts_time_aligned_object_values_from_observations():
    observations = [
        {
            "mug_pos": [0.1, 0.2, 0.3],
            "mug_quat": [0.0, 0.0, 0.0, 1.0],
            "mug_contact": 0,
            "plate_pos": [0.4, 0.5, 0.6],
            "plate_quat": [0.0, 0.0, 1.0, 0.0],
            "plate_contact": 1,
        },
        {
            "mug_pos": [0.2, 0.3, 0.4],
            "mug_quat": [0.0, 0.0, 0.0, 1.0],
            "mug_contact": 1,
            "plate_pos": [0.5, 0.6, 0.7],
            "plate_quat": [0.0, 0.0, 1.0, 0.0],
            "plate_contact": 1,
        },
    ]

    result = capture_libero_context(observations)

    assert result.tables["objects"]["object_name"].tolist() == ["mug", "plate"]
    assert result.arrays["scene_object_pos"].array.shape == (2, 2, 3)
    assert result.arrays["scene_object_pos"].axes == ["timestep", "object", "xyz"]
    assert result.arrays["scene_predicates"].axes == ["timestep", "predicate"]
    assert result.arrays["scene_predicates"].array.tolist() == [[0.0, 1.0], [1.0, 1.0]]


def test_context_arrays_can_be_written_to_trace_bundle(tmp_path):
    observations = [
        {
            "robot0_joint_pos": [0.0, 0.1],
            "robot0_joint_vel": [0.2, 0.3],
            "robot0_eef_pos": [0.4, 0.5, 0.6],
            "robot0_eef_quat": [0.0, 0.0, 0.0, 1.0],
            "robot0_gripper_qpos": [0.7, 0.8],
            "robot0_gripper_qvel": [0.9, 1.0],
            "agentview_image": np.zeros((4, 5, 3), dtype=np.uint8),
        }
    ]
    result = capture_libero_context(observations)
    arrays = {
        name: spec
        for name, spec in result.arrays.items()
        if name
        in {
            "robot_joint_pos",
            "robot_joint_vel",
            "eef_pos",
            "eef_quat",
            "eef_mat",
            "gripper_qpos",
            "gripper_qvel",
            "camera_resolution",
        }
    }

    bundle = TraceBundle.create(
        tmp_path / "context",
        manifest=TraceManifest(
            trace_id="context",
            episode_id="context",
            task_id="fake",
            prompt="fake",
            model_id="fake",
            env_id="libero",
            robot_id="panda",
            outcome="unknown",
            length=1,
        ),
        episode_arrays=arrays,
    )

    expected_eef_pos = np.array([[0.4, 0.5, 0.6]], dtype=np.float32)
    np.testing.assert_allclose(bundle.array("eef_pos"), expected_eef_pos)
    assert "robot_joint_pos" in set(bundle.array_index["name"])
