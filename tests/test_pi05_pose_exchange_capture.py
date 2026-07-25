from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

from vla_lens.pi05.pose_exchange_capture import (
    _existing_trace_ids,
    expand_pose_exchange_capture_job,
)


def test_pose_exchange_capture_plan_is_paired_deterministic_and_replayable():
    job = {
        "benchmark": "libero_90",
        "task_id": 73,
        "capture_profile": "rollout",
        "pairs": [
            {
                "pair_id": "rq019-layout0",
                "seed": 3100,
                "layout_id": 0,
                "target_object": "black_book_1",
                "distractor_object": "white_yellow_mug_1",
            },
            {
                "pair_id": "rq019-layout1",
                "seed": 3101,
                "layout_id": 1,
                "target_object": "black_book_1",
                "distractor_object": "white_yellow_mug_1",
            },
        ],
    }

    first = expand_pose_exchange_capture_job(job)
    second = expand_pose_exchange_capture_job(job)

    assert first == second
    assert len(first) == 4
    assert [row.role for row in first] == ["recipient", "donor", "recipient", "donor"]
    recipient, donor = first[:2]
    assert recipient.paired_trace_id == donor.trace_id
    assert donor.paired_trace_id == recipient.trace_id
    assert recipient.scene_mutation == {"kind": "identity", "objects": []}
    assert donor.scene_mutation == {
        "kind": "pose_exchange",
        "objects": ["black_book_1", "white_yellow_mug_1"],
    }
    assert recipient.layout_id == donor.layout_id == 0
    assert recipient.seed == donor.seed == 3100
    assert recipient.trace_id.endswith("rq019_layout0_recipient")
    assert donor.trace_id.endswith("rq019_layout0_donor")


def test_resume_reads_trace_identity_from_complete_episode_manifest(tmp_path, monkeypatch):
    manifest = tmp_path / "vla_lens" / "episodes" / "episode_000007" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"trace_id": "named-trace"}), encoding="utf-8")
    bundle = SimpleNamespace(
        scene_state=pd.DataFrame({"object_index": [0], "object_name": ["object"]}),
        tokens=pd.DataFrame({"token_index": [0]}),
        policy_calls=pd.DataFrame({"policy_call_index": [0]}),
        array=lambda _name, mmap=True: [0],
    )
    monkeypatch.setattr(
        "vla_lens.pi05.pose_exchange_capture.TraceBundle.open",
        lambda _path: bundle,
    )

    assert _existing_trace_ids(tmp_path) == {"named-trace"}
