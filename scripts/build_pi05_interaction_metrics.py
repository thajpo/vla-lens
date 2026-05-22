"""Build PI0.5 interaction-label metrics from a VLA-lens trace dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from vla_lens.pi05.interaction_metrics import save_pi05_interaction_metrics_artifact
from vla_lens.traces import TraceDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Trace dataset root")
    parser.add_argument("--name", default="PI0.5 interaction metrics")
    parser.add_argument("--movement-threshold-m", type=float, default=None)
    parser.add_argument("--lift-threshold-m", type=float, default=None)
    parser.add_argument("--consecutive-frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = {}
    if args.movement_threshold_m is not None:
        thresholds["movement_distance_m"] = args.movement_threshold_m
    if args.lift_threshold_m is not None:
        thresholds["lift_z_m"] = args.lift_threshold_m
    if args.consecutive_frames is not None:
        thresholds["consecutive_frames"] = args.consecutive_frames
    dataset = TraceDataset.open(args.root)
    saved = save_pi05_interaction_metrics_artifact(
        dataset,
        name=args.name,
        thresholds=thresholds,
    )
    print(f"artifact_id={saved.artifact.artifact_id}")
    print(f"artifact_type={saved.artifact.artifact_type}")
    print(f"path={saved.artifact.path}")
    print(f"episodes={len(saved.episode_labels)}")
    print(f"object_rows={len(saved.object_metrics)}")
    print(f"target_parse_success_rate={saved.artifact.metrics['target_parse_success_rate']}")
    print(f"no_object_moved_rate={saved.artifact.metrics['no_object_moved_rate']}")


if __name__ == "__main__":
    main()
