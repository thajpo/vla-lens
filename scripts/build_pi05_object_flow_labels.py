"""Build derived PI0.5 object-role and object-flow labels for probe training."""

from __future__ import annotations

import argparse
from pathlib import Path

from vla_lens.pi05.object_flow import save_pi05_object_flow_artifact
from vla_lens.traces import TraceDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Trace dataset root")
    parser.add_argument("--name", default="PI0.5 object flow labels")
    parser.add_argument("--movement-threshold-m", type=float, default=None)
    parser.add_argument("--lift-threshold-m", type=float, default=None)
    parser.add_argument("--contact-threshold-m", type=float, default=None)
    parser.add_argument("--consecutive-frames", type=int, default=None)
    parser.add_argument(
        "--no-rebuild-index",
        action="store_true",
        help="Skip refreshing the dashboard artifact index after saving labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = {}
    if args.movement_threshold_m is not None:
        thresholds["movement_distance_m"] = args.movement_threshold_m
    if args.lift_threshold_m is not None:
        thresholds["lift_z_m"] = args.lift_threshold_m
    if args.contact_threshold_m is not None:
        thresholds["contact_center_distance_m"] = args.contact_threshold_m
    if args.consecutive_frames is not None:
        thresholds["consecutive_frames"] = args.consecutive_frames

    dataset = TraceDataset.open(args.root)
    saved = save_pi05_object_flow_artifact(
        dataset,
        name=args.name,
        thresholds=thresholds,
        rebuild_index=not args.no_rebuild_index,
    )
    print(f"artifact_id={saved.artifact.artifact_id}")
    print(f"artifact_type={saved.artifact.artifact_type}")
    print(f"path={saved.artifact.path}")
    for key, value in dict(saved.artifact.method.get("outputs") or {}).items():
        print(f"{key}={value}")
    print(f"episodes={saved.artifact.metrics['episode_count']}")
    print(f"object_role_rows={len(saved.object_roles)}")
    print(f"interaction_events={len(saved.interaction_events)}")
    print(f"flow_steps={len(saved.flow_steps)}")
    print(f"timestep_labels={len(saved.timestep_labels)}")


if __name__ == "__main__":
    main()
